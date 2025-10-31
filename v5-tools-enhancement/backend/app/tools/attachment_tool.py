"""
Attachment processing tool using MarkItDown.
Handles various file formats and extracts content for analysis.

Version: 3.0.0 (Enhanced with input validation and security)

Changes:
- Added Pydantic input validation
- Enhanced file validation
- Improved error handling
- Better security checks
"""
import logging
import os
import tempfile
import hashlib
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiofiles

from pydantic import ValidationError

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MarkItDown = None
    MARKITDOWN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MarkItDown not installed. Attachment processing will be limited.")

from ..config import settings
from ..config.tool_settings import tool_settings
from ..schemas.tool_requests import AttachmentProcessRequest, AttachmentSaveRequest
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    # Documents
    '.pdf': 'PDF document',
    '.docx': 'Word document',
    '.doc': 'Word document',
    '.xlsx': 'Excel spreadsheet',
    '.xls': 'Excel spreadsheet',
    '.pptx': 'PowerPoint presentation',
    '.ppt': 'PowerPoint presentation',
    
    # Text
    '.txt': 'Text file',
    '.md': 'Markdown file',
    '.rtf': 'Rich text file',
    '.csv': 'CSV file',
    '.json': 'JSON file',
    '.xml': 'XML file',
    '.yaml': 'YAML file',
    '.yml': 'YAML file',
    
    # Web
    '.html': 'HTML file',
    '.htm': 'HTML file',
    
    # Images (OCR if available)
    '.jpg': 'JPEG image',
    '.jpeg': 'JPEG image',
    '.png': 'PNG image',
    '.gif': 'GIF image',
    '.bmp': 'Bitmap image',
    
    # Audio (transcription if available)
    '.mp3': 'MP3 audio',
    '.wav': 'WAV audio',
    '.m4a': 'M4A audio',
    '.ogg': 'OGG audio',
}


class AttachmentTool(BaseTool):
    """
    Tool for processing file attachments and extracting content.
    Uses MarkItDown to convert various formats to readable text.
    
    Version 3.0.0:
    - Enhanced with Pydantic input validation
    - Improved file security checks
    - Better error handling
    - Support for multiple file formats
    """
    
    def __init__(self):
        """Initialize attachment processing tool."""
        super().__init__(
            name="attachment_processor",
            description="Process and extract content from uploaded files",
            version="3.0.0"
        )
        
        # Resources initialized in async initialize()
        self.markitdown = None
        self.temp_dir = None
    
    async def initialize(self) -> None:
        """Initialize attachment tool resources (async-safe)."""
        try:
            logger.info(f"Initializing Attachment tool '{self.name}'...")
            
            # Initialize MarkItDown (CPU-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_markitdown
            )
            
            # Create temporary directory
            self.temp_dir = Path(tempfile.gettempdir()) / "cs_agent_attachments"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Attachment temp directory: {self.temp_dir}")
            
            self.initialized = True
            logger.info(
                f"✓ Attachment tool '{self.name}' initialized successfully "
                f"(MarkItDown: {'enabled' if self.markitdown else 'disabled'})"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Attachment tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup attachment tool resources."""
        try:
            logger.info(f"Cleaning up Attachment tool '{self.name}'...")
            
            # Cleanup temporary files
            deleted = await asyncio.get_event_loop().run_in_executor(
                None,
                self.cleanup_temp_files,
                0  # Delete all files
            )
            
            logger.info(f"Cleaned up {deleted} temporary files")
            
            # Clear references
            self.markitdown = None
            self.temp_dir = None
            
            self.initialized = False
            logger.info(f"✓ Attachment tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Attachment tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute attachment processing (async-first).
        
        Version 3.0.0: Added input validation.
        """
        action = kwargs.get("action", "process")
        
        try:
            if action == "process":
                # Validate process request
                try:
                    request = AttachmentProcessRequest(
                        file_path=kwargs.get("file_path", ""),
                        filename=kwargs.get("filename"),
                        extract_metadata=kwargs.get("extract_metadata", True),
                        chunk_for_rag=kwargs.get("chunk_for_rag", False)
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid process request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                result = await self.process_attachment_async(
                    file_path=request.file_path,
                    filename=request.filename,
                    extract_metadata=request.extract_metadata,
                    chunk_for_rag=request.chunk_for_rag
                )
                
                if result["success"]:
                    return ToolResult.success_result(
                        data=result,
                        metadata={
                            "tool": self.name,
                            "filename": result.get("filename"),
                            "word_count": result.get("word_count", 0)
                        }
                    )
                else:
                    return ToolResult.error_result(
                        error=result.get("error", "Unknown error"),
                        metadata={
                            "tool": self.name,
                            "filename": result.get("filename")
                        }
                    )
            
            elif action == "save":
                # Validate save request
                try:
                    request = AttachmentSaveRequest(
                        filename=kwargs.get("filename", "")
                    )
                except ValidationError as e:
                    return ToolResult.error_result(
                        error=f"Invalid save request: {e}",
                        metadata={"tool": self.name, "validation_errors": e.errors()}
                    )
                
                file_data = kwargs.get("file_data")
                if not file_data:
                    return ToolResult.error_result(
                        error="file_data is required for save action",
                        metadata={"tool": self.name}
                    )
                
                saved_path = await self.save_uploaded_file_async(file_data, request.filename)
                
                return ToolResult.success_result(
                    data={"saved_path": saved_path, "filename": request.filename},
                    metadata={"tool": self.name}
                )
            
            else:
                return ToolResult.error_result(
                    error=f"Unknown action: {action}. Valid actions: process, save",
                    metadata={"tool": self.name}
                )
                
        except Exception as e:
            logger.error(f"Attachment execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "action": action}
            )
    
    async def process_attachment_async(
        self,
        file_path: str,
        filename: Optional[str] = None,
        extract_metadata: bool = True,
        chunk_for_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Process an attachment and extract content (async).
        
        Args:
            file_path: Path to the file
            filename: Original filename (optional)
            extract_metadata: Whether to extract file metadata
            chunk_for_rag: Whether to chunk content for RAG indexing
            
        Returns:
            Processing results with extracted content
        """
        # Get file info (sync operation)
        file_info = self.get_file_info(file_path)
        
        if not file_info["exists"]:
            return {
                "success": False,
                "error": "File not found",
                "file_path": file_path
            }
        
        # Use provided filename or extract from path
        if not filename:
            filename = file_info["filename"]
        
        # Check if file is supported
        if not file_info["supported"]:
            return {
                "success": False,
                "error": f"Unsupported file type: {file_info['extension']}",
                "supported_types": list(SUPPORTED_EXTENSIONS.keys()),
                "filename": filename
            }
        
        # Check file size
        max_size = tool_settings.attachment_max_file_size
        if file_info["size_bytes"] > max_size:
            return {
                "success": False,
                "error": f"File too large. Max size: {max_size / (1024*1024):.1f}MB",
                "filename": filename,
                "size_mb": file_info["size_mb"]
            }
        
        try:
            # Process file (CPU-bound, run in thread pool)
            content = await asyncio.get_event_loop().run_in_executor(
                None,
                self._process_file_content,
                file_path,
                file_info
            )
            
            if not content:
                return {
                    "success": False,
                    "error": "Failed to extract content from file",
                    "filename": filename
                }
            
            # Build response
            result = {
                "success": True,
                "filename": filename,
                "file_type": file_info["file_type"],
                "extension": file_info["extension"],
                "size_mb": file_info["size_mb"],
                "content": content,
                "content_length": len(content),
                "word_count": len(content.split())
            }
            
            # Add metadata if requested
            if extract_metadata:
                result["metadata"] = {
                    "processed_with": "MarkItDown" if self.markitdown else "fallback",
                    "file_path": file_path,
                    "original_size": file_info["size_bytes"],
                    "processed_at": asyncio.get_event_loop().time()
                }
            
            # Add chunks if requested (CPU-bound)
            if chunk_for_rag and len(content.split()) > 500:
                chunks = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.chunk_content,
                    content
                )
                result["chunks"] = chunks
                result["chunk_count"] = len(chunks)
            
            # Add preview
            preview_length = min(500, len(content))
            result["preview"] = content[:preview_length] + ("..." if len(content) > preview_length else "")
            
            # Add specific notes based on file type
            if file_info["extension"] in ['.xlsx', '.xls', '.csv']:
                result["note"] = "Tabular data extracted and converted to text format"
            elif file_info["extension"] in ['.jpg', '.jpeg', '.png']:
                result["note"] = "Image processed for text extraction (OCR)"
            elif file_info["extension"] in ['.pdf']:
                result["note"] = "PDF content extracted, formatting may vary"
            elif file_info["extension"] in ['.docx', '.doc']:
                result["note"] = "Word document converted to plain text"
            
            logger.info(
                f"Successfully processed attachment: {filename} "
                f"({result['word_count']} words extracted)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing attachment {filename}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "filename": filename
            }
    
    async def save_uploaded_file_async(
        self,
        file_data: bytes,
        filename: str
    ) -> str:
        """
        Save uploaded file to temporary location (async).
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        # Generate unique filename
        file_hash = hashlib.md5(file_data).hexdigest()[:8]
        safe_filename = f"{file_hash}_{Path(filename).name}"
        temp_path = self.temp_dir / safe_filename
        
        try:
            # Write file asynchronously
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_data)
            
            logger.info(f"Saved uploaded file: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise
    
    async def process_multiple_async(
        self,
        file_paths: List[str],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Process multiple attachments concurrently (async).
        
        Args:
            file_paths: List of file paths
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_attachment_async(file_path)
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "success": False,
                    "error": str(result),
                    "file_path": file_paths[i]
                })
            else:
                formatted_results.append(result)
        
        return formatted_results
    
    async def extract_and_index_async(
        self,
        file_path: str,
        filename: str,
        rag_tool: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract content and optionally index in RAG system (async).
        
        Args:
            file_path: Path to file
            filename: Original filename
            rag_tool: RAG tool instance for indexing
            
        Returns:
            Processing and indexing results
        """
        # Process the attachment
        result = await self.process_attachment_async(
            file_path,
            filename,
            chunk_for_rag=True
        )
        
        if not result["success"]:
            return result
        
        # Index in RAG if tool provided and chunks available
        if rag_tool and "chunks" in result:
            try:
                # Prepare metadata for each chunk
                metadatas = [
                    {
                        "source": filename,
                        "chunk_index": i,
                        "file_type": result["file_type"],
                        "total_chunks": result["chunk_count"]
                    }
                    for i in range(result["chunk_count"])
                ]
                
                # Add to RAG using async method
                index_result = await rag_tool.add_documents_async(
                    documents=result["chunks"],
                    metadatas=metadatas
                )
                
                # Handle ToolResult or dict
                if isinstance(index_result, ToolResult):
                    result["indexed"] = index_result.success
                    result["documents_indexed"] = index_result.data.get("chunks_created", 0)
                else:
                    result["indexed"] = index_result.get("success", False)
                    result["documents_indexed"] = index_result.get("chunks_created", 0)
                
                if result["indexed"]:
                    logger.info(f"Indexed {result['documents_indexed']} chunks from {filename}")
                
            except Exception as e:
                logger.error(f"Failed to index attachment in RAG: {e}")
                result["indexed"] = False
                result["index_error"] = str(e)
        
        return result
    
    # ===========================
    # Private Helper Methods
    # ===========================
    
    def _init_markitdown(self) -> None:
        """Initialize MarkItDown (sync, called in thread pool)."""
        if not MARKITDOWN_AVAILABLE:
            self.markitdown = None
            logger.warning("MarkItDown not available. Limited file processing enabled.")
            return
        
        try:
            self.markitdown = MarkItDown(
                enable_plugins=True
            )
            logger.info("MarkItDown initialized with all plugins")
        except Exception as e:
            logger.warning(f"Failed to initialize MarkItDown with plugins: {e}")
            try:
                self.markitdown = MarkItDown()
                logger.info("MarkItDown initialized without plugins")
            except Exception as e2:
                logger.error(f"Failed to initialize MarkItDown: {e2}")
                self.markitdown = None
    
    def _process_file_content(
        self,
        file_path: str,
        file_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Process file content using available methods (sync).
        Called in thread pool from async method.
        """
        # Try MarkItDown first
        content = self.process_with_markitdown(file_path)
        
        # Fallback for text files
        if not content and file_info["extension"] in ['.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml']:
            content = self.process_text_file(file_path)
        
        return content
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about a file (sync)."""
        path = Path(file_path)
        
        if not path.exists():
            return {
                "exists": False,
                "error": "File not found"
            }
        
        stat = path.stat()
        extension = path.suffix.lower()
        
        return {
            "exists": True,
            "filename": path.name,
            "extension": extension,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "file_type": SUPPORTED_EXTENSIONS.get(extension, "Unknown"),
            "supported": extension in SUPPORTED_EXTENSIONS,
            "modified": stat.st_mtime
        }
    
    def process_with_markitdown(self, file_path: str) -> Optional[str]:
        """Process file with MarkItDown (sync)."""
        if not self.markitdown:
            return None
        
        try:
            result = self.markitdown.convert(file_path)
            
            # Extract text content based on result type
            if hasattr(result, 'text_content'):
                content = result.text_content
            elif hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)
            
            return content
            
        except Exception as e:
            logger.error(f"MarkItDown processing failed: {e}")
            return None
    
    def process_text_file(self, file_path: str) -> Optional[str]:
        """Process plain text files (sync)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file: {e}")
            return None
    
    def chunk_content(
        self,
        content: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """Split content into overlapping chunks for processing (sync)."""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """Clean up old temporary files (sync)."""
        import time
        
        if not self.temp_dir or not self.temp_dir.exists():
            return 0
        
        deleted = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted += 1
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old attachment files")
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
        
        return deleted


__all__ = ['AttachmentTool']
