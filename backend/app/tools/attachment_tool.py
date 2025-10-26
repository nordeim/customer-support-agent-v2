"""
Attachment processing tool using MarkItDown.
Handles various file formats and extracts content for analysis.
"""
import logging
import os
import tempfile
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import aiofiles
import asyncio

try:
    from markitdown import MarkItDown
except ImportError:
    logger.warning("MarkItDown not installed. Attachment processing will be limited.")
    MarkItDown = None

from ..config import settings
from .base_tool import BaseTool

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
    """
    
    def __init__(self):
        """Initialize attachment processing tool."""
        super().__init__(
            name="attachment_processor",
            description="Process and extract content from uploaded files"
        )
    
    def _setup(self) -> None:
        """Setup MarkItDown and temporary directory."""
        # Initialize MarkItDown if available
        if MarkItDown:
            try:
                self.markitdown = MarkItDown(
                    enable_plugins=True  # Enable all available plugins
                )
                logger.info("MarkItDown initialized with all plugins")
            except Exception as e:
                logger.warning(f"Failed to initialize MarkItDown with plugins: {e}")
                self.markitdown = MarkItDown() if MarkItDown else None
        else:
            self.markitdown = None
            logger.warning("MarkItDown not available. Limited file processing enabled.")
        
        # Create temporary directory for file processing
        self.temp_dir = Path(tempfile.gettempdir()) / "cs_agent_attachments"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attachment temp directory: {self.temp_dir}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information dictionary
        """
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
    
    async def save_uploaded_file(
        self,
        file_data: bytes,
        filename: str
    ) -> str:
        """
        Save uploaded file to temporary location.
        
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
    
    def process_with_markitdown(
        self,
        file_path: str
    ) -> Optional[str]:
        """
        Process file with MarkItDown.
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text content or None if failed
        """
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
        """
        Process plain text files.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content or None if failed
        """
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
        """
        Split content into overlapping chunks for processing.
        
        Args:
            content: Text content to chunk
            chunk_size: Size of each chunk in words
            overlap: Overlap between chunks in words
            
        Returns:
            List of text chunks
        """
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    async def process_attachment(
        self,
        file_path: str,
        filename: Optional[str] = None,
        extract_metadata: bool = True,
        chunk_for_rag: bool = False
    ) -> Dict[str, Any]:
        """
        Process an attachment and extract content.
        
        Args:
            file_path: Path to the file
            filename: Original filename (optional)
            extract_metadata: Whether to extract file metadata
            chunk_for_rag: Whether to chunk content for RAG indexing
            
        Returns:
            Processing results with extracted content
        """
        # Get file info
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
        if file_info["size_bytes"] > settings.max_file_size:
            return {
                "success": False,
                "error": f"File too large. Max size: {settings.max_file_size / (1024*1024)}MB",
                "filename": filename,
                "size_mb": file_info["size_mb"]
            }
        
        try:
            # Try MarkItDown first
            content = self.process_with_markitdown(file_path)
            
            # Fallback for text files
            if not content and file_info["extension"] in ['.txt', '.md', '.csv', '.json', '.xml']:
                content = self.process_text_file(file_path)
            
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
                    "original_size": file_info["size_bytes"]
                }
            
            # Add chunks if requested
            if chunk_for_rag and len(content.split()) > 500:
                chunks = self.chunk_content(content)
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
                "filename": filename
            }
    
    async def process_multiple(
        self,
        file_paths: List[str],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Process multiple attachments concurrently.
        
        Args:
            file_paths: List of file paths
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_attachment(file_path)
        
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def extract_and_index(
        self,
        file_path: str,
        filename: str,
        rag_tool: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract content and optionally index in RAG system.
        
        Args:
            file_path: Path to file
            filename: Original filename
            rag_tool: RAG tool instance for indexing
            
        Returns:
            Processing and indexing results
        """
        # Process the attachment
        result = await self.process_attachment(
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
                
                # Add to RAG
                index_result = rag_tool.add_documents(
                    documents=result["chunks"],
                    metadatas=metadatas
                )
                
                result["indexed"] = index_result.get("success", False)
                result["documents_indexed"] = index_result.get("chunks_created", 0)
                
                if result["indexed"]:
                    logger.info(f"Indexed {result['documents_indexed']} chunks from {filename}")
                
            except Exception as e:
                logger.error(f"Failed to index attachment in RAG: {e}")
                result["indexed"] = False
                result["index_error"] = str(e)
        
        return result
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Delete files older than this
            
        Returns:
            Number of files deleted
        """
        import time
        
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
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute attachment processing.
        
        Accepts:
            file_path: Path to file (required)
            filename: Original filename (optional)
            extract_metadata: Whether to include metadata (default: True)
            chunk_for_rag: Whether to chunk for RAG (default: False)
            
        Returns:
            Processing results
        """
        file_path = kwargs.get("file_path")
        
        if not file_path:
            return {
                "success": False,
                "error": "file_path is required"
            }
        
        return await self.process_attachment(
            file_path=file_path,
            filename=kwargs.get("filename"),
            extract_metadata=kwargs.get("extract_metadata", True),
            chunk_for_rag=kwargs.get("chunk_for_rag", False)
        )
    
    async def cleanup(self) -> None:
        """Cleanup temporary files on shutdown."""
        deleted = self.cleanup_temp_files(max_age_hours=0)
        logger.info(f"Attachment tool cleanup: {deleted} files removed")
