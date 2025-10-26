"""
Test suite for agent tools.
"""
import pytest
import asyncio
from pathlib import Path
import tempfile

from app.tools.rag_tool import RAGTool
from app.tools.memory_tool import MemoryTool
from app.tools.attachment_tool import AttachmentTool
from app.tools.escalation_tool import EscalationTool


@pytest.fixture
def rag_tool():
    """Create RAG tool instance."""
    return RAGTool()


@pytest.fixture
def memory_tool():
    """Create memory tool instance."""
    return MemoryTool()


@pytest.fixture
def attachment_tool():
    """Create attachment tool instance."""
    return AttachmentTool()


@pytest.fixture
def escalation_tool():
    """Create escalation tool instance."""
    return EscalationTool()


class TestRAGTool:
    """Test RAG tool functionality."""
    
    @pytest.mark.asyncio
    async def test_search_basic(self, rag_tool):
        """Test basic search functionality."""
        result = await rag_tool.search(
            query="password reset",
            k=3
        )
        
        assert result["query"] == "password reset"
        assert "sources" in result
        assert isinstance(result["sources"], list)
    
    @pytest.mark.asyncio
    async def test_search_with_threshold(self, rag_tool):
        """Test search with similarity threshold."""
        result = await rag_tool.search(
            query="refund policy",
            k=5,
            threshold=0.8
        )
        
        # Only highly relevant results should be returned
        for source in result["sources"]:
            assert source["relevance_score"] >= 0.8
    
    def test_document_chunking(self, rag_tool):
        """Test document chunking functionality."""
        long_text = " ".join(["word"] * 1500)  # 1500 words
        chunks = rag_tool.chunk_document(long_text)
        
        assert len(chunks) > 1
        assert all(len(chunk[0].split()) <= 500 for chunk in chunks)
    
    def test_add_documents(self, rag_tool):
        """Test adding documents to knowledge base."""
        docs = [
            "New policy document about returns.",
            "Updated FAQ about shipping."
        ]
        
        result = rag_tool.add_documents(docs)
        
        assert result["success"]
        assert result["documents_added"] == 2


class TestMemoryTool:
    """Test memory tool functionality."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory_tool):
        """Test storing and retrieving memories."""
        session_id = "test-session-123"
        
        # Store memory
        store_result = await memory_tool.store_memory(
            session_id=session_id,
            content="User prefers email communication",
            content_type="preference",
            importance=0.8
        )
        
        assert store_result["success"]
        assert "memory_id" in store_result
        
        # Retrieve memory
        memories = await memory_tool.retrieve_memories(
            session_id=session_id,
            content_type="preference"
        )
        
        assert len(memories) > 0
        assert memories[0]["content"] == "User prefers email communication"
    
    @pytest.mark.asyncio
    async def test_session_summary(self, memory_tool):
        """Test session summarization."""
        session_id = "test-session-456"
        
        # Store various memories
        await memory_tool.store_memory(
            session_id=session_id,
            content="John Doe",
            content_type="user_info"
        )
        await memory_tool.store_memory(
            session_id=session_id,
            content="Prefers morning calls",
            content_type="preference"
        )
        
        # Get summary
        summary = await memory_tool.summarize_session(session_id)
        
        assert isinstance(summary, str)
        assert "John Doe" in summary or "User Information" in summary
    
    @pytest.mark.asyncio
    async def test_importance_filtering(self, memory_tool):
        """Test filtering by importance."""
        session_id = "test-session-789"
        
        # Store memories with different importance
        await memory_tool.store_memory(
            session_id=session_id,
            content="Low importance fact",
            importance=0.2
        )
        await memory_tool.store_memory(
            session_id=session_id,
            content="High importance fact",
            importance=0.9
        )
        
        # Retrieve only high importance
        memories = await memory_tool.retrieve_memories(
            session_id=session_id,
            min_importance=0.5
        )
        
        assert all(m["importance"] >= 0.5 for m in memories)


class TestAttachmentTool:
    """Test attachment tool functionality."""
    
    @pytest.mark.asyncio
    async def test_process_text_file(self, attachment_tool):
        """Test processing a text file."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document with some content.")
            temp_path = f.name
        
        try:
            result = await attachment_tool.process_attachment(temp_path)
            
            assert result["success"]
            assert "content" in result
            assert "test document" in result["content"]
        finally:
            Path(temp_path).unlink()
    
    def test_file_info(self, attachment_tool):
        """Test getting file information."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"PDF content")
            temp_path = f.name
        
        try:
            info = attachment_tool.get_file_info(temp_path)
            
            assert info["exists"]
            assert info["extension"] == ".pdf"
            assert info["file_type"] == "PDF document"
            assert info["supported"]
        finally:
            Path(temp_path).unlink()
    
    def test_unsupported_file_type(self, attachment_tool):
        """Test handling unsupported file types."""
        info = attachment_tool.get_file_info("test.xyz")
        
        if info["exists"]:  # File might not exist
            assert not info.get("supported", False)


class TestEscalationTool:
    """Test escalation tool functionality."""
    
    @pytest.mark.asyncio
    async def test_keyword_detection(self, escalation_tool):
        """Test escalation keyword detection."""
        result = await escalation_tool.should_escalate(
            message="This is urgent! I need to speak to a manager immediately!"
        )
        
        assert result["escalate"]
        assert result["confidence"] > 0.7
        assert len(result["reasons"]) > 0
    
    @pytest.mark.asyncio
    async def test_sentiment_based_escalation(self, escalation_tool):
        """Test sentiment-based escalation."""
        result = await escalation_tool.should_escalate(
            message="This service is absolutely terrible and disgusting. I hate it!"
        )
        
        assert result["sentiment"] < 0
        assert "Negative sentiment" in str(result["reasons"])
    
    @pytest.mark.asyncio
    async def test_no_escalation_needed(self, escalation_tool):
        """Test when escalation is not needed."""
        result = await escalation_tool.should_escalate(
            message="Thank you for the help. That answers my question."
        )
        
        assert not result["escalate"]
        assert result["confidence"] < 0.7
    
    @pytest.mark.asyncio
    async def test_explicit_escalation_request(self, escalation_tool):
        """Test explicit escalation requests."""
        messages = [
            "I want to speak to a human",
            "Get me a manager",
            "Transfer me to a real person"
        ]
        
        for message in messages:
            result = await escalation_tool.should_escalate(message=message)
            assert result["escalate"], f"Failed for: {message}"
            assert result["confidence"] == 1.0
    
    def test_ticket_creation(self, escalation_tool):
        """Test escalation ticket creation."""
        escalation_result = {
            "escalate": True,
            "priority": "high",
            "category": "complaint",
            "reasons": ["Negative sentiment", "Keywords detected"],
            "urgency": 0.8,
            "sentiment": -0.6
        }
        
        ticket = escalation_tool.create_escalation_ticket(
            session_id="test-123",
            escalation_result=escalation_result
        )
        
        assert "ticket_id" in ticket
        assert ticket["priority"] == "high"
        assert ticket["category"] == "complaint"
        assert ticket["status"] == "pending"


# Integration test combining multiple tools
class TestToolIntegration:
    """Test integration between different tools."""
    
    @pytest.mark.asyncio
    async def test_attachment_to_rag_pipeline(self, attachment_tool, rag_tool):
        """Test processing attachment and indexing in RAG."""
        # Create a test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Important policy: All refunds must be processed within 30 days.")
            temp_path = f.name
        
        try:
            # Process and index
            result = await attachment_tool.extract_and_index(
                file_path=temp_path,
                filename="policy.txt",
                rag_tool=rag_tool
            )
            
            assert result["success"]
            
            # Search for the indexed content
            search_result = await rag_tool.search("refund processing time")
            
            # Should find the policy document
            assert len(search_result["sources"]) > 0
            
        finally:
            Path(temp_path).unlink()
