"""
Tests for async tool contract and ToolResult.
Validates Phase 1 implementation.
"""
import pytest
import asyncio
from app.tools.base_tool import BaseTool, ToolResult, ToolStatus
from app.tools.tool_adapters import (
    sync_to_async_adapter,
    ensure_async,
    AsyncToolAdapter,
    ensure_tool_async
)


# ===========================
# ToolResult Tests
# ===========================

@pytest.mark.unit
def test_tool_result_creation():
    """Test ToolResult creation and fields."""
    result = ToolResult(
        success=True,
        data={"key": "value"},
        metadata={"tool": "test"}
    )
    
    assert result.success is True
    assert result.data == {"key": "value"}
    assert result.metadata == {"tool": "test"}
    assert result.error is None
    assert result.status == ToolStatus.SUCCESS


@pytest.mark.unit
def test_tool_result_error():
    """Test error result creation."""
    result = ToolResult.error_result(
        error="Something went wrong",
        metadata={"tool": "test"}
    )
    
    assert result.success is False
    assert result.error == "Something went wrong"
    assert result.status == ToolStatus.ERROR
    assert result.metadata["tool"] == "test"


@pytest.mark.unit
def test_tool_result_success_helper():
    """Test success result helper."""
    result = ToolResult.success_result(
        data={"count": 5},
        metadata={"source": "cache"}
    )
    
    assert result.success is True
    assert result.data["count"] == 5
    assert result.metadata["source"] == "cache"
    assert result.status == ToolStatus.SUCCESS


@pytest.mark.unit
def test_tool_result_to_dict():
    """Test ToolResult serialization to dict."""
    result = ToolResult(
        success=True,
        data={"key": "value"},
        metadata={"tool": "test"},
        status=ToolStatus.SUCCESS
    )
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert result_dict["success"] is True
    assert result_dict["data"] == {"key": "value"}
    assert result_dict["status"] == "success"


@pytest.mark.unit
def test_tool_result_from_dict():
    """Test ToolResult deserialization from dict."""
    data = {
        "success": True,
        "data": {"key": "value"},
        "metadata": {"tool": "test"},
        "status": "success"
    }
    
    result = ToolResult.from_dict(data)
    
    assert result.success is True
    assert result.data["key"] == "value"
    assert result.status == ToolStatus.SUCCESS


# ===========================
# Adapter Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_to_async_adapter():
    """Test sync function wrapper."""
    def sync_function(x: int, y: int) -> int:
        return x + y
    
    async_function = sync_to_async_adapter(sync_function)
    result = await async_function(5, 3)
    
    assert result == 8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_async_with_sync_function():
    """Test ensure_async decorator with sync function."""
    @ensure_async
    def my_function(x: int) -> int:
        return x * 2
    
    result = await my_function(5)
    assert result == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_async_with_async_function():
    """Test ensure_async decorator with already-async function."""
    @ensure_async
    async def my_function(x: int) -> int:
        return x * 2
    
    result = await my_function(5)
    assert result == 10


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_tool_adapter():
    """Test AsyncToolAdapter with mock sync tool."""
    class MockSyncTool:
        def __init__(self):
            self.name = "mock_sync"
            self.description = "Mock sync tool"
            self.initialized = False
        
        def _initialize(self):
            self.initialized = True
        
        def execute(self, query: str) -> dict:
            return {"result": f"Processed: {query}"}
    
    sync_tool = MockSyncTool()
    async_tool = AsyncToolAdapter(sync_tool)
    
    await async_tool.initialize()
    assert async_tool.initialized is True
    
    result = await async_tool.execute(query="test")
    assert result["result"] == "Processed: test"
    
    await async_tool.cleanup()


@pytest.mark.unit
def test_ensure_tool_async_with_async_tool():
    """Test ensure_tool_async with already-async tool."""
    class AsyncTool(BaseTool):
        async def initialize(self):
            self.initialized = True
        
        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult.success_result(data=kwargs)
        
        async def cleanup(self):
            pass
    
    tool = AsyncTool(name="test", description="Test tool")
    wrapped_tool = ensure_tool_async(tool)
    
    # Should return same instance
    assert wrapped_tool is tool


@pytest.mark.unit
def test_ensure_tool_async_with_sync_tool():
    """Test ensure_tool_async with sync tool."""
    class SyncTool:
        def __init__(self):
            self.name = "sync_tool"
            self.initialized = False
        
        def execute(self, **kwargs):
            return {"data": kwargs}
    
    sync_tool = SyncTool()
    async_tool = ensure_tool_async(sync_tool)
    
    # Should wrap in adapter
    assert isinstance(async_tool, AsyncToolAdapter)
    assert async_tool.name == "sync_tool"


# ===========================
# Integration Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_tool_full_lifecycle():
    """Test complete async tool lifecycle."""
    class TestAsyncTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="test_async_tool",
                description="Test async tool implementation"
            )
            self.setup_called = False
            self.cleanup_called = False
        
        async def initialize(self):
            self.setup_called = True
            self.initialized = True
        
        async def execute(self, **kwargs) -> ToolResult:
            query = kwargs.get("query", "")
            return ToolResult.success_result(
                data={"response": f"Processed: {query}"},
                metadata={"tool": self.name}
            )
        
        async def cleanup(self):
            self.cleanup_called = True
    
    # Create tool
    tool = TestAsyncTool()
    assert tool.setup_called is False
    
    # Initialize
    await tool.initialize()
    assert tool.setup_called is True
    assert tool.initialized is True
    
    # Execute
    result = await tool.execute(query="test query")
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["response"] == "Processed: test query"
    assert result.metadata["tool"] == "test_async_tool"
    
    # Cleanup
    await tool.cleanup()
    assert tool.cleanup_called is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_tool_execution():
    """Test concurrent execution of async tools."""
    class ConcurrentTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="concurrent_test",
                description="Concurrent execution test"
            )
            self.call_count = 0
        
        async def initialize(self):
            self.initialized = True
        
        async def execute(self, **kwargs) -> ToolResult:
            await asyncio.sleep(0.1)  # Simulate async work
            self.call_count += 1
            return ToolResult.success_result(
                data={"call": self.call_count}
            )
        
        async def cleanup(self):
            pass
    
    tool = ConcurrentTool()
    await tool.initialize()
    
    # Execute 5 calls concurrently
    tasks = [tool.execute() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    for result in results:
        assert isinstance(result, ToolResult)
        assert result.success is True
