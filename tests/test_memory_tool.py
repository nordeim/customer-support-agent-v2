import asyncio
from backend.app.tools.memory_tool import MemoryTool

async def test_memory_tool():
    tool = MemoryTool()
    await tool.initialize()
    
    # Test 1: Valid store
    result = await tool.store_memory_async(
        session_id="test-123",
        content="User likes coffee",
        content_type="preference"
    )
    print("✓ Valid store:", result['success'])
    
    # Test 2: Invalid session_id (should fail)
    result = await tool.store_memory_async(
        session_id="bad'; DROP TABLE--",
        content="Test"
    )
    print("✓ Invalid session_id blocked:", not result['success'])
    
    # Test 3: Retrieve
    memories = await tool.retrieve_memories_async(
        session_id="test-123"
    )
    print(f"✓ Retrieved {len(memories)} memories")
    
    await tool.cleanup()

asyncio.run(test_memory_tool())
