"""
Tests for Session Store implementations (Phase 4).
Validates both InMemorySessionStore and RedisSessionStore.
"""
import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List

from app.session import SessionStore, SessionData, InMemorySessionStore

# Conditionally import Redis store
try:
    from app.session import RedisSessionStore, REDIS_AVAILABLE
except ImportError:
    RedisSessionStore = None
    REDIS_AVAILABLE = False


# ===========================
# Fixtures
# ===========================

@pytest.fixture
async def in_memory_store():
    """Create in-memory session store for testing."""
    store = InMemorySessionStore(
        max_sessions=100,
        default_ttl=300  # 5 minutes for tests
    )
    yield store
    # Cleanup
    await store.cleanup_expired()


@pytest.fixture
async def redis_store():
    """Create Redis session store for testing (if available)."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available")
    
    store = RedisSessionStore(
        redis_url="redis://localhost:6379/15",  # Use test DB
        key_prefix="test:session:",
        default_ttl=300
    )
    
    # Test connection
    if not await store.ping():
        pytest.skip("Redis not running")
    
    yield store
    
    # Cleanup - delete all test sessions
    session_ids = await store.list_active()
    for session_id in session_ids:
        await store.delete(session_id)
    
    await store.close()


@pytest.fixture(params=["in_memory"])
def session_store(request, in_memory_store, redis_store):
    """Parametrized fixture to test both store implementations."""
    if request.param == "in_memory":
        return in_memory_store
    elif request.param == "redis" and REDIS_AVAILABLE:
        return redis_store
    else:
        pytest.skip(f"Store type {request.param} not available")


# ===========================
# SessionData Tests
# ===========================

@pytest.mark.unit
def test_session_data_creation():
    """Test SessionData creation."""
    session_data = SessionData(
        session_id="test-session-001",
        user_id="user-123",
        thread_id=str(uuid.uuid4())
    )
    
    assert session_data.session_id == "test-session-001"
    assert session_data.user_id == "user-123"
    assert session_data.message_count == 0
    assert session_data.escalated is False
    assert isinstance(session_data.metadata, dict)


@pytest.mark.unit
def test_session_data_to_dict():
    """Test SessionData serialization to dict."""
    now = datetime.utcnow()
    session_data = SessionData(
        session_id="test-session-002",
        message_count=5,
        created_at=now,
        updated_at=now
    )
    
    data_dict = session_data.to_dict()
    
    assert isinstance(data_dict, dict)
    assert data_dict["session_id"] == "test-session-002"
    assert data_dict["message_count"] == 5
    assert isinstance(data_dict["created_at"], str)  # ISO format


@pytest.mark.unit
def test_session_data_from_dict():
    """Test SessionData deserialization from dict."""
    data_dict = {
        "session_id": "test-session-003",
        "user_id": "user-456",
        "message_count": 10,
        "escalated": True,
        "created_at": "2024-01-01T12:00:00",
        "metadata": {"custom": "value"}
    }
    
    session_data = SessionData.from_dict(data_dict)
    
    assert session_data.session_id == "test-session-003"
    assert session_data.user_id == "user-456"
    assert session_data.message_count == 10
    assert session_data.escalated is True
    assert isinstance(session_data.created_at, datetime)


@pytest.mark.unit
def test_session_data_json_serialization():
    """Test SessionData JSON serialization/deserialization."""
    original = SessionData(
        session_id="test-session-004",
        user_id="user-789",
        message_count=3,
        metadata={"key": "value"}
    )
    
    # Serialize to JSON
    json_str = original.to_json()
    assert isinstance(json_str, str)
    
    # Deserialize from JSON
    restored = SessionData.from_json(json_str)
    
    assert restored.session_id == original.session_id
    assert restored.user_id == original.user_id
    assert restored.message_count == original.message_count
    assert restored.metadata == original.metadata


# ===========================
# InMemorySessionStore Tests
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_set_and_get(in_memory_store):
    """Test setting and getting session data."""
    session_id = "test-session-100"
    session_data = SessionData(
        session_id=session_id,
        user_id="user-100"
    )
    
    # Set session
    result = await in_memory_store.set(session_id, session_data)
    assert result is True
    
    # Get session
    retrieved = await in_memory_store.get(session_id)
    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert retrieved.user_id == "user-100"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_update(in_memory_store):
    """Test updating session data."""
    session_id = "test-session-101"
    session_data = SessionData(session_id=session_id, message_count=0)
    
    await in_memory_store.set(session_id, session_data)
    
    # Update session
    result = await in_memory_store.update(
        session_id,
        {"message_count": 5, "user_id": "user-101"}
    )
    assert result is True
    
    # Verify update
    updated = await in_memory_store.get(session_id)
    assert updated.message_count == 5
    assert updated.user_id == "user-101"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_delete(in_memory_store):
    """Test deleting session data."""
    session_id = "test-session-102"
    session_data = SessionData(session_id=session_id)
    
    await in_memory_store.set(session_id, session_data)
    
    # Delete session
    result = await in_memory_store.delete(session_id)
    assert result is True
    
    # Verify deletion
    retrieved = await in_memory_store.get(session_id)
    assert retrieved is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_exists(in_memory_store):
    """Test checking session existence."""
    session_id = "test-session-103"
    
    # Should not exist initially
    exists = await in_memory_store.exists(session_id)
    assert exists is False
    
    # Create session
    session_data = SessionData(session_id=session_id)
    await in_memory_store.set(session_id, session_data)
    
    # Should exist now
    exists = await in_memory_store.exists(session_id)
    assert exists is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_increment_counter(in_memory_store):
    """Test atomic counter increment."""
    session_id = "test-session-104"
    session_data = SessionData(session_id=session_id, message_count=0)
    
    await in_memory_store.set(session_id, session_data)
    
    # Increment counter
    new_value = await in_memory_store.increment_counter(
        session_id,
        "message_count",
        delta=1
    )
    assert new_value == 1
    
    # Increment again
    new_value = await in_memory_store.increment_counter(
        session_id,
        "message_count",
        delta=1
    )
    assert new_value == 2
    
    # Verify in session data
    updated = await in_memory_store.get(session_id)
    assert updated.message_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_list_active(in_memory_store):
    """Test listing active sessions."""
    # Create multiple sessions
    session_ids = [f"test-session-{i}" for i in range(200, 205)]
    
    for session_id in session_ids:
        session_data = SessionData(session_id=session_id)
        await in_memory_store.set(session_id, session_data)
    
    # List active sessions
    active = await in_memory_store.list_active()
    
    assert len(active) >= 5
    for session_id in session_ids:
        assert session_id in active


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_count_active(in_memory_store):
    """Test counting active sessions."""
    # Create sessions
    for i in range(210, 215):
        session_data = SessionData(session_id=f"test-session-{i}")
        await in_memory_store.set(session_id=f"test-session-{i}", session_data=session_data)
    
    # Count active
    count = await in_memory_store.count_active()
    assert count >= 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_ttl_expiration(in_memory_store):
    """Test TTL-based expiration."""
    session_id = "test-session-300"
    session_data = SessionData(session_id=session_id)
    
    # Set session with short TTL
    await in_memory_store.set(session_id, session_data, ttl=1)
    
    # Should exist immediately
    exists = await in_memory_store.exists(session_id)
    assert exists is True
    
    # Wait for expiration
    await asyncio.sleep(2)
    
    # Should be expired now
    retrieved = await in_memory_store.get(session_id)
    assert retrieved is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_cleanup_expired(in_memory_store):
    """Test cleanup of expired sessions."""
    # Create sessions with short TTL
    for i in range(400, 405):
        session_data = SessionData(session_id=f"test-session-{i}")
        await in_memory_store.set(f"test-session-{i}", session_data, ttl=1)
    
    # Wait for expiration
    await asyncio.sleep(2)
    
    # Cleanup
    cleaned = await in_memory_store.cleanup_expired()
    assert cleaned >= 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_max_sessions_eviction(in_memory_store):
    """Test LRU eviction when max sessions reached."""
    # Create store with small max
    small_store = InMemorySessionStore(max_sessions=5, default_ttl=300)
    
    # Add 10 sessions (should evict oldest)
    for i in range(10):
        session_data = SessionData(session_id=f"evict-session-{i}")
        await small_store.set(f"evict-session-{i}", session_data)
    
    # First sessions should be evicted
    exists = await small_store.exists("evict-session-0")
    assert exists is False
    
    # Recent sessions should exist
    exists = await small_store.exists("evict-session-9")
    assert exists is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_in_memory_store_get_stats(in_memory_store):
    """Test getting store statistics."""
    # Create some sessions
    for i in range(500, 505):
        session_data = SessionData(session_id=f"test-session-{i}")
        await in_memory_store.set(f"test-session-{i}", session_data)
    
    # Get stats
    stats = await in_memory_store.get_stats()
    
    assert stats["store_type"] == "in_memory"
    assert "total_sessions" in stats
    assert "active_sessions" in stats
    assert stats["active_sessions"] >= 5


# ===========================
# RedisSessionStore Tests
# ===========================

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_set_and_get(redis_store):
    """Test Redis store set and get."""
    session_id = "redis-test-100"
    session_data = SessionData(
        session_id=session_id,
        user_id="redis-user-100"
    )
    
    # Set session
    result = await redis_store.set(session_id, session_data)
    assert result is True
    
    # Get session
    retrieved = await redis_store.get(session_id)
    assert retrieved is not None
    assert retrieved.session_id == session_id
    assert retrieved.user_id == "redis-user-100"


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_update(redis_store):
    """Test Redis store update."""
    session_id = "redis-test-101"
    session_data = SessionData(session_id=session_id, message_count=0)
    
    await redis_store.set(session_id, session_data)
    
    # Update
    result = await redis_store.update(
        session_id,
        {"message_count": 10, "escalated": True}
    )
    assert result is True
    
    # Verify
    updated = await redis_store.get(session_id)
    assert updated.message_count == 10
    assert updated.escalated is True


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_increment_counter(redis_store):
    """Test Redis store atomic counter increment."""
    session_id = "redis-test-102"
    session_data = SessionData(session_id=session_id, message_count=0)
    
    await redis_store.set(session_id, session_data)
    
    # Increment using Lua script
    new_value = await redis_store.increment_counter(
        session_id,
        "message_count",
        delta=1
    )
    assert new_value == 1
    
    # Increment again
    new_value = await redis_store.increment_counter(
        session_id,
        "message_count",
        delta=5
    )
    assert new_value == 6


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_delete(redis_store):
    """Test Redis store delete."""
    session_id = "redis-test-103"
    session_data = SessionData(session_id=session_id)
    
    await redis_store.set(session_id, session_data)
    
    # Delete
    result = await redis_store.delete(session_id)
    assert result is True
    
    # Verify
    retrieved = await redis_store.get(session_id)
    assert retrieved is None


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_list_active(redis_store):
    """Test Redis store list active sessions."""
    # Create sessions
    session_ids = [f"redis-list-{i}" for i in range(10)]
    
    for session_id in session_ids:
        session_data = SessionData(session_id=session_id)
        await redis_store.set(session_id, session_data)
    
    # List active
    active = await redis_store.list_active()
    
    for session_id in session_ids:
        assert session_id in active


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_ttl_auto_expiration(redis_store):
    """Test Redis TTL auto-expiration."""
    session_id = "redis-ttl-test"
    session_data = SessionData(session_id=session_id)
    
    # Set with short TTL
    await redis_store.set(session_id, session_data, ttl=2)
    
    # Should exist immediately
    exists = await redis_store.exists(session_id)
    assert exists is True
    
    # Wait for expiration
    await asyncio.sleep(3)
    
    # Should be auto-expired by Redis
    exists = await redis_store.exists(session_id)
    assert exists is False


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_ping(redis_store):
    """Test Redis store connection ping."""
    result = await redis_store.ping()
    assert result is True


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_store_get_stats(redis_store):
    """Test Redis store statistics."""
    # Create some sessions
    for i in range(20):
        session_data = SessionData(session_id=f"redis-stats-{i}")
        await redis_store.set(f"redis-stats-{i}", session_data)
    
    # Get stats
    stats = await redis_store.get_stats()
    
    assert stats["store_type"] == "redis"
    assert "redis_version" in stats
    assert "active_sessions" in stats
    assert stats["active_sessions"] >= 20


# ===========================
# Concurrency Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_increments(in_memory_store):
    """Test concurrent counter increments (in-memory)."""
    session_id = "concurrent-test-100"
    session_data = SessionData(session_id=session_id, message_count=0)
    
    await in_memory_store.set(session_id, session_data)
    
    # Run 10 concurrent increments
    tasks = [
        in_memory_store.increment_counter(session_id, "message_count", delta=1)
        for _ in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Final value should be 10
    final_data = await in_memory_store.get(session_id)
    assert final_data.message_count == 10


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_increments_redis(redis_store):
    """Test concurrent counter increments (Redis with Lua script)."""
    session_id = "redis-concurrent-100"
    session_data = SessionData(session_id=session_id, message_count=0)
    
    await redis_store.set(session_id, session_data)
    
    # Run 50 concurrent increments (stress test)
    tasks = [
        redis_store.increment_counter(session_id, "message_count", delta=1)
        for _ in range(50)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All increments should succeed atomically
    final_data = await redis_store.get(session_id)
    assert final_data.message_count == 50


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_sessions(in_memory_store):
    """Test handling multiple concurrent sessions."""
    session_count = 20
    session_ids = [f"multi-session-{i}" for i in range(session_count)]
    
    # Create sessions concurrently
    async def create_session(session_id):
        session_data = SessionData(session_id=session_id)
        await in_memory_store.set(session_id, session_data)
        return session_id
    
    tasks = [create_session(sid) for sid in session_ids]
    created = await asyncio.gather(*tasks)
    
    assert len(created) == session_count
    
    # Verify all exist
    for session_id in session_ids:
        exists = await in_memory_store.exists(session_id)
        assert exists is True


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_instance_simulation_redis(redis_store):
    """Simulate multi-instance access to same session (Redis)."""
    session_id = "multi-instance-session"
    
    # Instance 1 creates session
    session_data = SessionData(session_id=session_id, message_count=0)
    await redis_store.set(session_id, session_data)
    
    # Instance 2 increments counter
    await redis_store.increment_counter(session_id, "message_count", delta=1)
    
    # Instance 3 increments counter
    await redis_store.increment_counter(session_id, "message_count", delta=1)
    
    # Instance 1 reads final value
    final_data = await redis_store.get(session_id)
    
    # Should see both increments from other instances
    assert final_data.message_count == 2


# ===========================
# Edge Cases & Error Handling
# ===========================

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_nonexistent_session(in_memory_store):
    """Test getting non-existent session."""
    retrieved = await in_memory_store.get("nonexistent-session")
    assert retrieved is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_nonexistent_session(in_memory_store):
    """Test updating non-existent session."""
    result = await in_memory_store.update(
        "nonexistent-session",
        {"message_count": 5}
    )
    assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_nonexistent_session(in_memory_store):
    """Test deleting non-existent session."""
    result = await in_memory_store.delete("nonexistent-session")
    assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_increment_nonexistent_session(in_memory_store):
    """Test incrementing counter on non-existent session."""
    result = await in_memory_store.increment_counter(
        "nonexistent-session",
        "message_count",
        delta=1
    )
    assert result == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_data_with_complex_metadata(in_memory_store):
    """Test session with complex metadata."""
    session_id = "complex-metadata-session"
    session_data = SessionData(
        session_id=session_id,
        metadata={
            "user_preferences": {"theme": "dark", "language": "en"},
            "custom_fields": ["field1", "field2"],
            "nested": {"deep": {"value": 123}}
        }
    )
    
    await in_memory_store.set(session_id, session_data)
    
    # Retrieve and verify
    retrieved = await in_memory_store.get(session_id)
    assert retrieved.metadata["user_preferences"]["theme"] == "dark"
    assert retrieved.metadata["nested"]["deep"]["value"] == 123


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_connection_failure_handling():
    """Test handling of Redis connection failures."""
    # Create store with invalid URL
    bad_store = RedisSessionStore(
        redis_url="redis://localhost:9999",  # Invalid port
        key_prefix="bad:",
        default_ttl=300
    )
    
    # Operations should fail gracefully
    session_data = SessionData(session_id="test")
    
    # Set should return False on connection error
    result = await bad_store.set("test", session_data)
    assert result is False
    
    # Get should return None on connection error
    retrieved = await bad_store.get("test")
    assert retrieved is None


# ===========================
# Performance Tests
# ===========================

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_in_memory_store_performance():
    """Test in-memory store performance with many sessions."""
    import time
    
    store = InMemorySessionStore(max_sessions=1000, default_ttl=300)
    
    # Create 500 sessions
    start = time.time()
    
    for i in range(500):
        session_data = SessionData(session_id=f"perf-session-{i}")
        await store.set(f"perf-session-{i}", session_data)
    
    create_time = time.time() - start
    
    # Read 500 sessions
    start = time.time()
    
    for i in range(500):
        await store.get(f"perf-session-{i}")
    
    read_time = time.time() - start
    
    # Performance assertions
    assert create_time < 1.0  # Should create 500 sessions in < 1 second
    assert read_time < 0.5  # Should read 500 sessions in < 0.5 seconds
    
    print(f"\nPerformance: Created 500 sessions in {create_time:.3f}s, read in {read_time:.3f}s")


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.slow
async def test_redis_store_performance(redis_store):
    """Test Redis store performance with many operations."""
    import time
    
    # Create 100 sessions
    start = time.time()
    
    for i in range(100):
        session_data = SessionData(session_id=f"redis-perf-{i}")
        await redis_store.set(f"redis-perf-{i}", session_data)
    
    create_time = time.time() - start
    
    # Increment counters
    start = time.time()
    
    tasks = [
        redis_store.increment_counter(f"redis-perf-{i}", "message_count", 1)
        for i in range(100)
    ]
    await asyncio.gather(*tasks)
    
    increment_time = time.time() - start
    
    print(f"\nRedis Performance: Created 100 sessions in {create_time:.3f}s, "
          f"100 atomic increments in {increment_time:.3f}s")
