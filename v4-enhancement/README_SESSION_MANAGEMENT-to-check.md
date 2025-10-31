# Session Management System Documentation

## Overview

The Customer Support Agent now supports **externalized session state** with Redis for multi-instance deployments. This enables:

- ✅ **Horizontal Scaling**: Multiple agent instances share session state
- ✅ **Distributed Locking**: Prevents race conditions across instances
- ✅ **Session Fingerprinting**: Detects session hijacking attempts
- ✅ **Session Encryption**: Protects sensitive session data
- ✅ **L1 Caching**: Reduces Redis round-trips for better performance

---

## Architecture

### Session Store Types

#### 1. **InMemorySessionStore** (Development/Single Instance)
- Fast, simple, no external dependencies
- Sessions lost on restart
- Not shared across instances

#### 2. **RedisSessionStore** (Production/Multi-Instance)
- Persistent, shared across instances
- Automatic expiration with TTL
- Atomic operations with Lua scripts
- Optional L1 cache for performance

---

## Configuration

### Environment Variables

```bash
# Session Store Type
SESSION_STORE_TYPE=redis  # or 'in_memory'

# Redis Connection
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5
REDIS_RETRY_ON_TIMEOUT=true
REDIS_HEALTH_CHECK_INTERVAL=30

# Session Settings
SESSION_TIMEOUT_SECONDS=1800  # 30 minutes
REDIS_SESSION_TTL_SECONDS=3600  # 1 hour
REDIS_SESSION_KEY_PREFIX=agent:session:

# Distributed Locking
SESSION_LOCKING_ENABLED=true
SESSION_LOCK_TIMEOUT_SECONDS=30
SESSION_LOCK_RETRY_ATTEMPTS=3

# Security Features
SESSION_FINGERPRINTING_ENABLED=true
SESSION_FINGERPRINT_STRICT=false  # Set to true to reject mismatched sessions
SESSION_ENCRYPTION_ENABLED=true
SESSION_ENCRYPTION_KEY=your-32-byte-base64-encoded-key-here

# Performance Optimization
SESSION_L1_CACHE_ENABLED=true
SESSION_L1_CACHE_SIZE=1000
SESSION_L1_CACHE_TTL_SECONDS=60
```

---

## Deployment Scenarios

### Development (Single Instance)

```bash
# .env
SESSION_STORE_TYPE=in_memory
SESSION_TIMEOUT_SECONDS=1800
```

```python
# No special setup required
python -m app.main
```

### Production (Multi-Instance with Redis)

```bash
# .env
SESSION_STORE_TYPE=redis
REDIS_URL=redis://redis-cluster:6379/0
SESSION_LOCKING_ENABLED=true
SESSION_FINGERPRINTING_ENABLED=true
SESSION_ENCRYPTION_ENABLED=true
SESSION_ENCRYPTION_KEY=<your-key>
```

#### Docker Compose Example

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  agent-1:
    build: .
    environment:
      - SESSION_STORE_TYPE=redis
      - REDIS_URL=redis://redis:6379/0
      - SESSION_LOCKING_ENABLED=true
      - SESSION_FINGERPRINTING_ENABLED=true
      - SESSION_ENCRYPTION_ENABLED=true
      - SESSION_ENCRYPTION_KEY=${SESSION_ENCRYPTION_KEY}
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - backend

  agent-2:
    build: .
    environment:
      - SESSION_STORE_TYPE=redis
      - REDIS_URL=redis://redis:6379/0
      - SESSION_LOCKING_ENABLED=true
      - SESSION_FINGERPRINTING_ENABLED=true
      - SESSION_ENCRYPTION_ENABLED=true
      - SESSION_ENCRYPTION_KEY=${SESSION_ENCRYPTION_KEY}
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - backend

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - agent-1
      - agent-2
    networks:
      - backend

volumes:
  redis_data:

networks:
  backend:
```

---

## Security Features

### 1. Session Fingerprinting

Prevents session hijacking by binding sessions to client characteristics:

```python
# Automatic fingerprinting on session creation
session_data = await agent.get_or_create_session(
    session_id="abc123",
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0..."
)

# Verification on subsequent requests
# Raises SecurityError if fingerprint doesn't match (strict mode)
```

**Configuration:**
- `SESSION_FINGERPRINTING_ENABLED=true`: Enable feature
- `SESSION_FINGERPRINT_STRICT=true`: Reject mismatched sessions (recommended for production)

### 2. Session Encryption

Encrypts session data at rest in Redis:

```python
# Data is automatically encrypted before storage
# and decrypted on retrieval

# Generate encryption key:
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Configuration:**
- `SESSION_ENCRYPTION_ENABLED=true`: Enable feature
- `SESSION_ENCRYPTION_KEY=<key>`: 32-byte base64-encoded key

### 3. Distributed Locking

Prevents race conditions when multiple instances modify the same session:

```python
# Automatic locking in process_message()
# Lock is acquired, operation performed, lock released

# Manual usage:
lock = await agent._acquire_session_lock(session_id, timeout=30)
try:
    # Critical section
    await session_store.update(session_id, {...})
finally:
    await lock.release()
```

**Configuration:**
- `SESSION_LOCKING_ENABLED=true`: Enable feature
- `SESSION_LOCK_TIMEOUT_SECONDS=30`: Lock expiration
- `SESSION_LOCK_RETRY_ATTEMPTS=3`: Retry on failure

---

## Performance Optimization

### L1 Cache (Local In-Memory Cache)

Reduces Redis round-trips by caching frequently accessed sessions locally:

```
Request → L1 Cache (hit) → Return (fast)
       ↓
       L1 Cache (miss) → Redis → Update L1 → Return
```

**Configuration:**
- `SESSION_L1_CACHE_ENABLED=true`: Enable feature
- `SESSION_L1_CACHE_SIZE=1000`: Max entries
- `SESSION_L1_CACHE_TTL_SECONDS=60`: Cache TTL

**Benefits:**
- 🚀 ~10x faster for cache hits
- 📉 Reduced Redis load
- 🔄 Automatic invalidation on updates

---

## Monitoring & Metrics

### Session Store Stats

```python
# Get session store statistics
stats = await session_store.get_stats()

# Example output:
{
    "store_type": "redis",
    "active_sessions": 1234,
    "redis_version": "7.0.5",
    "connected_clients": 10,
    "used_memory_human": "45.2M",
    "total_commands_processed": 567890,
    "l1_cache_hit_rate": 0.85,  # 85% cache hits
    "l1_cache_size": 850
}
```

### Health Checks

```bash
# Application health check
curl http://localhost:8000/health

# Response includes session store status:
{
    "status": "healthy",
    "session_store": {
        "type": "RedisSessionStore",
        "connected": true,
        "active_sessions": 1234
    },
    "distributed_locking": {
        "enabled": true,
        "active_locks": 5
    }
}
```

---

## Migration Guide

### From In-Memory to Redis

1. **Setup Redis:**
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```

2. **Update Configuration:**
   ```bash
   # .env
   SESSION_STORE_TYPE=redis
   REDIS_URL=redis://localhost:6379/0
   SESSION_LOCKING_ENABLED=true
   ```

3. **Restart Application:**
   ```bash
   # Existing sessions will be lost
   # New sessions will use Redis
   python -m app.main
   ```

4. **Verify:**
   ```bash
   # Check Redis keys
   redis-cli KEYS "agent:session:*"
   
   # Monitor session operations
   redis-cli MONITOR
   ```

---

## Troubleshooting

### Redis Connection Issues

```python
# Check Redis connectivity
redis-cli -h localhost -p 6379 ping
# Should return: PONG

# Check Redis logs
docker logs <redis-container-id>
```

### Session Lock Timeout

```
ERROR: Failed to acquire lock for session abc123: Lock acquisition timeout
```

**Causes:**
- Another instance holding lock too long
- Lock timeout too short

**Solutions:**
- Increase `SESSION_LOCK_TIMEOUT_SECONDS`
- Check for deadlocks in application logic
- Review operation that holds lock

### L1 Cache Inconsistency

If sessions seem stale:

```bash
# Disable L1 cache temporarily
SESSION_L1_CACHE_ENABLED=false
```

Then investigate:
- Cache TTL too long
- Missing invalidation on updates

---

## Best Practices

### Production Deployment

1. ✅ **Use Redis for session store**
2. ✅ **Enable distributed locking**
3. ✅ **Enable session fingerprinting (strict mode)**
4. ✅ **Enable session encryption**
5. ✅ **Use L1 cache for performance**
6. ✅ **Monitor Redis health**
7. ✅ **Set appropriate TTLs**
8. ✅ **Use Redis Sentinel or Cluster for HA**

### Security

1. ✅ **Rotate encryption keys regularly**
2. ✅ **Use strong session IDs (UUIDs)**
3. ✅ **Enable strict fingerprinting in production**
4. ✅ **Monitor for session hijacking attempts**
5. ✅ **Use TLS for Redis connections**

### Performance

1. ✅ **Enable L1 cache**
2. ✅ **Tune Redis maxmemory policy**
3. ✅ **Monitor cache hit rates**
4. ✅ **Use connection pooling**
5. ✅ **Set appropriate lock timeouts**

---

## API Reference

### SessionData

```python
@dataclass
class SessionData:
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_count: int = 0
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    fingerprint_hash: Optional[str] = None  # For security
    
    # Methods
    def create_fingerprint(ip: str, user_agent: str) -> None
    def verify_fingerprint(ip: str, user_agent: str) -> bool
    def to_dict() -> Dict[str, Any]
    def from_dict(data: Dict) -> SessionData
```

### SessionStore Interface

```python
class SessionStore(ABC):
    async def get(session_id: str) -> Optional[SessionData]
    async def set(session_id: str, session_data: SessionData, ttl: int) -> bool
    async def update(session_id: str, updates: Dict, atomic: bool) -> bool
    async def delete(session_id: str) -> bool
    async def exists(session_id: str) -> bool
    async def increment_counter(session_id: str, field: str, delta: int) -> int
    async def get_stats() -> Dict[str, Any]
```

---

## Support

For issues or questions:
1. Check logs: `logs/app.log`
2. Review health endpoint: `/health`
3. Check session stats: `/` (root endpoint)
4. Review this documentation

---

**Version**: 2.0.0  
**Last Updated**: 2024-01-XX  
**Author**: Customer Support Agent Team
