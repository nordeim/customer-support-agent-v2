# Session Management System - Version 2.0

## Overview

The Customer Support AI Agent now features a production-ready session management system with:

- **Dual Storage Backends**: In-memory (development) and Redis (production)
- **Distributed Locking**: Prevents race conditions in multi-instance deployments
- **Session Encryption**: AES-256 encryption for sensitive session data
- **Session Fingerprinting**: Prevents session hijacking
- **L1 Caching**: In-memory cache reduces Redis load
- **Atomic Operations**: Lua scripts ensure data consistency

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                    │
├─────────────────────────────────────────────────────────┤
│                 CustomerSupportAgent                     │
├─────────────────────────────────────────────────────────┤
│                    SessionStore (Abstract)               │
├──────────────────────┬──────────────────────────────────┤
│ InMemorySessionStore │     RedisSessionStore            │
│  - Development       │  - Production                     │
│  - Single instance   │  - Multi-instance                 │
│  - No persistence    │  - Persistent                     │
│                      │  - L1 Cache                       │
│                      │  - Encryption                     │
│                      │  - Distributed Locks              │
└──────────────────────┴──────────────────────────────────┘
```

## Quick Start

### Development (In-Memory)

```bash
# .env
SESSION_STORE_TYPE=in_memory
SESSION_MAX_SESSIONS=10000
SESSION_TIMEOUT_MINUTES=30
```

### Production (Redis)

```bash
# .env
SESSION_STORE_TYPE=redis
REDIS_URL=redis://localhost:6379/0

# Session Security
SESSION_ENCRYPTION_ENABLED=true
SESSION_ENCRYPTION_KEY=<your-base64-key>
SESSION_FINGERPRINTING_ENABLED=true

# Distributed Locking
SESSION_LOCKING_ENABLED=true
SESSION_LOCK_TIMEOUT_SECONDS=30

# L1 Cache
SESSION_L1_CACHE_ENABLED=true
SESSION_L1_CACHE_SIZE=1000
SESSION_L1_CACHE_TTL_SECONDS=60
```

## Key Features

### 1. Encryption at Rest

Session data is encrypted using Fernet (AES-128 CBC + HMAC):

```python
# Generate encryption key
from cryptography.fernet import Fernet
key = Fernet.generate_key()
print(key.decode())
```

Add to `.env`:
```bash
SESSION_ENCRYPTION_KEY=<generated-key>
```

### 2. Session Fingerprinting

Binds sessions to IP + User-Agent to prevent hijacking:

```python
# Enabled by default
SESSION_FINGERPRINTING_ENABLED=true

# Strict mode (reject on mismatch)
SESSION_FINGERPRINT_STRICT=false
```

### 3. Distributed Locking

Prevents race conditions in concurrent operations:

```python
# Automatic locking in process_message()
async def process_message(self, session_id, message, ...):
    lock = await self._acquire_session_lock(session_id)
    try:
        # Safe concurrent operations
        session_data = await self.get_or_create_session(...)
        new_count = await self.session_store.increment_counter(...)
    finally:
        if lock:
            await lock.release()
```

### 4. L1 Cache

Reduces Redis load with in-memory caching:

```python
# Configuration
SESSION_L1_CACHE_ENABLED=true
SESSION_L1_CACHE_SIZE=1000
SESSION_L1_CACHE_TTL_SECONDS=60

# Cache hit rate monitoring
stats = await agent.session_store.get_stats()
print(stats['l1_cache_size'])
```

## API Usage

### Create/Get Session

```python
from app.agents.chat_agent import CustomerSupportAgent

agent = CustomerSupportAgent()

# Get or create session with fingerprinting
session_data = await agent.get_or_create_session(
    session_id="user-123",
    user_id="user-123",
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0..."
)
```

### Process Message (Thread-Safe)

```python
response = await agent.process_message(
    session_id="user-123",
    message="Help me reset my password",
    ip_address="192.168.1.1",
    user_agent="Mozilla/5.0..."
)

print(response.message)
print(response.tool_metadata['session_stats'])
```

### Manual Session Operations

```python
# Get session
session_data = await agent.session_store.get("user-123")

# Update session
await agent.session_store.update(
    "user-123",
    {"escalated": True},
    atomic=True
)

# Increment counter atomically
new_count = await agent.session_store.increment_counter(
    "user-123",
    "message_count",
    delta=1
)

# Delete session
await agent.session_store.delete("user-123")
```

## Monitoring & Health Checks

### Session Store Health

```python
health = await agent.session_store.health_check()
print(health)
# {
#   "healthy": True,
#   "operations": {"set": True, "get": True, "delete": True},
#   "stats": {...}
# }
```

### Statistics

```python
stats = await agent.session_store.get_stats()
print(stats)
# {
#   "store_type": "redis",
#   "active_sessions": 1234,
#   "l1_cache_enabled": True,
#   "l1_cache_size": 456,
#   "encryption_enabled": True
# }
```

### Periodic Cleanup

Automatic cleanup runs every `SESSION_CLEANUP_INTERVAL_SECONDS`:

```python
# Manual cleanup
cleaned = await agent.session_store.cleanup_expired()
print(f"Cleaned {cleaned} expired sessions")
```

## Deployment

### Single Instance

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - SESSION_STORE_TYPE=in_memory
      - SESSION_MAX_SESSIONS=10000
```

### Multi-Instance with Redis

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  app:
    build: .
    environment:
      - SESSION_STORE_TYPE=redis
      - REDIS_URL=redis://redis:6379/0
      - SESSION_ENCRYPTION_ENABLED=true
      - SESSION_ENCRYPTION_KEY=${SESSION_ENCRYPTION_KEY}
      - SESSION_LOCKING_ENABLED=true
      - SESSION_L1_CACHE_ENABLED=true
    depends_on:
      - redis
    deploy:
      replicas: 3

volumes:
  redis_data:
```

## Performance Tuning

### L1 Cache Sizing

```bash
# High-traffic applications
SESSION_L1_CACHE_SIZE=5000
SESSION_L1_CACHE_TTL_SECONDS=30

# Low-traffic applications
SESSION_L1_CACHE_SIZE=500
SESSION_L1_CACHE_TTL_SECONDS=120
```

### Redis Connection Pool

```bash
REDIS_MAX_CONNECTIONS=100  # High concurrency
REDIS_SOCKET_TIMEOUT=3     # Fast timeout
REDIS_HEALTH_CHECK_INTERVAL=15
```

### Lock Timeout

```bash
SESSION_LOCK_TIMEOUT_SECONDS=30  # Default
SESSION_LOCK_RETRY_ATTEMPTS=3
```

## Troubleshooting

### Session Fingerprint Mismatch

```
WARNING: Session fingerprint mismatch for user-123
```

**Solution**: Client changed IP or User-Agent. Options:
1. Disable strict mode: `SESSION_FINGERPRINT_STRICT=false`
2. Disable fingerprinting: `SESSION_FINGERPRINTING_ENABLED=false`

### Redis Connection Failed

```
ERROR: Redis ping failed: Connection refused
```

**Solution**:
1. Check Redis is running: `redis-cli ping`
2. Verify REDIS_URL in `.env`
3. Check firewall/network

### Lock Acquisition Failed

```
ERROR: Failed to acquire lock for session user-123
```

**Solution**:
1. Increase timeout: `SESSION_LOCK_TIMEOUT_SECONDS=60`
2. Check Redis health
3. Review lock contention in logs

## Security Best Practices

1. **Always enable encryption in production**
   ```bash
   SESSION_ENCRYPTION_ENABLED=true
   ```

2. **Use strong encryption keys**
   ```bash
   # Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   SESSION_ENCRYPTION_KEY=<64-char-base64-key>
   ```

3. **Enable fingerprinting**
   ```bash
   SESSION_FINGERPRINTING_ENABLED=true
   SESSION_FINGERPRINT_STRICT=true  # Production
   ```

4. **Secure Redis**
   ```bash
   REDIS_PASSWORD=strong-password
   # Use TLS: rediss://...
   ```

5. **Set appropriate TTLs**
   ```bash
   SESSION_TIMEOUT_MINUTES=30  # Not too long
   REDIS_SESSION_TTL_SECONDS=1800
   ```

## Migration Guide

### From v1.0 to v2.0

1. **Update dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Update `.env`**
   ```bash
   cp .env .env.backup
   cp .env.example .env
   # Merge your settings
   ```

3. **Add encryption key**
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   # Add to .env as SESSION_ENCRYPTION_KEY
   ```

4. **Test locally**
   ```bash
   SESSION_STORE_TYPE=in_memory python -m app.main
   ```

5. **Deploy to production**
   ```bash
   SESSION_STORE_TYPE=redis docker-compose up -d
   ```

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/customer-support-ai
- Documentation: https://docs.your-org.com
- Email: support@your-org.com
