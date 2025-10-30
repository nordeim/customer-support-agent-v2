# Get statistics
stats = await agent.session_store.get_stats()

# In-Memory:
{
    "store_type": "in_memory",
    "total_sessions": 1234,
    "active_sessions": 987,
    "expired_sessions": 247,
    "max_sessions": 10000,
    "utilization": "12.3%"
}

# Redis:
{
    "store_type": "redis",
    "active_sessions": 5678,
    "redis_version": "7.2.3",
    "connected_clients": 10,
    "used_memory_human": "2.5M",
    "total_commands_processed": 123456
}
