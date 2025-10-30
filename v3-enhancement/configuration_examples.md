# Configuration Examples

## Retry Configuration:

```Python
retry_config = RetryConfig(
    max_attempts=5,          # Try 5 times total
    wait_multiplier=2.0,     # Double wait time each retry
    wait_min=1.0,            # Start with 1 second
    wait_max=30.0,           # Max 30 seconds between retries
    retry_exceptions=(ConnectionError, TimeoutError)
)
```

## Circuit Breaker Configuration:

```Python
cb_config = CircuitBreakerConfig(
    fail_max=10,             # Open after 10 failures
    timeout=120,             # Try to close after 2 minutes
    expected_exception=Exception,
    name="custom_tool"
)
```
