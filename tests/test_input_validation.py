# Test in Python REPL or create test_validation.py

from backend.app.schemas.tool_requests import MemoryStoreRequest
from pydantic import ValidationError

# Test 1: Valid request
try:
    request = MemoryStoreRequest(
        session_id="user-123",
        content="User prefers email communication",
        content_type="preference"
    )
    print("✓ Valid request accepted")
except ValidationError as e:
    print("✗ Should not fail:", e)

# Test 2: SQL injection attempt (should fail)
try:
    request = MemoryStoreRequest(
        session_id="user'; DROP TABLE memories;--",
        content="Test"
    )
    print("✗ SQL injection not blocked!")
except ValidationError as e:
    print("✓ SQL injection blocked:", e.errors()[0]['msg'])

# Test 3: Path traversal attempt (should fail)
try:
    request = MemoryStoreRequest(
        session_id="../../../etc/passwd",
        content="Test"
    )
    print("✗ Path traversal not blocked!")
except ValidationError as e:
    print("✓ Path traversal blocked:", e.errors()[0]['msg'])

# Test 4: Content sanitization
request = MemoryStoreRequest(
    session_id="user-123",
    content="Test\x00with\x00null\x00bytes   and    extra     spaces",
    content_type="context"
)
print(f"✓ Sanitized content: '{request.content}'")
# Should be: "Test with null bytes and extra spaces"
