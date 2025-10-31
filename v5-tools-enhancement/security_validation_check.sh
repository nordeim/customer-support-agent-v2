# Create test_security.py
cat > test_security.py << 'EOF'
from backend.app.schemas.tool_requests import MemoryStoreRequest
from pydantic import ValidationError

def test_sql_injection():
    """Test SQL injection prevention."""
    payloads = [
        "'; DROP TABLE memories;--",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM users--"
    ]
    
    for payload in payloads:
        try:
            MemoryStoreRequest(session_id=payload, content="test")
            print(f"✗ FAILED: {payload} was accepted!")
            return False
        except ValidationError:
            print(f"✓ Blocked: {payload}")
    
    return True

def test_path_traversal():
    """Test path traversal prevention."""
    payloads = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "~/../../etc/shadow"
    ]
    
    for payload in payloads:
        try:
            MemoryStoreRequest(session_id=payload, content="test")
            print(f"✗ FAILED: {payload} was accepted!")
            return False
        except ValidationError:
            print(f"✓ Blocked: {payload}")
    
    return True

def test_xss_prevention():
    """Test XSS prevention in content."""
    payloads = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>"
    ]
    
    for payload in payloads:
        try:
            request = MemoryStoreRequest(
                session_id="test-123",
                content=payload
            )
            # Content should be sanitized
            if '<' in request.content or 'script' in request.content.lower():
                print(f"✗ FAILED: XSS not sanitized: {payload}")
                return False
            print(f"✓ Sanitized: {payload}")
        except ValidationError as e:
            print(f"✓ Rejected: {payload}")
    
    return True

if __name__ == "__main__":
    print("Running security tests...\n")
    
    print("1. SQL Injection Tests:")
    assert test_sql_injection(), "SQL injection tests failed"
    
    print("\n2. Path Traversal Tests:")
    assert test_path_traversal(), "Path traversal tests failed"
    
    print("\n3. XSS Prevention Tests:")
    assert test_xss_prevention(), "XSS prevention tests failed"
    
    print("\n✅ All security tests passed!")
EOF

python test_security.py
