#!/usr/bin/env python3
"""
Quick backend test - minimal dependencies version
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_backend():
    """Quick test of backend functionality."""
    print("🚀 Quick Backend Test")
    print("=" * 40)
    
    # Test 1: Health check
    print("1. Testing health endpoint...", end=" ")
    try:
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200:
            print("✅ PASSED")
        else:
            print(f"❌ FAILED (status: {r.status_code})")
    except Exception as e:
        print(f"❌ FAILED ({e})")
        return False
    
    # Test 2: Create session
    print("2. Creating session...", end=" ")
    try:
        r = requests.post(f"{BASE_URL}/api/sessions", json={"user_id": "test"})
        if r.status_code == 200:
            session_id = r.json().get("session_id") or r.json().get("sessionId")
            print(f"✅ PASSED (ID: {session_id[:12]}...)")
        else:
            print(f"❌ FAILED (status: {r.status_code})")
            return False
    except Exception as e:
        print(f"❌ FAILED ({e})")
        return False
    
    # Test 3: Send message
    print("3. Sending message...", end=" ")
    try:
        r = requests.post(
            f"{BASE_URL}/api/chat/sessions/{session_id}/messages",
            data={"message": "Hello!"}
        )
        if r.status_code == 200:
            response = r.json().get("message", "")[:30]
            print(f"✅ PASSED (Response: {response}...)")
        else:
            print(f"❌ FAILED (status: {r.status_code})")
    except Exception as e:
        print(f"❌ FAILED ({e})")
    
    print("=" * 40)
    print("✨ Basic tests completed!")
    return True

if __name__ == "__main__":
    try:
        success = test_backend()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
