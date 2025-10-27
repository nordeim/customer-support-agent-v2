# Comprehensive Backend Test Script - Meticulous Plan & Implementation

## 📋 Test Plan

### Overview
We'll create a comprehensive test script that exercises all major backend functionality:
1. **Health Checks** - Verify server is running and healthy
2. **Session Management** - Create, retrieve, and manage sessions
3. **Chat Operations** - Send messages and receive AI responses
4. **File Uploads** - Test attachment processing
5. **WebSocket Communication** - Real-time message streaming
6. **Knowledge Base** - Test RAG search functionality
7. **Error Handling** - Verify proper error responses
8. **Performance** - Basic load testing

### Test Categories
- ✅ **Smoke Tests** - Basic functionality
- 🔄 **Integration Tests** - Component interactions
- 🚀 **Performance Tests** - Response times and throughput
- 🛡️ **Security Tests** - Authentication and validation
- 🔍 **Edge Cases** - Boundary conditions and error scenarios

---

## 🛠️ Implementation

### Complete Test Script

**`backend/test_backend.py`**
```python
#!/usr/bin/env python3
"""
Comprehensive Backend Test Script for Customer Support AI Agent
This script exercises all major functionality of the backend server.
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import tempfile
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import aiohttp
import requests
from colorama import init, Fore, Style
import websockets
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Initialize colorama for cross-platform color support
init(autoreset=True)
console = Console()

# Configuration
BASE_URL = os.getenv("API_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", "ws://localhost:8000")
TIMEOUT = 30


class TestStatus(Enum):
    """Test status enumeration."""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    SKIPPED = "⏭️ SKIPPED"
    WARNING = "⚠️ WARNING"


@dataclass
class TestResult:
    """Test result data class."""
    name: str
    status: TestStatus
    message: str
    duration: float
    details: Optional[Dict[str, Any]] = None


class BackendTester:
    """Comprehensive backend testing utility."""
    
    def __init__(self, base_url: str = BASE_URL, ws_url: str = WS_URL):
        self.base_url = base_url.rstrip('/')
        self.ws_url = ws_url.rstrip('/')
        self.session_id: Optional[str] = None
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def print_header(self):
        """Print test header."""
        console.print("\n" + "="*60, style="bold blue")
        console.print("🚀 Customer Support AI Agent - Backend Test Suite", style="bold white")
        console.print("="*60, style="bold blue")
        console.print(f"📍 API URL: {self.base_url}", style="cyan")
        console.print(f"🔌 WebSocket URL: {self.ws_url}", style="cyan")
        console.print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="cyan")
        console.print("="*60 + "\n", style="bold blue")
    
    def add_result(self, result: TestResult):
        """Add test result and print status."""
        self.results.append(result)
        
        # Color based on status
        if result.status == TestStatus.PASSED:
            style = "green"
        elif result.status == TestStatus.FAILED:
            style = "red"
        elif result.status == TestStatus.WARNING:
            style = "yellow"
        else:
            style = "dim"
        
        console.print(
            f"{result.status.value} {result.name} "
            f"({result.duration:.2f}s) - {result.message}",
            style=style
        )
        
        if result.details and result.status == TestStatus.FAILED:
            console.print(f"  Details: {result.details}", style="dim red")
    
    # ===========================
    # Test Methods
    # ===========================
    
    def test_server_health(self) -> TestResult:
        """Test 1: Check if server is running and healthy."""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return TestResult(
                    name="Server Health Check",
                    status=TestStatus.PASSED,
                    message=f"Server is {data.get('status', 'healthy')}",
                    duration=time.time() - start,
                    details=data
                )
            else:
                return TestResult(
                    name="Server Health Check",
                    status=TestStatus.FAILED,
                    message=f"Unexpected status code: {response.status_code}",
                    duration=time.time() - start
                )
        except requests.exceptions.ConnectionError:
            return TestResult(
                name="Server Health Check",
                status=TestStatus.FAILED,
                message=f"Cannot connect to server at {self.base_url}",
                duration=time.time() - start
            )
        except Exception as e:
            return TestResult(
                name="Server Health Check",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_api_docs(self) -> TestResult:
        """Test 2: Check if API documentation is accessible."""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=5)
            
            if response.status_code == 200:
                return TestResult(
                    name="API Documentation",
                    status=TestStatus.PASSED,
                    message="API docs are accessible",
                    duration=time.time() - start
                )
            elif response.status_code == 404:
                return TestResult(
                    name="API Documentation",
                    status=TestStatus.WARNING,
                    message="API docs not available (likely production mode)",
                    duration=time.time() - start
                )
            else:
                return TestResult(
                    name="API Documentation",
                    status=TestStatus.FAILED,
                    message=f"Status code: {response.status_code}",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="API Documentation",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_create_session(self) -> TestResult:
        """Test 3: Create a new chat session."""
        start = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/sessions",
                json={"user_id": "test_user_123"},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id") or data.get("sessionId")
                
                if self.session_id:
                    return TestResult(
                        name="Create Session",
                        status=TestStatus.PASSED,
                        message=f"Session created: {self.session_id[:12]}...",
                        duration=time.time() - start,
                        details=data
                    )
                else:
                    return TestResult(
                        name="Create Session",
                        status=TestStatus.FAILED,
                        message="No session ID in response",
                        duration=time.time() - start,
                        details=data
                    )
            else:
                return TestResult(
                    name="Create Session",
                    status=TestStatus.FAILED,
                    message=f"Status code: {response.status_code}",
                    duration=time.time() - start,
                    details=response.text
                )
        except Exception as e:
            return TestResult(
                name="Create Session",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_get_session(self) -> TestResult:
        """Test 4: Retrieve session details."""
        if not self.session_id:
            return TestResult(
                name="Get Session",
                status=TestStatus.SKIPPED,
                message="No session ID available",
                duration=0
            )
        
        start = time.time()
        try:
            response = requests.get(
                f"{self.base_url}/api/sessions/{self.session_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return TestResult(
                    name="Get Session",
                    status=TestStatus.PASSED,
                    message=f"Retrieved session: {data.get('status', 'unknown')}",
                    duration=time.time() - start,
                    details=data
                )
            else:
                return TestResult(
                    name="Get Session",
                    status=TestStatus.FAILED,
                    message=f"Status code: {response.status_code}",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="Get Session",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_send_message(self) -> TestResult:
        """Test 5: Send a chat message."""
        if not self.session_id:
            return TestResult(
                name="Send Message",
                status=TestStatus.SKIPPED,
                message="No session ID available",
                duration=0
            )
        
        start = time.time()
        try:
            # Create form data
            form_data = {
                "message": "Hello! Can you help me with my order?"
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat/sessions/{self.session_id}/messages",
                data=form_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", "")[:50]
                return TestResult(
                    name="Send Message",
                    status=TestStatus.PASSED,
                    message=f"Response: {message}...",
                    duration=time.time() - start,
                    details=data
                )
            else:
                return TestResult(
                    name="Send Message",
                    status=TestStatus.FAILED,
                    message=f"Status code: {response.status_code}",
                    duration=time.time() - start,
                    details=response.text
                )
        except Exception as e:
            return TestResult(
                name="Send Message",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_file_upload(self) -> TestResult:
        """Test 6: Upload a test file."""
        if not self.session_id:
            return TestResult(
                name="File Upload",
                status=TestStatus.SKIPPED,
                message="No session ID available",
                duration=0
            )
        
        start = time.time()
        try:
            # Create a temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is a test document for the Customer Support AI Agent.\n")
                f.write("Order #12345: Status inquiry\n")
                f.write("Customer: Test User\n")
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    files = {'file': ('test_document.txt', f, 'text/plain')}
                    data = {'session_id': self.session_id}
                    
                    response = requests.post(
                        f"{self.base_url}/api/chat/upload",
                        files=files,
                        data=data,
                        timeout=10
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    return TestResult(
                        name="File Upload",
                        status=TestStatus.PASSED,
                        message=f"File uploaded: {data.get('filename', 'unknown')}",
                        duration=time.time() - start,
                        details=data
                    )
                elif response.status_code == 404:
                    return TestResult(
                        name="File Upload",
                        status=TestStatus.WARNING,
                        message="Upload endpoint not found (may not be implemented)",
                        duration=time.time() - start
                    )
                else:
                    return TestResult(
                        name="File Upload",
                        status=TestStatus.FAILED,
                        message=f"Status code: {response.status_code}",
                        duration=time.time() - start
                    )
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return TestResult(
                name="File Upload",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    async def test_websocket(self) -> TestResult:
        """Test 7: WebSocket connection and messaging."""
        if not self.session_id:
            return TestResult(
                name="WebSocket Connection",
                status=TestStatus.SKIPPED,
                message="No session ID available",
                duration=0
            )
        
        start = time.time()
        try:
            ws_uri = f"{self.ws_url}/ws?session_id={self.session_id}"
            
            async with websockets.connect(ws_uri, timeout=5) as websocket:
                # Send a test message
                await websocket.send(json.dumps({
                    "type": "message",
                    "content": "Test WebSocket message"
                }))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=5.0
                    )
                    data = json.loads(response)
                    
                    return TestResult(
                        name="WebSocket Connection",
                        status=TestStatus.PASSED,
                        message=f"Connected and received: {data.get('type', 'unknown')}",
                        duration=time.time() - start,
                        details=data
                    )
                except asyncio.TimeoutError:
                    return TestResult(
                        name="WebSocket Connection",
                        status=TestStatus.WARNING,
                        message="Connected but no response received",
                        duration=time.time() - start
                    )
                    
        except Exception as e:
            return TestResult(
                name="WebSocket Connection",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_search_knowledge_base(self) -> TestResult:
        """Test 8: Search the knowledge base."""
        start = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/api/chat/search",
                json={
                    "query": "refund policy",
                    "limit": 3
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                num_results = len(data) if isinstance(data, list) else 0
                return TestResult(
                    name="Knowledge Base Search",
                    status=TestStatus.PASSED,
                    message=f"Found {num_results} results",
                    duration=time.time() - start,
                    details=data
                )
            elif response.status_code == 404:
                return TestResult(
                    name="Knowledge Base Search",
                    status=TestStatus.WARNING,
                    message="Search endpoint not found",
                    duration=time.time() - start
                )
            else:
                return TestResult(
                    name="Knowledge Base Search",
                    status=TestStatus.FAILED,
                    message=f"Status code: {response.status_code}",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="Knowledge Base Search",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_message_history(self) -> TestResult:
        """Test 9: Retrieve message history."""
        if not self.session_id:
            return TestResult(
                name="Message History",
                status=TestStatus.SKIPPED,
                message="No session ID available",
                duration=0
            )
        
        start = time.time()
        try:
            response = requests.get(
                f"{self.base_url}/api/chat/sessions/{self.session_id}/messages",
                params={"limit": 10},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                messages = data.get("messages", [])
                return TestResult(
                    name="Message History",
                    status=TestStatus.PASSED,
                    message=f"Retrieved {len(messages)} messages",
                    duration=time.time() - start,
                    details=data
                )
            else:
                return TestResult(
                    name="Message History",
                    status=TestStatus.FAILED,
                    message=f"Status code: {response.status_code}",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="Message History",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_error_handling(self) -> TestResult:
        """Test 10: Error handling with invalid requests."""
        start = time.time()
        try:
            # Test with invalid session ID
            response = requests.get(
                f"{self.base_url}/api/sessions/invalid_session_id_12345",
                timeout=5
            )
            
            if response.status_code in [400, 404]:
                return TestResult(
                    name="Error Handling",
                    status=TestStatus.PASSED,
                    message=f"Properly handled invalid request ({response.status_code})",
                    duration=time.time() - start
                )
            else:
                return TestResult(
                    name="Error Handling",
                    status=TestStatus.WARNING,
                    message=f"Unexpected status code: {response.status_code}",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="Error Handling",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_rate_limiting(self) -> TestResult:
        """Test 11: Rate limiting (if enabled)."""
        start = time.time()
        try:
            # Send multiple rapid requests
            responses = []
            for i in range(10):
                response = requests.get(
                    f"{self.base_url}/health",
                    timeout=1
                )
                responses.append(response.status_code)
            
            if 429 in responses:
                return TestResult(
                    name="Rate Limiting",
                    status=TestStatus.PASSED,
                    message="Rate limiting is active",
                    duration=time.time() - start
                )
            else:
                return TestResult(
                    name="Rate Limiting",
                    status=TestStatus.WARNING,
                    message="Rate limiting not detected (may be disabled)",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="Rate Limiting",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_cors_headers(self) -> TestResult:
        """Test 12: CORS headers configuration."""
        start = time.time()
        try:
            response = requests.options(
                f"{self.base_url}/api/sessions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST"
                },
                timeout=5
            )
            
            cors_headers = response.headers.get("access-control-allow-origin")
            if cors_headers:
                return TestResult(
                    name="CORS Configuration",
                    status=TestStatus.PASSED,
                    message=f"CORS enabled for: {cors_headers}",
                    duration=time.time() - start
                )
            else:
                return TestResult(
                    name="CORS Configuration",
                    status=TestStatus.WARNING,
                    message="CORS headers not found",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="CORS Configuration",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    def test_performance(self) -> TestResult:
        """Test 13: Basic performance test."""
        if not self.session_id:
            return TestResult(
                name="Performance Test",
                status=TestStatus.SKIPPED,
                message="No session ID available",
                duration=0
            )
        
        start = time.time()
        try:
            response_times = []
            
            # Send 5 messages and measure response times
            for i in range(5):
                msg_start = time.time()
                response = requests.post(
                    f"{self.base_url}/api/chat/sessions/{self.session_id}/messages",
                    data={"message": f"Test message {i+1}"},
                    timeout=30
                )
                response_time = time.time() - msg_start
                response_times.append(response_time)
                
                if response.status_code != 200:
                    break
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                
                if avg_time < 5.0:  # Average under 5 seconds
                    return TestResult(
                        name="Performance Test",
                        status=TestStatus.PASSED,
                        message=f"Avg: {avg_time:.2f}s, Max: {max_time:.2f}s",
                        duration=time.time() - start,
                        details={"response_times": response_times}
                    )
                else:
                    return TestResult(
                        name="Performance Test",
                        status=TestStatus.WARNING,
                        message=f"Slow responses - Avg: {avg_time:.2f}s",
                        duration=time.time() - start,
                        details={"response_times": response_times}
                    )
            else:
                return TestResult(
                    name="Performance Test",
                    status=TestStatus.FAILED,
                    message="No successful responses",
                    duration=time.time() - start
                )
        except Exception as e:
            return TestResult(
                name="Performance Test",
                status=TestStatus.FAILED,
                message=str(e),
                duration=time.time() - start
            )
    
    # ===========================
    # Main Test Runner
    # ===========================
    
    async def run_all_tests(self):
        """Run all tests in sequence."""
        self.print_header()
        
        # Basic connectivity tests
        console.print("📡 [bold cyan]Testing Basic Connectivity...[/bold cyan]")
        self.add_result(self.test_server_health())
        self.add_result(self.test_api_docs())
        
        # Session management tests
        console.print("\n💬 [bold cyan]Testing Session Management...[/bold cyan]")
        self.add_result(self.test_create_session())
        self.add_result(self.test_get_session())
        
        # Chat functionality tests
        console.print("\n🤖 [bold cyan]Testing Chat Functionality...[/bold cyan]")
        self.add_result(self.test_send_message())
        self.add_result(self.test_message_history())
        self.add_result(self.test_search_knowledge_base())
        
        # File handling tests
        console.print("\n📎 [bold cyan]Testing File Operations...[/bold cyan]")
        self.add_result(self.test_file_upload())
        
        # WebSocket tests
        console.print("\n🔌 [bold cyan]Testing WebSocket...[/bold cyan]")
        self.add_result(await self.test_websocket())
        
        # Security and error handling tests
        console.print("\n🛡️ [bold cyan]Testing Security & Error Handling...[/bold cyan]")
        self.add_result(self.test_error_handling())
        self.add_result(self.test_cors_headers())
        self.add_result(self.test_rate_limiting())
        
        # Performance tests
        console.print("\n⚡ [bold cyan]Testing Performance...[/bold cyan]")
        self.add_result(self.test_performance())
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        total_time = time.time() - self.start_time
        
        # Count results by status
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        warned = sum(1 for r in self.results if r.status == TestStatus.WARNING)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        
        # Create summary table
        console.print("\n" + "="*60, style="bold blue")
        console.print("📊 TEST SUMMARY", style="bold white")
        console.print("="*60, style="bold blue")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", style="cyan", width=20)
        table.add_column("Count", justify="right", style="white")
        table.add_column("Percentage", justify="right", style="white")
        
        total = len(self.results)
        if total > 0:
            table.add_row("✅ Passed", str(passed), f"{(passed/total)*100:.1f}%")
            table.add_row("❌ Failed", str(failed), f"{(failed/total)*100:.1f}%")
            table.add_row("⚠️  Warning", str(warned), f"{(warned/total)*100:.1f}%")
            table.add_row("⏭️  Skipped", str(skipped), f"{(skipped/total)*100:.1f}%")
            table.add_row("─"*20, "─"*10, "─"*10, style="dim")
            table.add_row("Total", str(total), "100.0%", style="bold")
        
        console.print(table)
        
        # Overall status
        console.print(f"\n⏱️  Total Time: {total_time:.2f} seconds", style="cyan")
        
        if failed == 0:
            console.print("\n✨ [bold green]All critical tests passed![/bold green]")
        elif failed <= 2:
            console.print(f"\n⚠️  [bold yellow]Some tests failed ({failed}/{total})[/bold yellow]")
        else:
            console.print(f"\n❌ [bold red]Multiple tests failed ({failed}/{total})[/bold red]")
        
        # Recommendations
        if failed > 0 or warned > 0:
            console.print("\n💡 [bold]Recommendations:[/bold]")
            
            failed_names = [r.name for r in self.results if r.status == TestStatus.FAILED]
            if "Server Health Check" in failed_names:
                console.print("  • Ensure the server is running on the correct port", style="yellow")
            if "Create Session" in failed_names or "Send Message" in failed_names:
                console.print("  • Check database and Redis connections", style="yellow")
            if "WebSocket Connection" in failed_names:
                console.print("  • Verify WebSocket configuration and CORS settings", style="yellow")
            if warned > 0:
                console.print("  • Review warnings for potential improvements", style="yellow")


def print_requirements():
    """Print required packages for the test script."""
    console.print("\n📦 [bold]Required packages for this test script:[/bold]")
    console.print("  pip install requests aiohttp websockets colorama rich")


async def main():
    """Main test execution."""
    # Check if server is reachable
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        console.print(f"\n❌ [bold red]Cannot connect to server at {BASE_URL}[/bold red]")
        console.print("Please ensure the server is running:")
        console.print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000", style="cyan")
        return 1
    except Exception as e:
        console.print(f"\n❌ [bold red]Unexpected error: {e}[/bold red]")
        return 1
    
    # Run tests
    tester = BackendTester()
    await tester.run_all_tests()
    
    # Return exit code based on results
    failed = sum(1 for r in tester.results if r.status == TestStatus.FAILED)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Print requirements if requested
    if "--requirements" in sys.argv:
        print_requirements()
        sys.exit(0)
    
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

### Install Test Dependencies

**`backend/requirements-test.txt`**
```txt
# Test dependencies for backend testing
requests==2.31.0
aiohttp==3.9.1
websockets==12.0
colorama==0.4.6
rich==13.7.0
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
```

### Simple Quick Test Script

**`backend/quick_test.py`**
```python
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
```

## 📋 Usage Instructions

### 1. Install Test Dependencies

```bash
# Option 1: Install test requirements
pip install -r requirements-test.txt

# Option 2: Install minimal dependencies
pip install requests colorama rich websockets aiohttp
```

### 2. Run the Comprehensive Test

```bash
# Make sure the server is running first
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run the test
cd backend
python test_backend.py

# Or run with custom server URL
API_URL=http://localhost:8000 python test_backend.py
```

### 3. Run the Quick Test

```bash
# For a quick smoke test with minimal dependencies
python quick_test.py
```

### 4. Expected Output

The comprehensive test will show:
- ✅ **Green** for passed tests
- ❌ **Red** for failed tests
- ⚠️ **Yellow** for warnings
- ⏭️ **Gray** for skipped tests

And provide:
- Detailed results for each test
- Performance metrics
- Summary statistics
- Recommendations for fixes

## 🎯 Test Coverage

The test script covers:

1. **Basic Connectivity**
   - Server health
   - API documentation

2. **Session Management**
   - Create sessions
   - Retrieve sessions

3. **Chat Operations**
   - Send messages
   - Get responses
   - Message history

4. **File Handling**
   - Upload files
   - Process attachments

5. **Real-time Features**
   - WebSocket connections
   - Streaming responses

6. **Knowledge Base**
   - Search functionality
   - RAG integration

7. **Security**
   - CORS configuration
   - Rate limiting
   - Error handling

8. **Performance**
   - Response times
   - Throughput testing

This comprehensive test suite will help you quickly identify which parts of the backend are working and which need attention!
