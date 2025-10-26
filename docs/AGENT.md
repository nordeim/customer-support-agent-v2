# 🤖 AGENT.md - AI Agent Development Guide

## Executive Summary

This document provides comprehensive guidance for AI coding agents (Claude Code, Codex, GitHub Copilot, etc.) to independently implement and enhance the Customer Support AI Agent system. This is a production-ready, microservices-based application combining React frontend, FastAPI backend, Microsoft Agent Framework, and RAG-powered knowledge retrieval.

**Project Status**: Architecture designed, ready for implementation  
**Current Phase**: Initial development  
**Target Completion**: 20 business days  

## 🎯 Project Context & Objectives

### What You're Building

A sophisticated customer support AI system that:
- Provides intelligent, context-aware responses using Microsoft Agent Framework
- Implements RAG (Retrieval-Augmented Generation) with Google's EmbeddingGemma model
- Processes diverse document formats via MarkItDown
- Maintains conversation memory using SQLite
- Offers real-time chat via WebSockets
- Scales horizontally with Docker/Kubernetes

### Success Criteria

Your implementation is successful when:
1. ✅ All tests pass with >80% coverage
2. ✅ Docker Compose stack runs without errors
3. ✅ API endpoints respond correctly per specifications
4. ✅ Frontend displays chat interface and processes messages
5. ✅ Agent successfully uses all tools (RAG, Memory, Attachment, Escalation)
6. ✅ Monitoring dashboards show healthy metrics

## 🏗️ Project Structure & Architecture

### Directory Structure with Implementation Status

```
customer-support-ai-agent/
├── backend/                        [TO IMPLEMENT]
│   ├── app/
│   │   ├── __init__.py            [Priority: 1]
│   │   ├── main.py                [Priority: 1] - FastAPI entry point
│   │   ├── config.py              [Priority: 1] - Settings management
│   │   ├── agents/
│   │   │   ├── chat_agent.py     [Priority: 2] - Core agent logic
│   │   │   └── agent_factory.py  [Priority: 3]
│   │   ├── tools/
│   │   │   ├── rag_tool.py       [Priority: 2] - ChromaDB integration
│   │   │   ├── memory_tool.py    [Priority: 2] - SQLite persistence
│   │   │   ├── attachment_tool.py [Priority: 3] - MarkItDown
│   │   │   └── escalation_tool.py [Priority: 4]
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── chat.py       [Priority: 2]
│   │   │   │   └── sessions.py   [Priority: 2]
│   │   │   └── websocket.py      [Priority: 3]
│   │   ├── models/
│   │   │   ├── session.py        [Priority: 2]
│   │   │   └── message.py        [Priority: 2]
│   │   └── services/
│   │       ├── embedding_service.py [Priority: 2]
│   │       └── cache_service.py  [Priority: 3]
│   ├── requirements.txt           [Priority: 1]
│   └── Dockerfile                  [Priority: 4]
├── frontend/                       [TO IMPLEMENT]
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatInterface.tsx  [Priority: 2]
│   │   ├── hooks/
│   │   │   └── useChat.ts        [Priority: 2]
│   │   └── App.tsx                [Priority: 1]
│   ├── package.json               [Priority: 1]
│   └── Dockerfile                 [Priority: 4]
├── docker-compose.yml             [Priority: 3]
└── .env.example                   [Priority: 1]
```

### Architecture Patterns to Follow

```python
# Pattern 1: Async Service Pattern
class ServiceName:
    """Service docstring with purpose"""
    
    def __init__(self):
        """Initialize with dependency injection"""
        self.dependencies = self._initialize_dependencies()
    
    async def operation(self, params: dict) -> dict:
        """Async operation with type hints"""
        try:
            result = await self._process(params)
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            return {"success": False, "error": str(e)}

# Pattern 2: Tool Implementation Pattern
class ToolName:
    """Microsoft Agent Framework tool"""
    
    def __init__(self):
        self.name = "tool_name"
        self.description = "Tool purpose"
    
    async def __call__(self, **kwargs) -> dict:
        """Make tool callable for agent framework"""
        return await self.execute(**kwargs)
    
    async def execute(self, **kwargs) -> dict:
        """Tool logic implementation"""
        pass
```

## 📋 Implementation Phases

### Phase 1: Foundation (Days 1-3) ✅ READY TO START

#### Task 1.1: Project Setup
```bash
# Commands to execute
mkdir customer-support-ai-agent
cd customer-support-ai-agent
git init

# Create directory structure
mkdir -p backend/app/{agents,tools,api/routes,models,services,utils}
mkdir -p frontend/src/{components,hooks,services,types}
mkdir -p monitoring scripts

# Initialize backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create requirements.txt with exact versions
cat > requirements.txt << 'EOF'
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
sqlalchemy==2.0.35
alembic==1.13.3
chromadb==0.5.20
sentence-transformers==3.2.1
redis==5.2.0
markitdown==0.1.0
opentelemetry-api==1.29.0
opentelemetry-instrumentation-fastapi==0.49b0
prometheus-client==0.21.0
pydantic==2.9.2
pydantic-settings==2.6.1
python-dotenv==1.0.1
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
EOF

pip install -r requirements.txt
```

#### Task 1.2: Configuration Implementation

**File: `backend/app/config.py`**
```python
"""Application configuration management using Pydantic Settings."""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden via environment variables.
    """
    
    # Application
    app_name: str = "Customer Support AI Agent"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"  # development, staging, production
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"
    cors_origins: List[str] = ["http://localhost:3000"]
    
    # Security
    secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Database
    database_url: str = "sqlite:///./customer_support.db"
    database_echo: bool = False
    
    # Redis Cache
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 3600
    cache_enabled: bool = True
    
    # ChromaDB Vector Store
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "customer_support_docs"
    
    # Embedding Configuration
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    
    # Agent Configuration
    agent_model: str = "gpt-4o-mini"
    agent_temperature: float = 0.7
    agent_max_tokens: int = 2000
    agent_timeout: int = 30
    
    # OpenAI / Azure OpenAI
    openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_api_version: str = "2024-10-01-preview"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # Monitoring
    enable_telemetry: bool = True
    otlp_endpoint: str = "http://localhost:4317"
    metrics_enabled: bool = True
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".txt", ".md", ".csv", ".json",
        ".jpg", ".jpeg", ".png"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def validate_settings(self) -> None:
        """Validate critical settings on startup."""
        if not self.secret_key or self.secret_key == "change-this-in-production":
            if self.environment == "production":
                raise ValueError("SECRET_KEY must be set in production")
        
        if not self.openai_api_key and not self.azure_openai_endpoint:
            raise ValueError("Either OPENAI_API_KEY or Azure OpenAI config required")

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.validate_settings()
    return settings

# Global settings instance
settings = get_settings()
```

### Phase 2: Backend Core (Days 4-8)

#### Task 2.1: Main Application

**File: `backend/app/main.py`**
```python
"""FastAPI application entry point with middleware and lifecycle management."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Any, Dict

from .config import settings
from .api.routes import chat, sessions, health
from .api.websocket import websocket_endpoint
from .agents.chat_agent import CustomerSupportAgent
from .utils.telemetry import setup_telemetry
from .utils.middleware import RequestIDMiddleware, TimingMiddleware

# Configure structured logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(request_id)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if settings.environment != 'development' else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance: CustomerSupportAgent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    Initialize resources on startup, cleanup on shutdown.
    """
    global agent_instance
    
    # Startup
    try:
        logger.info(f"Starting {settings.app_name} v{settings.version}")
        logger.info(f"Environment: {settings.environment}")
        
        # Initialize telemetry if enabled
        if settings.enable_telemetry:
            setup_telemetry(app)
            logger.info("Telemetry initialized")
        
        # Initialize the AI agent
        agent_instance = CustomerSupportAgent()
        app.state.agent = agent_instance
        logger.info("Customer Support Agent initialized successfully")
        
        # Perform health checks
        await perform_startup_checks()
        
        logger.info("Application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Cleanup resources
    if agent_instance:
        await agent_instance.cleanup()
    
    logger.info("Application shutdown complete")

async def perform_startup_checks():
    """Perform critical health checks on startup."""
    checks = []
    
    # Check database connection
    try:
        from .models import get_db
        async with get_db() as db:
            await db.execute("SELECT 1")
        checks.append("Database: ✓")
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        checks.append("Database: ✗")
    
    # Check Redis if enabled
    if settings.cache_enabled:
        try:
            from .services.cache_service import CacheService
            cache = CacheService()
            await cache.ping()
            checks.append("Redis: ✓")
        except Exception as e:
            logger.warning(f"Redis check failed: {e}")
            checks.append("Redis: ✗")
    
    logger.info(f"Startup checks: {', '.join(checks)}")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered customer support system with RAG and conversation memory",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# Add custom middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimingMiddleware)

# Include API routes
app.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    sessions.router,
    prefix=f"{settings.api_prefix}/sessions",
    tags=["sessions"]
)

app.include_router(
    chat.router,
    prefix=f"{settings.api_prefix}/chat",
    tags=["chat"]
)

# Add WebSocket endpoint
app.add_api_websocket_route("/ws", websocket_endpoint)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions gracefully."""
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "docs": "/docs" if settings.debug else None,
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
        }
    )
```

#### Task 2.2: Agent Implementation

**File: `backend/app/agents/chat_agent.py`**
```python
"""Customer Support Agent implementation using Microsoft Agent Framework."""
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import logging
import json

# Note: These imports assume Microsoft Agent Framework packages
# Adjust based on actual package availability
try:
    from agent_framework import ChatAgent, AgentThread
    from agent_framework.azure import AzureOpenAIChatClient
    from agent_framework.openai import OpenAIChatClient
except ImportError:
    # Fallback for development without actual packages
    ChatAgent = object
    AgentThread = object
    AzureOpenAIChatClient = object
    OpenAIChatClient = object

from ..config import settings
from ..tools.rag_tool import RAGTool
from ..tools.memory_tool import MemoryTool
from ..tools.attachment_tool import AttachmentTool
from ..tools.escalation_tool import EscalationTool

logger = logging.getLogger(__name__)

class CustomerSupportAgent:
    """
    Production-ready customer support agent using Microsoft Agent Framework.
    Orchestrates multiple tools for intelligent customer support.
    """
    
    # System prompt defining agent behavior
    SYSTEM_PROMPT = """You are an expert customer support AI assistant with the following capabilities:

    TOOLS AVAILABLE:
    1. **rag_search**: Search our knowledge base for relevant information
    2. **memory_management**: Store and retrieve conversation context
    3. **attachment_processor**: Analyze uploaded documents
    4. **escalation_check**: Determine if human intervention is needed

    GUIDELINES:
    - Be helpful, professional, and empathetic in all interactions
    - Provide accurate information based on available knowledge
    - Always cite sources when using information from the knowledge base
    - Admit when you don't know something rather than guessing
    - Suggest escalation to human support when:
      * The issue is complex or requires human judgment
      * The customer expresses frustration or dissatisfaction
      * Legal or compliance matters are involved
      * You cannot find relevant information after searching
    
    CONVERSATION FLOW:
    1. Understand the customer's query completely
    2. Search for relevant information if needed
    3. Check conversation history for context
    4. Provide a clear, helpful response
    5. Offer additional assistance

    Remember: Customer satisfaction is the top priority."""
    
    def __init__(self):
        """Initialize the customer support agent."""
        self.agent = None
        self.tools = []
        self.threads: Dict[str, AgentThread] = {}
        self.message_history: Dict[str, List[Dict]] = {}
        
        # Initialize agent on creation
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """
        Initialize the agent with appropriate client and tools.
        Handles both OpenAI and Azure OpenAI configurations.
        """
        try:
            # Initialize chat client based on configuration
            chat_client = self._create_chat_client()
            
            # Initialize all tools
            self.tools = self._initialize_tools()
            
            # Create the agent
            self.agent = self._create_agent(chat_client)
            
            logger.info(
                f"Agent initialized with {len(self.tools)} tools: "
                f"{[tool.name for tool in self.tools]}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise
    
    def _create_chat_client(self):
        """Create appropriate chat client based on configuration."""
        if settings.azure_openai_endpoint:
            logger.info("Using Azure OpenAI client")
            return AzureOpenAIChatClient(
                endpoint=settings.azure_openai_endpoint,
                deployment_name=settings.azure_openai_deployment,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version
            )
        elif settings.openai_api_key:
            logger.info("Using OpenAI client")
            return OpenAIChatClient(
                api_key=settings.openai_api_key,
                model=settings.agent_model
            )
        else:
            # Fallback for development/testing
            logger.warning("No AI client configured, using mock client")
            return self._create_mock_client()
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize and return all agent tools."""
        tools = []
        
        try:
            tools.append(RAGTool())
            logger.info("RAG tool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {e}")
        
        try:
            tools.append(MemoryTool())
            logger.info("Memory tool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Memory tool: {e}")
        
        try:
            tools.append(AttachmentTool())
            logger.info("Attachment tool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Attachment tool: {e}")
        
        try:
            tools.append(EscalationTool())
            logger.info("Escalation tool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Escalation tool: {e}")
        
        return tools
    
    def _create_agent(self, chat_client) -> Any:
        """Create the ChatAgent instance."""
        return ChatAgent(
            chat_client=chat_client,
            instructions=self.SYSTEM_PROMPT,
            name="CustomerSupportBot",
            tools=self.tools,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens
        )
    
    def _create_mock_client(self):
        """Create a mock client for development without API keys."""
        class MockChatClient:
            async def complete(self, messages, **kwargs):
                return {
                    "choices": [{
                        "message": {
                            "content": "This is a mock response for development.",
                            "role": "assistant"
                        }
                    }]
                }
        return MockChatClient()
    
    async def get_or_create_thread(self, session_id: str) -> AgentThread:
        """
        Get existing thread or create new one for session.
        Threads maintain conversation state.
        """
        if session_id not in self.threads:
            self.threads[session_id] = AgentThread()
            self.message_history[session_id] = []
            logger.info(f"Created new thread for session: {session_id}")
        
        return self.threads[session_id]
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user message and return structured response.
        
        Args:
            session_id: Unique session identifier
            message: User's message text
            attachments: Optional list of file attachments
            metadata: Optional metadata about the request
        
        Returns:
            Dictionary containing response, sources, and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Get or create thread for this session
            thread = await self.get_or_create_thread(session_id)
            
            # Process attachments if present
            attachment_context = await self._process_attachments(attachments)
            
            # Load conversation memory
            memory_context = await self._load_memory_context(session_id)
            
            # Combine contexts with user message
            full_message = self._build_full_message(
                message, 
                attachment_context, 
                memory_context,
                metadata
            )
            
            # Store user message in history
            self._add_to_history(session_id, "user", message)
            
            # Run agent to get response
            response = await self._run_agent(thread, full_message)
            
            # Parse and structure the response
            result = self._parse_agent_response(response)
            
            # Store assistant response in history
            self._add_to_history(session_id, "assistant", result["message"])
            
            # Store important information in memory
            await self._update_memory(session_id, message, result)
            
            # Add metadata
            result["session_id"] = session_id
            result["timestamp"] = datetime.utcnow().isoformat()
            result["processing_time"] = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"Processed message for session {session_id} in "
                f"{result['processing_time']:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "message": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e) if settings.debug else "Internal error",
                "sources": [],
                "requires_escalation": True,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _process_attachments(
        self, 
        attachments: Optional[List[Any]]
    ) -> str:
        """Process attachments and return extracted context."""
        if not attachments:
            return ""
        
        attachment_context = "\n\n--- Attached Documents ---\n"
        
        for attachment in attachments:
            try:
                # Use attachment tool to process file
                tool = next((t for t in self.tools if t.name == "attachment_processor"), None)
                if tool:
                    result = await tool.process_attachment(
                        attachment.get("path"),
                        attachment.get("filename")
                    )
                    if result["success"]:
                        attachment_context += f"\n[{result['filename']}]:\n{result['content'][:1000]}\n"
            except Exception as e:
                logger.error(f"Failed to process attachment: {e}")
        
        return attachment_context
    
    async def _load_memory_context(self, session_id: str) -> str:
        """Load relevant memory context for the session."""
        try:
            tool = next((t for t in self.tools if t.name == "memory_management"), None)
            if tool:
                summary = await tool.summarize_session(session_id)
                if summary:
                    return f"\n\n--- Previous Context ---\n{summary}\n"
        except Exception as e:
            logger.error(f"Failed to load memory context: {e}")
        
        return ""
    
    def _build_full_message(
        self,
        message: str,
        attachment_context: str,
        memory_context: str,
        metadata: Optional[Dict]
    ) -> str:
        """Build complete message with all context."""
        full_message = message
        
        if memory_context:
            full_message = memory_context + "\n\n" + full_message
        
        if attachment_context:
            full_message = full_message + "\n\n" + attachment_context
        
        if metadata:
            full_message = full_message + f"\n\n[Metadata: {json.dumps(metadata)}]"
        
        return full_message
    
    async def _run_agent(self, thread: AgentThread, message: str) -> Any:
        """Run the agent with the given message."""
        if self.agent:
            return await self.agent.run(
                message,
                thread=thread,
                stream=False
            )
        else:
            # Fallback for development
            return {
                "text": "This is a development response. Agent not fully initialized.",
                "messages": []
            }
    
    def _parse_agent_response(self, response: Any) -> Dict[str, Any]:
        """Parse agent response into structured format."""
        result = {
            "message": "",
            "sources": [],
            "requires_escalation": False,
            "confidence": 0.0,
            "tools_used": []
        }
        
        # Extract main message
        if hasattr(response, "text"):
            result["message"] = response.text
        elif isinstance(response, dict) and "text" in response:
            result["message"] = response["text"]
        
        # Parse tool results if available
        if hasattr(response, "messages"):
            for msg in response.messages:
                if hasattr(msg, "content"):
                    for content in msg.content:
                        if hasattr(content, "tool_result"):
                            self._process_tool_result(content.tool_result, result)
        
        return result
    
    def _process_tool_result(
        self, 
        tool_result: Any, 
        result: Dict[str, Any]
    ) -> None:
        """Process individual tool results."""
        if hasattr(tool_result, "name"):
            result["tools_used"].append(tool_result.name)
            
            if tool_result.name == "rag_search" and hasattr(tool_result, "output"):
                sources = tool_result.output.get("sources", [])
                result["sources"].extend(sources)
            
            elif tool_result.name == "escalation_check" and hasattr(tool_result, "output"):
                result["requires_escalation"] = tool_result.output.get("escalate", False)
                result["escalation_reason"] = tool_result.output.get("reason", "")
    
    def _add_to_history(
        self, 
        session_id: str, 
        role: str, 
        content: str
    ) -> None:
        """Add message to conversation history."""
        if session_id not in self.message_history:
            self.message_history[session_id] = []
        
        self.message_history[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last 50 messages
        if len(self.message_history[session_id]) > 50:
            self.message_history[session_id] = self.message_history[session_id][-50:]
    
    async def _update_memory(
        self, 
        session_id: str, 
        user_message: str, 
        response: Dict
    ) -> None:
        """Update memory with important information from conversation."""
        try:
            tool = next((t for t in self.tools if t.name == "memory_management"), None)
            if tool:
                # Store user query as context
                await tool.store_memory(
                    session_id=session_id,
                    content=user_message,
                    content_type="context",
                    importance=0.7
                )
                
                # Store sources used
                if response.get("sources"):
                    await tool.store_memory(
                        session_id=session_id,
                        content=json.dumps(response["sources"][:3]),
                        content_type="fact",
                        importance=0.8
                    )
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
    
    async def stream_response(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response for real-time interaction.
        Yields updates as they're generated.
        """
        try:
            thread = await self.get_or_create_thread(session_id)
            
            # Process message context
            attachment_context = await self._process_attachments(attachments)
            memory_context = await self._load_memory_context(session_id)
            full_message = self._build_full_message(
                message, 
                attachment_context, 
                memory_context, 
                None
            )
            
            # Stream from agent
            if self.agent and hasattr(self.agent, "run_stream"):
                async for update in self.agent.run_stream(full_message, thread=thread):
                    if hasattr(update, "text") and update.text:
                        yield {
                            "type": "text",
                            "content": update.text,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    elif hasattr(update, "tool_call"):
                        yield {
                            "type": "tool_call",
                            "tool": update.tool_call.name,
                            "status": "running",
                            "timestamp": datetime.utcnow().isoformat()
                        }
            else:
                # Fallback for development
                yield {
                    "type": "text",
                    "content": "Streaming not available in development mode.",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Final completion message
            yield {
                "type": "complete",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up agent resources")
        
        # Clear threads and history
        self.threads.clear()
        self.message_history.clear()
        
        # Cleanup tools
        for tool in self.tools:
            if hasattr(tool, "cleanup"):
                await tool.cleanup()
        
        logger.info("Agent cleanup complete")
```

### Phase 3: Tools Implementation (Days 9-12)

#### Task 3.1: RAG Tool

**Implementation Instructions for `backend/app/tools/rag_tool.py`:**

1. Initialize SentenceTransformer with EmbeddingGemma
2. Set up ChromaDB persistent client
3. Implement embedding generation with proper prefixes
4. Add caching layer for repeated queries
5. Implement document chunking strategy
6. Handle similarity search with metadata filtering

**Key Implementation Points:**
```python
# Embedding prefixes for EmbeddingGemma
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "

# Chunking parameters
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50  # words

# Search parameters
DEFAULT_K = 5
SIMILARITY_THRESHOLD = 0.7
```

#### Task 3.2: Memory Tool

**Implementation for `backend/app/tools/memory_tool.py`:**

1. Create SQLAlchemy models for memory storage
2. Implement CRUD operations for memories
3. Add importance scoring for memory retrieval
4. Implement session summarization
5. Add cleanup for old memories

#### Task 3.3: Attachment Tool

**Implementation for `backend/app/tools/attachment_tool.py`:**

1. Initialize MarkItDown with all plugins
2. Handle file upload and temporary storage
3. Process supported file types
4. Extract and structure content
5. Optional: Index in RAG for searchability

### Phase 4: API Routes (Days 13-15)

#### Task 4.1: Chat Routes

**File: `backend/app/api/routes/chat.py`**
```python
"""Chat API routes for message handling."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Optional
import uuid
from datetime import datetime

from ...agents.chat_agent import CustomerSupportAgent
from ...models.schemas import ChatRequest, ChatResponse, MessageHistory
from ...services.auth_service import get_current_user
from ...config import settings

router = APIRouter()

@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
async def send_message(
    session_id: str,
    message: str = Form(...),
    attachments: List[UploadFile] = File(None),
    agent: CustomerSupportAgent = Depends(get_agent)
):
    """Send a message and receive AI response."""
    
    # Validate session exists
    # Process attachments if provided
    # Call agent.process_message
    # Return structured response
    
    pass  # Implementation here

@router.get("/sessions/{session_id}/messages", response_model=MessageHistory)
async def get_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0
):
    """Retrieve message history for a session."""
    pass  # Implementation here
```

### Phase 5: Frontend Implementation (Days 16-18)

#### Task 5.1: React Setup

**File: `frontend/package.json`**
```json
{
  "name": "customer-support-frontend",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "jest",
    "lint": "eslint src --ext ts,tsx --report-unused-disable-directives"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "socket.io-client": "^4.6.0",
    "react-markdown": "^9.0.0",
    "clsx": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.16",
    "eslint": "^8.56.0",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
```

#### Task 5.2: Chat Interface Component

**Implementation pattern for `frontend/src/components/ChatInterface.tsx`:**

1. Set up WebSocket connection on mount
2. Implement message sending with file upload
3. Handle streaming responses
4. Display sources in sidebar
5. Show escalation notifications
6. Implement auto-scroll and typing indicators

### Phase 6: Testing & Deployment (Days 19-20)

#### Task 6.1: Test Suite

**File: `backend/tests/test_agents.py`**
```python
"""Test suite for agent functionality."""
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initializes with all tools."""
    pass

@pytest.mark.asyncio
async def test_message_processing():
    """Test message processing flow."""
    pass

@pytest.mark.asyncio
async def test_error_handling():
    """Test agent handles errors gracefully."""
    pass
```

#### Task 6.2: Docker Configuration

**File: `docker-compose.yml`**
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///app/data/app.db
    volumes:
      - ./backend/data:/app/data
    depends_on:
      - redis
  
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  redis_data:
```

## 🧪 Testing Guidelines

### Unit Test Requirements

Each module must have corresponding tests with minimum 80% coverage:

```bash
# Backend testing
cd backend
pytest tests/ -v --cov=app --cov-report=html

# Frontend testing  
cd frontend
npm test -- --coverage
```

### Integration Test Checklist

- [ ] API endpoints respond correctly
- [ ] WebSocket connection maintains state
- [ ] File uploads process successfully
- [ ] Agent tools execute properly
- [ ] Error handling works as expected
- [ ] Rate limiting functions correctly

## 🚀 Deployment Checklist

### Pre-Deployment Verification

```bash
# 1. Run all tests
./scripts/run_tests.sh

# 2. Check code quality
./scripts/lint.sh

# 3. Build Docker images
docker-compose build

# 4. Run locally
docker-compose up

# 5. Test endpoints
curl http://localhost:8000/health
curl http://localhost:3000
```

### Production Deployment Steps

1. **Environment Variables**: Set all production values
2. **Database Migration**: Run any pending migrations
3. **Static Files**: Build and deploy frontend assets
4. **SSL Certificates**: Configure HTTPS
5. **Monitoring**: Verify Prometheus/Grafana connection
6. **Logging**: Confirm log aggregation working
7. **Backups**: Test backup/restore procedures
8. **Load Testing**: Verify performance under load

## 🎯 Success Metrics

### Code Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test Coverage | >80% | `pytest --cov` |
| Linting Score | 0 errors | `flake8`, `eslint` |
| Type Coverage | 100% | `mypy`, `tsc` |
| Documentation | All public APIs | Manual review |

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Response Time | <500ms p95 | Prometheus metrics |
| WebSocket Latency | <100ms | Custom metrics |
| Agent Response Time | <3s | Application logs |
| Concurrent Users | >100 | Load testing |

## 📚 Required Reading

### Framework Documentation
- [Microsoft Agent Framework Docs](https://github.com/microsoft/agent-framework)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React 18 Documentation](https://react.dev/)

### AI/ML Resources
- [EmbeddingGemma Model Card](https://huggingface.co/google/embeddinggemma-300m)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [MarkItDown Usage](https://github.com/microsoft/markitdown)

## 🤖 AI Agent Instructions

### For Code Generation

When implementing any component:
1. Follow the exact patterns shown in this document
2. Use type hints for all Python functions
3. Include comprehensive docstrings
4. Add error handling for all external calls
5. Log important operations at appropriate levels
6. Write corresponding tests immediately

### For Problem Solving

When encountering issues:
1. Check configuration settings first
2. Verify all dependencies are installed
3. Review logs for specific error messages
4. Consult the troubleshooting section
5. Test in isolation before integration

### For Optimization

Focus areas for performance:
1. Implement caching for repeated operations
2. Use connection pooling for databases
3. Batch embedding generation
4. Optimize chunk sizes for RAG
5. Implement request debouncing in frontend

## 🔧 Common Implementation Patterns

### Async Service Pattern
```python
class ServiceName:
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Lazy initialization pattern."""
        if not self.initialized:
            await self._setup()
            self.initialized = True
    
    async def operation(self, **kwargs):
        await self.initialize()
        return await self._execute(**kwargs)
```

### Error Handling Pattern
```python
try:
    result = await risky_operation()
    return {"success": True, "data": result}
except SpecificException as e:
    logger.warning(f"Expected error: {e}")
    return {"success": False, "error": str(e)}
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal server error")
```

### React Hook Pattern
```typescript
export const useCustomHook = (initialValue: any) => {
  const [state, setState] = useState(initialValue);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const execute = useCallback(async (params: any) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiCall(params);
      setState(result);
      return result;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  return { state, loading, error, execute };
};
```

## 📝 Final Notes

### Priority Order for Implementation

1. **Critical Path** (Must work first):
   - Basic FastAPI server (`main.py`, `config.py`)
   - Simple chat agent without tools
   - Basic chat API endpoint
   - Minimal React frontend

2. **Core Features** (Primary functionality):
   - RAG tool with ChromaDB
   - Memory tool with SQLite
   - WebSocket support
   - Frontend chat interface

3. **Enhanced Features** (Complete experience):
   - Attachment processing
   - Escalation logic
   - Monitoring/metrics
   - Production Docker setup

### Development Tips

1. **Start Simple**: Get a basic message echo working first
2. **Incremental Testing**: Test each component in isolation
3. **Mock External Services**: Use mocks for OpenAI initially
4. **Version Control**: Commit after each working feature
5. **Documentation**: Update docs as you implement

### When Stuck

If you encounter blockers:
1. Implement a simpler version first
2. Add TODO comments for complex parts
3. Create mock implementations to continue
4. Focus on the critical path
5. Ask for clarification on requirements

---

**Remember**: The goal is a working, production-ready system. Prioritize functionality over perfection in the initial implementation, then iterate to improve.
