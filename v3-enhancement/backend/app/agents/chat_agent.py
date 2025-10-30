"""
Customer Support Agent implementation with full tool integration.
This agent orchestrates RAG, Memory, Attachment, and Escalation tools.

Phase 3 Update: All tool calls use telemetry wrapper with retry and circuit breaker.
"""
import asyncio
import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field

from ..config import settings
from ..config.tool_settings import tool_settings
from ..tools.base_tool import ToolResult
from ..tools.tool_call_wrapper import (
    call_tool_with_wrapper,
    tool_call_context,
    RetryConfig,
    get_tool_metrics
)
from ..models.session import Session
from ..models.message import Message
from ..models.memory import Memory

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent processing."""
    session_id: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message_count: int = 0
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None  # Phase 3: Track request ID


class AgentResponse:
    """Structured agent response."""
    
    def __init__(
        self,
        message: str,
        sources: List[Dict] = None,
        requires_escalation: bool = False,
        confidence: float = 0.0,
        tools_used: List[str] = None,
        processing_time: float = 0.0
    ):
        self.message = message
        self.sources = sources or []
        self.requires_escalation = requires_escalation
        self.confidence = confidence
        self.tools_used = tools_used or []
        self.processing_time = processing_time
        self.tool_metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "sources": self.sources,
            "requires_escalation": self.requires_escalation,
            "confidence": self.confidence,
            "tools_used": self.tools_used,
            "processing_time": self.processing_time,
            "metadata": self.tool_metadata
        }


class CustomerSupportAgent:
    """
    Production-ready customer support agent with full tool integration.
    Orchestrates multiple AI tools for comprehensive support capabilities.
    
    Phase 3: All tool calls wrapped with retry, circuit breaker, and telemetry.
    """
    
    # System prompt with tool instructions
    SYSTEM_PROMPT = """You are an expert customer support AI assistant with access to the following tools:

AVAILABLE TOOLS:
1. **rag_search**: Search our knowledge base for relevant information
   - Use this when users ask questions about policies, procedures, or general information
   - Always cite sources when using information from this tool

2. **memory_management**: Store and retrieve conversation context
   - Use this to remember important user information and preferences
   - Check memory at the start of each conversation for context

3. **attachment_processor**: Process and analyze uploaded documents
   - Use this when users upload files
   - Extract and analyze content from various file formats

4. **escalation_check**: Determine if human intervention is needed
   - Monitor for signs that require human support
   - Check sentiment and urgency of user messages

INSTRUCTIONS:
1. Always be helpful, professional, and empathetic
2. Use tools appropriately to provide accurate information
3. Cite your sources when providing information from the knowledge base
4. Remember important details about the user and their issues
5. Escalate to human support when:
   - The user explicitly asks for human assistance
   - The issue involves legal or compliance matters
   - The user expresses high frustration or dissatisfaction
   - You cannot resolve the issue after multiple attempts

RESPONSE FORMAT:
- Provide clear, concise answers
- Break down complex information into steps
- Offer additional help and next steps
- Maintain a friendly, professional tone

Remember: Customer satisfaction is the top priority."""
    
    def __init__(self, use_registry: Optional[bool] = None):
        """
        Initialize the agent with all tools.
        
        Args:
            use_registry: Whether to use registry mode (None = auto-detect from settings)
        """
        self.tools = {}
        self.contexts = {}  # Store session contexts (in-memory, Phase 4 will externalize)
        self.initialized = False
        
        # Determine initialization mode
        if use_registry is None:
            registry_mode = getattr(settings, 'agent_tool_registry_mode', 'legacy')
            self.use_registry = (registry_mode == 'registry')
        else:
            self.use_registry = use_registry
        
        # Retry configuration for tool calls
        self.retry_config = RetryConfig(
            max_attempts=getattr(settings, 'agent_max_retries', 3),
            wait_multiplier=1.0,
            wait_min=1.0,
            wait_max=10.0,
            retry_exceptions=(Exception,)
        )
        
        logger.info(f"Agent initialization mode: {'registry' if self.use_registry else 'legacy'}")
        
        # Initialize on creation (legacy mode only)
        if not self.use_registry:
            self._initialize_legacy()
    
    async def initialize_async(self) -> None:
        """
        Initialize agent asynchronously (registry mode).
        Must be called explicitly when using registry mode.
        """
        if not self.use_registry:
            logger.warning("initialize_async called in legacy mode - tools already initialized")
            return
        
        try:
            logger.info("Initializing agent in registry mode...")
            await self._initialize_registry()
            self.initialized = True
            logger.info(f"✓ Agent initialized with {len(self.tools)} tools (registry mode)")
        except Exception as e:
            logger.error(f"Failed to initialize agent in registry mode: {e}", exc_info=True)
            raise
    
    def _initialize_legacy(self) -> None:
        """Initialize all tools using legacy method."""
        try:
            logger.info("Initializing agent tools (legacy mode)...")
            
            from ..tools import RAGTool, MemoryTool, AttachmentTool, EscalationTool
            
            if tool_settings.enable_rag_tool:
                self.tools['rag'] = RAGTool()
                logger.info("✓ RAG tool initialized")
            
            if tool_settings.enable_memory_tool:
                self.tools['memory'] = MemoryTool()
                logger.info("✓ Memory tool initialized")
            
            if tool_settings.enable_attachment_tool:
                self.tools['attachment'] = AttachmentTool()
                logger.info("✓ Attachment tool initialized")
            
            if tool_settings.enable_escalation_tool:
                self.tools['escalation'] = EscalationTool()
                logger.info("✓ Escalation tool initialized")
            
            self.initialized = True
            logger.info(f"Agent initialized with {len(self.tools)} tools (legacy mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent (legacy mode): {e}", exc_info=True)
            raise
    
    async def _initialize_registry(self) -> None:
        """Initialize all tools using registry."""
        try:
            from ..tools.registry import ToolRegistry, ToolDependencies
            
            dependencies = ToolDependencies(
                settings=settings,
                tool_settings=tool_settings
            )
            
            self.tools = await ToolRegistry.create_and_initialize_tools(
                dependencies=dependencies,
                enabled_only=True,
                concurrent_init=True
            )
            
            if not self.tools:
                logger.warning("No tools were created by registry")
            
        except Exception as e:
            logger.error(f"Failed to initialize tools via registry: {e}", exc_info=True)
            raise
    
    def get_or_create_context(self, session_id: str, request_id: Optional[str] = None) -> AgentContext:
        """
        Get or create context for a session.
        
        Args:
            session_id: Session identifier
            request_id: Request correlation ID (Phase 3)
        """
        if session_id not in self.contexts:
            self.contexts[session_id] = AgentContext(
                session_id=session_id,
                thread_id=str(uuid.uuid4()),
                request_id=request_id
            )
            logger.info(
                f"Created new context for session: {session_id}",
                extra={"session_id": session_id, "request_id": request_id}
            )
        else:
            # Update request_id for existing context
            self.contexts[session_id].request_id = request_id
        
        return self.contexts[session_id]
    
    async def load_session_context(
        self,
        session_id: str,
        request_id: Optional[str] = None
    ) -> str:
        """
        Load conversation context from memory with telemetry.
        
        Args:
            session_id: Session identifier
            request_id: Request correlation ID
        """
        try:
            memory_tool = self.tools.get('memory')
            if not memory_tool:
                logger.warning("Memory tool not available")
                return ""
            
            # Call with telemetry wrapper
            async with tool_call_context(
                tool_name='memory',
                operation='load_context',
                request_id=request_id,
                session_id=session_id
            ):
                # Summarize session
                summary = await memory_tool.summarize_session(session_id)
                
                # Retrieve recent memories
                memories = await memory_tool.retrieve_memories(
                    session_id=session_id,
                    content_type="context",
                    limit=5
                )
                
                if memories:
                    recent_context = "\nRecent conversation points:\n"
                    for memory in memories[:3]:
                        recent_context += f"- {memory['content']}\n"
                    summary += recent_context
                
                return summary
            
        except Exception as e:
            logger.error(
                f"Error loading session context: {e}",
                extra={"session_id": session_id, "request_id": request_id}
            )
            return ""
    
    async def search_knowledge_base(
        self,
        query: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base using RAG tool with telemetry.
        
        Args:
            query: Search query
            request_id: Request correlation ID
            session_id: Session identifier
            k: Number of results
        """
        try:
            rag_tool = self.tools.get('rag')
            if not rag_tool:
                logger.warning("RAG tool not available")
                return []
            
            # Call with wrapper (includes retry and circuit breaker)
            result = await call_tool_with_wrapper(
                tool=rag_tool,
                method_name='search',
                request_id=request_id,
                session_id=session_id,
                retry_config=self.retry_config,
                timeout=30.0,
                query=query,
                k=k,
                threshold=0.7
            )
            
            # Handle ToolResult
            if isinstance(result, ToolResult):
                if result.success:
                    return result.data.get("sources", [])
                else:
                    logger.error(
                        f"RAG search failed: {result.error}",
                        extra={"request_id": request_id, "session_id": session_id}
                    )
                    return []
            else:
                # Legacy dict response
                return result.get("sources", [])
            
        except Exception as e:
            logger.error(
                f"RAG search error: {e}",
                extra={"request_id": request_id, "session_id": session_id}
            )
            return []
    
    async def process_attachments(
        self,
        attachments: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Process uploaded attachments with telemetry.
        
        Args:
            attachments: List of attachment metadata
            request_id: Request correlation ID
            session_id: Session identifier
        """
        if not attachments:
            return ""
        
        attachment_tool = self.tools.get('attachment')
        rag_tool = self.tools.get('rag')
        
        if not attachment_tool:
            logger.warning("Attachment tool not available")
            return ""
        
        processed_content = "\n📎 Attached Documents:\n"
        
        for attachment in attachments:
            try:
                # Call with wrapper
                result = await call_tool_with_wrapper(
                    tool=attachment_tool,
                    method_name='process_attachment',
                    request_id=request_id,
                    session_id=session_id,
                    timeout=60.0,
                    file_path=attachment.get("path"),
                    filename=attachment.get("filename"),
                    chunk_for_rag=True
                )
                
                # Handle ToolResult
                if isinstance(result, ToolResult):
                    result = result.data
                
                if result.get("success"):
                    processed_content += f"\n[{result['filename']}]:\n"
                    processed_content += f"{result.get('preview', '')}\n"
                    
                    # Index in RAG if chunks available
                    if rag_tool and "chunks" in result:
                        rag_tool.add_documents(
                            documents=result["chunks"],
                            metadatas=[
                                {
                                    "source": result['filename'],
                                    "type": "user_upload",
                                    "session_id": session_id,
                                    "request_id": request_id
                                }
                                for _ in result["chunks"]
                            ]
                        )
                        logger.info(
                            f"Indexed {len(result['chunks'])} chunks from {result['filename']}",
                            extra={"request_id": request_id, "session_id": session_id}
                        )
                
            except Exception as e:
                logger.error(
                    f"Error processing attachment: {e}",
                    extra={"request_id": request_id, "session_id": session_id}
                )
                processed_content += f"\n[Error processing {attachment.get('filename', 'file')}]\n"
        
        return processed_content
    
    async def check_escalation(
        self,
        message: str,
        context: AgentContext,
        message_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Check if escalation is needed with telemetry.
        
        Args:
            message: User message
            context: Agent context
            message_history: Previous messages
        """
        try:
            escalation_tool = self.tools.get('escalation')
            if not escalation_tool:
                logger.warning("Escalation tool not available")
                return {"escalate": False, "confidence": 0.0}
            
            # Call with wrapper
            result = await call_tool_with_wrapper(
                tool=escalation_tool,
                method_name='should_escalate',
                request_id=context.request_id,
                session_id=context.session_id,
                timeout=10.0,
                message=message,
                message_history=message_history,
                metadata={
                    "session_id": context.session_id,
                    "message_count": context.message_count,
                    "already_escalated": context.escalated
                }
            )
            
            # Handle ToolResult
            if isinstance(result, ToolResult):
                result = result.data
            
            # Create ticket if escalation needed
            if result.get("escalate") and not context.escalated:
                result["ticket"] = escalation_tool.create_escalation_ticket(
                    session_id=context.session_id,
                    escalation_result=result,
                    user_info={"user_id": context.user_id}
                )
                context.escalated = True
                logger.info(
                    f"Escalation triggered for session {context.session_id}",
                    extra={"session_id": context.session_id, "request_id": context.request_id}
                )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Escalation check error: {e}",
                extra={"session_id": context.session_id, "request_id": context.request_id}
            )
            return {"escalate": False, "confidence": 0.0}
    
    async def store_conversation_memory(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        important_facts: List[str] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Store important information in memory with telemetry.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            agent_response: Agent's response
            important_facts: Extracted facts
            request_id: Request correlation ID
        """
        try:
            memory_tool = self.tools.get('memory')
            if not memory_tool:
                logger.warning("Memory tool not available")
                return
            
            async with tool_call_context(
                tool_name='memory',
                operation='store_conversation',
                request_id=request_id,
                session_id=session_id
            ):
                # Store user message
                await memory_tool.store_memory(
                    session_id=session_id,
                    content=f"User: {user_message[:200]}",
                    content_type="context",
                    importance=0.5
                )
                
                # Store agent response
                if len(agent_response) > 100:
                    await memory_tool.store_memory(
                        session_id=session_id,
                        content=f"Agent: {agent_response[:200]}",
                        content_type="context",
                        importance=0.4
                    )
                
                # Store important facts
                if important_facts:
                    for fact in important_facts:
                        await memory_tool.store_memory(
                            session_id=session_id,
                            content=fact,
                            content_type="fact",
                            importance=0.8
                        )
            
        except Exception as e:
            logger.error(
                f"Error storing memory: {e}",
                extra={"session_id": session_id, "request_id": request_id}
            )
    
    def extract_important_facts(
        self,
        message: str,
        response: str
    ) -> List[str]:
        """Extract important facts from conversation."""
        facts = []
        
        import re
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        for email in emails:
            facts.append(f"User email: {email}")
        
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, message)
        for phone in phones:
            facts.append(f"User phone: {phone}")
        
        order_pattern = r'\b(?:order|ticket|reference|confirmation)\s*#?\s*([A-Z0-9-]+)\b'
        orders = re.findall(order_pattern, message, re.IGNORECASE)
        for order in orders:
            facts.append(f"Reference number: {order}")
        
        return facts
    
    async def generate_response(
        self,
        message: str,
        context: str,
        sources: List[Dict],
        escalation: Dict[str, Any]
    ) -> str:
        """Generate agent response based on context and tools."""
        response_parts = []
        
        if context == "No previous context available for this session.":
            response_parts.append("Hello! I'm here to help you today.")
        
        if sources:
            response_parts.append("Based on our information:")
            for i, source in enumerate(sources[:2], 1):
                response_parts.append(f"{i}. {source['content'][:200]}...")
        
        if escalation.get("escalate"):
            response_parts.append(
                "\nI understand this is important to you. "
                "I'm connecting you with a human support specialist who can better assist you."
            )
            if escalation.get("ticket"):
                response_parts.append(
                    f"Your ticket number is: {escalation['ticket']['ticket_id']}"
                )
        
        if not response_parts:
            response_parts.append(
                "I'm here to help! Could you please provide more details about your inquiry?"
            )
        
        return "\n\n".join(response_parts)
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        message_history: Optional[List[Dict]] = None,
        request_id: Optional[str] = None
    ) -> AgentResponse:
        """
        Process a user message and generate response.
        
        Args:
            session_id: Session identifier
            message: User message
            attachments: Optional file attachments
            user_id: Optional user identifier
            message_history: Previous messages
            request_id: Request correlation ID (Phase 3)
            
        Returns:
            AgentResponse with generated response and metadata
        """
        start_time = datetime.utcnow()
        
        # Generate request_id if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            # Get or create context with request_id
            context = self.get_or_create_context(session_id, request_id)
            context.user_id = user_id
            context.message_count += 1
            
            logger.info(
                f"Processing message for session {session_id}",
                extra={
                    "session_id": session_id,
                    "request_id": request_id,
                    "message_count": context.message_count
                }
            )
            
            # Load session context with telemetry
            session_context = await self.load_session_context(session_id, request_id)
            
            # Process attachments with telemetry
            attachment_context = ""
            if attachments:
                attachment_context = await self.process_attachments(
                    attachments,
                    request_id,
                    session_id
                )
            
            # Search knowledge base with telemetry
            sources = await self.search_knowledge_base(
                message,
                request_id,
                session_id
            )
            
            # Check escalation with telemetry
            escalation = await self.check_escalation(message, context, message_history)
            
            # Generate response
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            # Extract and store important facts with telemetry
            facts = self.extract_important_facts(message, response_text)
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text,
                important_facts=facts,
                request_id=request_id
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Build response
            response = AgentResponse(
                message=response_text,
                sources=sources[:3],
                requires_escalation=escalation.get("escalate", False),
                confidence=escalation.get("confidence", 0.95),
                tools_used=list(self.tools.keys()),
                processing_time=processing_time
            )
            
            # Add metadata
            response.tool_metadata = {
                "session_id": session_id,
                "request_id": request_id,
                "message_count": context.message_count,
                "has_context": bool(session_context),
                "facts_extracted": len(facts),
                "initialization_mode": "registry" if self.use_registry else "legacy",
                "circuit_breaker_status": get_tool_metrics()
            }
            
            if escalation.get("ticket"):
                response.tool_metadata["ticket_id"] = escalation["ticket"]["ticket_id"]
            
            logger.info(
                f"Processed message for session {session_id} in {processing_time:.2f}s",
                extra={
                    "session_id": session_id,
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "escalated": response.requires_escalation
                }
            )
            
            return response
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.error(
                f"Error processing message: {e}",
                extra={
                    "session_id": session_id,
                    "request_id": request_id,
                    "processing_time": processing_time
                },
                exc_info=True
            )
            
            return AgentResponse(
                message="I apologize, but I encountered an error processing your request. "
                        "Please try again or contact support directly.",
                requires_escalation=True,
                confidence=0.0,
                processing_time=processing_time
            )
    
    async def stream_response(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response for real-time interaction with telemetry."""
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            yield {
                "type": "start",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            yield {
                "type": "status",
                "message": "Loading conversation context...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            context = self.get_or_create_context(session_id, request_id)
            session_context = await self.load_session_context(session_id, request_id)
            
            if attachments:
                yield {
                    "type": "status",
                    "message": "Processing attachments...",
                    "timestamp": datetime.utcnow().isoformat()
                }
                attachment_context = await self.process_attachments(
                    attachments,
                    request_id,
                    session_id
                )
            
            yield {
                "type": "status",
                "message": "Searching knowledge base...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sources = await self.search_knowledge_base(message, request_id, session_id)
            
            if sources:
                yield {
                    "type": "sources",
                    "sources": sources[:3],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            escalation = await self.check_escalation(message, context)
            
            if escalation.get("escalate"):
                yield {
                    "type": "escalation",
                    "required": True,
                    "reason": escalation.get("reasons", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            words = response_text.split()
            chunk_size = 5
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                yield {
                    "type": "text",
                    "content": chunk + ' ',
                    "timestamp": datetime.utcnow().isoformat()
                }
                await asyncio.sleep(0.05)
            
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text,
                request_id=request_id
            )
            
            yield {
                "type": "complete",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(
                f"Streaming error: {e}",
                extra={"session_id": session_id, "request_id": request_id}
            )
            yield {
                "type": "error",
                "error": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        try:
            if session_id in self.contexts:
                del self.contexts[session_id]
            
            memory_tool = self.tools.get('memory')
            if memory_tool:
                await memory_tool.cleanup_old_memories(days=30)
            
            logger.info(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all agent resources."""
        logger.info("Cleaning up agent resources...")
        
        for session_id in list(self.contexts.keys()):
            await self.cleanup_session(session_id)
        
        if self.use_registry:
            from ..tools.registry import ToolRegistry
            await ToolRegistry.cleanup_tools(self.tools, concurrent=True)
        else:
            for tool_name, tool in self.tools.items():
                if hasattr(tool, 'cleanup'):
                    try:
                        await tool.cleanup()
                        logger.info(f"Cleaned up {tool_name} tool")
                    except Exception as e:
                        logger.error(f"Error cleaning up {tool_name} tool: {e}")
        
        logger.info("Agent cleanup complete")
