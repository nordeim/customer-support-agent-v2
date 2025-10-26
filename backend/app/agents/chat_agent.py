"""
Customer Support Agent implementation with full tool integration.
This agent orchestrates RAG, Memory, Attachment, and Escalation tools.
"""
import asyncio
import json
import logging
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from ..config import settings
from ..tools import RAGTool, MemoryTool, AttachmentTool, EscalationTool
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
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


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
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "sources": self.sources,
            "requires_escalation": self.requires_escalation,
            "confidence": self.confidence,
            "tools_used": self.tools_used,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }


class CustomerSupportAgent:
    """
    Production-ready customer support agent with full tool integration.
    Orchestrates multiple AI tools for comprehensive support capabilities.
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
    
    def __init__(self):
        """Initialize the agent with all tools."""
        self.tools = {}
        self.contexts = {}  # Store session contexts
        self.initialized = False
        
        # Initialize on creation
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize all tools and components."""
        try:
            # Initialize tools
            logger.info("Initializing agent tools...")
            
            self.tools['rag'] = RAGTool()
            logger.info("✓ RAG tool initialized")
            
            self.tools['memory'] = MemoryTool()
            logger.info("✓ Memory tool initialized")
            
            self.tools['attachment'] = AttachmentTool()
            logger.info("✓ Attachment tool initialized")
            
            self.tools['escalation'] = EscalationTool()
            logger.info("✓ Escalation tool initialized")
            
            self.initialized = True
            logger.info(f"Agent initialized with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise
    
    def get_or_create_context(self, session_id: str) -> AgentContext:
        """Get or create context for a session."""
        if session_id not in self.contexts:
            self.contexts[session_id] = AgentContext(
                session_id=session_id,
                thread_id=str(uuid.uuid4())
            )
            logger.info(f"Created new context for session: {session_id}")
        
        return self.contexts[session_id]
    
    async def load_session_context(self, session_id: str) -> str:
        """Load conversation context from memory."""
        try:
            # Get memory summary
            memory_tool = self.tools['memory']
            summary = await memory_tool.summarize_session(session_id)
            
            # Get recent messages
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
            logger.error(f"Error loading session context: {e}")
            return ""
    
    async def search_knowledge_base(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base using RAG tool."""
        try:
            rag_tool = self.tools['rag']
            result = await rag_tool.search(
                query=query,
                k=k,
                threshold=0.7
            )
            
            return result.get("sources", [])
            
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    async def process_attachments(
        self,
        attachments: List[Dict[str, Any]]
    ) -> str:
        """Process uploaded attachments."""
        if not attachments:
            return ""
        
        attachment_tool = self.tools['attachment']
        rag_tool = self.tools['rag']
        
        processed_content = "\n📎 Attached Documents:\n"
        
        for attachment in attachments:
            try:
                # Process attachment
                result = await attachment_tool.process_attachment(
                    file_path=attachment.get("path"),
                    filename=attachment.get("filename"),
                    chunk_for_rag=True
                )
                
                if result["success"]:
                    # Add summary to context
                    processed_content += f"\n[{result['filename']}]:\n"
                    processed_content += f"{result['preview']}\n"
                    
                    # Index in RAG if chunks available
                    if "chunks" in result:
                        rag_tool.add_documents(
                            documents=result["chunks"],
                            metadatas=[
                                {
                                    "source": result['filename'],
                                    "type": "user_upload",
                                    "session_id": attachment.get("session_id")
                                }
                                for _ in result["chunks"]
                            ]
                        )
                        logger.info(f"Indexed {len(result['chunks'])} chunks from {result['filename']}")
                
            except Exception as e:
                logger.error(f"Error processing attachment: {e}")
                processed_content += f"\n[Error processing {attachment.get('filename', 'file')}]\n"
        
        return processed_content
    
    async def check_escalation(
        self,
        message: str,
        context: AgentContext,
        message_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """Check if escalation is needed."""
        try:
            escalation_tool = self.tools['escalation']
            
            result = await escalation_tool.should_escalate(
                message=message,
                message_history=message_history,
                metadata={
                    "session_id": context.session_id,
                    "message_count": context.message_count,
                    "already_escalated": context.escalated
                }
            )
            
            # Create ticket if escalation needed
            if result["escalate"] and not context.escalated:
                result["ticket"] = escalation_tool.create_escalation_ticket(
                    session_id=context.session_id,
                    escalation_result=result,
                    user_info={"user_id": context.user_id}
                )
                context.escalated = True
                logger.info(f"Escalation triggered for session {context.session_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Escalation check error: {e}")
            return {"escalate": False, "confidence": 0.0}
    
    async def store_conversation_memory(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        important_facts: List[str] = None
    ) -> None:
        """Store important information in memory."""
        try:
            memory_tool = self.tools['memory']
            
            # Store user message as context
            await memory_tool.store_memory(
                session_id=session_id,
                content=f"User: {user_message[:200]}",
                content_type="context",
                importance=0.5
            )
            
            # Store agent response summary
            if len(agent_response) > 100:
                await memory_tool.store_memory(
                    session_id=session_id,
                    content=f"Agent: {agent_response[:200]}",
                    content_type="context",
                    importance=0.4
                )
            
            # Store any identified important facts
            if important_facts:
                for fact in important_facts:
                    await memory_tool.store_memory(
                        session_id=session_id,
                        content=fact,
                        content_type="fact",
                        importance=0.8
                    )
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
    
    def extract_important_facts(
        self,
        message: str,
        response: str
    ) -> List[str]:
        """Extract important facts from conversation."""
        facts = []
        
        # Look for user information patterns
        import re
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        for email in emails:
            facts.append(f"User email: {email}")
        
        # Phone pattern
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, message)
        for phone in phones:
            facts.append(f"User phone: {phone}")
        
        # Order/ticket number pattern
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
        # Build response based on available information
        response_parts = []
        
        # Add greeting if first message
        if context == "No previous context available for this session.":
            response_parts.append("Hello! I'm here to help you today.")
        
        # Add information from knowledge base
        if sources:
            response_parts.append("Based on our information:")
            for i, source in enumerate(sources[:2], 1):
                response_parts.append(f"{i}. {source['content'][:200]}...")
        
        # Add escalation message if needed
        if escalation.get("escalate"):
            response_parts.append(
                "\nI understand this is important to you. "
                "I'm connecting you with a human support specialist who can better assist you."
            )
            if escalation.get("ticket"):
                response_parts.append(
                    f"Your ticket number is: {escalation['ticket']['ticket_id']}"
                )
        
        # Default helpful response if no specific information
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
        message_history: Optional[List[Dict]] = None
    ) -> AgentResponse:
        """
        Process a user message and generate response.
        
        Args:
            session_id: Session identifier
            message: User message
            attachments: Optional file attachments
            user_id: Optional user identifier
            message_history: Previous messages
            
        Returns:
            AgentResponse with generated response and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Get or create context
            context = self.get_or_create_context(session_id)
            context.user_id = user_id
            context.message_count += 1
            
            # Load session context from memory
            session_context = await self.load_session_context(session_id)
            
            # Process attachments if any
            attachment_context = await self.process_attachments(attachments) if attachments else ""
            
            # Search knowledge base for relevant information
            sources = await self.search_knowledge_base(message)
            
            # Check for escalation
            escalation = await self.check_escalation(message, context, message_history)
            
            # Generate response
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            # Extract and store important facts
            facts = self.extract_important_facts(message, response_text)
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text,
                important_facts=facts
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Build response
            response = AgentResponse(
                message=response_text,
                sources=sources[:3],  # Limit sources in response
                requires_escalation=escalation.get("escalate", False),
                confidence=escalation.get("confidence", 0.95),
                tools_used=["rag", "memory", "escalation"],
                processing_time=processing_time
            )
            
            # Add metadata
            response.metadata = {
                "session_id": session_id,
                "message_count": context.message_count,
                "has_context": bool(session_context),
                "facts_extracted": len(facts)
            }
            
            if escalation.get("ticket"):
                response.metadata["ticket_id"] = escalation["ticket"]["ticket_id"]
            
            logger.info(
                f"Processed message for session {session_id} in {processing_time:.2f}s "
                f"(escalate: {response.requires_escalation})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            
            # Return error response
            return AgentResponse(
                message="I apologize, but I encountered an error processing your request. "
                        "Please try again or contact support directly.",
                requires_escalation=True,
                confidence=0.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def stream_response(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response for real-time interaction.
        
        Yields:
            Updates as they're generated
        """
        try:
            # Initial processing
            yield {
                "type": "start",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Load context
            yield {
                "type": "status",
                "message": "Loading conversation context...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            context = self.get_or_create_context(session_id)
            session_context = await self.load_session_context(session_id)
            
            # Process attachments
            if attachments:
                yield {
                    "type": "status",
                    "message": "Processing attachments...",
                    "timestamp": datetime.utcnow().isoformat()
                }
                attachment_context = await self.process_attachments(attachments)
            
            # Search knowledge base
            yield {
                "type": "status",
                "message": "Searching knowledge base...",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            sources = await self.search_knowledge_base(message)
            
            if sources:
                yield {
                    "type": "sources",
                    "sources": sources[:3],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check escalation
            escalation = await self.check_escalation(message, context)
            
            if escalation.get("escalate"):
                yield {
                    "type": "escalation",
                    "required": True,
                    "reason": escalation.get("reasons", []),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Generate and stream response
            response_text = await self.generate_response(
                message=message,
                context=session_context,
                sources=sources,
                escalation=escalation
            )
            
            # Simulate streaming by sending response in chunks
            words = response_text.split()
            chunk_size = 5
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i+chunk_size])
                yield {
                    "type": "text",
                    "content": chunk + ' ',
                    "timestamp": datetime.utcnow().isoformat()
                }
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Store in memory
            await self.store_conversation_memory(
                session_id=session_id,
                user_message=message,
                agent_response=response_text
            )
            
            # Final completion
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
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        try:
            # Remove context
            if session_id in self.contexts:
                del self.contexts[session_id]
            
            # Clean up old memories (optional)
            memory_tool = self.tools['memory']
            await memory_tool.cleanup_old_memories(days=30)
            
            logger.info(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def cleanup(self) -> None:
        """Clean up all agent resources."""
        logger.info("Cleaning up agent resources...")
        
        # Clean up all sessions
        for session_id in list(self.contexts.keys()):
            await self.cleanup_session(session_id)
        
        # Clean up tools
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'cleanup'):
                await tool.cleanup()
                logger.info(f"Cleaned up {tool_name} tool")
        
        logger.info("Agent cleanup complete")
