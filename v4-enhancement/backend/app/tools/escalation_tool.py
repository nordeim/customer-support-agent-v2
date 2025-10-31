"""
Escalation tool for detecting when human intervention is needed.
Analyzes conversation context to determine escalation requirements.

Version: 2.0.0 (Optimized - removed unnecessary thread pool)
"""
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..config import settings
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Escalation trigger keywords and phrases
ESCALATION_KEYWORDS = {
    "urgent": 1.0,
    "emergency": 1.0,
    "complaint": 0.9,
    "angry": 0.9,
    "frustrated": 0.8,
    "disappointed": 0.8,
    "unacceptable": 0.8,
    "legal": 0.9,
    "lawsuit": 1.0,
    "lawyer": 0.9,
    "sue": 0.9,
    "refund": 0.7,
    "compensation": 0.7,
    "manager": 0.8,
    "supervisor": 0.8,
    "human": 0.7,
    "speak to someone": 0.8,
    "talk to a person": 0.8,
    "not helping": 0.7,
    "doesn't work": 0.6,
    "broken": 0.6,
    "critical": 0.9,
    "immediate": 0.8,
    "asap": 0.8,
    "right now": 0.8
}

# Sentiment thresholds
NEGATIVE_SENTIMENT_THRESHOLD = -0.5
ESCALATION_CONFIDENCE_THRESHOLD = 0.7


class EscalationTool(BaseTool):
    """
    Tool for detecting when a conversation should be escalated.
    
    Version 2.0.0:
    - FIXED: Removed unnecessary thread pool for _analyze_message
    - Optimized performance by running analysis directly
    - Enhanced error handling
    """
    
    def __init__(self):
        """Initialize escalation detection tool."""
        super().__init__(
            name="escalation_check",
            description="Determine if human intervention is needed",
            version="2.0.0"
        )
        
        # Resources initialized in async initialize()
        self.keywords = None
        self.escalation_reasons = []
    
    async def initialize(self) -> None:
        """Initialize escalation tool resources (async-safe)."""
        try:
            logger.info(f"Initializing Escalation tool '{self.name}'...")
            
            # Load custom keywords from settings
            self.keywords = ESCALATION_KEYWORDS.copy()
            
            # Add custom keywords if available
            if hasattr(settings, 'escalation_keywords'):
                custom_keywords = settings.escalation_keywords
                if isinstance(custom_keywords, dict):
                    self.keywords.update(custom_keywords)
                elif isinstance(custom_keywords, list):
                    for keyword in custom_keywords:
                        if keyword not in self.keywords:
                            self.keywords[keyword] = 0.8
            
            # Initialize escalation tracking
            self.escalation_reasons = []
            
            self.initialized = True
            logger.info(
                f"✓ Escalation tool '{self.name}' initialized "
                f"with {len(self.keywords)} keywords"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Escalation tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup escalation tool resources."""
        try:
            logger.info(f"Cleaning up Escalation tool '{self.name}'...")
            
            self.escalation_reasons = []
            self.keywords = None
            
            self.initialized = False
            logger.info(f"✓ Escalation tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Escalation tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute escalation check (async-first)."""
        message = kwargs.get("message")
        
        if not message:
            return ToolResult.error_result(
                error="message parameter is required",
                metadata={"tool": self.name}
            )
        
        try:
            # Perform escalation check
            result = await self.should_escalate_async(
                message=message,
                message_history=kwargs.get("message_history"),
                confidence_threshold=kwargs.get(
                    "confidence_threshold",
                    ESCALATION_CONFIDENCE_THRESHOLD
                ),
                metadata=kwargs.get("metadata")
            )
            
            # Create ticket if requested and escalation needed
            if result["escalate"] and kwargs.get("create_ticket", False):
                ticket = self.create_escalation_ticket(
                    session_id=kwargs.get("session_id", "unknown"),
                    escalation_result=result,
                    user_info=kwargs.get("user_info")
                )
                result["ticket"] = ticket
                
                # Send notification if configured
                if kwargs.get("notify", False):
                    notification = await self.notify_human_support_async(
                        ticket,
                        kwargs.get("notification_channel", "email")
                    )
                    result["notification"] = notification
            
            return ToolResult.success_result(
                data=result,
                metadata={
                    "tool": self.name,
                    "escalated": result["escalate"],
                    "confidence": result["confidence"],
                    "reasons_count": len(result.get("reasons", []))
                }
            )
            
        except Exception as e:
            logger.error(f"Escalation execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "message_preview": message[:100]}
            )
    
    async def should_escalate_async(
        self,
        message: str,
        message_history: Optional[List[Dict[str, Any]]] = None,
        confidence_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine if conversation should be escalated.
        
        Version 2.0.0: FIXED - Runs analysis directly without thread pool.
        """
        escalation_signals = []
        total_confidence = 0.0
        
        # FIXED: Run analysis directly (it's fast enough, no need for thread pool)
        analysis_result = self._analyze_message(
            message,
            message_history,
            metadata
        )
        
        # Unpack analysis results
        keyword_score = analysis_result['keyword_score']
        found_keywords = analysis_result['found_keywords']
        sentiment = analysis_result['sentiment']
        urgency = analysis_result['urgency']
        patterns = analysis_result['patterns']
        explicit_request = analysis_result['explicit_request']
        
        # 1. Check for escalation keywords
        if keyword_score > 0:
            escalation_signals.append(f"Keywords detected: {', '.join(found_keywords)}")
            total_confidence += keyword_score * 0.4  # 40% weight
        
        # 2. Analyze sentiment
        if sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            escalation_signals.append(f"Negative sentiment: {sentiment:.2f}")
            total_confidence += abs(sentiment) * 0.2  # 20% weight
        
        # 3. Check urgency
        if urgency > 0.5:
            escalation_signals.append(f"High urgency: {urgency:.2f}")
            total_confidence += urgency * 0.2  # 20% weight
        
        # 4. Analyze conversation patterns
        if patterns['repetitive_questions']:
            escalation_signals.append("Repetitive questions detected")
            total_confidence += 0.1
        
        if patterns['unresolved_issues']:
            escalation_signals.append("Long conversation without resolution")
            total_confidence += 0.1
        
        if patterns['degrading_sentiment']:
            escalation_signals.append("Degrading customer sentiment")
            total_confidence += 0.15
        
        if patterns['multiple_problems']:
            escalation_signals.append("Multiple issues reported")
            total_confidence += 0.1
        
        # 5. Check for explicit escalation request
        if explicit_request:
            escalation_signals.append("Explicit escalation request")
            total_confidence = 1.0  # Always escalate on explicit request
        
        # Determine if should escalate
        should_escalate = total_confidence >= confidence_threshold
        
        # Build response
        result = {
            "escalate": should_escalate,
            "confidence": min(total_confidence, 1.0),
            "reasons": escalation_signals,
            "urgency": urgency,
            "sentiment": sentiment,
            "threshold": confidence_threshold
        }
        
        # Add escalation category if escalating
        if should_escalate:
            if "legal" in message.lower() or "lawsuit" in message.lower():
                result["category"] = "legal"
                result["priority"] = "high"
            elif urgency > 0.7:
                result["category"] = "urgent"
                result["priority"] = "high"
            elif sentiment < -0.7:
                result["category"] = "complaint"
                result["priority"] = "medium"
            else:
                result["category"] = "general"
                result["priority"] = "normal"
        
        logger.info(
            f"Escalation check: {should_escalate} "
            f"(confidence: {total_confidence:.2f}, reasons: {len(escalation_signals)})"
        )
        
        return result
    
    async def notify_human_support_async(
        self,
        ticket: Dict[str, Any],
        notification_channel: str = "email"
    ) -> Dict[str, Any]:
        """Notify human support about escalation (async)."""
        # Simulate notification sending
        await asyncio.sleep(0.1)
        
        notification = {
            "channel": notification_channel,
            "ticket_id": ticket["ticket_id"],
            "sent_at": datetime.utcnow().isoformat(),
            "status": "sent"
        }
        
        if notification_channel == "email":
            logger.info(f"Email notification sent for ticket {ticket['ticket_id']}")
            notification["recipient"] = getattr(
                settings,
                'escalation_notification_email',
                'support@example.com'
            )
            
        elif notification_channel == "slack":
            logger.info(f"Slack notification sent for ticket {ticket['ticket_id']}")
            notification["channel_id"] = "#customer-support"
        
        return notification
    
    # ===========================
    # Private Helper Methods (Sync - Fast Operations)
    # ===========================
    
    def _analyze_message(
        self,
        message: str,
        message_history: Optional[List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze message for escalation signals (sync, CPU-bound but fast).
        
        Version 2.0.0: FIXED - Runs directly, no thread pool needed.
        """
        # Detect keywords
        keyword_score, found_keywords = self.detect_keywords(message)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(message)
        
        # Calculate urgency
        urgency = self.calculate_urgency_score(message, metadata)
        
        # Check conversation patterns
        patterns = self.check_conversation_patterns(message_history or [
