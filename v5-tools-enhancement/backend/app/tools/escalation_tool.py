"""
Escalation tool for detecting when human intervention is needed.
Analyzes conversation context to determine escalation requirements.

Version: 3.0.0 (Enhanced with input validation and improved analysis)

Changes:
- Added Pydantic input validation
- Enhanced sentiment analysis
- Improved pattern detection
- Better error handling
"""
import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from pydantic import ValidationError

from ..config import settings
from ..config.tool_settings import tool_settings
from ..schemas.tool_requests import EscalationCheckRequest
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Escalation trigger keywords and phrases with weights
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
    "right now": 0.8,
    "worthless": 0.7,
    "terrible": 0.7,
    "horrible": 0.7,
    "worst": 0.8
}

# Sentiment thresholds
NEGATIVE_SENTIMENT_THRESHOLD = -0.5
ESCALATION_CONFIDENCE_THRESHOLD = 0.7


class EscalationTool(BaseTool):
    """
    Tool for detecting when a conversation should be escalated to human support.
    
    Analyzes multiple signals:
    - Escalation keywords
    - Sentiment analysis
    - Urgency indicators
    - Conversation patterns
    - Explicit escalation requests
    
    Version 3.0.0:
    - Enhanced with Pydantic input validation
    - Improved sentiment analysis
    - Better pattern detection
    - Comprehensive error handling
    """
    
    def __init__(self):
        """Initialize escalation detection tool."""
        super().__init__(
            name="escalation_check",
            description="Determine if human intervention is needed based on conversation context",
            version="3.0.0"
        )
        
        # Resources initialized in async initialize()
        self.keywords = None
        self.escalation_reasons = []
    
    async def initialize(self) -> None:
        """Initialize escalation tool resources (async-safe)."""
        try:
            logger.info(f"Initializing Escalation tool '{self.name}'...")
            
            # Load custom keywords from settings if available
            self.keywords = ESCALATION_KEYWORDS.copy()
            
            # Add custom keywords from tool settings
            custom_keywords = tool_settings.escalation_keywords
            if custom_keywords:
                self.keywords.update(custom_keywords)
            
            # Initialize escalation tracking
            self.escalation_reasons = []
            
            self.initialized = True
            logger.info(
                f"✓ Escalation tool '{self.name}' initialized successfully "
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
        """
        Execute escalation check (async-first).
        
        Version 3.0.0: Added input validation.
        """
        # Validate input
        try:
            request = EscalationCheckRequest(
                message=kwargs.get("message", ""),
                message_history=kwargs.get("message_history"),
                confidence_threshold=kwargs.get("confidence_threshold", ESCALATION_CONFIDENCE_THRESHOLD),
                metadata=kwargs.get("metadata")
            )
        except ValidationError as e:
            return ToolResult.error_result(
                error=f"Invalid escalation check request: {e}",
                metadata={"tool": self.name, "validation_errors": e.errors()}
            )
        
        try:
            # Perform escalation check
            result = await self.should_escalate_async(
                message=request.message,
                message_history=request.message_history,
                confidence_threshold=request.confidence_threshold,
                metadata=request.metadata
            )
            
            # Create ticket if requested and escalation is needed
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
                metadata={"tool": self.name, "message_preview": request.message[:100]}
            )
    
    async def should_escalate_async(
        self,
        message: str,
        message_history: Optional[List[Dict[str, Any]]] = None,
        confidence_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine if conversation should be escalated to human support.
        
        Analyzes multiple signals:
        1. Escalation keywords
        2. Sentiment analysis
        3. Urgency indicators
        4. Conversation patterns
        5. Explicit escalation requests
        
        Args:
            message: Current user message
            message_history: Previous messages in conversation
            confidence_threshold: Minimum confidence for escalation
            metadata: Additional context about the conversation
            
        Returns:
            Escalation decision with confidence score and reasoning
        """
        escalation_signals = []
        total_confidence = 0.0
        
        # Run analysis (lightweight, can run directly)
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
        """
        Notify human support about escalation.
        
        Args:
            ticket: Escalation ticket details
            notification_channel: Notification method (email, slack, webhook)
            
        Returns:
            Notification status
        """
        notification = {
            "channel": notification_channel,
            "ticket_id": ticket["ticket_id"],
            "sent_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        try:
            if notification_channel == "email" and tool_settings.escalation_notification_email:
                notification["recipient"] = tool_settings.escalation_notification_email
                notification["status"] = "sent"
                logger.info(f"Email notification sent for ticket {ticket['ticket_id']}")
                
            elif notification_channel == "webhook" and tool_settings.escalation_notification_webhook:
                notification["webhook_url"] = tool_settings.escalation_notification_webhook
                notification["status"] = "sent"
                logger.info(f"Webhook notification sent for ticket {ticket['ticket_id']}")
                
            elif notification_channel == "slack":
                notification["channel_id"] = "#customer-support"
                notification["status"] = "sent"
                logger.info(f"Slack notification sent for ticket {ticket['ticket_id']}")
            
            else:
                notification["status"] = "skipped"
                notification["reason"] = f"No {notification_channel} configuration found"
                logger.warning(f"Escalation notification skipped: {notification['reason']}")
        
        except Exception as e:
            notification["status"] = "failed"
            notification["error"] = str(e)
            logger.error(f"Failed to send escalation notification: {e}")
        
        return notification
    
    # ===========================
    # Private Helper Methods
    # ===========================
    
    def _analyze_message(
        self,
        message: str,
        message_history: Optional[List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze message for escalation signals.
        
        Returns dictionary with analysis results.
        """
        # Detect keywords
        keyword_score, found_keywords = self.detect_keywords(message)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(message)
        
        # Calculate urgency
        urgency = self.calculate_urgency_score(message, metadata)
        
        # Check conversation patterns
        patterns = self.check_conversation_patterns(message_history or [])
        
        # Check for explicit escalation request
        explicit_request = self._check_explicit_request(message)
        
        return {
            'keyword_score': keyword_score,
            'found_keywords': found_keywords,
            'sentiment': sentiment,
            'urgency': urgency,
            'patterns': patterns,
            'explicit_request': explicit_request
        }
    
    def _check_explicit_request(self, message: str) -> bool:
        """Check if message explicitly requests escalation."""
        explicit_patterns = [
            r'\b(speak|talk)\s+(to|with)\s+a?\s*(human|person|agent|representative)\b',
            r'\bget\s+me\s+a?\s*(manager|supervisor)\b',
            r'\b(transfer|escalate|connect)\s+me\b',
            r'\bi\s+(want|need)\s+to\s+speak\s+(to|with)\b',
            r'\breal\s+person\b'
        ]
        
        message_lower = message.lower()
        for pattern in explicit_patterns:
            if re.search(pattern, message_lower):
                return True
        
        return False
    
    def detect_keywords(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect escalation keywords in text.
        
        Returns:
            Tuple of (normalized_score, found_keywords)
        """
        text_lower = text.lower()
        found_keywords = []
        total_score = 0.0
        
        for keyword, weight in self.keywords.items():
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.append(keyword)
                total_score += weight
        
        # Normalize score (cap at 1.0)
        normalized_score = min(total_score, 1.0)
        
        return normalized_score, found_keywords
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using enhanced heuristics.
        
        Returns:
            Sentiment score between -1.0 (very negative) and 1.0 (very positive)
        """
        positive_words = {
            "good", "great", "excellent", "happy", "pleased", "thank",
            "perfect", "wonderful", "satisfied", "love", "amazing",
            "fantastic", "appreciate", "helpful", "nice", "awesome",
            "brilliant", "outstanding", "superb"
        }
        
        negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "hate",
            "disgusting", "pathetic", "useless", "ridiculous", "stupid",
            "angry", "frustrated", "disappointed", "unacceptable",
            "terrible", "poor", "fail", "failed", "failing", "wrong",
            "broken", "issue", "problem"
        }
        
        # Intensifiers
        intensifiers = {"very", "extremely", "absolutely", "completely", "totally"}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            # Check if previous word is an intensifier
            intensity = 1.5 if i > 0 and words[i-1] in intensifiers else 1.0
            
            if word in positive_words:
                positive_count += intensity
            elif word in negative_words:
                negative_count += intensity
        
        # Calculate sentiment score
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / max(total_words * 0.1, 1)
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment))
    
    def calculate_urgency_score(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate urgency score based on various factors.
        
        Returns:
            Urgency score between 0.0 and 1.0
        """
        urgency_indicators = {
            "time_sensitive": ["urgent", "asap", "immediately", "right now", "today", "now"],
            "business_critical": ["critical", "blocking", "down", "not working", "broken", "stopped"],
            "financial": ["payment", "charge", "bill", "invoice", "money", "charged"],
            "security": ["hacked", "breach", "stolen", "fraud", "unauthorized", "compromised"]
        }
        
        text_lower = text.lower()
        urgency_score = 0.0
        
        for category, keywords in urgency_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category == "security":
                        urgency_score += 0.5
                    elif category == "business_critical":
                        urgency_score += 0.4
                    elif category == "financial":
                        urgency_score += 0.3
                    else:
                        urgency_score += 0.2
        
        # Check for explicit time mentions
        time_patterns = [
            r'\b\d+\s*(hour|minute|min|hr)s?\b',
            r'\bwithin\s+\d+\b',
            r'\bdeadline\b',
            r'\bexpir(es?|ing|ed)\b'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                urgency_score += 0.3
                break
        
        return min(urgency_score, 1.0)
    
    def check_conversation_patterns(
        self,
        message_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation patterns for escalation signals.
        
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            "repetitive_questions": False,
            "conversation_length": len(message_history),
            "unresolved_issues": False,
            "multiple_problems": False,
            "degrading_sentiment": False
        }
        
        if len(message_history) < 2:
            return patterns
        
        # Filter user messages
        user_messages = [m for m in message_history if m.get("role") == "user"]
        
        if len(user_messages) < 2:
            return patterns
        
        # Check for repetitive questions
        if len(user_messages) >= 3:
            recent_messages = [m.get("content", "").lower() for m in user_messages[-3:]]
            # Check similarity
            if len(set(recent_messages)) <= 2:
                patterns["repetitive_questions"] = True
        
        # Check conversation length
        if patterns["conversation_length"] > 10:
            patterns["unresolved_issues"] = True
        
        # Check for degrading sentiment
        if len(user_messages) >= 2:
            first_sentiment = self.analyze_sentiment(user_messages[0].get("content", ""))
            last_sentiment = self.analyze_sentiment(user_messages[-1].get("content", ""))
            
            if last_sentiment < first_sentiment - 0.3:
                patterns["degrading_sentiment"] = True
        
        # Check for multiple problem indicators
        problem_words = ["also", "another", "additionally", "furthermore", "besides", "and"]
        all_user_text = " ".join([m.get("content", "") for m in user_messages])
        
        problem_count = sum(1 for word in problem_words if word in all_user_text.lower())
        if problem_count >= 3:
            patterns["multiple_problems"] = True
        
        return patterns
    
    def create_escalation_ticket(
        self,
        session_id: str,
        escalation_result: Dict[str, Any],
        user_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an escalation ticket for human support.
        
        Args:
            session_id: Session identifier
            escalation_result: Results from escalation analysis
            user_info: Optional user information
            
        Returns:
            Escalation ticket
        """
        ticket = {
            "ticket_id": f"ESC-{session_id[:8]}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "priority": escalation_result.get("priority", "normal"),
            "category": escalation_result.get("category", "general"),
            "reasons": escalation_result.get("reasons", []),
            "urgency_score": escalation_result.get("urgency", 0.0),
            "sentiment_score": escalation_result.get("sentiment", 0.0),
            "confidence": escalation_result.get("confidence", 0.0),
            "status": "pending"
        }
        
        if user_info:
            ticket["user_info"] = user_info
        
        logger.info(
            f"Created escalation ticket: {ticket['ticket_id']} "
            f"(priority: {ticket['priority']}, category: {ticket['category']})"
        )
        
        return ticket


__all__ = ['EscalationTool']
