"""
Escalation tool for detecting when human intervention is needed.
Analyzes conversation context to determine escalation requirements.
"""
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config import settings
from .base_tool import BaseTool

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
    Tool for detecting when a conversation should be escalated to human support.
    Analyzes various signals including keywords, sentiment, and context.
    """
    
    def __init__(self):
        """Initialize escalation detection tool."""
        super().__init__(
            name="escalation_check",
            description="Determine if human intervention is needed based on conversation context"
        )
    
    def _setup(self) -> None:
        """Setup escalation rules and configurations."""
        # Load custom keywords from settings if available
        self.keywords = ESCALATION_KEYWORDS.copy()
        
        # Add any custom keywords from configuration
        if hasattr(settings, 'escalation_keywords'):
            self.keywords.update(settings.escalation_keywords)
        
        # Escalation reasons for better tracking
        self.escalation_reasons = []
        
        logger.info(f"Escalation tool initialized with {len(self.keywords)} keywords")
    
    def detect_keywords(self, text: str) -> Tuple[float, List[str]]:
        """
        Detect escalation keywords in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (escalation score, found keywords)
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
        Analyze sentiment of text using basic heuristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        # Simple sentiment analysis using word lists
        positive_words = {
            "good", "great", "excellent", "happy", "pleased", "thank",
            "perfect", "wonderful", "satisfied", "love", "amazing"
        }
        
        negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "hate",
            "disgusting", "pathetic", "useless", "ridiculous", "stupid"
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Calculate sentiment score
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / max(total_words * 0.1, 1)
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment))
    
    def check_conversation_patterns(
        self,
        message_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation patterns for escalation signals.
        
        Args:
            message_history: List of previous messages
            
        Returns:
            Pattern analysis results
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
        
        # Check for repetitive questions (user asking same thing multiple times)
        user_messages = [m for m in message_history if m.get("role") == "user"]
        if len(user_messages) >= 3:
            # Simple check: similar messages
            recent_messages = [m.get("content", "").lower() for m in user_messages[-3:]]
            if len(set(recent_messages)) == 1:  # All same
                patterns["repetitive_questions"] = True
        
        # Check conversation length (too long might indicate unresolved issue)
        if patterns["conversation_length"] > 10:
            patterns["unresolved_issues"] = True
        
        # Check for degrading sentiment
        if len(user_messages) >= 2:
            first_sentiment = self.analyze_sentiment(user_messages[0].get("content", ""))
            last_sentiment = self.analyze_sentiment(user_messages[-1].get("content", ""))
            
            if last_sentiment < first_sentiment - 0.3:
                patterns["degrading_sentiment"] = True
        
        # Check for multiple problem indicators
        problem_words = ["also", "another", "additionally", "furthermore", "besides"]
        all_user_text = " ".join([m.get("content", "") for m in user_messages])
        
        problem_count = sum(1 for word in problem_words if word in all_user_text.lower())
        if problem_count >= 2:
            patterns["multiple_problems"] = True
        
        return patterns
    
    def calculate_urgency_score(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate urgency score based on various factors.
        
        Args:
            text: Current message text
            metadata: Optional metadata about the conversation
            
        Returns:
            Urgency score (0.0 to 1.0)
        """
        urgency_indicators = {
            "time_sensitive": ["urgent", "asap", "immediately", "right now", "today"],
            "business_critical": ["critical", "blocking", "down", "not working", "broken"],
            "financial": ["payment", "charge", "bill", "invoice", "money"],
            "security": ["hacked", "breach", "stolen", "fraud", "unauthorized"]
        }
        
        text_lower = text.lower()
        urgency_score = 0.0
        
        for category, keywords in urgency_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category == "security":
                        urgency_score += 0.5  # Security issues are highest priority
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
    
    async def should_escalate(
        self,
        message: str,
        message_history: Optional[List[Dict[str, Any]]] = None,
        confidence_threshold: float = ESCALATION_CONFIDENCE_THRESHOLD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine if conversation should be escalated to human support.
        
        Args:
            message: Current user message
            message_history: Previous messages in conversation
            confidence_threshold: Minimum confidence for escalation
            metadata: Additional context about the conversation
            
        Returns:
            Escalation decision with reasoning
        """
        escalation_signals = []
        total_confidence = 0.0
        
        # 1. Check for escalation keywords
        keyword_score, found_keywords = self.detect_keywords(message)
        if keyword_score > 0:
            escalation_signals.append(f"Keywords detected: {', '.join(found_keywords)}")
            total_confidence += keyword_score * 0.4  # 40% weight
        
        # 2. Analyze sentiment
        sentiment = self.analyze_sentiment(message)
        if sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
            escalation_signals.append(f"Negative sentiment: {sentiment:.2f}")
            total_confidence += abs(sentiment) * 0.2  # 20% weight
        
        # 3. Check urgency
        urgency = self.calculate_urgency_score(message, metadata)
        if urgency > 0.5:
            escalation_signals.append(f"High urgency: {urgency:.2f}")
            total_confidence += urgency * 0.2  # 20% weight
        
        # 4. Analyze conversation patterns
        if message_history:
            patterns = self.check_conversation_patterns(message_history)
            
            if patterns["repetitive_questions"]:
                escalation_signals.append("Repetitive questions detected")
                total_confidence += 0.1
            
            if patterns["unresolved_issues"]:
                escalation_signals.append("Long conversation without resolution")
                total_confidence += 0.1
            
            if patterns["degrading_sentiment"]:
                escalation_signals.append("Degrading customer sentiment")
                total_confidence += 0.15
            
            if patterns["multiple_problems"]:
                escalation_signals.append("Multiple issues reported")
                total_confidence += 0.1
        
        # 5. Check for explicit escalation request
        explicit_patterns = [
            r'\b(speak|talk)\s+(to|with)\s+a?\s*(human|person|agent|representative)\b',
            r'\bget\s+me\s+a?\s*(manager|supervisor)\b',
            r'\b(transfer|escalate|connect)\s+me\b'
        ]
        
        for pattern in explicit_patterns:
            if re.search(pattern, message.lower()):
                escalation_signals.append("Explicit escalation request")
                total_confidence = 1.0  # Always escalate on explicit request
                break
        
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
    
    def create_escalation_ticket(
        self,
        session_id: str,
        escalation_result: Dict[str, Any],
        user_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an escalation ticket for human support.
        
        Args:
            session_id: Current session ID
            escalation_result: Result from should_escalate
            user_info: Optional user information
            
        Returns:
            Ticket information
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
            "status": "pending"
        }
        
        if user_info:
            ticket["user_info"] = user_info
        
        logger.info(f"Created escalation ticket: {ticket['ticket_id']}")
        
        return ticket
    
    async def notify_human_support(
        self,
        ticket: Dict[str, Any],
        notification_channel: str = "email"
    ) -> Dict[str, Any]:
        """
        Notify human support about escalation.
        
        Args:
            ticket: Escalation ticket
            notification_channel: How to notify (email, slack, etc.)
            
        Returns:
            Notification status
        """
        # This would integrate with actual notification systems
        # For now, we'll simulate the notification
        
        notification = {
            "channel": notification_channel,
            "ticket_id": ticket["ticket_id"],
            "sent_at": datetime.utcnow().isoformat(),
            "status": "sent"
        }
        
        if notification_channel == "email":
            # Simulate email sending
            logger.info(f"Email notification sent for ticket {ticket['ticket_id']}")
            notification["recipient"] = "support@example.com"
            
        elif notification_channel == "slack":
            # Simulate Slack notification
            logger.info(f"Slack notification sent for ticket {ticket['ticket_id']}")
            notification["channel_id"] = "#customer-support"
        
        return notification
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute escalation check.
        
        Accepts:
            message: Current user message (required)
            message_history: Conversation history (optional)
            confidence_threshold: Threshold for escalation (optional)
            create_ticket: Whether to create a ticket if escalated (optional)
            
        Returns:
            Escalation decision and details
        """
        message = kwargs.get("message")
        
        if not message:
            return {
                "success": False,
                "error": "message parameter is required"
            }
        
        # Perform escalation check
        result = await self.should_escalate(
            message=message,
            message_history=kwargs.get("message_history"),
            confidence_threshold=kwargs.get("confidence_threshold", ESCALATION_CONFIDENCE_THRESHOLD),
            metadata=kwargs.get("metadata")
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
                notification = await self.notify_human_support(
                    ticket,
                    kwargs.get("notification_channel", "email")
                )
                result["notification"] = notification
        
        result["success"] = True
        return result
