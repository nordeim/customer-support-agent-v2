"""
Tool-specific configuration settings.
Defines feature flags and per-tool configurations for the agent system.

Version: 2.0.0 (Validated and enhanced)
"""
from typing import Dict, Any, Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

import logging

logger = logging.getLogger(__name__)


class ToolSettings(BaseSettings):
    """
    Tool-specific configuration with feature flags.
    Each tool can be enabled/disabled and configured independently.
    
    Version 2.0.0: Validated all settings and improved documentation.
    """
    
    # ===========================
    # Tool Feature Flags
    # ===========================
    
    enable_rag_tool: bool = Field(
        default=True,
        description="Enable RAG (Retrieval-Augmented Generation) tool"
    )
    
    enable_memory_tool: bool = Field(
        default=True,
        description="Enable Memory management tool"
    )
    
    enable_escalation_tool: bool = Field(
        default=True,
        description="Enable Escalation detection tool"
    )
    
    enable_attachment_tool: bool = Field(
        default=True,
        description="Enable Attachment processing tool"
    )
    
    # Future tools (disabled by default)
    enable_crm_tool: bool = Field(
        default=False,
        description="Enable CRM lookup tool"
    )
    
    enable_billing_tool: bool = Field(
        default=False,
        description="Enable Billing/invoice tool"
    )
    
    enable_inventory_tool: bool = Field(
        default=False,
        description="Enable Inventory lookup tool"
    )
    
    # ===========================
    # RAG Tool Configuration
    # ===========================
    
    rag_chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="RAG document chunk size in words"
    )
    
    rag_chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in words"
    )
    
    rag_search_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of RAG search results"
    )
    
    rag_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for RAG results"
    )
    
    rag_cache_enabled: bool = Field(
        default=True,
        description="Enable caching for RAG search results"
    )
    
    rag_cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="RAG cache TTL in seconds"
    )
    
    # ===========================
    # Memory Tool Configuration
    # ===========================
    
    memory_max_entries: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum memory entries per session"
    )
    
    memory_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Memory TTL in hours"
    )
    
    memory_cleanup_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days before cleaning old memories"
    )
    
    memory_importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum importance for memory retrieval"
    )
    
    # ===========================
    # Escalation Tool Configuration
    # ===========================
    
    escalation_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for escalation"
    )
    
    escalation_keywords: Dict[str, float] = Field(
        default_factory=lambda: {
            "urgent": 1.0,
            "emergency": 1.0,
            "complaint": 0.9,
            "legal": 0.9,
            "lawsuit": 1.0,
            "manager": 0.8,
            "supervisor": 0.8
        },
        description="Escalation keywords with weights"
    )
    
    escalation_notification_enabled: bool = Field(
        default=False,
        description="Enable automatic escalation notifications"
    )
    
    escalation_notification_email: Optional[str] = Field(
        default=None,
        description="Email address for escalation notifications"
    )
    
    escalation_notification_webhook: Optional[str] = Field(
        default=None,
        description="Webhook URL for escalation notifications"
    )
    
    # ===========================
    # Attachment Tool Configuration
    # ===========================
    
    attachment_max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum attachment file size in bytes"
    )
    
    attachment_allowed_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".doc", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png"
        ],
        description="Allowed file extensions for attachments"
    )
    
    attachment_chunk_for_rag: bool = Field(
        default=True,
        description="Automatically chunk attachments for RAG indexing"
    )
    
    attachment_temp_cleanup_hours: int = Field(
        default=24,
        ge=1,
        description="Hours before cleaning up temporary attachment files"
    )
    
    # ===========================
    # CRM Tool Configuration
    # ===========================
    
    crm_api_endpoint: Optional[str] = Field(
        default=None,
        description="CRM API endpoint URL"
    )
    
    crm_api_key: Optional[str] = Field(
        default=None,
        description="CRM API key (use secrets manager in production)"
    )
    
    crm_timeout: int = Field(
        default=10,
        ge=1,
        description="CRM API timeout in seconds"
    )
    
    crm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum CRM API retry attempts"
    )
    
    # ===========================
    # Validators
    # ===========================
    
    @field_validator('escalation_keywords', mode='before')
    @classmethod
    def parse_escalation_keywords(cls, v):
        """Parse escalation keywords from various formats."""
        if v is None:
            return {
                "urgent": 1.0,
                "emergency": 1.0,
                "complaint": 0.9,
                "legal": 0.9,
                "lawsuit": 1.0,
                "manager": 0.8,
                "supervisor": 0.8
            }
        
        if isinstance(v, dict):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('{'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated key=value pairs
            result = {}
            for pair in v.split(','):
                if '=' in pair:
                    key, value = pair.strip().split('=', 1)
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = 0.8
                else:
                    result[pair.strip()] = 0.8
            return result if result else cls.parse_escalation_keywords(None)
        
        return v
    
    @field_validator('attachment_allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from various formats."""
        default = [
            ".pdf", ".docx", ".doc", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png"
        ]
        
        if v is None:
            return default
        
        if isinstance(v, list):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('['):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        
        return default
    
    # ===========================
    # Helper Methods
    # ===========================
    
    def get_enabled_tools(self) -> List[str]:
        """
        Get list of enabled tool names.
        
        Returns:
            List of enabled tool identifiers
        """
        enabled = []
        
        if self.enable_rag_tool:
            enabled.append('rag')
        if self.enable_memory_tool:
            enabled.append('memory')
        if self.enable_escalation_tool:
            enabled.append('escalation')
        if self.enable_attachment_tool:
            enabled.append('attachment')
        if self.enable_crm_tool:
            enabled.append('crm')
        if self.enable_billing_tool:
            enabled.append('billing')
        if self.enable_inventory_tool:
            enabled.append('inventory')
        
        return enabled
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier ('rag', 'memory', etc.)
            
        Returns:
            Dictionary of tool-specific configuration
        """
        if tool_name == 'rag':
            return {
                'chunk_size': self.rag_chunk_size,
                'chunk_overlap': self.rag_chunk_overlap,
                'search_k': self.rag_search_k,
                'similarity_threshold': self.rag_similarity_threshold,
                'cache_enabled': self.rag_cache_enabled,
                'cache_ttl': self.rag_cache_ttl
            }
        
        elif tool_name == 'memory':
            return {
                'max_entries': self.memory_max_entries,
                'ttl_hours': self.memory_ttl_hours,
                'cleanup_days': self.memory_cleanup_days,
                'importance_threshold': self.memory_importance_threshold
            }
        
        elif tool_name == 'escalation':
            return {
                'confidence_threshold': self.escalation_confidence_threshold,
                'keywords': self.escalation_keywords,
                'notification_enabled': self.escalation_notification_enabled,
                'notification_email': self.escalation_notification_email,
                'notification_webhook': self.escalation_notification_webhook
            }
        
        elif tool_name == 'attachment':
            return {
                'max_file_size': self.attachment_max_file_size,
                'allowed_extensions': self.attachment_allowed_extensions,
                'chunk_for_rag': self.attachment_chunk_for_rag,
                'temp_cleanup_hours': self.attachment_temp_cleanup_hours
            }
        
        elif tool_name == 'crm':
            return {
                'api_endpoint': self.crm_api_endpoint,
                'api_key': self.crm_api_key,
                'timeout': self.crm_timeout,
                'max_retries': self.crm_max_retries
            }
        
        else:
            return {}
    
    def validate_tool_config(self, tool_name: str) -> List[str]:
        """
        Validate configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        if tool_name == 'crm' and self.enable_crm_tool:
            if not self.crm_api_endpoint:
                warnings.append("CRM tool enabled but no API endpoint configured")
            if not self.crm_api_key:
                warnings.append("CRM tool enabled but no API key configured")
        
        if tool_name == 'escalation' and self.escalation_notification_enabled:
            if not self.escalation_notification_email and not self.escalation_notification_webhook:
                warnings.append(
                    "Escalation notifications enabled but no email or webhook configured"
                )
        
        return warnings


# Create global instance
tool_settings = ToolSettings()

# Export
__all__ = ['ToolSettings', 'tool_settings']
