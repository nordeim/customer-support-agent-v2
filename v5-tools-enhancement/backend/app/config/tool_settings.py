"""
Tool-specific configuration settings.
Defines feature flags and per-tool configurations for the agent system.

Version: 3.0.0 (Enhanced with SecretStr for API keys and secrets management)

Changes:
- Added SecretStr for sensitive fields
- Added environment variable and secrets manager support
- Added production validation
- Enhanced error messages and documentation
"""
from typing import Dict, Any, Optional, List, Union
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings
import os
import logging

logger = logging.getLogger(__name__)


class ToolSettings(BaseSettings):
    """
    Tool-specific configuration with feature flags.
    Each tool can be enabled/disabled and configured independently.
    
    Version 3.0.0: Enhanced security with SecretStr and secrets management.
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
    # CRM Tool Configuration (ENHANCED WITH SECRETSTR)
    # ===========================
    
    crm_api_endpoint: Optional[str] = Field(
        default=None,
        description="CRM API endpoint URL"
    )
    
    crm_api_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "CRM API key. Supports:\n"
            "- Direct value (dev only)\n"
            "- env://VAR_NAME (load from environment)\n"
            "- secretsmanager://aws/secret-name (AWS Secrets Manager)\n"
            "Production must use env:// or secretsmanager:// prefix"
        )
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
    # Billing Tool Configuration (ENHANCED WITH SECRETSTR)
    # ===========================
    
    billing_api_endpoint: Optional[str] = Field(
        default=None,
        description="Billing API endpoint URL"
    )
    
    billing_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Billing API key (supports env:// and secretsmanager:// prefixes)"
    )
    
    # ===========================
    # Inventory Tool Configuration (ENHANCED WITH SECRETSTR)
    # ===========================
    
    inventory_api_endpoint: Optional[str] = Field(
        default=None,
        description="Inventory API endpoint URL"
    )
    
    inventory_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Inventory API key (supports env:// and secretsmanager:// prefixes)"
    )
    
    # ===========================
    # Validators (ENHANCED WITH SECRETS MANAGEMENT)
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
    
    @field_validator('crm_api_key', 'billing_api_key', 'inventory_api_key', mode='before')
    @classmethod
    def load_api_key_from_source(cls, v: Optional[Union[str, SecretStr]]) -> Optional[SecretStr]:
        """
        Load API key from various sources.
        
        ADDED in Version 3.0.0 for secure secrets management.
        
        Supports:
        - Direct value: "sk-abc123" (development only)
        - Environment variable: "env://CRM_API_KEY"
        - AWS Secrets Manager: "secretsmanager://aws/crm-api-key"
        
        Args:
            v: API key value or reference
            
        Returns:
            SecretStr with loaded value or None
            
        Raises:
            ValueError: If production uses direct value or loading fails
        """
        if v is None:
            return None
        
        # Already a SecretStr
        if isinstance(v, SecretStr):
            return v
        
        if not isinstance(v, str):
            raise ValueError(f"API key must be string or SecretStr, got {type(v)}")
        
        # Empty string = None
        if not v.strip():
            return None
        
        # Load from environment variable
        if v.startswith('env://'):
            env_var = v.replace('env://', '')
            env_value = os.getenv(env_var)
            
            if not env_value:
                logger.warning(f"Environment variable not set: {env_var}")
                return None
            
            logger.info(f"Loaded API key from environment variable: {env_var}")
            return SecretStr(env_value)
        
        # Load from AWS Secrets Manager
        elif v.startswith('secretsmanager://aws/'):
            secret_name = v.replace('secretsmanager://aws/', '')
            
            try:
                import boto3
                from botocore.exceptions import ClientError
                
                client = boto3.client('secretsmanager')
                
                try:
                    response = client.get_secret_value(SecretId=secret_name)
                    secret_value = response['SecretString']
                    
                    logger.info(f"Loaded API key from AWS Secrets Manager: {secret_name}")
                    return SecretStr(secret_value)
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    logger.error(f"Failed to load secret '{secret_name}': {error_code}")
                    raise ValueError(f"Cannot load secret from AWS: {error_code}")
                    
            except ImportError:
                raise ValueError(
                    "boto3 not installed. Install with: pip install boto3"
                )
        
        # Direct value (check if production)
        else:
            # Get environment from settings (if available)
            try:
                from ..config import settings
                if settings.environment == 'production':
                    raise ValueError(
                        "In production, API keys must use env:// or secretsmanager:// prefix. "
                        f"Example: env://CRM_API_KEY or secretsmanager://aws/crm-api-key"
                    )
            except ImportError:
                # Settings not available yet (during config loading)
                pass
            
            logger.warning("Using direct API key value (development only)")
            return SecretStr(v)
    
    # ===========================
    # Helper Methods (ENHANCED)
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
                'has_api_key': self.crm_api_key is not None,  # Don't expose actual key
                'timeout': self.crm_timeout,
                'max_retries': self.crm_max_retries
            }
        
        elif tool_name == 'billing':
            return {
                'api_endpoint': self.billing_api_endpoint,
                'has_api_key': self.billing_api_key is not None
            }
        
        elif tool_name == 'inventory':
            return {
                'api_endpoint': self.inventory_api_endpoint,
                'has_api_key': self.inventory_api_key is not None
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
        
        if tool_name == 'billing' and self.enable_billing_tool:
            if not self.billing_api_endpoint:
                warnings.append("Billing tool enabled but no API endpoint configured")
            if not self.billing_api_key:
                warnings.append("Billing tool enabled but no API key configured")
        
        if tool_name == 'inventory' and self.enable_inventory_tool:
            if not self.inventory_api_endpoint:
                warnings.append("Inventory tool enabled but no API endpoint configured")
            if not self.inventory_api_key:
                warnings.append("Inventory tool enabled but no API key configured")
        
        if tool_name == 'escalation' and self.escalation_notification_enabled:
            if not self.escalation_notification_email and not self.escalation_notification_webhook:
                warnings.append(
                    "Escalation notifications enabled but no email or webhook configured"
                )
        
        return warnings
    
    # ===========================
    # Secure API Key Access (NEW)
    # ===========================
    
    def get_crm_api_key(self) -> Optional[str]:
        """
        Get CRM API key value (use this instead of accessing field directly).
        
        ADDED in Version 3.0.0 for secure access.
        
        Returns:
            API key string or None if not set
        """
        if self.crm_api_key:
            return self.crm_api_key.get_secret_value()
        return None
    
    def get_billing_api_key(self) -> Optional[str]:
        """
        Get Billing API key value.
        
        Returns:
            API key string or None if not set
        """
        if self.billing_api_key:
            return self.billing_api_key.get_secret_value()
        return None
    
    def get_inventory_api_key(self) -> Optional[str]:
        """
        Get Inventory API key value.
        
        Returns:
            API key string or None if not set
        """
        if self.inventory_api_key:
            return self.inventory_api_key.get_secret_value()
        return None


# Create global instance
tool_settings = ToolSettings()

# Export
__all__ = ['ToolSettings', 'tool_settings']
