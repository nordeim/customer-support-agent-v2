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
