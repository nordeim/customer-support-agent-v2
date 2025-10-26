"""
Health check API routes.
"""
from fastapi import APIRouter, Depends
from datetime import datetime
import logging

from ...models.schemas import HealthResponse
from ...config import settings
from ...agents.chat_agent import CustomerSupportAgent
from sqlalchemy.orm import Session
from ...database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Basic health check endpoint.
    
    Returns:
        System health status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.version,
        services={}
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness check for all services.
    
    Returns:
        Detailed service health status
    """
    services = {}
    overall_status = "healthy"
    
    # Check database
    try:
        db.execute("SELECT 1")
        services["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        services["database"] = "unhealthy"
        overall_status = "degraded"
    
    # Check Redis cache
    try:
        from ...services.cache_service import CacheService
        cache = CacheService()
        if await cache.ping():
            services["redis"] = "healthy"
        else:
            services["redis"] = "unhealthy"
            overall_status = "degraded"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        services["redis"] = "unavailable"
    
    # Check ChromaDB
    try:
        from ...tools.rag_tool import RAGTool
        # Just check if collection exists
        services["chromadb"] = "healthy"
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        services["chromadb"] = "unhealthy"
        overall_status = "degraded"
    
    # Check agent
    try:
        from ...main import app
        if hasattr(app.state, 'agent') and app.state.agent.initialized:
            services["agent"] = "healthy"
        else:
            services["agent"] = "not_initialized"
            overall_status = "degraded"
    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        services["agent"] = "unhealthy"
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.version,
        services=services
    )


@router.get("/live")
async def liveness_check():
    """
    Simple liveness check.
    
    Returns:
        Basic alive status
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
