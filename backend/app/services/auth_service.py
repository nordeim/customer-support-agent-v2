"""
Authentication and authorization service.
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import logging
from passlib.context import CryptContext

from ..config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthService:
    """Handles authentication and authorization."""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.jwt_algorithm
        self.expiration_hours = settings.jwt_expiration_hours
    
    def create_token(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a JWT token.
        
        Args:
            user_id: User identifier
            metadata: Additional claims
            
        Returns:
            JWT token string
        """
        payload = {
            "sub": user_id,
            "exp": datetime.utcnow() + timedelta(hours=self.expiration_hours),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        if metadata:
            payload.update(metadata)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.debug(f"Created token for user: {user_id}")
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)


# Global auth service instance
auth_service = AuthService()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[str]:
    """
    Get current user from JWT token.
    
    Args:
        credentials: Bearer token from request
        
    Returns:
        User ID if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        payload = auth_service.verify_token(credentials.credentials)
        return payload.get("sub")
    
    except HTTPException:
        return None


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Require authentication for an endpoint.
    
    Args:
        credentials: Bearer token from request
        
    Returns:
        User ID
        
    Raises:
        HTTPException: If not authenticated
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = auth_service.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        return user_id
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


class RoleChecker:
    """Check user roles for authorization."""
    
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles
    
    def __call__(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> bool:
        """
        Check if user has required role.
        
        Args:
            credentials: Bearer token
            
        Returns:
            True if authorized
            
        Raises:
            HTTPException: If not authorized
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        try:
            payload = auth_service.verify_token(credentials.credentials)
            user_role = payload.get("role", "user")
            
            if user_role not in self.allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return True
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Authorization failed"
            )


# Role checkers
require_admin = RoleChecker(["admin"])
require_support = RoleChecker(["admin", "support"])
