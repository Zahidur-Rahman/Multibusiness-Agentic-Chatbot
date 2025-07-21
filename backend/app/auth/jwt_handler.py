"""
JWT Authentication Handler for Multi-Business Chatbot.
Handles token creation, validation, and business access control.
"""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.app.config import get_settings, validate_business_access

settings = get_settings()
security = HTTPBearer()

class JWTHandler:
    """JWT token handler with business isolation"""
    
    def __init__(self):
        self.secret_key = settings.jwt.secret_key
        self.algorithm = settings.jwt.algorithm
        self.expiration = settings.jwt.expiration
        self.refresh_expiration = settings.jwt.refresh_expiration
    
    def create_access_token(self, user_id: str, business_id: str, 
                          username: str, role: str = "user") -> str:
        """Create JWT access token with business context"""
        payload = {
            "user_id": user_id,
            "business_id": business_id,  # Critical for business isolation
            "username": username,
            "role": role,
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(seconds=self.expiration),
            "iat": datetime.now(timezone.utc)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str, business_id: str) -> str:
        """Create JWT refresh token"""
        payload = {
            "user_id": user_id,
            "business_id": business_id,
            "type": "refresh",
            "exp": datetime.now(timezone.utc) + timedelta(seconds=self.refresh_expiration),
            "iat": datetime.now(timezone.utc)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if datetime.now(timezone.utc) > datetime.fromtimestamp(payload["exp"], tz=timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.DecodeError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Get current user from JWT token"""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        # Ensure it's an access token
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        return {
            "user_id": payload["user_id"],
            "business_id": payload["business_id"],
            "username": payload["username"],
            "role": payload["role"]
        }
    
    async def validate_business_access(self, user_business_id: str, requested_business_id: str) -> bool:
        """Validate if user has access to the requested business"""
        # Use the validation function from config
        return await validate_business_access(user_business_id, requested_business_id)
    
    async def require_business_access(self, user_business_id: str, requested_business_id: str):
        """Require business access or raise HTTPException"""
        if not await self.validate_business_access(user_business_id, requested_business_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to business '{requested_business_id}'"
            )

# Global JWT handler instance
jwt_handler = JWTHandler()

# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """FastAPI dependency to get current user"""
    return jwt_handler.get_current_user(credentials)

async def require_business_access(user_business_id: str, requested_business_id: str):
    """FastAPI dependency to require business access"""
    await jwt_handler.require_business_access(user_business_id, requested_business_id)

async def require_admin(current_user: dict = Depends(get_current_user)):
    """FastAPI dependency to require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user 