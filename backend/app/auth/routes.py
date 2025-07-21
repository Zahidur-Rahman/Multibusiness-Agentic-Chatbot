"""
Authentication routes for Multi-Business Chatbot.
Handles login, logout, and token management with business isolation.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import logging
from fastapi.responses import JSONResponse

from backend.app.auth.jwt_handler import jwt_handler, get_current_user, security
from backend.app.config import get_settings
from backend.app.services.mongodb_service import mongodb_service
from backend.app.utils import verify_password
from fastapi_limiter.depends import RateLimiter

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["authentication"])

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str
    business_id: Optional[str] = None  # Optional for admin

class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    business_id: str
    role: str

class RefreshRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str

class UserInfo(BaseModel):
    """User information model"""
    user_id: str
    username: str
    business_id: str
    role: str

# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@router.post("/login", response_model=LoginResponse, dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def login(request: LoginRequest):
    """
    Login with business context.
    User must specify which business they want to access, unless they are admin.
    """
    try:
        # Use MongoDB for user authentication
        mongo_service = mongodb_service
        await mongo_service.connect()

        # First, try to find the user by username (and business_id if provided)
        user_query = {"username": request.username}
        if request.business_id:
            user_query["business_id"] = request.business_id
        user = await mongo_service._collections['users'].find_one(user_query)

        # If not found and business_id was provided, try again without business_id (for admin login)
        if not user and not request.business_id:
            user = await mongo_service._collections['users'].find_one({"username": request.username})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Check user status
        if user.get("status", "active") != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is not active"
            )
        # Verify password
        if not verify_password(request.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        user_id = user["user_id"]
        role = user.get("role", "user")
        username = user["username"]

        # For non-admins, require business_id and check allowed businesses
        if role != "admin":
            if not request.business_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="business_id is required for non-admin users"
                )
            # Validate business exists
            try:
                await mongodb_service.get_business_config(request.business_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Business '{request.business_id}' not configured"
                )
            # Check if user is allowed for this business
            allowed_businesses = user.get("allowed_businesses", [])
            if request.business_id not in allowed_businesses:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"User does not have access to business '{request.business_id}'"
                )
            business_id = request.business_id
        else:
            # For admin, allow login without business_id and set allowed_businesses
            allowed_businesses = user.get("allowed_businesses", [])
            business_id = request.business_id if request.business_id else None

        # Create tokens with business context
        access_token = jwt_handler.create_access_token(
            user_id=user_id,
            business_id=business_id if business_id else "all",
            username=username,
            role=role
        )
        refresh_token = jwt_handler.create_refresh_token(
            user_id=user_id,
            business_id=business_id if business_id else "all"
        )
        logger.info(f"User '{username}' logged in as '{role}'")
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt.expiration,
            user_id=user_id,
            username=username,
            business_id=business_id if business_id else "all",
            role=role
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(request: RefreshRequest):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = jwt_handler.verify_token(request.refresh_token)
        
        # Ensure it's a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Create new access token
        access_token = jwt_handler.create_access_token(
            user_id=payload["user_id"],
            business_id=payload["business_id"],
            username=payload.get("username", "unknown"),
            role=payload.get("role", "user")
        )
        
        # Create new refresh token
        refresh_token = jwt_handler.create_refresh_token(
            user_id=payload["user_id"],
            business_id=payload["business_id"]
        )
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt.expiration,
            user_id=payload["user_id"],
            username=payload.get("username", "unknown"),
            business_id=payload["business_id"],
            role=payload.get("role", "user")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return UserInfo(
        user_id=current_user["user_id"],
        username=current_user["username"],
        business_id=current_user["business_id"],
        role=current_user["role"]
    )

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout user (invalidate token).
    In production, you might want to blacklist the token.
    """
    try:
        # Verify token to get user info for logging
        payload = jwt_handler.verify_token(credentials.credentials)
        logger.info(f"User '{payload.get('username', 'unknown')}' logged out from business '{payload.get('business_id', 'unknown')}'")
        
        # TODO: In production, add token to blacklist
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Still return success even if token is invalid
        return {"message": "Successfully logged out"}

@router.get("/validate")
async def validate_token(current_user: dict = Depends(get_current_user)):
    """Validate current token and return user info"""
    return {
        "valid": True,
        "user": current_user
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def require_role(required_roles):
    def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role", "user")
        if isinstance(required_roles, str):
            roles = [required_roles]
        else:
            roles = required_roles
        if user_role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this resource."
            )
        return current_user
    return role_checker 