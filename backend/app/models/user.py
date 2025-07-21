"""
MongoDB models for user management and permissions.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from backend.app.models.business import PyObjectId

class UserPermissions(BaseModel):
    """User permissions model"""
    tables: List[str] = []  # Tables user can access
    operations: List[str] = []  # read, write, aggregate, etc.
    row_level_filters: Dict[str, Any] = {}  # Row-level security filters

class User(BaseModel):
    """User model"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(unique=True, index=True)
    username: str = Field(unique=True, index=True)
    email: EmailStr = Field(unique=True, index=True)
    hashed_password: str
    business_id: str = Field(index=True)  # Primary business assignment
    role: str = "user"  # user, admin, analyst, etc.
    status: str = "active"  # active, inactive, suspended
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    allowed_businesses: List[str] = []  # Added for admins
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserPermission(BaseModel):
    """User permission model for business access"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(index=True)
    business_id: str = Field(index=True)
    role: str = "user"
    permissions: UserPermissions
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserSession(BaseModel):
    """User session model for tracking active sessions"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    session_id: str = Field(unique=True, index=True)
    user_id: str = Field(index=True)
    business_id: str = Field(index=True)
    token_hash: str  # Hashed JWT token for session tracking
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserCreate(BaseModel):
    """Model for creating a user"""
    username: str
    email: EmailStr
    password: str
    business_id: str
    role: str = "user"

class UserUpdate(BaseModel):
    """Model for updating a user"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    business_id: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None

class UserPermissionCreate(BaseModel):
    """Model for creating user permissions"""
    user_id: str
    business_id: str
    role: str = "user"
    permissions: UserPermissions
    expires_at: Optional[datetime] = None

class UserPermissionUpdate(BaseModel):
    """Model for updating user permissions"""
    role: Optional[str] = None
    permissions: Optional[UserPermissions] = None
    expires_at: Optional[datetime] = None

class UserLogin(BaseModel):
    """Model for user login"""
    username: str
    password: str
    business_id: str

class UserResponse(BaseModel):
    """Model for user response (without sensitive data)"""
    user_id: str
    username: str
    email: EmailStr
    business_id: str
    role: str
    status: str
    created_at: datetime
    last_login: Optional[datetime] = None

def validate_business_access(user: dict, requested_business_id: str) -> bool:
    if user["role"] == "admin":
        return requested_business_id in user.get("allowed_businesses", [])
    return user.get("business_id") == requested_business_id 