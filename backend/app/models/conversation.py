"""
MongoDB models for conversation sessions and memory management.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from backend.app.models.business import PyObjectId

class ChatMessage(BaseModel):
    role: str
    content: str
    # Add any additional fields if needed (e.g., timestamp, metadata)
    # timestamp: Optional[datetime] = None
    # metadata: Optional[dict] = None

class Message(BaseModel):
    """Individual message in a conversation"""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}  # Additional message metadata
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

class ConversationContext(BaseModel):
    """Conversation context for memory management"""
    active_tables: List[str] = []  # Tables currently being discussed
    recent_queries: List[str] = []  # Recent SQL queries
    user_intent: Optional[str] = None  # Current user intent
    business_context: Dict[str, Any] = {}  # Business-specific context

class UserPreferences(BaseModel):
    """User preferences for conversation"""
    language: str = "en"
    response_style: str = "friendly"  # friendly, formal, concise
    data_format: str = "table"  # table, json, csv

class ConversationMemory(BaseModel):
    """Conversation memory model"""
    messages: List[Message] = []
    context: ConversationContext = Field(default_factory=ConversationContext)
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    session_variables: Dict[str, Any] = {}  # Session-specific variables

class ConversationSession(BaseModel):
    """Conversation session model"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    session_id: str = Field(unique=True, index=True)
    user_id: str = Field(index=True)
    business_id: str = Field(index=True)
    conversation_memory: ConversationMemory = Field(default_factory=ConversationMemory)
    status: str = "active"  # active, paused, ended
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    cached_schema_context: Optional[List[dict]] = None  # Cached vector search result
    pause_context: Optional[dict] = None  # For pause/resume support
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ConversationSessionCreate(BaseModel):
    """Model for creating new conversation sessions"""
    session_id: str
    user_id: str
    business_id: str
    expires_at: Optional[datetime] = None

class AuditLog(BaseModel):
    """Audit log for tracking database write operations"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(index=True)
    business_id: str = Field(index=True)
    session_id: str = Field(index=True)
    operation_type: str  # INSERT, UPDATE, DELETE
    table_name: str
    sql_query: str
    affected_rows: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ConversationAnalytics(BaseModel):
    """Conversation analytics model"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    session_id: str = Field(index=True)
    user_id: str = Field(index=True)
    business_id: str = Field(index=True)
    total_messages: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    user_satisfaction_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ConversationSessionUpdate(BaseModel):
    """Model for updating a conversation session"""
    status: Optional[str] = None
    expires_at: Optional[datetime] = None

class MessageCreate(BaseModel):
    """Model for creating a message"""
    role: str
    content: str
    metadata: Dict[str, Any] = {}

class ConversationContextUpdate(BaseModel):
    """Model for updating conversation context"""
    active_tables: Optional[List[str]] = None
    recent_queries: Optional[List[str]] = None
    user_intent: Optional[str] = None
    business_context: Optional[Dict[str, Any]] = None

class UserPreferencesUpdate(BaseModel):
    """Model for updating user preferences"""
    preferred_output_format: Optional[str] = None
    language: Optional[str] = None
    detail_level: Optional[str] = None
    timezone: Optional[str] = None

class ConversationResponse(BaseModel):
    """Model for conversation response"""
    session_id: str
    message: Message
    context: ConversationContext
    analytics: Optional[Dict[str, Any]] = None

