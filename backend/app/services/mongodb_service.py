"""
MongoDB service for handling all database operations.
Manages business configs, users, conversations, and schemas.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from backend.app.config import get_settings
from backend.app.models.business import BusinessConfig, BusinessSchema, BusinessConfigCreate, BusinessConfigUpdate
from backend.app.models.user import User, UserPermission, UserSession, UserCreate, UserUpdate
from backend.app.models.conversation import ConversationSession, ConversationAnalytics, ConversationSessionCreate, AuditLog

logger = logging.getLogger(__name__)

class MongoDBService:
    """MongoDB service for multi-business chatbot"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self._collections = {}
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.settings.mongodb.uri)
            self.db = self.client[self.settings.mongodb.database]
            
            # Initialize collections
            self._collections = {
                'business_configs': self.db.business_configs,
                'business_schemas': self.db.business_schemas,
                'users': self.db.users,
                'user_permissions': self.db.user_permissions,
                'user_sessions': self.db.user_sessions,
                'conversation_sessions': self.db.conversation_sessions,
                'conversation_analytics': self.db.conversation_analytics,
                'audit_logs': self.db.audit_logs
            }
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create database indexes"""
        try:
            # Business configs indexes
            await self._collections['business_configs'].create_index([("business_id", ASCENDING)], unique=True)
            
            # Business schemas indexes
            await self._collections['business_schemas'].create_index([("business_id", ASCENDING)])
            await self._collections['business_schemas'].create_index([("table_name", ASCENDING)])
            await self._collections['business_schemas'].create_index([("vector_id", ASCENDING)])
            
            # Users indexes
            await self._collections['users'].create_index([("user_id", ASCENDING)], unique=True)
            await self._collections['users'].create_index([("username", ASCENDING)], unique=True)
            await self._collections['users'].create_index([("email", ASCENDING)], unique=True)
            await self._collections['users'].create_index([("business_id", ASCENDING)])
            
            # User permissions indexes
            await self._collections['user_permissions'].create_index([("user_id", ASCENDING), ("business_id", ASCENDING)], unique=True)
            
            # User sessions indexes
            await self._collections['user_sessions'].create_index([("session_id", ASCENDING)], unique=True)
            await self._collections['user_sessions'].create_index([("user_id", ASCENDING)])
            await self._collections['user_sessions'].create_index([("expires_at", ASCENDING)])
            
            # Conversation sessions indexes
            await self._collections['conversation_sessions'].create_index([("session_id", ASCENDING)], unique=True)
            await self._collections['conversation_sessions'].create_index([("user_id", ASCENDING)])
            await self._collections['conversation_sessions'].create_index([("business_id", ASCENDING)])
            await self._collections['conversation_sessions'].create_index([("last_activity", DESCENDING)])
            
            # Conversation analytics indexes
            await self._collections['conversation_analytics'].create_index([("session_id", ASCENDING)], unique=True)
            await self._collections['conversation_analytics'].create_index([("user_id", ASCENDING)])
            await self._collections['conversation_analytics'].create_index([("business_id", ASCENDING)])
            
            # Audit logs indexes
            await self._collections['audit_logs'].create_index([("user_id", ASCENDING)])
            await self._collections['audit_logs'].create_index([("business_id", ASCENDING)])
            await self._collections['audit_logs'].create_index([("session_id", ASCENDING)])
            await self._collections['audit_logs'].create_index([("operation_type", ASCENDING)])
            await self._collections['audit_logs'].create_index([("table_name", ASCENDING)])
            await self._collections['audit_logs'].create_index([("timestamp", DESCENDING)])
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise
    
    # =============================================================================
    # BUSINESS CONFIGURATION METHODS
    # =============================================================================
    
    async def create_business_config(self, business_config: BusinessConfigCreate) -> BusinessConfig:
        """Create a new business configuration"""
        try:
            config = BusinessConfig(**business_config.dict())
            result = await self._collections['business_configs'].insert_one(config.dict(by_alias=True))
            config.id = result.inserted_id
            return config
        except DuplicateKeyError:
            raise ValueError(f"Business '{business_config.business_id}' already exists")
    
    async def get_business_config(self, business_id: str) -> Optional[BusinessConfig]:
        """Get business configuration by business_id"""
        doc = await self._collections['business_configs'].find_one({"business_id": business_id})
        return BusinessConfig(**doc) if doc else None
    
    async def get_all_business_configs(self) -> List[BusinessConfig]:
        """Get all business configurations"""
        cursor = self._collections['business_configs'].find({"status": "active"})
        configs = []
        async for doc in cursor:
            configs.append(BusinessConfig(**doc))
        return configs
    
    async def update_business_config(self, business_id: str, update_data: BusinessConfigUpdate) -> Optional[BusinessConfig]:
        """Update business configuration"""
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        update_dict["updated_at"] = datetime.now(timezone.utc)
        
        result = await self._collections['business_configs'].find_one_and_update(
            {"business_id": business_id},
            {"$set": update_dict},
            return_document=True
        )
        return BusinessConfig(**result) if result else None
    
    async def delete_business_config(self, business_id: str) -> bool:
        """Delete business configuration"""
        result = await self._collections['business_configs'].delete_one({"business_id": business_id})
        return result.deleted_count > 0
    
    # =============================================================================
    # BUSINESS SCHEMA METHODS
    # =============================================================================
    
    async def create_business_schema(self, schema: BusinessSchema) -> BusinessSchema:
        """Create a new business schema"""
        try:
            result = await self._collections['business_schemas'].insert_one(schema.dict(by_alias=True))
            schema.id = result.inserted_id
            return schema
        except DuplicateKeyError:
            raise ValueError(f"Schema for table '{schema.table_name}' in business '{schema.business_id}' already exists")
    
    async def get_business_schemas(self, business_id: str) -> List[BusinessSchema]:
        """Get all schemas for a business"""
        cursor = self._collections['business_schemas'].find({"business_id": business_id})
        schemas = []
        async for doc in cursor:
            schemas.append(BusinessSchema(**doc))
        return schemas
    
    async def get_business_schema(self, business_id: str, table_name: str) -> Optional[BusinessSchema]:
        """Get specific schema for a business and table"""
        doc = await self._collections['business_schemas'].find_one({
            "business_id": business_id,
            "table_name": table_name
        })
        return BusinessSchema(**doc) if doc else None
    
    async def get_business_schema_by_id(self, business_id: str, schema_id: str) -> Optional[BusinessSchema]:
        """Get specific schema for a business by schema ID"""
        from bson import ObjectId
        try:
            doc = await self._collections['business_schemas'].find_one({
                "business_id": business_id,
                "_id": ObjectId(schema_id)
            })
            return BusinessSchema(**doc) if doc else None
        except Exception as e:
            logger.error(f"Error getting schema by ID {schema_id} for business {business_id}: {e}")
            return None
    
    async def update_schema_vector_id(self, business_id: str, table_name: str, vector_id: str) -> bool:
        """Update vector ID for a schema"""
        result = await self._collections['business_schemas'].update_one(
            {"business_id": business_id, "table_name": table_name},
            {
                "$set": {
                    "vector_id": vector_id,
                    "indexed_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        return result.modified_count > 0
    
    # =============================================================================
    # USER MANAGEMENT METHODS
    # =============================================================================
    
    async def create_user(self, user_data: UserCreate, hashed_password: str) -> User:
        """Create a new user"""
        try:
            user = User(
                user_id=f"user_{user_data.username}_{user_data.business_id}",
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password,
                business_id=user_data.business_id,
                role=user_data.role
            )
            result = await self._collections['users'].insert_one(user.dict(by_alias=True))
            user.id = result.inserted_id
            return user
        except DuplicateKeyError:
            raise ValueError(f"User '{user_data.username}' already exists")
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        doc = await self._collections['users'].find_one({"username": username})
        return User(**doc) if doc else None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        doc = await self._collections['users'].find_one({"email": email})
        return User(**doc) if doc else None

    async def get_user_by_id(self, user_id: str):
        if not self._collections or 'users' not in self._collections:
            await self.connect()
        doc = await self._collections['users'].find_one({"user_id": user_id})
        return User(**doc) if doc else None
    
    async def get_user_permissions(self, user_id: str, business_id: str) -> Optional[UserPermission]:
        """Get user permissions for a specific business"""
        doc = await self._collections['user_permissions'].find_one({
            "user_id": user_id,
            "business_id": business_id
        })
        return UserPermission(**doc) if doc else None
    
    async def update_user_last_login(self, user_id: str) -> bool:
        """Update user's last login time"""
        result = await self._collections['users'].update_one(
            {"user_id": user_id},
            {"$set": {"last_login": datetime.now(timezone.utc)}}
        )
        return result.modified_count > 0
    
    # =============================================================================
    # SESSION MANAGEMENT METHODS
    # =============================================================================
    
    async def create_user_session(self, session: UserSession) -> UserSession:
        """Create a new user session"""
        try:
            result = await self._collections['user_sessions'].insert_one(session.dict(by_alias=True))
            session.id = result.inserted_id
            return session
        except DuplicateKeyError:
            raise ValueError(f"Session '{session.session_id}' already exists")
    
    async def get_user_session(self, session_id: str) -> Optional[UserSession]:
        """Get user session by session_id"""
        doc = await self._collections['user_sessions'].find_one({"session_id": session_id})
        return UserSession(**doc) if doc else None
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity"""
        result = await self._collections['user_sessions'].update_one(
            {"session_id": session_id},
            {"$set": {"last_activity": datetime.now(timezone.utc)}}
        )
        return result.modified_count > 0
    
    async def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a user session"""
        result = await self._collections['user_sessions'].update_one(
            {"session_id": session_id},
            {"$set": {"is_active": False}}
        )
        return result.modified_count > 0
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        result = await self._collections['user_sessions'].delete_many({
            "expires_at": {"$lt": datetime.now(timezone.utc)}
        })
        return result.deleted_count
    
    # =============================================================================
    # CONVERSATION METHODS
    # =============================================================================
    
    async def create_conversation_session(self, session_data: ConversationSessionCreate) -> ConversationSession:
        """Create a new conversation session"""
        try:
            session = ConversationSession(
                session_id=f"sess_{session_data.user_id}_{session_data.business_id}_{int(datetime.now(timezone.utc).timestamp())}",
                user_id=session_data.user_id,
                business_id=session_data.business_id,
                expires_at=session_data.expires_at
            )
            result = await self._collections['conversation_sessions'].insert_one(session.dict(by_alias=True))
            session.id = result.inserted_id
            return session
        except DuplicateKeyError:
            raise ValueError(f"Conversation session already exists")
    
    async def create_conversation_session_with_id(self, session_id: str, user_id: str, business_id: str, expires_at: datetime) -> ConversationSession:
        """Create a new conversation session with a specific session_id"""
        try:
            # Clean the session_id before creating the session
            cleaned_session_id = session_id.replace('"', '').replace('\\"', '')
            session = ConversationSession(
                session_id=cleaned_session_id,
                user_id=user_id,
                business_id=business_id,
                expires_at=expires_at
            )
            result = await self._collections['conversation_sessions'].insert_one(session.dict(by_alias=True))
            session.id = result.inserted_id
            return session
        except DuplicateKeyError:
            raise ValueError(f"Conversation session with ID '{session_id}' already exists")
    
    async def migrate_session_id_format(self, old_session_id: str, new_session_id: str) -> bool:
        """Migrate a session from old session_id format to new clean format"""
        try:
            # Update the session_id field
            result = await self._collections['conversation_sessions'].update_one(
                {"session_id": old_session_id},
                {"$set": {"session_id": new_session_id}}
            )
            
            if result.modified_count > 0:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error migrating session: {e}")
            return False
    
    async def get_conversation_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session by session_id"""
        
        # Try the exact session_id first
        doc = await self._collections['conversation_sessions'].find_one({"session_id": session_id})
        
        if not doc:
            # Try with escaped quotes (common in Swagger UI)
            escaped_session_id = f'"{session_id}"'
            doc = await self._collections['conversation_sessions'].find_one({"session_id": escaped_session_id})
        
        if not doc:
            # Try with double escaped quotes
            double_escaped_session_id = f'\\"{session_id}\\"'
            doc = await self._collections['conversation_sessions'].find_one({"session_id": double_escaped_session_id})
        
        if not doc:
            # Try removing quotes from the session_id
            clean_session_id = session_id.replace('"', '').replace('\\"', '')
            if clean_session_id != session_id:
                doc = await self._collections['conversation_sessions'].find_one({"session_id": clean_session_id})
        
        return ConversationSession(**doc) if doc else None
    
    async def update_conversation_session(self, session: ConversationSession) -> bool:
        """Update an existing conversation session by session_id."""
        
        # Always include conversation_memory in updates, even if it's empty
        update_data = session.dict(exclude={"session_id"})
        update_data["updated_at"] = datetime.now(timezone.utc)
        
        result = await self._collections['conversation_sessions'].update_one(
            {"session_id": session.session_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    async def update_conversation_memory(self, session_id: str, memory_data: Dict[str, Any]) -> bool:
        """Update conversation memory"""
        # Clean the session_id before using it
        cleaned_session_id = session_id.replace('"', '').replace('\\"', '')
        
        result = await self._collections['conversation_sessions'].update_one(
            {"session_id": cleaned_session_id},
            {
                "$set": {
                    "conversation_memory": memory_data,
                    "last_activity": datetime.now(timezone.utc)
                }
            }
        )
        
        return result.modified_count > 0
    
    async def update_conversation_memory_upsert(self, session_id: str, memory_data: Dict[str, Any], user_id: str, business_id: str) -> bool:
        """Update conversation memory with upsert functionality"""
        # Clean the session_id before using it
        cleaned_session_id = session_id.replace('"', '').replace('\\"', '')
        
        result = await self._collections['conversation_sessions'].update_one(
            {"session_id": cleaned_session_id},
            {
                "$set": {
                    "conversation_memory": memory_data,
                    "last_activity": datetime.now(timezone.utc),
                    "user_id": user_id,
                    "business_id": business_id,
                    "status": "active",
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc) + timedelta(hours=24)
                }
            },
            upsert=True
        )
        
        return result.modified_count > 0 or result.upserted_id is not None
    
    async def get_user_conversations(self, user_id: str, business_id: str, limit: int = 10) -> List[ConversationSession]:
        """Get recent conversations for a user"""
        cursor = self._collections['conversation_sessions'].find({
            "user_id": user_id,
            "business_id": business_id
        }).sort("last_activity", DESCENDING).limit(limit)
        
        sessions = []
        async for doc in cursor:
            sessions.append(ConversationSession(**doc))
        return sessions

    async def set_cached_schema_context(self, session_id: str, schema_context: list) -> bool:
        """Set cached schema context for a session."""
        # Clean the session_id before using it
        cleaned_session_id = session_id.replace('"', '').replace('\\"', '')
        result = await self._collections['conversation_sessions'].update_one(
            {"session_id": cleaned_session_id},
            {"$set": {"cached_schema_context": schema_context, "last_activity": datetime.now(timezone.utc)}},
            upsert=True
        )
        return result.modified_count > 0

    async def get_cached_schema_context(self, session_id: str) -> Optional[list]:
        """Get cached schema context for a session."""
        # Clean the session_id before using it
        cleaned_session_id = session_id.replace('"', '').replace('\\"', '')
        doc = await self._collections['conversation_sessions'].find_one({"session_id": cleaned_session_id})
        if doc and "cached_schema_context" in doc:
            return doc["cached_schema_context"]
        return None

    async def add_message_to_conversation(self, session_id: str, message: dict) -> bool:
        """Append a message to the conversation session's messages array."""
        # Clean the session_id before using it
        cleaned_session_id = session_id.replace('"', '').replace('\\"', '')
        result = await self._collections['conversation_sessions'].update_one(
            {"session_id": cleaned_session_id},
            {"$push": {"messages": message}, "$set": {"last_activity": datetime.now(timezone.utc)}},
            upsert=True
        )
        return result.modified_count > 0 or result.upserted_id is not None

    async def get_last_n_messages(self, session_id: str, n: int = 10) -> list:
        """Get the last n messages for a conversation session, ordered oldest to newest."""
        # Clean the session_id before using it
        cleaned_session_id = session_id.replace('"', '').replace('\\"', '')
        doc = await self._collections['conversation_sessions'].find_one({"session_id": cleaned_session_id}, {"messages": 1})
        if doc and "messages" in doc:
            return doc["messages"][-n:]
        return []

    # ===================== ADMIN BUSINESS & USER MANAGEMENT =====================
    async def add_business(self, business_id: str, config: dict) -> bool:
        """Add a new business config to MongoDB."""
        result = await self._collections['business_configs'].update_one(
            {"business_id": business_id},
            {"$set": {**config, "business_id": business_id}},
            upsert=True
        )
        return result.upserted_id is not None or result.modified_count > 0

    async def remove_business(self, business_id: str) -> bool:
        """Remove a business config from MongoDB."""
        result = await self._collections['business_configs'].delete_one({"business_id": business_id})
        return result.deleted_count > 0

    async def add_user(self, user: dict) -> bool:
        """Add a new user to MongoDB."""
        result = await self._collections['users'].update_one(
            {"user_id": user["user_id"]},
            {"$set": user},
            upsert=True
        )
        return result.upserted_id is not None or result.modified_count > 0

    async def remove_user(self, user_id: str) -> bool:
        """Remove a user from MongoDB."""
        result = await self._collections['users'].delete_one({"user_id": user_id})
        return result.deleted_count > 0

    async def assign_user_to_businesses(self, user_id: str, business_ids: list) -> bool:
        """Assign a user to one or more businesses."""
        result = await self._collections['users'].update_one(
            {"user_id": user_id},
            {"$set": {"allowed_businesses": business_ids}}
        )
        return result.modified_count > 0

    async def add_or_update_business_schema(self, business_id: str, schema: dict) -> bool:
        """Add or update a business schema in MongoDB."""
        result = await self._collections['business_schemas'].update_one(
            {"business_id": business_id, "table_name": schema["table_name"]},
            {"$set": schema},
            upsert=True
        )
        return result.upserted_id is not None or result.modified_count > 0

    # =============================================================================
    # AUDIT LOGGING METHODS
    # =============================================================================
    
    async def create_audit_log(self, audit_log: AuditLog) -> AuditLog:
        """Create a new audit log entry"""
        try:
            result = await self._collections['audit_logs'].insert_one(audit_log.dict(by_alias=True))
            audit_log.id = result.inserted_id
            return audit_log
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
            raise
    
    async def get_audit_logs(self, user_id: Optional[str] = None, business_id: Optional[str] = None, 
                           operation_type: Optional[str] = None, limit: int = 100) -> List[AuditLog]:
        """Get audit logs with optional filtering"""
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if business_id:
            filter_dict["business_id"] = business_id
        if operation_type:
            filter_dict["operation_type"] = operation_type
        
        cursor = self._collections['audit_logs'].find(filter_dict).sort("timestamp", DESCENDING).limit(limit)
        logs = []
        async for doc in cursor:
            logs.append(AuditLog(**doc))
        return logs
    
    async def get_audit_logs_by_session(self, session_id: str, limit: int = 50) -> List[AuditLog]:
        """Get audit logs for a specific session"""
        cursor = self._collections['audit_logs'].find({"session_id": session_id}).sort("timestamp", DESCENDING).limit(limit)
        logs = []
        async for doc in cursor:
            logs.append(AuditLog(**doc))
        return logs

# Global MongoDB service instance
mongodb_service = MongoDBService()

async def get_mongodb_service():
    if mongodb_service.db is None or not mongodb_service._collections:
        await mongodb_service.connect()
    return mongodb_service 