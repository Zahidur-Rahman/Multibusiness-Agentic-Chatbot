"""
Main FastAPI application for the Multi-Business Conversational Chatbot.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from typing import Dict, Any, List, Optional
import re
import os
import json
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from fastapi.responses import JSONResponse

from backend.app.config import get_settings, Settings
from backend.app.auth.routes import router as auth_router
from backend.app.services.business_service import router as business_admin_router
from backend.app.services.mistral_llm_service import MistralLLMService
from backend.app.services.vector_search import FaissVectorSearchService
from backend.app.services.mongodb_service import MongoDBService, get_mongodb_service
from backend.app.services.redis_service import redis_service
from backend.app.models.conversation import ConversationSession
from datetime import datetime
from backend.app.mcp.mcp_client import MCPClient
from backend.app.auth.jwt_handler import get_current_user, require_business_access, require_admin
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as aioredis
from backend.app.models.business import BusinessConfig, BusinessConfigCreate, BusinessConfigUpdate
from backend.app.models.user import User, UserCreate, UserUpdate
from backend.app.utils import hash_password
from backend.app.auth.routes import require_role
from backend.app.services.chat_graph import chat_graph
from backend.app.utils.query_classifier import is_database_related_query_dynamic
from backend.app.models.conversation import ChatMessage, ConversationSessionCreate
from backend.app.models.chat_graph_state import ChatGraphState
from backend.app.utils.chat_helpers import format_db_result
import copy
from backend.app.models.conversation import ConversationMemory, Message
import bson
from backend.app.services.agent_registry import agent_registry

# Helper to recursively extract DB-like responses for user-friendly formatting
import re, json, ast

def extract_db_like(text):
    def try_json(val):
        try:
            return json.loads(val)
        except Exception:
            try:
                # Try to fix single quotes and parse
                fixed = val.replace("'", '"')
                return json.loads(fixed)
            except Exception:
                try:
                    # Try Python literal eval as last resort
                    return ast.literal_eval(val)
                except Exception:
                    return None

    # If it's a string, try to extract JSON object
    if isinstance(text, str):
        match = re.search(r'({.*})', text, re.DOTALL)
        if match:
            data = try_json(match.group(1))
            if data:
                return extract_db_like(data)
        return None
    # If it's a dict, check for DB-like keys or nested content
    if isinstance(text, dict):
        if any(k in text for k in ("results", "success", "row_count")):
            return text
        if "content" in text and isinstance(text["content"], list):
            for item in text["content"]:
                if isinstance(item, dict) and "text" in item:
                    found = extract_db_like(item["text"])
                    if found:
                        return found
        if "text" in text:
            found = extract_db_like(text["text"])
            if found:
                return found
    return None

def convert_objectid_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(i) for i in obj]
    elif hasattr(obj, 'to_str'):
        return str(obj)
    elif 'ObjectId' in str(type(obj)):
        return str(obj)
    return obj

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

mcp_client = None  # Global placeholder
vector_search_service = FaissVectorSearchService()
llm_service = MistralLLMService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_client
    logger.info("Starting Multi-Business Conversational Chatbot...")
    
    logger.info("Connecting to Redis...")
    settings = get_settings()
    redis_url = f"redis://{settings.redis.host}:{settings.redis.port}/0"
    redis = await aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)
    
    # Initialize Redis service
    await redis_service.connect()
    logger.info("Connected to Redis.")

    try:
        logger.info("Connecting to MongoDB...")
        mongo_service = await get_mongodb_service()
        logger.info("Connected to MongoDB.")
        business_configs = await mongo_service.get_all_business_configs()
        business_ids = [b.business_id for b in business_configs]
        logger.info(f"Found business IDs: {business_ids}")
        for business_id in business_ids:
            logger.info(f"Indexing schemas for business: {business_id}")
            await vector_search_service.index_business_schemas(business_id)
            logger.info(f"Finished indexing for business: {business_id}")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        # Don't raise the exception - let the app start even if indexing fails
        logger.warning("Continuing startup despite indexing errors")
    
    # Start MCP client here
    MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp", "server_enhanced.py")
    mcp_client = MCPClient(MCP_SERVER_PATH)
    logger.info("MCP client started.")
    logger.info("Application startup complete")
    yield  # Don't forget this!
    # Optionally: shutdown/cleanup MCP client here
    if mcp_client:
        await mcp_client.aclose()
        logger.info("MCP client closed.")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Multi-Business Conversational Chatbot",
        description="A production-ready, dynamic multi-business conversational chatbot with PostgreSQL integration, vector-based schema discovery, and LangChain conversational AI.",
        version="1.0.0",
        docs_url="/docs",  # Always enable docs for debugging
        redoc_url="/redoc",  # Always enable redoc for debugging
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.server.allowed_methods,
        allow_headers=settings.server.allowed_headers,
    )
    
    # Add trusted host middleware for production
    if not settings.server.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure based on your domain
        )
    
    # Include authentication routes
    app.include_router(auth_router)
    # Include business/user admin routes
    app.include_router(business_admin_router)
    
    return app

# Create the application instance
app = create_app()

# Connect to MongoDB at startup
# @app.on_event("startup")
# async def startup_event():
#     # Get all business IDs from MongoDB
#     mongo_service = get_mongodb_service()
#     business_configs = await mongo_service.get_all_business_configs()
#     business_ids = [b.business_id for b in business_configs]
#     # Index schemas for each business using the global instance
#     for business_id in business_ids:
#         await vector_search_service.index_business_schemas(business_id)

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Business Conversational Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "multi-business-chatbot",
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check(settings: Settings = Depends(get_settings)):
    """Detailed health check with service status"""
    health_status = {
        "status": "healthy",
        "service": "multi-business-chatbot",
        "version": "1.0.0",
        "features": {
            "multi_business": settings.features.enable_multi_business,
            "vector_search": settings.features.enable_vector_search,
            "schema_discovery": settings.features.enable_schema_discovery,
            "conversation_memory": settings.features.enable_conversation_memory,
            "analytics": settings.features.enable_analytics
        },
        "businesses": settings.business_ids,
        "services": {
            "database": "unknown",  # TODO: Add database health check
            "mongodb": "unknown",   # TODO: Add MongoDB health check
            "redis": "unknown",     # TODO: Add Redis health check
            "mcp_server": "unknown" # TODO: Add MCP server health check
        }
    }
    
    return health_status

# =============================================================================
# BUSINESS MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/businesses")
async def list_businesses(mongo_service: MongoDBService = Depends(get_mongodb_service)):
    businesses = await mongo_service.get_all_business_configs()
    return {
        "businesses": [b.business_id for b in businesses],
        "count": len(businesses)
    }

@app.get("/businesses/{business_id}/config")
async def get_business_config(business_id: str, mongo_service: MongoDBService = Depends(get_mongodb_service)):
    config = await mongo_service.get_business_config(business_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found")
    return {
        "business_id": business_id,
        "host": config.host,
        "database": config.database,
        "port": config.port,
        "user": config.user,
        # Don't expose password in response
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_session_id(session_id: str) -> str:
    """Clean session ID by removing escaped quotes and normalizing format."""
    if not session_id:
        return session_id
    
    # Remove escaped quotes that Swagger UI might add
    cleaned = session_id.replace('\\"', '').replace('"', '')
    
    # Ensure it starts with 'sess_'
    if not cleaned.startswith('sess_'):
        return session_id  # Return original if it doesn't match expected format
    
    return cleaned

def serialize_message_for_redis(msg) -> dict:
    """Serialize message for Redis storage, handling datetime objects"""
    if isinstance(msg, dict):
        # Already a dict, just ensure datetime fields are serialized
        serialized = {}
        for key, value in msg.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
        return serialized
    else:
        # Message object, convert to dict and handle datetime
        msg_dict = msg.model_dump()
        if 'timestamp' in msg_dict and isinstance(msg_dict['timestamp'], datetime):
            msg_dict['timestamp'] = msg_dict['timestamp'].isoformat()
        return msg_dict

# =============================================================================
# CONVERSATION ENDPOINTS (Placeholder)
# =============================================================================

@app.post("/conversations")
async def create_conversation():
    """Create a new conversation session"""
    # TODO: Implement conversation creation
    return {
        "message": "Conversation creation endpoint - to be implemented",
        "status": "placeholder"
    }

@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str):
    """Send a message in a conversation"""
    # TODO: Implement message handling
    return {
        "message": "Message handling endpoint - to be implemented",
        "conversation_id": conversation_id,
        "status": "placeholder"
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details"""
    # TODO: Implement conversation retrieval
    return {
        "message": "Conversation retrieval endpoint - to be implemented",
        "conversation_id": conversation_id,
        "status": "placeholder"
    }

# =============================================================================
# SCHEMA DISCOVERY ENDPOINTS (Placeholder)
# =============================================================================

@app.get("/businesses/{business_id}/schemas")
async def list_schemas(business_id: str):
    """List all schemas for a business"""
    # TODO: Implement schema listing
    return {
        "message": "Schema listing endpoint - to be implemented",
        "business_id": business_id,
        "status": "placeholder"
    }

@app.get("/businesses/{business_id}/schemas/{table_name}")
async def get_schema(business_id: str, table_name: str):
    """Get schema for a specific table"""
    # TODO: Implement schema retrieval
    return {
        "message": "Schema retrieval endpoint - to be implemented",
        "business_id": business_id,
        "table_name": table_name,
        "status": "placeholder"
    }

# =============================================================================
# ADMIN-ONLY BUSINESS & USER MANAGEMENT ENDPOINTS
# =============================================================================

from pydantic import BaseModel
from typing import Dict, Any

class BusinessCreateRequest(BaseModel):
    business_id: str
    config: Dict[str, Any]

@app.post("/admin/businesses", dependencies=[Depends(require_admin)])
async def admin_add_business(
    request: BusinessCreateRequest,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    logger.info(f"Adding business: {request.business_id}")
    success = await mongo_service.add_business(request.business_id, request.config)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add business")
    return {"message": f"Business '{request.business_id}' added", "status": "success"}

@app.delete("/admin/businesses/{business_id}", dependencies=[Depends(require_admin)])
async def admin_remove_business(business_id: str, current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Remove a business"""
    success = await mongo_service.remove_business(business_id)
    if not success:
        raise HTTPException(status_code=404, detail="Business not found or could not be removed")
    return {"message": f"Business '{business_id}' removed", "status": "success"}

@app.post("/admin/users", dependencies=[Depends(require_admin)])
async def admin_add_user(user: Dict[str, Any], current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Add a new user and assign to business(es)"""
    success = await mongo_service.add_user(user)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add user")
    return {"message": f"User '{user.get('username')}' added", "status": "success"}

@app.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def admin_remove_user(user_id: str, current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Remove a user"""
    success = await mongo_service.remove_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found or could not be removed")
    return {"message": f"User '{user_id}' removed", "status": "success"}

@app.post("/admin/users/{user_id}/assign", dependencies=[Depends(require_admin)])
async def admin_assign_user_to_business(user_id: str, business_ids: List[str], current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Assign user to one or more businesses"""
    success = await mongo_service.assign_user_to_businesses(user_id, business_ids)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign user to businesses")
    return {"message": f"User '{user_id}' assigned to businesses {business_ids}", "status": "success"}

@app.post("/admin/businesses/{business_id}/schemas", dependencies=[Depends(require_admin)])
async def admin_add_or_update_business_schema(business_id: str, schema: Dict[str, Any], current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Add or update a business schema"""
    success = await mongo_service.add_or_update_business_schema(business_id, schema)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add/update schema")
    # AUTOMATION: Re-index vector embeddings after schema change using the global instance
    await vector_search_service.index_business_schemas(business_id)
    return {"message": f"Schema for business '{business_id}' added/updated and vector index refreshed", "status": "success"}

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": "The requested resource was not found",
            "status_code": 404
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    )

# =============================================================================
# DEVELOPMENT ENDPOINTS (only in debug mode)
# =============================================================================

@app.get("/debug/config")
async def debug_config(mongo_service: MongoDBService = Depends(get_mongodb_service)):
    businesses = await mongo_service.get_all_business_configs()
    return {
        "businesses": [b.business_id for b in businesses],
        "count": len(businesses)
    }

@app.get("/debug/schemas/{business_id}")
async def debug_schemas(business_id: str, mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Debug endpoint to check available schemas for a business"""
    schemas = await mongo_service.get_business_schemas(business_id)
    return {
        "business_id": business_id,
        "schemas": [
            {
                "table_name": s.table_name,
                "description": s.schema_description,
                "columns": [{"name": c.name, "type": c.type, "description": c.description} for c in s.columns]
            }
            for s in schemas
        ],
        "count": len(schemas)
    }

@app.get("/debug/vector-search/{business_id}")
async def debug_vector_search(
    business_id: str, 
    query: str, 
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to test vector search for a business"""
    results = await vector_search_service.search_schemas(business_id, query, top_k=5)
    return {
        "business_id": business_id,
        "query": query,
        "results": [
            {
                "table_name": r.get("table_name"),
                "description": r.get("schema_description"),
                "columns": [c.get("name") for c in r.get("columns", [])]
            }
            for r in results
        ],
        "count": len(results)
    }

@app.get("/debug/conversations/user/{user_id}")
async def debug_user_conversations(
    user_id: str,
    business_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to list all conversations for a user"""
    try:
        conversations = await mongo_service.get_user_conversations(user_id, business_id, limit=10)
        return {
            "user_id": user_id,
            "business_id": business_id,
            "conversation_count": len(conversations),
            "conversations": [
                {
                    "session_id": conv.session_id,
                    "message_count": len(conv.conversation_memory.messages) if conv.conversation_memory else 0,
                    "last_activity": conv.last_activity,
                    "status": conv.status
                }
                for conv in conversations
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving user conversations: {e}")
        return {"error": str(e)}

@app.get("/debug/session-info/{session_id}")
async def debug_session_info(
    session_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to show session ID variations and find matching sessions"""
    # Clean session ID to handle Swagger UI escaping
    original_session_id = session_id
    cleaned_session_id = clean_session_id(session_id)
    
    # Try different variations
    variations = [
        session_id,
        f'"{session_id}"',
        f'\\"{session_id}\\"',
        cleaned_session_id
    ]
    
    results = {}
    for var in variations:
        try:
            session = await mongo_service.get_conversation_session(var)
            if session:
                results[var] = {
                    "found": True,
                    "message_count": len(session.conversation_memory.messages) if session.conversation_memory and session.conversation_memory.messages else 0,
                    "session_id_in_db": session.session_id
                }
            else:
                results[var] = {"found": False}
        except Exception as e:
            results[var] = {"found": False, "error": str(e)}
    
    return {
        "original_session_id": original_session_id,
        "cleaned_session_id": cleaned_session_id,
        "variations_tested": variations,
        "results": results
    }

@app.get("/debug/cleanup-sessions")
async def debug_cleanup_sessions(
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to clean up sessions with escaped quotes"""
    try:
        # Find all sessions with escaped quotes
        cursor = mongo_service._collections['conversation_sessions'].find({
            "session_id": {"$regex": r'^".*"$'}
        })
        
        migrated_sessions = []
        async for doc in cursor:
            old_session_id = doc['session_id']
            new_session_id = old_session_id.replace('"', '').replace('\\"', '')
            
            if old_session_id != new_session_id:
                success = await mongo_service.migrate_session_id_format(old_session_id, new_session_id)
                if success:
                    migrated_sessions.append({
                        "old_id": old_session_id,
                        "new_id": new_session_id
                    })
        
        return {
            "message": f"Migrated {len(migrated_sessions)} sessions",
            "migrated_sessions": migrated_sessions
        }
    except Exception as e:
        return {"error": f"Error cleaning up sessions: {str(e)}"}

@app.get("/debug/migrate-escaped-sessions")
async def debug_migrate_escaped_sessions(
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to migrate all sessions with escaped quotes to clean format"""
    try:
        # Find all sessions with escaped quotes
        escaped_sessions = []
        cursor = mongo_service._collections['conversation_sessions'].find({
            "session_id": {"$regex": r'^".*"$'}  # Session IDs wrapped in quotes
        })
        
        async for doc in cursor:
            escaped_sessions.append(doc)
        
        logger.info(f"[Debug] Found {len(escaped_sessions)} sessions with escaped quotes")
        
        migrated_count = 0
        for doc in escaped_sessions:
            old_session_id = doc['session_id']
            # Clean the session ID
            new_session_id = old_session_id.replace('"', '').replace('\\"', '')
            
            if new_session_id != old_session_id:
                success = await mongo_service.migrate_session_id_format(old_session_id, new_session_id)
                if success:
                    migrated_count += 1
                    logger.info(f"[Debug] Migrated session from '{old_session_id}' to '{new_session_id}'")
        
        return {
            "message": "Session migration completed",
            "total_escaped_sessions": len(escaped_sessions),
            "migrated_sessions": migrated_count,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Session migration failed: {e}")
        return {
            "message": f"Session migration failed: {str(e)}",
            "status": "error"
        }

@app.get("/debug/list-all-sessions")
async def debug_list_all_sessions(
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to list all sessions and their formats"""
    try:
        cursor = mongo_service._collections['conversation_sessions'].find({})
        
        sessions = []
        async for doc in cursor:
            session_info = {
                "session_id": doc.get('session_id'),
                "user_id": doc.get('user_id'),
                "business_id": doc.get('business_id'),
                "has_escaped_quotes": doc.get('session_id', '').startswith('"') and doc.get('session_id', '').endswith('"'),
                "message_count": len(doc.get('conversation_memory', {}).get('messages', [])),
                "created_at": doc.get('created_at'),
                "last_activity": doc.get('last_activity')
            }
            sessions.append(session_info)
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions,
            "escaped_sessions": [s for s in sessions if s['has_escaped_quotes']],
            "clean_sessions": [s for s in sessions if not s['has_escaped_quotes']]
        }
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return {
            "message": f"Failed to list sessions: {str(e)}",
            "status": "error"
        }

@app.get("/debug/conversations/{session_id}")
async def debug_conversation(
    session_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    # Clean session ID to handle Swagger UI escaping
    original_session_id = session_id
    session_id = clean_session_id(session_id)
    if original_session_id != session_id:
        logger.info(f"[Debug] Cleaned session_id from '{original_session_id}' to '{session_id}'")
    """Debug endpoint to check conversation storage"""
    try:
        session = await mongo_service.get_conversation_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "business_id": session.business_id,
                "status": session.status,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "expires_at": session.expires_at,
                "conversation_memory": {
                    "message_count": len(session.conversation_memory.messages) if session.conversation_memory and session.conversation_memory.messages else 0,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                            "timestamp": msg.timestamp
                        }
                        for msg in (session.conversation_memory.messages if session.conversation_memory and session.conversation_memory.messages else [])
                    ]
                },
                "cached_schema_context": len(session.cached_schema_context) if session.cached_schema_context else 0
            }
        else:
            return {"error": f"Session '{session_id}' not found"}
    except Exception as e:
        return {"error": f"Error retrieving session: {str(e)}"}

@app.get("/debug/session/{session_id}")
async def debug_session(
    session_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to check the current state of a session"""
    # Clean session ID to handle Swagger UI escaping
    original_session_id = session_id
    session_id = clean_session_id(session_id)
    if original_session_id != session_id:
        logger.info(f"[Debug] Cleaned session_id from '{original_session_id}' to '{session_id}'")
    
    try:
        session = await mongo_service.get_conversation_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "business_id": session.business_id,
                "status": session.status,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "expires_at": session.expires_at,
                "conversation_memory": {
                    "message_count": len(session.conversation_memory.messages) if session.conversation_memory and session.conversation_memory.messages else 0,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                            "timestamp": msg.timestamp
                        }
                        for msg in (session.conversation_memory.messages if session.conversation_memory and session.conversation_memory.messages else [])
                    ]
                },
                "cached_schema_context": len(session.cached_schema_context) if session.cached_schema_context else 0
            }
        else:
            return {"error": f"Session '{session_id}' not found"}
    except Exception as e:
        return {"error": f"Error retrieving session: {str(e)}"}

@app.get("/debug/generate-session-id")
async def debug_generate_session_id(
    current_user: dict = Depends(get_current_user)
):
    """Debug endpoint to generate a clean session ID for testing"""
    user_id = current_user["user_id"]
    business_id = current_user["business_id"]
    session_id = f"sess_{user_id}_{business_id}_{int(datetime.now().timestamp())}"
    return {
        "session_id": session_id,
        "user_id": user_id,
        "business_id": business_id,
        "timestamp": int(datetime.now().timestamp()),
        "note": "Use this session_id in your chat requests to maintain conversation context"
    }

@app.get("/debug/mcp-test/{business_id}")
async def debug_mcp_test(
    business_id: str, 
    query: str = "SELECT 1 as test",
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to test MCP connection directly"""
    try:
        result = await mcp_client.execute_query(query, business_id)
        return {
            "business_id": business_id,
            "query": query,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "business_id": business_id,
            "query": query,
            "error": str(e),
            "status": "error"
        }

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Made optional
    business_id: Optional[str] = None  # Optional business_id for admin users
    conversation_history: Optional[List[Dict[str, Any]]] = []
    refresh_schema_context: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    
    @validator('session_id', pre=True)
    def clean_session_id(cls, v):
        if v is None:
            return v
        # Remove escaped quotes that might be present
        cleaned = v.replace('\\"', '').replace('"', '')
        return cleaned
    
    class Config:
        # Ensure proper JSON serialization
        json_encoders = {
            str: lambda v: v.replace('\\"', '').replace('"', '') if isinstance(v, str) else v
        }

# Add this function to extract user_id from the request state (assuming authentication middleware sets it)
async def get_user_id(request: Request):
    # If you use request.state.user, adjust as needed
    user = getattr(request.state, 'user', None)
    if user and hasattr(user, 'user_id'):
        return str(user.user_id)
    # Fallback: use IP if user_id is not available
    return request.client.host

# Add this helper at the top-level (after imports)
def ensure_chat_messages(messages):
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(ChatMessage(**msg))
        elif isinstance(msg, ChatMessage):
            result.append(msg)
        else:
            # Handles LangGraph/LangChain Message or any other type with .role and .content
            result.append(ChatMessage(role=getattr(msg, 'role', 'user'), content=getattr(msg, 'content', '')))
    return result

# Remove the is_database_related_query_dynamic function definition from this file.
# If needed, import it from backend.app.utils.query_classifier instead.

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(RateLimiter(times=10, seconds=60, identifier=get_user_id))])
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    import logging
    logger = logging.getLogger(__name__)
    # 1. Load session from Redis first, then MongoDB if not found
    session = None
    conversation_history = []
    pause_context = None
    session_id = request.session_id
    if session_id:
        redis_data = await redis_service.get_session_conversation(session_id)
        if redis_data:
            # Build session and conversation_history from Redis data
            conversation_memory = ConversationMemory(
                messages=[Message(**msg) for msg in redis_data.get("messages", [])]
            )
            session = ConversationSession(
                session_id=redis_data.get("session_id"),
                user_id=redis_data.get("user_id", current_user["user_id"]),
                business_id=redis_data.get("business_id", request.business_id or current_user["business_id"]),
                conversation_memory=conversation_memory,
                pause_context=redis_data.get("pause_context"),
                created_at=redis_data.get("created_at"),
                last_activity=redis_data.get("last_updated"),
                expires_at=None,
                status=redis_data.get("status", "active"),
                cached_schema_context=redis_data.get("cached_schema_context"),
                updated_at=None,
                id=None
            )
            conversation_history = conversation_memory.messages
            pause_context = redis_data.get("pause_context")
        else:
            # Fallback to MongoDB
            session = await mongo_service.get_conversation_session(session_id)
            if session and session.conversation_memory and session.conversation_memory.messages:
                conversation_history = session.conversation_memory.messages
                pause_context = getattr(session, "pause_context", None)
            else:
                conversation_history = []
                pause_context = None
    else:
        # New session
        conversation_history = []
        raw_user_id = str(current_user['user_id'])
        clean_user_id = raw_user_id.replace('"', '').replace("'", "")
        session_id = f"sess_{clean_user_id}_{request.business_id or current_user['business_id']}_{int(datetime.now(timezone.utc).timestamp())}"
        session_data = ConversationSessionCreate(
            session_id=session_id,
            user_id=clean_user_id,
            business_id=request.business_id or current_user["business_id"],
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        session = await mongo_service.create_conversation_session(session_data)
        session.session_id = session_id
        request.session_id = session_id
        pause_context = None

    # After loading session
    logger.info(f"DEBUG: After loading session, pause_context={getattr(session, 'pause_context', None)}")

    # Ensure all messages are valid ChatMessage before passing to agentic workflow
    conversation_history = ensure_chat_messages(conversation_history)
    conversation_history = conversation_history[-20:]
    while conversation_history and conversation_history[-1].role == 'user':
        conversation_history.pop()

    # --- PAUSE/RESUME LOGIC ---
    confirm_triggers = {"confirm", "yes", "confirm update", "confirm delete", "yes, update", "yes, delete", "update confirmed", "delete confirmed"}
    cancel_triggers = {"no", "cancel", "stop", "abort", "never mind", "nevermind"}
    user_message = request.message.strip().lower()
    pause_sql = None
    pause_schema_context = None
    pause_business_id = None
    pause_reason = None
    pause_message = None
    if hasattr(session, 'pause_context') and session.pause_context:
        pause_reason = session.pause_context.get('pause_reason')
        pause_message = session.pause_context.get('pause_message')
        pause_sql = session.pause_context.get('sql')
        pause_schema_context = session.pause_context.get('schema_context')
        pause_business_id = session.pause_context.get('business_id')
        cleaned_history = [ChatMessage(role=m.role, content=m.content) for m in session.conversation_memory.messages]
        # Remove trailing (user: yes, assistant: confirm prompt) pairs
        while len(cleaned_history) >= 2 and \
            cleaned_history[-1].role == 'assistant' and 'confirm' in cleaned_history[-1].content.lower() and \
            cleaned_history[-2].role == 'user' and any(trigger in cleaned_history[-2].content.lower() for trigger in confirm_triggers):
            cleaned_history = cleaned_history[:-2]
        import string
        user_message_clean = request.message.strip().lower().translate(str.maketrans('', '', string.punctuation))
        is_confirmation = any(trigger in user_message_clean for trigger in confirm_triggers)
        is_cancel = any(trigger in user_message_clean for trigger in cancel_triggers)
        # --- CLEAR PAUSE IF INTERRUPTED ---
        if not is_confirmation and not is_cancel:
            # Clear pause from session and context
            session.pause_context = None
            pause_context = None
            pause_reason = None
            pause_message = None
            pause_sql = None
            pause_schema_context = None
            # Build state as normal (no pause fields at all)
            state = ChatGraphState(
                user_id=session.user_id,
                business_id=session.business_id,
                message=request.message,
                conversation_history=cleaned_history,
            )
            # Save cleared pause to MongoDB and Redis immediately
            await mongo_service.update_conversation_session(session)
            redis_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "business_id": session.business_id,
                "messages": [msg.model_dump() for msg in cleaned_history],
                "pause_context": None,
                "created_at": str(session.created_at),
                "last_updated": str(datetime.now(timezone.utc)),
            }
            redis_data = convert_objectid_to_str(redis_data)
            await redis_service.cache_session_conversation(session.session_id, redis_data)
        else:
            state = ChatGraphState(
                user_id=session.user_id,
                business_id=pause_business_id or session.business_id,
                message=request.message,
                conversation_history=cleaned_history,
                sql=pause_sql,
                schema_context=pause_schema_context,
                resume_from_pause=is_confirmation,
                pause_reason=pause_reason,
                pause_message=pause_message,
                confirm=is_confirmation,
            )
    else:
        state = ChatGraphState(
            user_id=session.user_id,
            business_id=session.business_id,
            message=request.message,
            conversation_history=[ChatMessage(role=m.role, content=m.content) for m in session.conversation_memory.messages],
        )

    # 4. Build LangGraph state and run the workflow
    # Use 'state' for workflow execution
    # After loading session and before invoking the workflow
    if hasattr(session, 'pause_context') and session.pause_context:
        pause_reason = session.pause_context.get('pause_reason')
        pause_message = session.pause_context.get('pause_message')
        pause_sql = session.pause_context.get('sql')
        pause_schema_context = session.pause_context.get('schema_context')
        pause_business_id = session.pause_context.get('business_id')
        user_message = request.message.strip().lower()
        import string
        user_message_clean = user_message.translate(str.maketrans('', '', string.punctuation))
        confirm_triggers = {"confirm", "yes", "confirm update", "confirm delete", "yes update", "yes delete", "update confirmed", "delete confirmed"}
        is_confirmation = any(trigger in user_message_clean for trigger in confirm_triggers)
        is_cancel = any(trigger in user_message_clean for trigger in cancel_triggers)
        logger.info(f"DEBUG: pause_context present. pause_reason={pause_reason}, user_message='{user_message}', user_message_clean='{user_message_clean}', is_confirmation={is_confirmation}, is_cancel={is_cancel}")
        if is_cancel:
            # Clear pause context and respond with cancellation
            session.pause_context = None
            pause_context = None
            assistant_response = "Okay, the operation has been cancelled. Let me know if you need anything else!"
            await mongo_service.update_conversation_session(session)
            redis_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "business_id": session.business_id,
                "messages": [msg.model_dump() for msg in conversation_history],
                "pause_context": pause_context,
                "created_at": str(session.created_at),
                "last_updated": str(datetime.now(timezone.utc)),
            }
            redis_data = convert_objectid_to_str(redis_data)
            await redis_service.cache_session_conversation(session.session_id, redis_data)
            return ChatResponse(
                response=assistant_response,
                session_id=session.session_id
            )
        if is_confirmation:
            state = ChatGraphState(
                message=request.message,
                business_id=pause_business_id or session.business_id,
                user_id=session.user_id,
                conversation_history=conversation_history,
                pause_reason=pause_reason,
                pause_message=pause_message,
                sql=pause_sql,
                schema_context=pause_schema_context,
                resume_from_pause=True,
                confirm=True,
            )
            logger.info(f"DEBUG: Constructed state for confirmation: pause_reason={pause_reason}, resume_from_pause=True, confirm=True, message='{request.message}'")
        else:
            state = ChatGraphState(
                message=request.message,
                business_id=pause_business_id or session.business_id,
                user_id=session.user_id,
                conversation_history=conversation_history,
                pause_reason=pause_reason,
                pause_message=pause_message,
                sql=pause_sql,
                schema_context=pause_schema_context,
                resume_from_pause=False,
                confirm=False,
            )
            logger.info(f"DEBUG: Constructed state for waiting confirmation: pause_reason={pause_reason}, resume_from_pause=False, confirm=False, message='{request.message}'")
    else:
        # Normal state construction
        state = ChatGraphState(
            message=request.message,
            business_id=session.business_id,
            user_id=session.user_id,
            conversation_history=conversation_history,
        )
        logger.info(f"DEBUG: Constructed normal state: message='{request.message}', business_id={session.business_id}, user_id={session.user_id}")
    # Log the state before workflow invocation
    logger.info(f"DEBUG: Passing state to workflow: pause_reason={getattr(state, 'pause_reason', None)}, resume_from_pause={getattr(state, 'resume_from_pause', None)}, confirm={getattr(state, 'confirm', None)}, message='{state.message}'")
    # Use the agent registry to get the 'general' agent (no logic change yet)
    agent = agent_registry.get_agent('general')
    langgraph_state = copy.deepcopy(state)
    result = await agent.ainvoke(langgraph_state)
    logger.info(f"DEBUG: Workflow returned result of type {type(result)}: {result}")
    assistant_response = str(result.get('response', '')).strip() if isinstance(result, dict) else str(getattr(result, 'response', ''))

    # 5. Handle delete/update confirmation pause (if present)
    if isinstance(result, dict) and result.get('pause_reason') in ['confirm_delete', 'confirm_update']:
        logger.info(f"DEBUG: Workflow result for pause: {result}")
        assistant_response = result.get('pause_message', assistant_response)
        pause_context = {
            'sql': result.get('sql'),
            'schema_context': result.get('schema_context'),
            'business_id': state.business_id,
            'user_id': state.user_id,
            'conversation_history': [msg.model_dump() for msg in state.conversation_history],
            'pause_reason': result.get('pause_reason'),
            'pause_message': result.get('pause_message'),
        }
        session.pause_context = pause_context  # <-- SAVE THE ACTUAL CONTEXT!
        logger.info(f"DEBUG: Saving session.pause_context={pause_context}")
    # Only clear pause_context after confirmed update/delete execution, not here!

    # 6. Post-process for user-friendliness if DB-like result
    db_like = extract_db_like(assistant_response)
    if db_like:
        assistant_response = format_db_result(db_like)

    # 7. Append new user and assistant messages AFTER the workflow
    conversation_history.append(ChatMessage(role="user", content=request.message))
    conversation_history.append(ChatMessage(role="assistant", content=assistant_response))
    # Remove trailing confirmation/yes pairs before saving
    cleaned_history = conversation_history
    while len(cleaned_history) >= 2 and \
        cleaned_history[-1].role == 'assistant' and 'confirm' in cleaned_history[-1].content.lower() and \
        cleaned_history[-2].role == 'user' and any(trigger in cleaned_history[-2].content.lower() for trigger in confirm_triggers):
        cleaned_history = cleaned_history[:-2]
    session.conversation_memory.messages = cleaned_history
    session.pause_context = pause_context
    logger.info(f"DEBUG: Saving session.pause_context={session.pause_context}")

    # After workflow execution and before saving session:
    if state.confirm and state.resume_from_pause:
        # Remove the last two messages (confirmation prompt and user "yes")
        if len(state.conversation_history) >= 2:
            state.conversation_history = state.conversation_history[:-2]
        # Optionally, add a result message from the assistant
        if state.response:
            state.conversation_history.append({'role': 'assistant', 'content': state.response})
        # Clear pause context ONLY after confirmation and update execution
        session.pause_context = None
        pause_context = None  # Also clear for Redis
        logger.info(f"DEBUG: Cleared pause_context after confirmed update/delete execution.")

    # 8. Save to MongoDB (for durability)
    await mongo_service.update_conversation_session(session)

    # 9. Save to Redis (for speed)
    # Guarantee pause_context is not present after confirmation
    if state.confirm and state.resume_from_pause:
        pause_context = None
        session.pause_context = None
    logger.debug(f"[ChatEndpoint] Saving to Redis. pause_context: {pause_context}")
    redis_data = {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "business_id": session.business_id,
        "messages": [msg.model_dump() for msg in conversation_history],
        "pause_context": pause_context,  # This will be None if cleared
        "created_at": str(session.created_at),
        "last_updated": str(datetime.now(timezone.utc)),
    }
    redis_data = convert_objectid_to_str(redis_data)
    await redis_service.cache_session_conversation(
        session.session_id,
        redis_data
    )

    # 10. Return response
    return ChatResponse(
        response=assistant_response,
        session_id=session.session_id
    )

@app.post("/debug/generate-sql")
async def debug_generate_sql(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to test SQL generation without execution"""
    user_id = current_user["user_id"]
    business_id = current_user["business_id"]
    
    # First, classify the query using the same logic as the main chat endpoint
    is_db_query = await is_database_related_query_dynamic(request.message, business_id, vector_search_service, [])
    
    if not is_db_query:
        # This is a general conversation, not a database query
        return {
            "business_id": business_id,
            "user_message": request.message,
            "classification": "GENERAL_CONVERSATION",
            "message": "This is a general conversation query, not a database query. Use the main /chat endpoint for general conversations.",
            "should_generate_sql": False
        }
    
    # Get schema context
    schema_context = await vector_search_service.search_schemas(business_id, request.message, top_k=5)
    
    # Format schema text
    schema_text = ""
    if schema_context:
        for schema in schema_context:
            schema_text += f"\nTable: {schema.get('table_name', 'Unknown')}\n"
            schema_text += f"Description: {schema.get('schema_description', 'No description')}\n"
            schema_text += "Columns:\n"
            for col in schema.get('columns', []):
                schema_text += f"  - {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')} ({col.get('description', 'No description')})\n"
    
    # Generate SQL prompt
    sql_prompt = (
        "You are an expert SQL assistant for a PostgreSQL database. "
        "Your task is to convert natural language requests into SQL queries.\n\n"
        "AVAILABLE DATABASE SCHEMAS:\n{schema_text}\n"
        "CURRENT USER REQUEST: {request.message}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the user's natural language request\n"
        "2. Identify relevant tables and columns from the schema above\n"
        "3. Generate a single, safe, syntactically correct SQL query\n"
        "4. Use SELECT, INSERT, UPDATE, or DELETE statements as appropriate\n"
        "5. For INSERT: Generate valid INSERT statements with proper values\n"
        "6. For UPDATE: Generate UPDATE statements with WHERE clauses to target specific records\n"
        "7. For DELETE: Generate DELETE statements with WHERE clauses to target specific records\n"
        "8. Never use DROP, TRUNCATE, or ALTER statements\n"
        "9. Use only the tables and columns provided in the schema context\n\n"
        "OUTPUT FORMAT: Generate ONLY the complete SQL query, no explanations, no markdown, no code blocks, no prefixes.\n"
        "Preserve all SQL clauses including WHERE, ORDER BY, GROUP BY, HAVING, etc.\n"
        "Example outputs:\n"
        "- SELECT: SELECT * FROM customers WHERE active = true;\n"
        "- INSERT: INSERT INTO customers (name, email, phone) VALUES ('John Doe', 'john@example.com', '1234567890');\n"
        "- UPDATE: UPDATE customers SET phone = '0987654321' WHERE id = 1;\n"
        "- DELETE: DELETE FROM customers WHERE id = 1;\n"
        "If the request cannot be handled with a valid SQL query, reply: 'Operation not allowed.'\n"
        "If no relevant tables are found in the schema, reply: 'No relevant tables found in schema.'\n\n"
        "SQL QUERY:"
    )
    
    # Generate SQL
    sql_response = await llm_service.chat([
        {"role": "system", "content": sql_prompt},
        {"role": "user", "content": request.message}
    ])
    
    # Clean SQL (same logic as main endpoint)
    sql_query = sql_response.strip()
    
    # Remove markdown code blocks
    if sql_query.startswith('```'):
        lines = sql_query.split('\n')
        if len(lines) > 1:
            sql_lines = []
            for line in lines[1:]:
                if line.strip() == '```':
                    break
                sql_lines.append(line)
            sql_query = '\n'.join(sql_lines).strip()
    
    # Remove prefixes from first line
    lines = sql_query.split('\n')
    if lines:
        first_line = lines[0].strip()
        prefixes_to_remove = ['sql query:', 'sql:', 'query:']
        for prefix in prefixes_to_remove:
            if first_line.lower().startswith(prefix):
                first_line = first_line[len(prefix):].strip()
                break
        lines[0] = first_line
        sql_query = '\n'.join(lines).strip()
    
    # Remove trailing semicolon to prevent syntax errors when LIMIT is added
    sql_query = sql_query.rstrip(';').strip()
    
    return {
        "business_id": business_id,
        "user_message": request.message,
        "classification": "DATABASE_QUERY",
        "raw_llm_response": sql_response,
        "cleaned_sql": sql_query,
        "schema_context": [s.get('table_name') for s in schema_context],
        "should_generate_sql": True
    }

@app.get("/debug/redis-conversation/{session_id}")
async def debug_redis_conversation(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Debug endpoint to check Redis conversation cache"""
    # Clean session ID to handle Swagger UI escaping
    cleaned_session_id = clean_session_id(session_id)
    
    # Get conversation from Redis
    cached_conversation = await redis_service.get_session_conversation(cleaned_session_id)
    
    if cached_conversation:
        # Get TTL
        ttl = await redis_service.get_session_ttl(cleaned_session_id)
        
        return {
            "session_id": cleaned_session_id,
            "found_in_redis": True,
            "ttl_seconds": ttl,
            "message_count": len(cached_conversation.get("messages", [])),
            "created_at": cached_conversation.get("created_at"),
            "last_updated": cached_conversation.get("last_updated"),
            "messages": cached_conversation.get("messages", [])
        }
    else:
        return {
            "session_id": cleaned_session_id,
            "found_in_redis": False,
            "message": "No conversation found in Redis cache"
        }

@app.get("/debug/redis-stats")
async def debug_redis_stats(
    current_user: dict = Depends(get_current_user)
):
    """Debug endpoint to check Redis cache statistics"""
    stats = await redis_service.get_cache_stats()
    return stats

@app.get("/debug/conversation-source/{session_id}")
async def debug_conversation_source(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Debug endpoint to check where conversation data comes from (Redis vs MongoDB)"""
    # Clean session ID to handle Swagger UI escaping
    cleaned_session_id = clean_session_id(session_id)
    
    # Check Redis first
    cached_conversation = await redis_service.get_session_conversation(cleaned_session_id)
    redis_has_data = cached_conversation and cached_conversation.get("messages")
    
    # Check MongoDB
    mongo_service = get_mongodb_service()
    session = await mongo_service.get_conversation_session(cleaned_session_id)
    mongo_has_data = session and session.conversation_memory and session.conversation_memory.messages
    
    return {
        "session_id": cleaned_session_id,
        "redis_has_data": redis_has_data,
        "redis_message_count": len(cached_conversation.get("messages", [])) if cached_conversation else 0,
        "redis_ttl": await redis_service.get_session_ttl(cleaned_session_id) if redis_has_data else None,
        "mongo_has_data": mongo_has_data,
        "mongo_message_count": len(session.conversation_memory.messages) if mongo_has_data else 0,
        "conversation_source": "Redis" if redis_has_data else "MongoDB" if mongo_has_data else "None",
        "recommended_source": "Redis (faster)" if redis_has_data else "MongoDB (persistent)" if mongo_has_data else "No data"
    }

@app.get("/debug/audit-logs")
async def debug_audit_logs(
    user_id: Optional[str] = None,
    business_id: Optional[str] = None,
    operation_type: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to view audit logs"""
    # Only admins can view audit logs
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can view audit logs"
        )
    
    logs = await mongo_service.get_audit_logs(
        user_id=user_id,
        business_id=business_id,
        operation_type=operation_type,
        limit=limit
    )
    
    return {
        "audit_logs": [
            {
                "id": str(log.id),
                "user_id": log.user_id,
                "business_id": log.business_id,
                "session_id": log.session_id,
                "operation_type": log.operation_type,
                "table_name": log.table_name,
                "sql_query": log.sql_query,
                "affected_rows": log.affected_rows,
                "timestamp": log.timestamp.isoformat(),
                "success": log.success,
                "error_message": log.error_message
            }
            for log in logs
        ],
        "total_count": len(logs)
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level=settings.server.log_level.lower()
    ) 