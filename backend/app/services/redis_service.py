"""
Redis service for caching and session management.
"""

import json
import logging
import redis.asyncio as redis
from typing import Dict, Any, Optional, List
from datetime import datetime
from backend.app.config import get_settings

settings = get_settings()
logger = logging.getLogger("redis_service")

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class RedisService:
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password,
                db=settings.redis.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            await self.redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {settings.redis.host}:{settings.redis.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("Disconnected from Redis")

    async def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self._connected or not self.redis:
            return False
        try:
            await self.redis.ping()
            return True
        except Exception:
            self._connected = False
            return False

    # =============================================================================
    # SESSION CONVERSATION CACHING (1 HOUR TTL)
    # =============================================================================

    async def cache_session_conversation(self, session_id: str, conversation_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """
        Cache session conversation in Redis (1 hour default TTL)
        Args:
            session_id: Session identifier
            conversation_data: Dictionary containing conversation data
            ttl: Time to live in seconds (1 hour = 3600 seconds)
        """
        if not await self.is_connected():
            logger.warning(f"Redis not connected, cannot cache session conversation: '{session_id}'")
            return False
        
        try:
            cache_key = f"session_conversation:{session_id}"
            logger.info(f"💾 Caching session conversation with key: '{cache_key}' for {ttl} seconds")
            await self.redis.setex(cache_key, ttl, json.dumps(conversation_data, cls=DateTimeEncoder))
            logger.info(f"✅ Cached session conversation for '{session_id}' for {ttl} seconds")
            return True
        except Exception as e:
            logger.error(f"Failed to cache session conversation: {e}")
            return False

    async def get_session_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session conversation from Redis"""
        if not await self.is_connected():
            logger.warning(f"Redis not connected, cannot get session conversation: '{session_id}'")
            return None
        
        try:
            cache_key = f"session_conversation:{session_id}"
            logger.info(f"🔍 Looking for Redis key: '{cache_key}'")
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                logger.info(f"✅ Cache hit for session conversation: '{session_id}'")
                return json.loads(cached_data)
            else:
                logger.info(f"❌ Cache miss for session conversation: '{session_id}'")
            return None
        except Exception as e:
            logger.error(f"Failed to get session conversation: {e}")
            return None

    async def update_session_conversation(self, session_id: str, conversation_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Update existing session conversation in Redis"""
        if not await self.is_connected():
            return False
        
        try:
            cache_key = f"session_conversation:{session_id}"
            await self.redis.setex(cache_key, ttl, json.dumps(conversation_data, cls=DateTimeEncoder))
            logger.info(f"Updated session conversation for '{session_id}' for {ttl} seconds")
            return True
        except Exception as e:
            logger.error(f"Failed to update session conversation: {e}")
            return False

    async def add_message_to_session(self, session_id: str, message: Dict[str, Any], ttl: int = 3600) -> bool:
        """Add a single message to existing session conversation"""
        if not await self.is_connected():
            return False
        
        try:
            # Get existing conversation
            existing_data = await self.get_session_conversation(session_id)
            if existing_data:
                # Add new message to existing conversation
                messages = existing_data.get("messages", [])
                messages.append(message)
                existing_data["messages"] = messages
                existing_data["last_updated"] = message.get("timestamp", "")
                
                # Update with new TTL
                await self.update_session_conversation(session_id, existing_data, ttl)
                logger.info(f"Added message to session conversation: '{session_id}'")
                return True
            else:
                # Create new conversation with single message
                new_data = {
                    "session_id": session_id,
                    "messages": [message],
                    "created_at": message.get("timestamp", ""),
                    "last_updated": message.get("timestamp", "")
                }
                await self.cache_session_conversation(session_id, new_data, ttl)
                logger.info(f"Created new session conversation with message: '{session_id}'")
                return True
        except Exception as e:
            logger.error(f"Failed to add message to session conversation: {e}")
            return False

    async def clear_session_conversation(self, session_id: str) -> bool:
        """Clear session conversation from Redis"""
        if not await self.is_connected():
            return False
        
        try:
            cache_key = f"session_conversation:{session_id}"
            await self.redis.delete(cache_key)
            logger.info(f"Cleared session conversation: '{session_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session conversation: {e}")
            return False

    async def get_session_ttl(self, session_id: str) -> Optional[int]:
        """Get remaining TTL for session conversation"""
        if not await self.is_connected():
            return None
        
        try:
            cache_key = f"session_conversation:{session_id}"
            ttl = await self.redis.ttl(cache_key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Failed to get session TTL: {e}")
            return None

    async def extend_session_ttl(self, session_id: str, ttl: int = 3600) -> bool:
        """Extend TTL for session conversation"""
        if not await self.is_connected():
            return False
        
        try:
            cache_key = f"session_conversation:{session_id}"
            # Get existing data
            existing_data = await self.get_session_conversation(session_id)
            if existing_data:
                # Update with new TTL
                await self.redis.setex(cache_key, ttl, json.dumps(existing_data, cls=DateTimeEncoder))
                logger.info(f"Extended TTL for session conversation: '{session_id}' to {ttl} seconds")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to extend session TTL: {e}")
            return False

    # =============================================================================
    # RATE LIMITING
    # =============================================================================

    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if rate limit is exceeded"""
        if not await self.is_connected():
            return True  # Allow if Redis is down
        
        try:
            current = await self.redis.get(key)
            if current is None:
                await self.redis.setex(key, window, 1)
                return True
            elif int(current) < limit:
                await self.redis.incr(key)
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow if Redis fails

    # =============================================================================
    # CACHE STATISTICS
    # =============================================================================

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not await self.is_connected():
            return {"error": "Redis not connected"}
        
        try:
            info = await self.redis.info()
            keys = await self.redis.dbsize()
            
            # Count session conversation keys
            session_keys = await self.redis.keys("session_conversation:*")
            
            return {
                "connected": True,
                "total_keys": keys,
                "session_conversations": len(session_keys),
                "memory_usage": info.get("used_memory_human", "Unknown"),
                "uptime": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

# Global Redis service instance
redis_service = RedisService() 