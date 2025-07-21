"""
Configuration management for the Multi-Business Conversational Chatbot.
Handles environment variables, validation, and provides typed configuration.
"""

import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from dotenv import load_dotenv
import asyncio
from bson import ObjectId
# from backend.app.services.vector_search import FaissVectorSearchService  # REMOVE this import
from backend.app.models.business import BusinessSchema

# Load environment variables
load_dotenv()

class BusinessConfig(BaseSettings):
    """Individual business configuration"""
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    
    @field_validator('port', mode='before')
    def validate_port(cls, v):
        return int(v) if isinstance(v, str) else v

class MongoDBConfig(BaseSettings):
    """MongoDB configuration settings"""
    uri: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    database: str = Field(default="chatbot_config", env="MONGODB_DB")
    user: Optional[str] = Field(default=None, env="MONGODB_USER")
    password: Optional[str] = Field(default=None, env="MONGODB_PASSWORD")
    
    class Config:
        env_prefix = "MONGODB_"

class RedisConfig(BaseSettings):
    """Redis configuration settings"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    class Config:
        env_prefix = "REDIS_"

class JWTSettings(BaseSettings):
    """JWT configuration settings"""
    secret_key: str = Field(env="JWT_SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    expiration: int = Field(default=7200, env="JWT_EXPIRATION")  # 2 hours (7200 seconds)
    refresh_expiration: int = Field(default=86400, env="JWT_REFRESH_EXPIRATION")
    
    class Config:
        env_prefix = "JWT_"

class MistralConfig(BaseSettings):
    """Mistral LLM configuration settings"""
    api_key: str = Field(env="MISTRAL_API_KEY")
    model: str = Field(default="mistral-large-2407", env="MISTRAL_MODEL")
    class Config:
        env_prefix = "MISTRAL_"

class VectorSearchConfig(BaseSettings):
    """Vector search configuration settings"""
    store_type: str = Field(default="faiss", env="VECTOR_STORE_TYPE")
    faiss_index_path: Optional[str] = Field(default=None, env="FAISS_INDEX_PATH")
    class Config:
        env_prefix = ""

class ServerConfig(BaseSettings):
    """Server configuration settings"""
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    show_sql_debug: bool = Field(default=False, env="SHOW_SQL_DEBUG")
    
    # CORS settings
    allowed_origins: List[str] = Field(default=["http://localhost:3000"], env="ALLOWED_ORIGINS")
    allowed_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], env="ALLOWED_METHODS")
    allowed_headers: List[str] = Field(default=["*"], env="ALLOWED_HEADERS")
    
    @field_validator('allowed_origins', 'allowed_methods', 'allowed_headers', mode='before')
    def parse_list_fields(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v

class SecurityConfig(BaseSettings):
    """Security configuration settings"""
    bcrypt_rounds: int = Field(default=12, env="BCRYPT_ROUNDS")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    class Config:
        env_prefix = ""

class FeatureFlags(BaseSettings):
    """Feature flags configuration"""
    enable_vector_search: bool = Field(default=True, env="ENABLE_VECTOR_SEARCH")
    enable_schema_discovery: bool = Field(default=True, env="ENABLE_SCHEMA_DISCOVERY")
    enable_conversation_memory: bool = Field(default=True, env="ENABLE_CONVERSATION_MEMORY")
    enable_multi_business: bool = Field(default=True, env="ENABLE_MULTI_BUSINESS")
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    enable_cross_session_context: bool = Field(default=False, env="ENABLE_CROSS_SESSION_CONTEXT")
    max_conversation_context_messages: int = Field(default=20, env="MAX_CONVERSATION_CONTEXT_MESSAGES")
    
    class Config:
        env_prefix = ""

class Settings(BaseSettings):
    """Main application settings"""
    
    # Core configurations
    mongodb: MongoDBConfig = MongoDBConfig()
    redis: RedisConfig = RedisConfig()
    jwt: JWTSettings = JWTSettings()
    vector_search: VectorSearchConfig = VectorSearchConfig()
    server: ServerConfig = ServerConfig()
    security: SecurityConfig = SecurityConfig()
    features: FeatureFlags = FeatureFlags()
    mistral: MistralConfig = MistralConfig()
    
    # NOTE: Per-business database configuration is NOT loaded here.
    # It must be fetched dynamically from MongoDB at runtime via the service layer.
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

# Remove get_business_config and business_configs logic

def get_database_url(business_config) -> str:
    """Get database URL for a specific business (pass BusinessConfig object)"""
    return f"postgresql://{business_config.user}:{business_config.password}@{business_config.host}:{business_config.port}/{business_config.database}"

def get_mongodb_url() -> str:
    """Get MongoDB connection URL"""
    config = settings.mongodb
    if config.user and config.password:
        return f"mongodb://{config.user}:{config.password}@{config.uri.replace('mongodb://', '')}/{config.database}"
    return f"{config.uri}/{config.database}"

def get_redis_url() -> str:
    """Get Redis connection URL"""
    config = settings.redis
    if config.password:
        return f"redis://:{config.password}@{config.host}:{config.port}/{config.db}"
    return f"redis://{config.host}:{config.port}/{config.db}"

async def validate_business_access(user_id: str, requested_business_id: str) -> bool:
    """Validate if user has access to the requested business by checking MongoDB user permissions."""
    from backend.app.services.mongodb_service import mongodb_service  # Moved import here to avoid circular import
    user = await mongodb_service.get_user_by_id(user_id)
    if not user:
        return False
    
    # Admin users should have access to all businesses
    if user.role == 'admin':
        return True
    
    # For regular users, check specific business access
    allowed_businesses = user.allowed_businesses or []
    return requested_business_id in allowed_businesses 

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # This tells Pydantic to treat it as a string in OpenAPI/JSON schema
        return {"type": "string"} 