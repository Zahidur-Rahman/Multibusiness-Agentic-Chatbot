"""
MongoDB models for business configurations and schemas.
Supports Pinecone, Weaviate, or Faiss for vector search.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        import pydantic_core
        return pydantic_core.core_schema.no_info_after_validator_function(
            cls.validate,
            handler(ObjectId)
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # This tells Pydantic/OpenAPI to treat it as a string in the schema
        return {'type': 'string', 'format': 'objectid'}

class DatabaseConfig(BaseModel):
    """Database configuration for a business"""
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    ssl_mode: str = "require"

class BusinessConfig(BaseModel):
    """Business configuration model"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    business_id: str = Field(unique=True, index=True)
    name: str
    description: Optional[str] = None  # Added description field for business
    db_config: DatabaseConfig
    status: str = "active"  # active, inactive, suspended
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ColumnInfo(BaseModel):
    """Column information for schema"""
    name: str
    type: str
    max_length: Optional[int] = None
    nullable: bool = True
    default: Optional[str] = None
    position: int
    business_meaning: Optional[str] = None
    description: Optional[str] = None  # Added description field for column

class TableRelationship(BaseModel):
    """Table relationship information"""
    table_name: str
    relationship_type: str  # foreign_key, primary_key, etc.
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None

class BusinessSchema(BaseModel):
    """Business schema model for vector embeddings"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    business_id: str = Field(index=True)
    table_name: str
    schema_description: str
    columns: List[ColumnInfo]
    relationships: List[TableRelationship] = []
    embedding_text: str  # Combined schema description for vector search
    vector_id: Optional[str] = None  # ID in vector store (Pinecone/Weaviate/Faiss)
    indexed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class BusinessConfigCreate(BaseModel):
    """Model for creating business configuration"""
    business_id: str
    name: str
    description: Optional[str] = None  # Added description field for business
    db_config: DatabaseConfig
    status: str = "active"

class BusinessConfigUpdate(BaseModel):
    """Model for updating business configuration"""
    name: Optional[str] = None
    description: Optional[str] = None  # Added description field for business
    db_config: Optional[DatabaseConfig] = None
    status: Optional[str] = None

class BusinessSchemaCreate(BaseModel):
    """Model for creating business schema"""
    business_id: str
    table_name: str
    schema_description: str
    columns: List[ColumnInfo]
    relationships: List[TableRelationship] = []
    embedding_text: str

class BusinessSchemaUpdate(BaseModel):
    """Model for updating business schema"""
    schema_description: Optional[str] = None
    columns: Optional[List[ColumnInfo]] = None
    relationships: Optional[List[TableRelationship]] = None
    embedding_text: Optional[str] = None 