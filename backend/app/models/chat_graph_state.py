from pydantic import BaseModel
from typing import List, Optional, Any
from backend.app.models.conversation import ChatMessage

class ChatGraphState(BaseModel):
    message: str
    business_id: str
    user_id: str
    conversation_history: List[ChatMessage]
    schema_context: Optional[Any] = None
    is_db_query: Optional[bool] = None
    sql_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    sql: Optional[str] = None
    db_result: Optional[Any] = None
    response: Optional[str] = None
    next: Optional[str] = None  # For routing in LangGraph
    # --- Add these fields for pause/resume support ---
    pause_reason: Optional[str] = None
    pause_message: Optional[str] = None
    confirm: Optional[bool] = None
    resume_from_pause: Optional[bool] = None 