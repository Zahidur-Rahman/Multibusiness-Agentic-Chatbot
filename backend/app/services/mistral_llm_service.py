from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.app.config import get_settings
from backend.app.services.mongodb_service import mongodb_service
from backend.app.models.business import BusinessConfig, BusinessSchema, BusinessConfigCreate, BusinessConfigUpdate
from backend.app.models.user import User, UserPermission, UserSession, UserCreate, UserUpdate
from backend.app.models.conversation import ConversationSession, ConversationAnalytics, ConversationSessionCreate
from backend.app.services.mongodb_service import MongoDBService
from backend.app.utils import hash_password, verify_password
from backend.app.auth.routes import require_role
import logging
logger = logging.getLogger(__name__)

settings = get_settings()

class MistralLLMService:
    def __init__(self):
        self.api_key = settings.mistral.api_key
        self.model = settings.mistral.model
        self.llm = ChatMistralAI(model=self.model, api_key=self.api_key)

    async def chat(self, messages: List[dict]) -> str:
        """
        messages: List of dicts with 'role' ('user' or 'system') and 'content'.
        Returns: LLM response as string.
        """
        lc_messages = []
        for msg in messages:
            if msg['role'] == 'user':
                lc_messages.append(HumanMessage(content=msg['content']))
            else:
                lc_messages.append(SystemMessage(content=msg['content']))
        response = await self.llm.ainvoke(lc_messages)
        return response.content

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using Mistral (if supported) or fallback to sentence-transformers.
        """
        # Mistral does not provide embeddings directly; use sentence-transformers as fallback
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode([text])[0].tolist() 