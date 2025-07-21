from typing import List
from backend.app.services.mistral_llm_service import MistralLLMService
import logging
import traceback

async def is_database_related_query_dynamic(message: str, business_id: str, vector_search_service, conversation_history: List = None) -> bool:
    """
    Fully LLM-based dynamic classification - no hardcoded rules.
    Uses the LLM to intelligently determine if a query should generate SQL.
    """
    try:
        # 1. Get schema context for the query
        schema_results = await vector_search_service.search_schemas(business_id, message, top_k=3)
        
        # 2. Create schema context for the LLM
        schema_context = ""
        if schema_results and len(schema_results) > 0:
            schema_context = "Available database schemas:\n"
            for schema in schema_results:
                schema_context += f"- {schema.get('table_name', 'Unknown')}: {schema.get('schema_description', 'No description')}\n"
        else:
            schema_context = "No relevant database schemas found."
        
        # 3. Use LLM to classify the query with conversation context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get last 4 messages for context (2 exchanges)
            recent_messages = conversation_history[-4:]
            conversation_context = "RECENT CONVERSATION CONTEXT:\n"
            for msg in recent_messages:
                role = msg.get('role', msg.role if hasattr(msg, 'role') else 'unknown')
                content = msg.get('content', msg.content if hasattr(msg, 'content') else 'unknown')
                conversation_context += f"{role.upper()}: {content}\n"
        else:
            conversation_context = "No recent conversation context available."
        
        classification_prompt = f"""
You are an intelligent, context-aware query classifier for a business chatbot system.

AVAILABLE DATABASE SCHEMAS:
{schema_context}

{conversation_context}

CURRENT USER QUERY: "{message}"

TASK: Decide if this query should generate a SQL database query or be treated as general conversation.

CLASSIFICATION RULES:
- If the user is asking for specific data, records, or facts that could be retrieved from the database, classify as DATABASE_QUERY.
- If the user is asking for general help, explanations, opinions, or casual conversation, classify as GENERAL_CONVERSATION.
- If the user mentions specific names, IDs, or entities that could be looked up, classify as DATABASE_QUERY.
- If the user is asking "how to" questions or seeking advice, classify as GENERAL_CONVERSATION.
- If the user asks "about" someone/something and relevant database tables exist, classify as DATABASE_QUERY.
- If the user asks for information that could be found in business data, classify as DATABASE_QUERY.
- If the query is ambiguous, ask the user for clarification rather than guessing.
- Use conversation context ONLY if the current query is ambiguous or uses pronouns.

EXAMPLES WITH CONTEXT:
- Previous: "show me customers" → Current: "and their orders" → DATABASE_QUERY
- Previous: "what about the menu?" → Current: "show me the prices" → DATABASE_QUERY
- Previous: "find John Smith" → Current: "what about his contact info?" → DATABASE_QUERY
- Previous: "hello" → Current: "how are you?" → GENERAL_CONVERSATION
- Previous: "show me customers" → Current: "thanks, bye" → GENERAL_CONVERSATION
- Previous: "show me customers" → Current: "what about them?" → DATABASE_QUERY

STANDALONE EXAMPLES:
- "give me all information about Zahid" → DATABASE_QUERY
- "About Zahid" → DATABASE_QUERY
- "show me the menu" → DATABASE_QUERY
- "how do I reset my password?" → GENERAL_CONVERSATION
- "what's the weather like?" → GENERAL_CONVERSATION
- "find customer John Smith" → DATABASE_QUERY
- "can you help me?" → GENERAL_CONVERSATION
- "tell me about the restaurant" → DATABASE_QUERY
- "hello" → GENERAL_CONVERSATION
- "I want to see all orders placed last week" → DATABASE_QUERY
- "Can you explain how invoices work?" → GENERAL_CONVERSATION
- "Show me all employees in the sales department" → DATABASE_QUERY
- "What is the refund policy?" → GENERAL_CONVERSATION

RESPONSE FORMAT: Respond with ONLY "DATABASE_QUERY" or "GENERAL_CONVERSATION" (no other text). If unsure, respond: "CLARIFICATION_NEEDED".

CLASSIFICATION:
"""
        # Use the LLM to classify
        llm_service = MistralLLMService()
        classification_response = await llm_service.chat([
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": message}
        ])
        # Parse the response
        classification = classification_response.strip().upper()
        if "CLARIFICATION_NEEDED" in classification:
            return False
        is_database_query = "DATABASE_QUERY" in classification
        
        return is_database_query
        
    except Exception as e:
        logging.error(f"[DynamicClassifier] Error in LLM-based classification: {e}\n{traceback.format_exc()}")
        # Conservative fallback - if LLM fails, check if we have schema matches
        try:
            schema_results = await vector_search_service.search_schemas(business_id, message, top_k=3)
            fallback_result = schema_results and len(schema_results) > 0
            return fallback_result
        except Exception as fallback_error:
            logging.error(f"[DynamicClassifier] Fallback classification also failed: {fallback_error}\n{traceback.format_exc()}")
            return False 