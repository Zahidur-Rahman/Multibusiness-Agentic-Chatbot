from langgraph.graph import StateGraph
from backend.app.services.mistral_llm_service import MistralLLMService
from backend.app.services.vector_search import FaissVectorSearchService
from backend.app.mcp.mcp_client import MCPClient
from backend.app.utils.query_classifier import is_database_related_query_dynamic
from backend.app.utils.chat_helpers import build_sql_prompt, clean_sql_from_llm, format_db_result, build_system_prompt
from backend.app.models.conversation import ChatMessage
from backend.app.models.chat_graph_state import ChatGraphState
import logging
import time
import string
import json
import re
logger = logging.getLogger(__name__)

# Instantiate your services (adjust script_path as needed)
llm_service = MistralLLMService()
vector_search_service = FaissVectorSearchService()
mcp_client = MCPClient(script_path="backend/app/mcp/server_enhanced.py")

def ensure_chat_messages(messages):
    return [msg if isinstance(msg, ChatMessage) else ChatMessage.model_validate(msg) for msg in messages]

# --- Node Functions ---

async def classify_message(context: ChatGraphState) -> ChatGraphState:
    context.conversation_history = ensure_chat_messages(context.conversation_history or [])
    context.is_db_query = await is_database_related_query_dynamic(
        context.message,
        context.business_id,
        vector_search_service,
        [msg.model_dump() for msg in context.conversation_history]
    )
    return context

async def router_node(context: ChatGraphState) -> ChatGraphState:
    # Decide next node based on classification
    if context.is_db_query:
        context.next = "VectorSearch"
    else:
        context.next = "LLMChat"
    return context

async def vector_search_node(context: ChatGraphState) -> ChatGraphState:
    context.schema_context = await vector_search_service.search_schemas(
        context.business_id, context.message, top_k=5
    )
    return context

async def llm_chat_node(context: ChatGraphState) -> ChatGraphState:
    context.conversation_history = ensure_chat_messages(context.conversation_history or [])
    # FINAL CHECK: Remove any trailing 'user' messages before building LLM input
    while context.conversation_history and (
        getattr(context.conversation_history[-1], 'role', None) == 'user' or
        (isinstance(context.conversation_history[-1], dict) and context.conversation_history[-1].get('role') == 'user')
    ):
        context.conversation_history.pop()
    context.system_prompt = build_system_prompt(
        context, context.conversation_history, context.schema_context or []
    )
    messages = [
        {"role": "system", "content": context.system_prompt},
    ] + [msg.model_dump() for msg in context.conversation_history]
    # Always add the new user message as the last message
    messages.append({"role": "user", "content": context.message})
    # ABSOLUTE GUARANTEE: If the last two messages are both 'user', remove the second-to-last one
    if len(messages) > 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "user":
        messages.pop(-2)
    # Assertion: no two consecutive messages have the same role
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            raise ValueError(f"Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}")
    # Truncate to last valid system/user/assistant turn if error persists
    if len(messages) > 3:
        messages = [messages[0]] + messages[-2:]
    response = await llm_service.chat(messages)
    context.response = response
    return context

# --- New Node: Generate SQL Only (no execution) ---
async def generate_sql_node(context: ChatGraphState) -> ChatGraphState:
    context.conversation_history = ensure_chat_messages(context.conversation_history or [])
    # Remove any trailing 'user' messages before building LLM input
    while context.conversation_history and (
        getattr(context.conversation_history[-1], 'role', None) == 'user' or
        (isinstance(context.conversation_history[-1], dict) and context.conversation_history[-1].get('role') == 'user')
    ):
        context.conversation_history.pop()
    context.sql_prompt = build_sql_prompt(
        context, context.conversation_history, context.schema_context or []
    )
    messages = [
        {"role": "system", "content": context.sql_prompt},
    ] + [msg.model_dump() for msg in context.conversation_history]
    messages.append({"role": "user", "content": context.message})
    if len(messages) > 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "user":
        messages.pop(-2)
    # Assertion: no two consecutive messages have the same role
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            raise ValueError(f"Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}")
    if len(messages) > 3:
        messages = [messages[0]] + messages[-2:]
    sql_response = await llm_service.chat(messages)
    sql_query = clean_sql_from_llm(sql_response)
    context.sql = sql_query
    return context

# --- New Node: Execute SQL via MCP ---
async def execute_sql_node(context: ChatGraphState) -> ChatGraphState:
    import logging
    logger = logging.getLogger(__name__)
    sql = getattr(context, 'sql', None)
    business_id = getattr(context, 'business_id', None)
    user_id = getattr(context, 'user_id', None)
    logger.info(f"[SQL_EXECUTION] Sending to MCP: user_id={user_id}, business_id={business_id}, sql={sql}")
    start_time = time.time()
    try:
        mcp_result = await mcp_client.execute_query(sql, business_id)
        logger.info(f"DEBUG: Executing SQL: {sql}")
        logger.info(f"DEBUG: Raw DB result: {mcp_result}")
        # --- FIX: Parse MCP result if it's a JSON string in content[0]['text'] ---
        rows = []
        error_message = None
        # Check for error in MCP result (in 'error' field or in 'content')
        if isinstance(mcp_result, dict):
            # Check for error in 'error' field
            if mcp_result.get('error'):
                error_message = mcp_result['error']
            # Check for error in 'content'
            if 'content' in mcp_result and isinstance(mcp_result['content'], list):
                for item in mcp_result['content']:
                    if 'text' in item and 'error' in item['text'].lower():
                        error_message = item['text']
                        break
            # Try to parse rows if present
            if (
                'content' in mcp_result
                and isinstance(mcp_result['content'], list)
                and len(mcp_result['content']) > 0
                and 'text' in mcp_result['content'][0]
            ):
                try:
                    parsed = json.loads(mcp_result['content'][0]['text'])
                    rows = parsed.get('results', [])
                    logger.info(f"DEBUG: Parsed rows from MCP result: {rows}")
                except Exception as e:
                    logger.error(f"Error parsing MCP result: {e}")
                    rows = []
        elif isinstance(mcp_result, list):
            rows = mcp_result
        elapsed = time.time() - start_time
        # --- ERROR HANDLING ---
        if error_message:
            logger.error(f"[SQL_EXECUTION] MCP ERROR for business_id={business_id}: {error_message}")
            # User-friendly error for foreign key constraint
            if 'foreign key constraint' in error_message.lower():
                context.response = (
                    "Unable to delete the customer because there are related records (such as orders) that reference this customer. "
                    "Please delete those records first or contact support."
                )
            else:
                context.response = (
                    "Sorry, I couldn't complete your request due to a database error: "
                    f"{error_message}"
                )
            # Clear pause/confirmation fields to break the loop
            context.pause_reason = None
            context.pause_message = None
            context.confirm = None
            context.resume_from_pause = None
            return context
        # Determine query type
        is_select = sql.strip().lower().startswith("select")
        is_update = sql.strip().lower().startswith("update")
        is_delete = sql.strip().lower().startswith("delete")
        if is_select:
            if not rows:
                context.response = (
                    "I couldn't find any matching records for your request. "
                    "Would you like to try a different name or provide more details?"
                )
            else:
                # Format the results for the user
                result_strings = []
                for row in rows:
                    result_strings.append(
                        ", ".join(f"{k}: {v}" for k, v in row.items())
                    )
                context.response = "Result: " + "; ".join(result_strings)
        elif is_update:
            context.response = (
                "I've updated the information as you requested! "
                "Is there anything else you'd like to change or check?"
            )
            # Clear pause/confirmation fields after success
            context.pause_reason = None
            context.pause_message = None
            context.confirm = None
            context.resume_from_pause = None
        elif is_delete:
            context.response = (
                "The record has been deleted. Let me know if you need help with anything else!"
            )
            # Clear pause/confirmation fields after success
            context.pause_reason = None
            context.pause_message = None
            context.confirm = None
            context.resume_from_pause = None
        else:
            context.response = "The operation was completed."
    except Exception as e:
        logger.error(f"[SQL_EXECUTION] Exception: {e}")
        context.response = f"Sorry, there was an error executing your request: {e}"
    return context

# --- Dependency Resolver logic after SQL generation (for delete confirmation) ---
async def db_tool_with_dependency_check(context: ChatGraphState) -> dict:
    import logging
    logger = logging.getLogger(__name__)
    sql = context.sql or ""
    # --- VALIDATE SQL FOR UPDATE/DELETE ---
    # Simple regex for valid UPDATE/DELETE with table and WHERE clause
    update_pattern = re.compile(r"^update\s+\w+\s+set\s+.+where\s+.+", re.IGNORECASE)
    delete_pattern = re.compile(r"^delete\s+from\s+\w+\s+where\s+.+", re.IGNORECASE)
    if ("update" in sql.lower() and not update_pattern.match(sql)) or ("delete" in sql.lower() and not delete_pattern.match(sql)):
        logger.error("[DBToolWithDepCheck] Invalid or ambiguous UPDATE/DELETE SQL detected. Replying conversationally.")
        context.response = "I'm not sure what you want to update or delete. Could you clarify your request?"
        context.next = "Respond"
        return context.model_dump()
    # Check for DELETE and require confirmation BEFORE executing
    if "delete" in sql.lower() and not getattr(context, "confirm", False):
        logger.error("[DBToolWithDepCheck] Pausing for delete confirmation")
        context.response = "You are about to delete data. Please confirm by replying 'confirm delete'."
        context.pause_reason = "confirm_delete"
        context.pause_message = context.response
        context.next = "PauseNode"
        return context.model_dump() # Return a dict for easy logging
    # Check for UPDATE and require confirmation BEFORE executing
    if "update" in sql.lower() and not getattr(context, "confirm", False):
        logger.error("[DBToolWithDepCheck] Pausing for update confirmation")
        pause_result = {
            "pause_reason": "confirm_update",
            "pause_message": "You are about to update data. Please confirm by replying 'confirm update'.",
            "sql": sql,
            "schema_context": getattr(context, 'schema_context', None),
            "next": "PauseNode" # Ensure next is set for the main graph
        }
        logger.info(f"DEBUG: DBToolWithDepCheck returning pause_result: {pause_result}")
        return pause_result
    # Only execute the SQL if not a DELETE/UPDATE or if already confirmed
    logger.error("[DBToolWithDepCheck] No confirmation needed, proceeding to ExecuteSQL")
    context.next = "ExecuteSQL"
    return context.model_dump() # Return a dict for easy logging

async def response_node(context: ChatGraphState) -> dict:
    import logging
    logger = logging.getLogger(__name__)
    output = {
        'message': getattr(context, 'message', None),
        'business_id': getattr(context, 'business_id', None),
        'user_id': getattr(context, 'user_id', None),
        'conversation_history': getattr(context, 'conversation_history', None),
        'schema_context': getattr(context, 'schema_context', None),
        'is_db_query': getattr(context, 'is_db_query', None),
        'sql_prompt': getattr(context, 'sql_prompt', None),
        'system_prompt': getattr(context, 'system_prompt', None),
        'sql': getattr(context, 'sql', None),
        'db_result': getattr(context, 'db_result', None),
        'response': getattr(context, 'response', None),
        'next': getattr(context, 'next', None),
        # --- PAUSE FIELDS ---
        'pause_reason': getattr(context, 'pause_reason', None),
        'pause_message': getattr(context, 'pause_message', None),
        'confirm': getattr(context, 'confirm', None),
        'resume_from_pause': getattr(context, 'resume_from_pause', None),
    }
    logger.info(f"DEBUG: response_node output: {output}")
    return output

async def pause_node(context: ChatGraphState) -> ChatGraphState:
    # Set the correct pause message based on the reason
    if getattr(context, "pause_reason", None) == "confirm_update":
        context.response = "You are about to update data. Please confirm by replying 'confirm update'."
        context.pause_message = context.response
    elif getattr(context, "pause_reason", None) == "confirm_delete":
        context.response = "You are about to delete data. Please confirm by replying 'confirm delete'."
        context.pause_message = context.response
    else:
        # Fallback in case pause_reason is missing
        context.response = "Confirmation required for this action. Please confirm to proceed."
        context.pause_message = context.response
    return context

# --- New Node: Resume or Classify ---
async def resume_or_classify_node(context: ChatGraphState) -> ChatGraphState:
    import logging
    logger = logging.getLogger(__name__)
    confirm_triggers = {"confirm", "yes", "confirm update", "confirm delete", "yes, update", "yes, delete", "update confirmed", "delete confirmed"}
    user_message = getattr(context, 'message', '').strip().lower()
    # Remove punctuation for more robust matching
    user_message_clean = user_message.translate(str.maketrans('', '', string.punctuation))
    logger.debug(f"[ResumeOrClassify] user_message: '{user_message}', cleaned: '{user_message_clean}', pause_reason: '{getattr(context, 'pause_reason', None)}', resume_from_pause: '{getattr(context, 'resume_from_pause', False)}'")
    # If we are in a pause state and the user confirms, treat as confirmation
    if not getattr(context, 'resume_from_pause', False) and getattr(context, 'pause_reason', None) and any(trigger in user_message_clean for trigger in confirm_triggers):
        logger.error(f"[ResumeOrClassify] Confirmation '{user_message}' received for pause_reason '{context.pause_reason}'. Resuming to ExecuteSQL.")
        context.resume_from_pause = True
        context.confirm = True
        context.next = "ExecuteSQL"
        return context
    if getattr(context, "resume_from_pause", False):
        context.next = "ExecuteSQL"
    else:
        context.next = "Classify"
    return context

# --- Build the LangGraph Workflow ---
builder = StateGraph(ChatGraphState)
builder.add_node("ResumeOrClassify", resume_or_classify_node)
builder.add_node("Classify", classify_message)
builder.add_node("Router", router_node)
builder.add_node("VectorSearch", vector_search_node)
builder.add_node("LLMChat", llm_chat_node)
builder.add_node("GenerateSQL", generate_sql_node)
builder.add_node("DBToolWithDepCheck", db_tool_with_dependency_check)
builder.add_node("ExecuteSQL", execute_sql_node)
builder.add_node("Respond", response_node)
builder.add_node("PauseNode", pause_node)

# Edges
builder.add_conditional_edges(
    "ResumeOrClassify",
    lambda x: x.next,
    {"Classify": "Classify", "ExecuteSQL": "ExecuteSQL"}
)
builder.add_edge("Classify", "Router")
builder.add_conditional_edges(
    "Router",
    lambda x: x.next,  # Use the 'next' attribute from router_node's return value
    {"LLMChat": "LLMChat", "VectorSearch": "VectorSearch"}
)
builder.add_edge("VectorSearch", "GenerateSQL")
builder.add_edge("GenerateSQL", "DBToolWithDepCheck")
builder.add_conditional_edges(
    "DBToolWithDepCheck",
    lambda x: getattr(x, "next", None),
    {"PauseNode": "PauseNode", "ExecuteSQL": "ExecuteSQL"}
)
builder.add_edge("ExecuteSQL", "Respond")
builder.add_edge("PauseNode", "Respond")

builder.set_entry_point("ResumeOrClassify")

chat_graph = builder.compile()

# --- Usage Example (in FastAPI endpoint) ---
# from .chat_graph import chat_graph
# result = await chat_graph.arun(context)
# return {"response": result['response'], ...}

# --- TODOs ---
# - Implement or import is_database_related_query_dynamic
# - Implement prompt-building and SQL-cleaning logic
# - Format DB results for user
# - Integrate conversation memory/history as needed
