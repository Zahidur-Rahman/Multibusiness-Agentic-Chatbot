def build_system_prompt(context, conversation_history, schema_context):
    schema_text = ""
    if schema_context:
        for schema in schema_context:
            schema_text += f"\nTable: {schema.get('table_name', 'Unknown')}\n"
            schema_text += f"Description: {schema.get('schema_description', 'No description')}\n"
            schema_text += "Columns:\n"
            for col in schema.get('columns', []):
                schema_text += f"  - {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')} ({col.get('description', 'No description')})\n"
            if schema.get('relationships'):
                schema_text += "Relationships:\n"
                for rel in schema.get('relationships', []):
                    schema_text += f"  - {rel.get('from_table', 'Unknown')}.{rel.get('from_column', 'Unknown')} -> {rel.get('to_table', 'Unknown')}.{rel.get('to_column', 'Unknown')}\n"
            schema_text += "\n"
    else:
        schema_text = "No relevant schema context found."

    system_prompt = (
        "You are a highly capable, friendly, and proactive AI assistant for business users. "
        "You can answer general questions, help with business data, and guide users through complex workflows.\n\n"
        f"AVAILABLE DATA (for reference):\n{schema_text}\n\n"
        "CONVERSATION HISTORY (for context):\n"
    )
    for msg in conversation_history:
        role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
        content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
        system_prompt += f"{role.upper()}: {content}\n"
    system_prompt += (
        f"\nCURRENT USER REQUEST: {context.message}\n\n"
        "INSTRUCTIONS:\n"
        "- Be friendly, clear, and concise.\n"
        "- If the user's request is ambiguous, ask a clarifying question before proceeding.\n"
        "- If you encounter an error or missing data, apologize and suggest next steps.\n"
        "- Use the schema and conversation history to provide accurate, context-aware answers.\n"
        "- If the user asks for something outside your scope, politely explain your limitations.\n"
        "- Always confirm with the user before making any changes to business data.\n"
        "- If the user seems confused, offer to clarify or provide examples.\n"
        "- If the user asks for a summary, provide a concise overview.\n"
        "- If the user asks for a list, present it in a clear, readable format.\n"
        "- If the user asks for help, offer step-by-step guidance.\n"
        "- If the user asks for sensitive or restricted data, remind them of privacy and security policies.\n"
        "- If you are unsure, ask the user for clarification rather than guessing.\n"
        "- Always maintain a professional and helpful tone.\n"
    )
    return system_prompt

def build_sql_prompt(context, conversation_history, schema_context):
    schema_text = ""
    if schema_context:
        for schema in schema_context:
            schema_text += f"\nTable: {schema.get('table_name', 'Unknown')}\n"
            schema_text += f"Description: {schema.get('schema_description', 'No description')}\n"
            schema_text += "Columns:\n"
            for col in schema.get('columns', []):
                schema_text += f"  - {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')} ({col.get('description', 'No description')})\n"
            if schema.get('relationships'):
                schema_text += "Relationships:\n"
                for rel in schema.get('relationships', []):
                    schema_text += f"  - {rel.get('from_table', 'Unknown')}.{rel.get('from_column', 'Unknown')} -> {rel.get('to_table', 'Unknown')}.{rel.get('to_column', 'Unknown')}\n"
            schema_text += "\n"
    else:
        schema_text = "No relevant schema context found."

    sql_prompt = (
        "You are an expert SQL assistant for a PostgreSQL business database. "
        "Your job is to convert natural language requests into safe, correct SQL queries.\n\n"
        f"AVAILABLE DATABASE SCHEMAS (for reference):\n{schema_text}\n"
        "CONVERSATION HISTORY (for context):\n"
    )
    for msg in conversation_history:
        role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
        content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
        sql_prompt += f"{role.upper()}: {content}\n"
    sql_prompt += (
        f"\nCURRENT USER REQUEST: {context.message}\n"
        "INSTRUCTIONS:\n"
        "- Analyze the user's request and use the schema above to generate a single, safe, syntactically correct SQL query.\n"
        "- Use SELECT, INSERT, UPDATE, or DELETE as appropriate, but never DROP, TRUNCATE, or ALTER.\n"
        "- Always use WHERE clauses for UPDATE/DELETE to avoid affecting all records.\n"
        "- For ambiguous requests, ask the user for clarification before generating SQL.\n"
        "- For INSERT, use realistic example values.\n"
        "- For UPDATE, only update fields explicitly mentioned by the user.\n"
        "- For DELETE, confirm with the user before proceeding.\n"
        "- Use JOINs if the user asks for related data across tables.\n"
        "- Use ILIKE and wildcards for partial, case-insensitive text matches.\n"
        "- Handle NULLs and missing values gracefully.\n"
        "- If the request cannot be handled with a valid SQL query, reply: 'Operation not allowed.'\n"
        "- If no relevant tables are found, reply: 'No relevant tables found in schema.'\n"
        "- Output ONLY the SQL query, no explanations, no markdown, no code blocks, no prefixes.\n"
        "- Preserve all SQL clauses including WHERE, ORDER BY, GROUP BY, HAVING, etc.\n"
        "- Example outputs:\n"
        "  - SELECT: SELECT * FROM customers WHERE active = true;\n"
        "  - INSERT: INSERT INTO customers (name, email, phone) VALUES ('John Doe', 'john@example.com', '1234567890');\n"
        "  - UPDATE: UPDATE customers SET phone = '0987654321' WHERE id = 1;\n"
        "  - DELETE: DELETE FROM customers WHERE id = 1;\n"
    )
    return sql_prompt

def clean_sql_from_llm(sql_response):
    # Remove markdown/code block and common prefixes, as in your main.py
    sql_query = sql_response.strip()
    if sql_query.startswith('```'):
        lines = sql_query.split('\n')
        if len(lines) > 1:
            sql_lines = []
            for line in lines[1:]:
                if line.strip() == '```':
                    break
                sql_lines.append(line)
            sql_query = '\n'.join(sql_lines).strip()
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
    sql_query = sql_query.rstrip(';').strip()
    return sql_query

def format_db_result(mcp_result):
    # User-friendly formatting for DB results
    if isinstance(mcp_result, dict):
        if mcp_result.get('success') and mcp_result.get('results'):
            rows = mcp_result['results']
            if not rows:
                return "No results found."
            headers = list(rows[0].keys())
            if len(rows) == 1:
                # Single row: return as a summary sentence
                row = rows[0]
                summary = ", ".join(f"{h}: {row.get(h, '')}" for h in headers)
                return f"Result: {summary}"
            # Multi-row: Build a table
            lines = [" | ".join(headers)]
            lines.append("-|-".join(["---"] * len(headers)))
            for row in rows:
                lines.append(" | ".join(str(row.get(h, '')) for h in headers))
            return "\n".join(lines)
        elif mcp_result.get('success'):
            return "Query executed successfully, but no results found."
        elif mcp_result.get('error'):
            return f"Error: {mcp_result['error']}"
        else:
            # Fallback: pretty-print JSON
            import json
            return json.dumps(mcp_result, indent=2)
    # Try to parse as JSON string
    try:
        import json
        parsed = json.loads(mcp_result)
        return format_db_result(parsed)
    except Exception:
        pass
    # Fallback: return as-is
    return str(mcp_result) 