import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from collections import defaultdict
import threading
import time

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv, find_dotenv # Import find_dotenv for robust path finding
import traceback
import pymongo

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

# Configure logging
# Set level to DEBUG temporarily to see all verbose logs
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce log spam
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("multi-business-mcp-server")

# --- DOTENV LOADING ---
# Try to find the .env file explicitly. This is more robust.
# It searches the current directory and its parents.
dotenv_path = find_dotenv()
if dotenv_path:
    logger.info(f"Found .env file at: {dotenv_path}")
    load_dotenv(dotenv_path)
    logger.info("Loaded .env file successfully.")
else:
    logger.warning("No .env file found by find_dotenv(). Ensure it's in the root or a parent directory.")
    # Fallback to default load_dotenv() behavior if find_dotenv fails, though less reliable
    load_dotenv()
# --- END DOTENV LOADING ---


class DummyNotificationOptions:
    tools_changed = None

class ConnectionPoolManager:
    """Manages database connection pools for multiple businesses"""
    
    def __init__(self):
        self.pools = {}
        self.business_configs = {}
        self.lock = threading.Lock()
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", "300")) # 5 minutes
        self.last_health_check = defaultdict(int)
        self.last_health_status = defaultdict(lambda: None)  # Track last health status
        self._load_business_configs()
    
    def _load_business_configs(self):
        """Load business configurations from MongoDB"""
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        mongodb_db = os.getenv("MONGODB_DB", "chatbot_config")
        try:
            client = pymongo.MongoClient(mongodb_uri)
            db = client[mongodb_db]
            collection = db["business_configs"]
            # Only load active businesses
            docs = collection.find({"status": "active"})
            count = 0
            for doc in docs:
                business_id = doc["business_id"]
                db_config = doc["db_config"]
                config = {
                    "host": db_config["host"],
                    "database": db_config["database"],
                    "user": db_config["user"],
                    "password": db_config["password"],
                    "port": db_config.get("port", 5432),
                    "minconn": 2,
                    "maxconn": 10,
                    "connect_timeout": 30,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                }
                self.business_configs[business_id] = config
                count += 1
            logger.info(f"Loaded {count} business configs from MongoDB.")
            logger.info(f"Loaded business configs: {list(self.business_configs.keys())}")
        except Exception as e:
            logger.error(f"Failed to load business configs from MongoDB: {e}")
            raise
    
    def _create_connection_pool(self, business_id: str) -> SimpleConnectionPool:
        """Create a new connection pool for a business"""
        config = self.business_configs[business_id].copy() # Use a copy to pop values

        # Extract pool-specific settings
        minconn = config.pop("minconn", 2)
        maxconn = config.pop("maxconn", 10)
        
        try:
            pool = SimpleConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                **config # Pass remaining config directly to psycopg2
            )
            logger.info(f"Created connection pool for business: {business_id} (min={minconn}, max={maxconn})")
            return pool
        except Exception as e:
            logger.error(f"Failed to create pool for business {business_id}: {e}", exc_info=True)
            raise
    
    def get_pool(self, business_id: str) -> SimpleConnectionPool:
        """Get or create a connection pool for a business"""
        with self.lock:
            if business_id not in self.pools:
                if business_id not in self.business_configs:
                    raise ValueError(f"Business '{business_id}' not configured. Available businesses: {list(self.business_configs.keys())}")
                
                self.pools[business_id] = self._create_connection_pool(business_id)
            
            return self.pools[business_id]
    
    def get_connection(self, business_id: str):
        """Get a connection from the pool with health check"""
        current_time = time.time()
        
        # Only perform health check if not already failing
        if current_time - self.last_health_check[business_id] > self.health_check_interval:
            try:
                self._health_check(business_id)
            except Exception as e:
                logger.error(f"Health check failed for business {business_id}: {e}")
                self.last_health_check[business_id] = current_time
                self.last_health_check[business_id] = current_time
        
        pool = self.get_pool(business_id)
        try:
            connection = pool.getconn()
            # Basic check if the connection is usable
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return connection
        except Exception as e:
            logger.warning(f"Failed to get usable connection for business {business_id}: {e}. Attempting to recreate pool.", exc_info=True)
            self._recreate_pool(business_id)
            # After recreation, try getting a connection one more time
            try:
                connection = pool.getconn()
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                return connection
            except Exception as e_retry:
                logger.error(f"Failed to get connection even after pool recreation for business {business_id}: {e_retry}", exc_info=True)
                raise
    
    def return_connection(self, business_id: str, connection, error=False):
        """Return a connection to the pool"""
        if business_id in self.pools:
            try:
                if error or connection.closed:
                    # Don't return bad/closed connections to the pool
                    connection.close()
                    logger.warning(f"Discarded bad or closed connection for business {business_id}.")
                else:
                    self.pools[business_id].putconn(connection)
            except Exception as e:
                logger.error(f"Error returning connection for business {business_id}: {e}", exc_info=True)
    
    def _health_check(self, business_id: str):
        """Perform health check on a business connection"""
        conn = None
        try:
            # FIX: Get connection directly from pool to avoid recursion
            pool = self.get_pool(business_id)
            conn = pool.getconn()
            if self.last_health_status[business_id] != "pass":
                    logger.info(f"Health check passed for business: {business_id}")
                    self.last_health_status[business_id] = "pass"
        except Exception as e:
            if self.last_health_status[business_id] != "fail":
                logger.error(f"Health check failed for business {business_id}: {e}")
            self.last_health_status[business_id] = "fail"
            self._recreate_pool(business_id)
        finally:
            if conn:
                self.return_connection(business_id, conn) # Return the connection regardless
    
    def _recreate_pool(self, business_id: str):
        """Recreate connection pool for a business"""
        with self.lock:
            logger.info(f"Attempting to recreate connection pool for business: {business_id}")
            if business_id in self.pools:
                try:
                    self.pools[business_id].closeall()
                    logger.info(f"Closed old connection pool for business {business_id}.")
                except Exception as e:
                    logger.error(f"Error closing old pool for business {business_id}: {e}", exc_info=True)
                del self.pools[business_id] # Remove the old pool reference
            
            try:
                self.pools[business_id] = self._create_connection_pool(business_id)
                logger.info(f"Successfully recreated connection pool for business: {business_id}")
            except Exception as e:
                logger.error(f"Failed to recreate connection pool for business {business_id}: {e}", exc_info=True)
                # If recreation fails, remove it from pools to indicate it's truly down
                if business_id in self.pools:
                    del self.pools[business_id]
    
    def close_all_pools(self):
        """Close all connection pools"""
        with self.lock:
            for business_id, pool in list(self.pools.items()): # Iterate over a copy
                try:
                    pool.closeall()
                    logger.info(f"Closed connection pool for business: {business_id}")
                except Exception as e:
                    logger.error(f"Error closing pool for business {business_id}: {e}", exc_info=True)
            self.pools.clear()
    
    def list_businesses(self) -> List[str]:
        """List all configured businesses"""
        return list(self.business_configs.keys())
    
    def get_business_info(self, business_id: str) -> Dict[str, Any]:
        """Get business configuration info (without sensitive data)"""
        if business_id not in self.business_configs:
            return {}
        
        config = self.business_configs[business_id]
        # Return a copy without password
        return {
            "business_id": business_id,
            "host": config["host"],
            "database": config["database"],
            "port": config["port"],
            "user": config["user"],
            "pool_status": "active" if business_id in self.pools else "inactive",
            "minconn": config.get("minconn"),
            "maxconn": config.get("maxconn"),
            "connect_timeout": config.get("connect_timeout"),
        }

class MultiBusinessPostgreSQLServer:
    """Multi-Business PostgreSQL MCP Server implementation"""
    
    def __init__(self):
        self.server = Server("multi-business-postgres-server")
        self.pool_manager = ConnectionPoolManager()
        self.query_timeout = int(os.getenv("QUERY_TIMEOUT", "30"))  # seconds
        self._setup_handlers()
    
    def get_db_connection(self, business_id: str):
        """Get database connection for a specific business"""
        try:
            return self.pool_manager.get_connection(business_id)
        except Exception as e:
            logger.error(f"Database connection error for business {business_id}: {e}", exc_info=True)
            raise
    
    def _setup_handlers(self):
        """Setup MCP request handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="execute_query",
                    description="Execute a SQL query and return results. Only SELECT queries are allowed for security.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL query to execute (SELECT statements only)"
                            },
                            "business_id": {
                                "type": "string",
                                "description": "Business ID to execute query against (REQUIRED)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Optional limit for result rows (default: 1000, max: 10000)",
                                "minimum": 1,
                                "maximum": 10000
                            }
                        },
                        "required": ["query", "business_id"]
                    }
                ),
                Tool(
                    name="get_table_schema",
                    description="Get detailed schema information for a specific table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to get schema for"
                            },
                            "business_id": {
                                "type": "string",
                                "description": "Business ID (REQUIRED)"
                            }
                        },
                        "required": ["table_name", "business_id"]
                    }
                ),
                Tool(
                    name="list_tables",
                    description="List all tables in the public schema",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "business_id": {
                                "type": "string",
                                "description": "Business ID (REQUIRED)"
                            }
                        },
                        "required": ["business_id"]
                    }
                ),
                Tool(
                    name="list_businesses",
                    description="List all available businesses with their connection status",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_business_info",
                    description="Get detailed information about a specific business",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "business_id": {
                                "type": "string",
                                "description": "Business ID (REQUIRED)"
                            }
                        },
                        "required": ["business_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "list_businesses":
                    result = await self._list_businesses()
                elif name == "get_business_info":
                    business_id = arguments.get("business_id")
                    if not business_id:
                        return [TextContent(type="text", text=json.dumps({
                            "error": "business_id is required for this operation"
                        }))]
                    result = await self._get_business_info(business_id)
                else:
                    # All other tools require business_id
                    business_id = arguments.get("business_id")
                    if not business_id:
                        return [TextContent(type="text", text=json.dumps({
                            "error": "business_id is required for this operation",
                            "tool": name,
                            "available_businesses": self.pool_manager.list_businesses()
                        }))]
                    
                    if name == "execute_query":
                        query = arguments.get("query", "")
                        limit = arguments.get("limit", 1000)
                        result = await self._execute_query(query, business_id, limit)
                    elif name == "get_table_schema":
                        result = await self._get_table_schema(arguments.get("table_name", ""), business_id)
                    elif name == "list_tables":
                        result = await self._list_tables(business_id)
                    else:
                        return [TextContent(type="text", text=json.dumps({
                            "error": f"Unknown tool: {name}"
                        }))]
                
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                logger.error(f"Tool call error for {name} with arguments {arguments}: {e}", exc_info=True)
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e),
                    "tool": name,
                    "arguments": arguments,
                    "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else None
                }))]
    
    async def _execute_query(self, query: str, business_id: str, limit: int = 1000) -> str:
        """Execute a SQL query with security checks"""
        conn = None
        error_occurred = False
        
        try:
            # Input validation
            query = query.strip()
            if not query:
                return json.dumps({"error": "Empty query provided"})
            
            # Validate limit
            limit = max(1, min(limit, 10000))
            
            # Security check - allow SELECT, INSERT, UPDATE, DELETE statements
            query_upper = query.upper().strip()
            allowed_operations = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
            operation_found = any(query_upper.startswith(op) for op in allowed_operations)
            
            if not operation_found:
                return json.dumps({
                    "error": "Only SELECT, INSERT, UPDATE, and DELETE queries are allowed for security reasons",
                    "provided_query": query[:100] + "..." if len(query) > 100 else query
                })
            
            # Check for dangerous patterns (only block destructive operations)
            # Use word boundaries to avoid false positives with SQL functions
            dangerous_patterns = [
                r'\bDROP\b', r'\bTRUNCATE\b', r'\bALTER\b', r'\bCREATE\b',
                r'\bGRANT\b', r'\bREVOKE\b', r'\bCOPY\b', r'\bCALL\b', r'\bEXECUTE\b', r'\bIMPORT\b', r'\bEXPORT\b'
            ]
            
            import re
            query_upper = query.upper()
            for pattern in dangerous_patterns:
                if re.search(pattern, query_upper):
                    return json.dumps({
                        "error": f"Query contains forbidden keyword: {pattern.replace(r'\b', '')}",
                        "provided_query": query[:100] + "..." if len(query) > 100 else query
                    })
            
            # Get business-specific connection
            conn = self.get_db_connection(business_id)
            
            # Handle different operation types
            is_select = query_upper.startswith('SELECT')
            
            # Add automatic LIMIT only for SELECT queries
            if is_select and 'LIMIT' not in query_upper and 'FETCH FIRST' not in query_upper and 'ROWNUM' not in query_upper:
                query += f" LIMIT {limit}"
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Set query timeout
                cursor.execute(f"SET statement_timeout = {self.query_timeout * 1000}")
                
                start_time = time.time()
                cursor.execute(query)
                execution_time = time.time() - start_time
                
                if is_select:
                    # Handle SELECT queries - fetch results
                    results = cursor.fetchall()
                    
                    # Convert to JSON-serializable format
                    json_results = []
                    for row in results:
                        json_row = {}
                        for key, value in row.items():
                            # Handle special types that aren't JSON serializable
                            if hasattr(value, 'isoformat'):  # datetime objects
                                json_row[key] = value.isoformat()
                            elif isinstance(value, (bytes, memoryview)):
                                json_row[key] = str(value)
                            elif hasattr(value, '__class__') and value.__class__.__name__ == 'Decimal':
                                # Handle Decimal objects from PostgreSQL NUMERIC/DECIMAL columns
                                json_row[key] = float(value)
                            elif isinstance(value, (int, float, str, bool, type(None))):
                                # Basic JSON-serializable types
                                json_row[key] = value
                            else:
                                # Fallback for any other non-serializable types
                                json_row[key] = str(value)
                        json_results.append(json_row)
                    
                    return json.dumps({
                        "success": True,
                        "business_id": business_id,
                        "query": query,
                        "row_count": len(json_results),
                        "execution_time_seconds": round(execution_time, 3),
                        "results": json_results
                    }, indent=2)
                else:
                    # Handle INSERT, UPDATE, DELETE queries - get affected rows
                    affected_rows = cursor.rowcount
                    conn.commit()  # Commit the transaction for write operations
                    
                    return json.dumps({
                        "success": True,
                        "business_id": business_id,
                        "query": query,
                        "affected_rows": affected_rows,
                        "execution_time_seconds": round(execution_time, 3),
                        "operation_type": "INSERT" if query_upper.startswith('INSERT') else "UPDATE" if query_upper.startswith('UPDATE') else "DELETE"
                    }, indent=2)
                    
        except psycopg2.Error as e:
            error_occurred = True
            logger.error(f"Database error executing query for business {business_id}: {e}", exc_info=True)
            return json.dumps({
                "error": f"Database error: {str(e)}",
                "business_id": business_id,
                "query": query[:100] + "..." if len(query) > 100 else query
            })
        except Exception as e:
            error_occurred = True
            logger.error(f"Unexpected error executing query for business {business_id}: {e}", exc_info=True)
            return json.dumps({
                "error": f"Unexpected error: {str(e)}",
                "business_id": business_id,
                "query": query[:100] + "..." if len(query) > 100 else query
            })
        finally:
            if conn:
                self.pool_manager.return_connection(business_id, conn, error=error_occurred)
    
    async def _get_table_schema(self, table_name: str, business_id: str) -> str:
        """Get detailed schema information for a table"""
        conn = None
        error_occurred = False
        
        try:
            if not table_name:
                return json.dumps({"error": "Table name is required"})
            
            conn = self.get_db_connection(business_id)
            with conn.cursor() as cursor:
                # Get column information
                cursor.execute("""
                    SELECT 
                        column_name,
                        data_type,
                        character_maximum_length,
                        is_nullable,
                        column_default,
                        ordinal_position
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position;
                """, (table_name,))
                
                columns = cursor.fetchall()
                if not columns:
                    return json.dumps({
                        "error": f"Table '{table_name}' not found in business '{business_id}' or is not in public schema",
                        "business_id": business_id,
                        "table_name_attempted": table_name
                    })
                
                # Get primary key information
                cursor.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                        AND tc.table_name = kcu.table_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_name = %s AND tc.table_schema = 'public';
                """, (table_name,))
                
                primary_keys = [row[0] for row in cursor.fetchall()]
                
                schema_info = []
                # `columns` is a list of tuples. psycopg2's default cursor returns tuples.
                # If you want dict-like access, use RealDictCursor for this too or convert manually.
                # Assuming `columns` are in the order of `SELECT` statement:
                for col in columns:
                    col_info = {
                        "name": col[0],
                        "type": col[1],
                        "max_length": col[2],
                        "nullable": col[3] == "YES",
                        "default": col[4],
                        "position": col[5],
                        "is_primary_key": col[0] in primary_keys
                    }
                    schema_info.append(col_info)
                
                return json.dumps({
                    "business_id": business_id,
                    "table_name": table_name,
                    "columns": schema_info,
                    "column_count": len(schema_info),
                    "primary_keys": primary_keys
                }, indent=2)
                    
        except Exception as e:
            error_occurred = True
            logger.error(f"Schema error for business {business_id}, table {table_name}: {e}", exc_info=True)
            return json.dumps({
                "error": f"Error getting schema: {str(e)}",
                "business_id": business_id,
                "table_name": table_name
            })
        finally:
            if conn:
                self.pool_manager.return_connection(business_id, conn, error=error_occurred)
    
    async def _list_tables(self, business_id: str) -> str:
        """List all tables in the public schema"""
        conn = None
        error_occurred = False
        
        try:
            conn = self.get_db_connection(business_id)
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        table_name,
                        table_type
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                
                tables = cursor.fetchall()
                table_list = [{"name": table[0], "type": table[1]} for table in tables]
                
                return json.dumps({
                    "business_id": business_id,
                    "tables": table_list,
                    "count": len(table_list)
                }, indent=2)
                    
        except Exception as e:
            error_occurred = True
            logger.error(f"List tables error for business {business_id}: {e}", exc_info=True)
            return json.dumps({
                "error": f"Error listing tables: {str(e)}",
                "business_id": business_id
            })
        finally:
            if conn:
                self.pool_manager.return_connection(business_id, conn, error=error_occurred)
    
    async def _list_businesses(self) -> str:
        """List all available businesses with their status"""
        try:
            businesses = self.pool_manager.list_businesses()
            business_info = []
            
            for business_id in businesses:
                info = self.pool_manager.get_business_info(business_id)
                business_info.append(info)
            
            return json.dumps({
                "businesses": business_info,
                "count": len(business_info)
            }, indent=2)
        except Exception as e:
            logger.error(f"List businesses error: {e}", exc_info=True)
            return json.dumps({"error": f"Error listing businesses: {str(e)}"})
    
    async def _get_business_info(self, business_id: str) -> str:
        """Get detailed information about a specific business"""
        try:
            info = self.pool_manager.get_business_info(business_id)
            if not info:
                return json.dumps({
                    "error": f"Business '{business_id}' not found or not configured",
                    "available_businesses": self.pool_manager.list_businesses()
                })
            
            return json.dumps(info, indent=2)
        except Exception as e:
            logger.error(f"Get business info error for {business_id}: {e}", exc_info=True)
            return json.dumps({"error": f"Error getting business info: {str(e)}"})

async def main():
    """Main function to run the MCP server"""
    postgres_server = None
    try:
        # Create server
        postgres_server = MultiBusinessPostgreSQLServer()
        
        # Check if any businesses are configured
        businesses = postgres_server.pool_manager.list_businesses()
        if not businesses:
            logger.error("No businesses configured. Please set BUSINESS_IDS and business-specific environment variables.")
            logger.error("Example: BUSINESS_IDS=biz1,biz2 BUSINESS_BIZ1_POSTGRES_HOST=localhost ...")
            sys.exit(1)
        
        logger.info(f"Configured businesses: {businesses}")
        
        # Test connections to all businesses
        logger.info("Attempting database connection tests for configured businesses...")
        for business_id in businesses:
            try:
                # The get_db_connection method already includes a basic health check and pool management
                conn = postgres_server.get_db_connection(business_id)
                postgres_server.pool_manager.return_connection(business_id, conn)
                logger.info(f"Database connection test successful for business: {business_id}")
            except Exception as e:
                logger.error(f"Database connection test failed for business {business_id}: {e}", exc_info=True)
                # Continue even if one fails, as other businesses might still be accessible.
                # However, for a production setup, you might want to exit if crucial connections fail.
        
        # Run the MCP server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
                logger.info("Starting Multi-Business PostgreSQL MCP server...")
                await postgres_server.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="multi-business-postgres-server",
                        server_version="1.0.0",
                        capabilities=postgres_server.server.get_capabilities(DummyNotificationOptions(), {}),
                    ),
                )
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"A critical server error occurred: {e}", exc_info=True)
        sys.exit(1) # Exit with an error code for critical failures
    finally:
        # Clean up connection pools
        if postgres_server:
            logger.info("Initiating connection pool closure.")
            postgres_server.pool_manager.close_all_pools()
            logger.info("Connection pools closed.")

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    logger.info("Checking OS compatibility...")
    if sys.platform.startswith('win'):
        logger.info("Windows detected, setting ProactorEventLoopPolicy")
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main asynchronous function
    asyncio.run(main())