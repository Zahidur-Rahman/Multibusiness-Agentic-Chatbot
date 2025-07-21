import asyncio
import json
import subprocess
import sys
import threading
import queue
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("mcp_client")

class MCPClient:
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.process = None
        self._message_id = 0
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._communication_thread = None
        self._initialized = False
        self._start_process()
        self._start_communication_thread()
        self._initialize_server()

    def _start_process(self):
        self.process = subprocess.Popen(
            [sys.executable, self.script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered for Windows
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith('win') else 0
        )

    def _initialize_server(self):
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "fastapi-client", "version": "1.0.0"}
                }
            }
            init_str = json.dumps(init_message) + "\n"
            self.process.stdin.write(init_str)
            self.process.stdin.flush()
            response_line = self.process.stdout.readline()
            if response_line:
                try:
                    response = json.loads(response_line.strip())
                    if response.get("id") == 1 and "result" in response:
                        logger.info("✅ MCP server initialized successfully")
                        self._initialized = True
                        initialized_notification = {
                            "jsonrpc": "2.0",
                            "method": "notifications/initialized"
                        }
                        notif_str = json.dumps(initialized_notification) + "\n"
                        self.process.stdin.write(notif_str)
                        self.process.stdin.flush()
                    else:
                        logger.error(f"Initialization failed: {response}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse initialization response: {e}")
            else:
                logger.error("No initialization response received")
        except Exception as e:
            logger.error(f"Initialization error: {e}")

    def _start_communication_thread(self):
        def communication_worker():
            try:
                while True:
                    try:
                        request = self._request_queue.get(timeout=1)
                        if request is None:
                            break
                        request_str = json.dumps(request) + "\n"
                        self.process.stdin.write(request_str)
                        self.process.stdin.flush()
                        response_line = self.process.stdout.readline()
                        if response_line:
                            try:
                                response = json.loads(response_line.strip())
                                self._response_queue.put(response)
                            except json.JSONDecodeError:
                                self._response_queue.put({"error": "Invalid JSON response"})
                        else:
                            self._response_queue.put({"error": "No response from server"})
                    except queue.Empty:
                        if self.process.poll() is not None:
                            logger.error("MCP process died")
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Communication thread error: {e}")
                        self._response_queue.put({"error": str(e)})
            except Exception as e:
                logger.error(f"Communication worker fatal error: {e}")
        self._communication_thread = threading.Thread(target=communication_worker, daemon=True)
        self._communication_thread.start()

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        if not self._initialized:
            return {"error": "MCP server not initialized"}
        try:
            self._message_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._message_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            self._request_queue.put(message)
            timeout = 60  # Increased from 30 to 60 seconds
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self._response_queue.get(timeout=1)
                    if response.get("id") == self._message_id:
                        if "result" in response:
                            return response["result"]
                        elif "error" in response:
                            return {"error": response["error"]}
                except queue.Empty:
                    continue
            logger.error(f"Request timeout after {timeout} seconds for tool {tool_name}")
            return {"error": f"Request timeout after {timeout} seconds"}
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    async def execute_query(self, sql_query: str, business_id: str, limit: int = 1000) -> Any:
        return await self.call_tool("execute_query", {"query": sql_query, "business_id": business_id, "limit": limit})

    async def aclose(self):
        if self._communication_thread:
            self._request_queue.put(None)
            self._communication_thread.join(timeout=5)
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill() 