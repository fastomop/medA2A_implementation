"""
Official MCP (Model Context Protocol) integration for medical-a2a framework.
Uses the official MCP Python SDK for proper protocol compliance.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from pathlib import Path
import os
from contextlib import asynccontextmanager

# Import official MCP client components
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Provide a warning if MCP is not available
    logging.warning("MCP SDK not installed. Install with: pip install mcp")

logger = logging.getLogger(__name__)

class Transport(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    
    @classmethod
    def from_url(cls, url: str) -> 'Transport':
        """Determine transport type from URL."""
        if url.startswith("stdio://"):
            return cls.STDIO
        elif url.startswith("http://") or url.startswith("https://"):
            return cls.SSE
        else:
            raise ValueError(f"Unknown transport type for URL: {url}")

@dataclass
class MCPServer:
    """MCP server configuration."""
    name: str
    description: str
    medical_speciality: Optional[str] = None
    stdio_params: Optional[StdioServerParameters] = None # Parameters for stdio transport
    url: Optional[str] = None # For SSE or other URL-based transports

@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    medical_context: Optional[Dict[str, Any]] = None

class MCPClient:
    """Client for a single MCP server using official MCP SDK."""
    
    def __init__(self, server: MCPServer):
        if not MCP_AVAILABLE:
            raise ImportError("MCP SDK is not installed. Install with: pip install mcp")
            
        self.server = server
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, MCPTool] = {}
        self._context_manager = None
        self._read_stream = None
        self._write_stream = None
        
    async def connect(self):
        """Connect to the MCP server."""
        if self.server.stdio_params:
            logger.debug(f"Attempting to connect to STDIO server with params: {self.server.stdio_params}")
            
            # --- TEMPORARY DEBUGGING CODE ---
            # Replace stdio_client with direct subprocess.Popen for detailed output
            try:
                process = await asyncio.create_subprocess_exec(
                    *self.server.stdio_params.command,
                    *self.server.stdio_params.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.server.stdio_params.cwd,
                    env=self.server.stdio_params.env
                )
                logger.debug(f"OMCP Server subprocess started with PID: {process.pid}")

                # Read stdout and stderr for a short period to capture startup errors
                stdout_data = b''
                stderr_data = b''
                try:
                    stdout_data, stderr_data = await asyncio.wait_for(process.communicate(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning("OMCP Server did not terminate within 5 seconds during debug capture.")
                    # If it didn't terminate, it's likely still running, so don't kill it yet

                if stdout_data:
                    logger.debug(f"OMCP Server STDOUT:\n{stdout_data.decode().strip()}")
                if stderr_data:
                    logger.error(f"OMCP Server STDERR:\n{stderr_data.decode().strip()}")

                if process.returncode is not None and process.returncode != 0:
                    raise RuntimeError(f"OMCP Server exited with code {process.returncode}")

                # Re-create the streams for ClientSession if the process is still alive
                if process.returncode is None:
                    self._read_stream = process.stdout
                    self._write_stream = process.stdin
                else:
                    raise RuntimeError("OMCP Server terminated prematurely during debug capture.")

            except Exception as e:
                logger.error(f"Error launching OMCP Server subprocess: {e}")
                raise
            # --- END TEMPORARY DEBUGGING CODE ---

        elif self.server.url:
            # SSE transport would be implemented here when available in official SDK
            raise NotImplementedError("SSE transport not yet implemented with official SDK")
        else:
            raise ValueError("MCPServer must have either stdio_params or url configured.")
            
        # Create session
        self.session = ClientSession(self._read_stream, self._write_stream)
        await self.session.__aenter__()
        
        # Initialize the session
        try:
            result = await self.session.initialize()
            logger.info(f"Initialized connection to {self.server.name}: {result}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.server.name}: {e}")
            await self.disconnect()
            raise
            
        # Discover tools
        await self._discover_tools()
        
    async def _discover_tools(self):
        """Discover available tools."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        # Use official list_tools method
        result = await self.session.list_tools()
        
        self.tools.clear()
        for tool in result.tools:
            mcp_tool = MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else None,
                medical_context=getattr(tool, 'medicalContext', None)
            )
            self.tools[tool.name] = mcp_tool
            
        logger.info(f"Discovered {len(self.tools)} tools from {self.server.name}")
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool using the official SDK."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in {self.server.name}")
            
        # Use official call_tool method
        result = await self.session.call_tool(name=tool_name, arguments=arguments)
        
        # Extract the content from the result
        if result.content:
            # Handle different content types
            if len(result.content) == 1 and result.content[0].type == "text":
                return result.content[0].text
            else:
                # Return all content for complex results
                return [{"type": c.type, "data": getattr(c, c.type)} for c in result.content]
        
        return result
        
    async def disconnect(self):
        """Disconnect from the server."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing session: {e}")
                
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing transport: {e}")
                
        self.session = None
        self._context_manager = None
        self._read_stream = None
        self._write_stream = None
        self.tools.clear()

class MCPManager:
    """Manages multiple MCP server connections using official SDK."""
    
    def __init__(self, initial_servers: List[MCPServer]):
        self.servers: Dict[str, MCPServer] = {s.name: s for s in initial_servers}
        self.clients: Dict[str, MCPClient] = {}
        self.available_tools: Dict[str, Dict] = {}
        
    async def discover_servers(self, discovery_endpoint: Optional[str] = None):
        """Discover MCP servers from a registry."""
        if discovery_endpoint:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(discovery_endpoint)
                servers_data = response.json()
                
                for server_data in servers_data:
                    server = MCPServer(**server_data)
                    self.servers[server.name] = server
                    
        # Connect to all servers
        await self._connect_all()
        
    async def register_server(self, server: MCPServer):
        """Register and connect to a new MCP server."""
        self.servers[server.name] = server
        await self._connect_server(server)
        
    async def _connect_all(self):
        """Connect to all registered servers."""
        tasks = [self._connect_server(server) for server in self.servers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for server, result in zip(self.servers.values(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to {server.name}: {result}")
        
    async def _connect_server(self, server: MCPServer):
        """Connect to a specific server."""
        try:
            # Disconnect existing client if any
            if server.name in self.clients:
                await self.clients[server.name].disconnect()
                
            # Create and connect new client
            client = MCPClient(server)
            await client.connect()
            self.clients[server.name] = client
            
            # Update available tools
            self._update_tool_registry(server.name, client)
            
        except Exception as e:
            logger.error(f"Failed to connect to {server.name}: {e}")
            raise
            
    def _update_tool_registry(self, server_name: str, client: MCPClient):
        """Update the tool registry with tools from a client."""
        # Remove old tools from this server
        self.available_tools = {
            k: v for k, v in self.available_tools.items() 
            if v["server"] != server_name
        }
        
        # Add new tools
        for tool_name, tool in client.tools.items():
            tool_id = f"{server_name}:{tool_name}"
            self.available_tools[tool_id] = {
                "server": server_name,
                "name": tool_name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "medical_context": tool.medical_context
            }
            
    async def get_available_tools(self) -> str:
        """Get formatted list of available tools."""
        import json
        tools_list = []
        for tool_id, tool_info in self.available_tools.items():
            params_str = json.dumps(tool_info.get("parameters", {}), indent=2)
            tools_list.append(
                f"- {tool_id}: {tool_info['description']}\n"
                f"  Parameters: {params_str}"
            )
            
        return "\n".join(tools_list)
        
    async def call_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool using format 'server_name:tool_name'."""
        parts = tool_id.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid tool_id format: '{tool_id}'. Use 'server_name:tool_name'")
            
        server_name, tool_name = parts
        
        if server_name not in self.clients:
            raise ValueError(f"Server '{server_name}' not connected")
            
        result = await self.clients[server_name].call_tool(tool_name, parameters)
        
        # Wrap result in expected format for backward compatibility
        return {"result": result}
        
    async def shutdown(self):
        """Disconnect all servers."""
        tasks = [client.disconnect() for client in self.clients.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.clients.clear()
        self.available_tools.clear()

class MCPDiscoveryMixin:
    """
    Mixin to add official MCP SDK support to medical agents.
    Compatible with existing framework while using official MCP client.
    """
    
    def __init__(self, *args, mcp_servers: Optional[List[MCPServer]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_manager = MCPManager(mcp_servers or [])
        self._mcp_initialized = False
        
    async def _ensure_mcp_initialized(self):
        """Ensure MCP connections are initialized (lazy initialization)."""
        if not self._mcp_initialized:
            try:
                await self.mcp_manager._connect_all()
                self._mcp_initialized = True
                logger.info("MCP connections initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MCP connections: {e}")
                raise
            
    async def discover_mcp_servers(self, discovery_endpoint: Optional[str] = None):
        """Discover available MCP servers."""
        await self.mcp_manager.discover_servers(discovery_endpoint)
        self._mcp_initialized = True
        
    async def register_mcp_server(self, server: MCPServer):
        """Register a new MCP server."""
        await self.mcp_manager.register_server(server)
        
    async def get_mcp_tools(self):
        """Get available MCP tools (ensures initialization)."""
        await self._ensure_mcp_initialized()
        return await self.mcp_manager.get_available_tools()
        
    async def call_mcp_tool(self, tool_id: str, parameters: Dict[str, Any]):
        """Call an MCP tool (ensures initialization)."""
        await self._ensure_mcp_initialized()
        return await self.mcp_manager.call_tool(tool_id, parameters)

# For backward compatibility, if MCP SDK is not installed, fall back to the original implementation
if not MCP_AVAILABLE:
    logger.warning("MCP SDK not available. Using the original implementation as fallback.")
    # This would import the original implementation
    # from .mcp import MCPServer, MCPTool, MCPClient, MCPManager, MCPDiscoveryMixin