"""
Official MCP Manager using Python SDK patterns with config-based server definitions.
Supports multiple MCP servers per agent with proper async context management.
"""
import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional, AsyncContextManager
from mcp import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

class MCPServerConfig:
    """Configuration for an MCP server."""
    
    def __init__(self, name: str, command: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
    
    def to_stdio_params(self) -> StdioServerParameters:
        """Convert to MCP StdioServerParameters."""
        params = {
            "command": self.command,
            "args": self.args,
            "env": self.env
        }
        if self.cwd:
            params["cwd"] = self.cwd
        return StdioServerParameters(**params)

class MCPClient:
    """Individual MCP client with official SDK patterns."""
    
    def __init__(self, config: MCPServerConfig, exit_stack: AsyncExitStack):
        self.config = config
        self.exit_stack = exit_stack
        self.session: Optional[ClientSession] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the MCP client with proper async context management."""
        if self._initialized:
            return
        
        try:
            logger.debug(f"Initializing MCP client for {self.config.name}")
            params = self.config.to_stdio_params()
            
            # Use official SDK pattern with AsyncExitStack
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(params)
            )
            read_stream, write_stream = stdio_transport
            
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # Initialize the session
            await self.session.initialize()
            
            self._initialized = True
            logger.info(f"MCP client {self.config.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client {self.config.name}: {e}")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on this MCP server with automatic recovery."""
        if not self._initialized:
            await self.initialize()
        
        if not self.session:
            raise RuntimeError(f"MCP session not initialized for {self.config.name}")
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                logger.error(f"Tool call attempt {attempt + 1} failed on {self.config.name}: {e}")
                
                # Check if this is a connection-related error that might benefit from restart
                if self._is_connection_error(e) and attempt < max_retries:
                    logger.warning(f"Connection error detected, attempting to restart MCP client {self.config.name}")
                    try:
                        await self._restart_client()
                        logger.info(f"Successfully restarted MCP client {self.config.name}")
                        continue  # Retry the call
                    except Exception as restart_error:
                        logger.error(f"Failed to restart MCP client {self.config.name}: {restart_error}")
                
                # If final attempt or non-connection error, re-raise
                if attempt == max_retries:
                    raise
    
    def _is_connection_error(self, error: Exception) -> bool:
        """Check if the error indicates a connection issue that might be recoverable."""
        error_msg = str(error).lower()
        connection_indicators = [
            "connection closed",
            "connection lost", 
            "broken pipe",
            "transport endpoint is not connected",
            "no such file or directory",  # Process died
            "connection refused",
            "timeout"
        ]
        return any(indicator in error_msg for indicator in connection_indicators)
    
    async def _restart_client(self) -> None:
        """Restart the MCP client session."""
        # Mark as uninitialized to force fresh start
        self._initialized = False
        self.session = None
        
        # Re-initialize with fresh connection
        await self.initialize()
    
    async def list_tools(self) -> List[Any]:
        """List available tools on this MCP server."""
        if not self._initialized:
            await self.initialize()
        
        if not self.session:
            raise RuntimeError(f"MCP session not initialized for {self.config.name}")
        
        try:
            result = await self.session.list_tools()
            return result.tools
        except Exception as e:
            logger.error(f"Failed to list tools on {self.config.name}: {e}")
            raise

class MCPManager:
    """
    Manager for multiple MCP servers using official SDK patterns.
    Supports config-based server definitions and per-agent server assignments.
    """
    
    def __init__(self, server_configs: Dict[str, Dict[str, Any]]):
        self.server_configs = {}
        self.clients: Dict[str, MCPClient] = {}
        self._exit_stack: Optional[AsyncExitStack] = None
        self._initialized = False
        
        # Parse server configurations (skip comment entries)
        for name, config in server_configs.items():
            # Skip comment entries
            if name.startswith('_') or not isinstance(config, dict) or 'command' not in config:
                continue
                
            self.server_configs[name] = MCPServerConfig(
                name=name,
                command=config['command'],
                args=config.get('args', []),
                env=config.get('env', {}),
                cwd=config.get('cwd')
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.cleanup()
    
    async def initialize(self) -> None:
        """Initialize all MCP clients."""
        if self._initialized:
            return
        
        self._exit_stack = AsyncExitStack()
        
        try:
            # Initialize all configured MCP clients
            for name, config in self.server_configs.items():
                client = MCPClient(config, self._exit_stack)
                self.clients[name] = client
                # Initialize immediately to catch connection issues early
                await client.initialize()
            
            self._initialized = True
            logger.info(f"MCPManager initialized with {len(self.clients)} servers")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCPManager: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self) -> None:
        """Clean up all MCP connections."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
                logger.info("MCPManager cleanup completed")
            except Exception as e:
                logger.error(f"Error during MCPManager cleanup: {e}")
            finally:
                self._exit_stack = None
                self._initialized = False
                self.clients.clear()
    
    def get_client(self, server_name: str) -> MCPClient:
        """Get a specific MCP client by server name."""
        if not self._initialized:
            raise RuntimeError("MCPManager not initialized")
        
        if server_name not in self.clients:
            raise ValueError(f"MCP server '{server_name}' not found. Available: {list(self.clients.keys())}")
        
        return self.clients[server_name]
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific MCP server."""
        client = self.get_client(server_name)
        return await client.call_tool(tool_name, arguments)
    
    async def list_tools(self, server_name: str) -> List[Any]:
        """List tools on a specific MCP server."""
        client = self.get_client(server_name)
        return await client.list_tools()
    
    async def list_all_tools(self) -> Dict[str, List[Any]]:
        """List tools on all MCP servers."""
        all_tools = {}
        for name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                all_tools[name] = tools
            except Exception as e:
                logger.error(f"Failed to list tools for {name}: {e}")
                all_tools[name] = []
        return all_tools
    
    def get_available_servers(self) -> List[str]:
        """Get list of available server names."""
        return list(self.clients.keys())

async def create_mcp_manager(server_configs: Dict[str, Dict[str, Any]]) -> MCPManager:
    """Factory function to create and initialize an MCPManager."""
    manager = MCPManager(server_configs)
    await manager.initialize()
    return manager