"""
MCP client for connecting to OMCP server.
"""

import asyncio
import logging
import os
import signal
import psutil
import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of different error types."""
    TRANSIENT = "transient"  # Temporary issues, retry same connection
    CONNECTION_BROKEN = "connection_broken"  # Reset database connection only
    LOCK_CONFLICT = "lock_conflict"  # Kill process and restart completely
    FATAL = "fatal"  # Mark connection as permanently failed
    UNKNOWN = "unknown"  # Unclassified errors

class ConnectionHealth:
    """Track connection health and failure patterns."""
    
    def __init__(self):
        self.consecutive_failures = 0
        self.total_failures = 0
        self.last_failure_time = 0
        self.last_success_time = time.time()
        self.is_circuit_open = False
        self.circuit_open_time = 0
        self.backoff_delay = 1.0
        self.max_consecutive_failures = 3
        self.circuit_open_duration = 30.0  # 30 seconds
        
    def record_success(self):
        """Record a successful operation."""
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.is_circuit_open = False
        self.backoff_delay = 1.0
        
    def record_failure(self, error_type: ErrorType):
        """Record a failure and update circuit breaker state."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        # Open circuit breaker for severe failures
        if (error_type in [ErrorType.LOCK_CONFLICT, ErrorType.FATAL] or 
            self.consecutive_failures >= self.max_consecutive_failures):
            self.is_circuit_open = True
            self.circuit_open_time = time.time()
            
        # Exponential backoff
        self.backoff_delay = min(self.backoff_delay * 2, 60.0)
        
    def should_attempt_connection(self) -> bool:
        """Check if we should attempt a connection based on circuit breaker state."""
        if not self.is_circuit_open:
            return True
            
        # Check if circuit breaker should be reset
        if time.time() - self.circuit_open_time > self.circuit_open_duration:
            logger.info("Circuit breaker reset, allowing connection attempt")
            self.is_circuit_open = False
            return True
            
        return False
        
    def get_backoff_delay(self) -> float:
        """Get current backoff delay."""
        return self.backoff_delay if self.consecutive_failures > 0 else 0

@dataclass
class MCPConfig:
    """Simple MCP configuration."""
    command: str
    args: list
    env: dict
    cwd: str

class SimpleMCPClient:
    """A bulletproof MCP client with connection cleanup and restart logic."""
    
    _instances = {}  # Class variable to track instances
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.session = None
        self._server_process = None
        self._connection_id = f"{config.command}:{':'.join(config.args)}"
        self._health = ConnectionHealth()
        self._last_health_check = 0
        self._health_check_interval = 30.0  # 30 seconds
        # Use AsyncExitStack for proper MCP resource management
        self._exit_stack = AsyncExitStack()
        self._is_connected = False
        
    @classmethod
    def get_or_create_instance(cls, config: MCPConfig):
        """Get existing instance or create new one (singleton pattern)."""
        connection_id = f"{config.command}:{':'.join(config.args)}"
        
        if connection_id in cls._instances:
            instance = cls._instances[connection_id]
            # Check if existing connection is still alive
            if instance.is_connected():
                logger.info(f"Reusing existing MCP connection: {connection_id}")
                return instance
            else:
                logger.info(f"Existing connection dead, cleaning up: {connection_id}")
                instance.cleanup_dead_connection()
                del cls._instances[connection_id]
        
        # Create new instance
        instance = cls(config)
        cls._instances[connection_id] = instance
        logger.info(f"Created new MCP connection: {connection_id}")
        return instance
        
    def is_connected(self):
        """Check if the connection is still alive."""
        return self._is_connected and self.session is not None and not self._health.is_circuit_open
        
    def _classify_error(self, error_message: str, is_mcp_error: bool = False) -> ErrorType:
        """Classify error type based on error message and context."""
        if not error_message:
            return ErrorType.UNKNOWN
            
        error_lower = error_message.lower()
        
        # Lock conflict detection
        if any(keyword in error_lower for keyword in [
            'lock', 'locked', 'conflicting lock', 'database is locked',
            'could not set lock', 'database lock'
        ]):
            return ErrorType.LOCK_CONFLICT
            
        # Connection broken detection
        if is_mcp_error or any(keyword in error_lower for keyword in [
            'connection closed', 'connection lost', 'connection refused',
            'broken pipe', 'transport error', 'session closed'
        ]):
            return ErrorType.CONNECTION_BROKEN
            
        # Fatal errors
        if any(keyword in error_lower for keyword in [
            'no such file', 'permission denied', 'access denied',
            'file not found', 'invalid database'
        ]):
            return ErrorType.FATAL
            
        # Transient errors
        if any(keyword in error_lower for keyword in [
            'timeout', 'temporary', 'busy', 'retry',
            'deadlock', 'rollback', 'constraint'
        ]):
            return ErrorType.TRANSIENT
            
        return ErrorType.UNKNOWN
        
    def _is_database_error(self, result) -> bool:
        """Check if MCP result contains a database error."""
        if not hasattr(result, 'isError') or not result.isError:
            return False
            
        if hasattr(result, 'content') and result.content:
            for content in result.content:
                if hasattr(content, 'text') and content.text:
                    error_type = self._classify_error(content.text)
                    return error_type != ErrorType.UNKNOWN
                    
        return False
        
    async def _health_check(self) -> bool:
        """Perform a lightweight health check on the database connection."""
        current_time = time.time()
        
        # Skip if we did a health check recently
        if current_time - self._last_health_check < self._health_check_interval:
            return True
            
        try:
            # Try a simple query to test the connection
            result = await self.session.call_tool(
                name="Select_Query", 
                arguments={"query": "SELECT 1 as health_check LIMIT 1"}
            )
            
            self._last_health_check = current_time
            
            # Check if health check returned an error
            if self._is_database_error(result):
                logger.warning("Health check failed - database error in result")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed with MCP exception: {e}")
            self._last_health_check = current_time
            return False
        
    def cleanup_dead_connection(self):
        """Clean up a dead connection."""
        try:
            if self._server_process and self._server_process.poll() is None:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
        except:
            pass
        
        # Reset connection state
        self.session = None
        self._server_process = None
        self._is_connected = False
        # Don't touch _exit_stack here as it's handled by async methods
        
    async def connect(self):
        """Connect to MCP server using official MCP pattern with AsyncExitStack."""
        # Only do aggressive cleanup if this is a recovery attempt
        if hasattr(self, '_is_recovery_attempt') and self._is_recovery_attempt:
            await self._cleanup_existing_processes()
        else:
            # Light cleanup - just check for obvious conflicts
            await self._light_cleanup()
        
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
            cwd=self.config.cwd
        )
        
        try:
            # Use the official MCP pattern with AsyncExitStack
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            read_stream, write_stream = stdio_transport
            
            # Create session using the official pattern
            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # Initialize the session
            await self.session.initialize()
            self._is_connected = True
            
            # Try to get server process reference for monitoring
            # Note: This is implementation detail, may not always be available
            if hasattr(self._exit_stack._exit_callbacks[-2][1], '_process'):
                self._server_process = self._exit_stack._exit_callbacks[-2][1]._process
            
            logger.info("Connected to MCP server using official pattern")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            # Clean up on failure
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
            raise
        
    async def _cleanup_existing_processes(self):
        """Enhanced cleanup to break database locks and kill all OMCP processes."""
        try:
            logger.info("🧹 Starting enhanced OMCP process cleanup...")
            
            # Step 1: Find ALL processes that might be holding DuckDB locks
            processes_to_kill = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if ('omcp' in cmdline.lower() or 
                        'synthea.duckdb' in cmdline or
                        ('python' in proc.info['name'] and 'main_robust.py' in cmdline) or
                        ('python' in proc.info['name'] and 'omcp_wrapper.py' in cmdline)):
                        
                        processes_to_kill.append((proc.info['pid'], cmdline))
                        logger.info(f"🎯 Found OMCP process to kill: PID {proc.info['pid']}")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Step 2: Kill all found processes (graceful then force)
            for pid, cmdline in processes_to_kill:
                try:
                    proc = psutil.Process(pid)
                    logger.info(f"⚡ Terminating OMCP process: {pid}")
                    proc.terminate()
                    
                    try:
                        proc.wait(timeout=3)
                        logger.info(f"✅ Process {pid} terminated gracefully")
                    except psutil.TimeoutExpired:
                        logger.warning(f"🔨 Force killing stubborn process: {pid}")
                        proc.kill()
                        proc.wait(timeout=2)
                        logger.info(f"💀 Process {pid} force killed")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"Process {pid} already gone or access denied: {e}")
                except Exception as e:
                    logger.error(f"Failed to kill process {pid}: {e}")
            
            # Step 3: Additional cleanup using system commands as fallback
            try:
                import subprocess
                
                # Kill any remaining main_robust.py processes
                result = subprocess.run(
                    ["pkill", "-f", "main_robust.py"], 
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    logger.info("🧹 Cleaned up remaining main_robust.py processes via pkill")
                
                # Kill any remaining omcp_wrapper.py processes  
                result = subprocess.run(
                    ["pkill", "-f", "omcp_wrapper.py"], 
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    logger.info("🧹 Cleaned up remaining omcp_wrapper.py processes via pkill")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                logger.debug("pkill not available or no additional processes found")
            
            # Step 4: Wait for database locks to clear
            logger.info("⏳ Waiting for database locks to clear...")
            await asyncio.sleep(3)
            
            # Step 5: Verify cleanup success
            remaining_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if ('omcp' in cmdline.lower() or 'main_robust.py' in cmdline):
                        remaining_procs.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if remaining_procs:
                logger.warning(f"⚠️ Some OMCP processes still running: {remaining_procs}")
            else:
                logger.info("✅ All OMCP processes successfully cleaned up")
                
        except Exception as e:
            logger.error(f"Error during enhanced process cleanup: {e}")
    
    async def _light_cleanup(self):
        """Light cleanup for normal startup - only handle obvious conflicts."""
        try:
            # Just check if our specific connection already exists and clean it up
            connection_id = f"{self.config.command}:{':'.join(self.config.args)}"
            if connection_id in SimpleMCPClient._instances:
                existing = SimpleMCPClient._instances[connection_id]
                if not existing.is_connected():
                    existing.cleanup_dead_connection()
                    del SimpleMCPClient._instances[connection_id]
                    logger.info("Cleaned up dead MCP connection")
        except Exception as e:
            logger.debug(f"Light cleanup error (non-critical): {e}")
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool with bulletproof connection management and authentic result preservation."""
        
        # DEBUG: Confirm our bulletproof code is running
        logger.info(f"🛡️ BULLETPROOF MCP CLIENT: calling {tool_name} with bulletproof logic")
        
        # Check circuit breaker
        if not self._health.should_attempt_connection():
            raise Exception(f"Connection circuit breaker open - too many recent failures")
            
        # Apply backoff delay if needed
        backoff_delay = self._health.get_backoff_delay()
        if backoff_delay > 0:
            logger.info(f"Applying backoff delay: {backoff_delay}s")
            await asyncio.sleep(backoff_delay)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure connection for this attempt
                if not self.session or not self.is_connected():
                    # Mark as recovery attempt if this isn't the first try
                    if attempt > 0:
                        self._is_recovery_attempt = True
                    await self.connect()
                    self._is_recovery_attempt = False
                
                # Perform health check on first attempt
                if attempt == 0 and not await self._health_check():
                    logger.warning("Health check failed, attempting connection reset")
                    await self._cleanup_and_restart()
                    continue
                
                # Make the actual MCP call
                result = await self.session.call_tool(name=tool_name, arguments=arguments)
                
                # Check if authentic MCP result contains database errors
                if self._is_database_error(result):
                    error_text = self._extract_error_text(result)
                    error_type = self._classify_error(error_text)
                    
                    logger.error(f"Database error in MCP result ({error_type.value}): {error_text[:200]}...")
                    self._health.record_failure(error_type)
                    
                    # If this is our last attempt, return the authentic MCP error
                    if attempt == max_retries - 1:
                        logger.error("All retry attempts exhausted, returning authentic MCP error")
                        return self._extract_result_content(result)
                    
                    # Attempt recovery based on error type
                    if error_type == ErrorType.LOCK_CONFLICT:
                        logger.info(f"Lock conflict detected, force restarting server (attempt {attempt + 1})")
                        await self._force_restart_server()
                    elif error_type in [ErrorType.CONNECTION_BROKEN, ErrorType.UNKNOWN]:
                        logger.info(f"Connection issue detected, cleaning up and restarting (attempt {attempt + 1})")
                        await self._cleanup_and_restart()
                    elif error_type == ErrorType.FATAL:
                        logger.error("Fatal error detected, no recovery possible")
                        return self._extract_result_content(result)
                    else:  # TRANSIENT
                        logger.info(f"Transient error detected, brief wait before retry (attempt {attempt + 1})")
                        await asyncio.sleep(2 ** attempt)
                        
                    continue
                    
                # Success - record it and return authentic MCP result
                self._health.record_success()
                logger.debug(f"Tool call successful on attempt {attempt + 1}")
                return self._extract_result_content(result)
                
            except Exception as e:
                error_type = self._classify_error(str(e), is_mcp_error=True)
                logger.error(f"MCP session error on attempt {attempt + 1} ({error_type.value}): {e}")
                
                self._health.record_failure(error_type)
                
                # If this is our last attempt, re-raise the original exception
                if attempt == max_retries - 1:
                    logger.error("All MCP session retry attempts exhausted")
                    raise
                
                # Always do cleanup and restart for MCP session failures
                logger.info(f"MCP session failure, cleaning up and restarting (attempt {attempt + 1})")
                await self._cleanup_and_restart()
                await asyncio.sleep(2 ** attempt)
                
        # This should never be reached
        raise Exception("Unexpected end of retry loop")
        
    def _extract_result_content(self, result) -> Any:
        """Extract content from MCP result in authentic form."""
        if hasattr(result, 'content') and result.content and len(result.content) > 0:
            return result.content[0].text
        return result
        
    def _extract_error_text(self, result) -> str:
        """Extract error text from MCP result for classification."""
        if not (hasattr(result, 'content') and result.content):
            return ""
            
        error_text = ""
        for content in result.content:
            if hasattr(content, 'text'):
                error_text += content.text
        return error_text
        
    async def _cleanup_and_restart(self):
        """Clean up connection and restart for any type of failure."""
        logger.info("Cleaning up connection and restarting...")
        try:
            # Disconnect current session
            await self.disconnect()
            
            # Clean up any lingering processes (force aggressive cleanup)
            self._is_recovery_attempt = True
            await self._cleanup_existing_processes()
            self._is_recovery_attempt = False
            
            # Brief pause for cleanup to complete
            await asyncio.sleep(2)
            
            logger.info("Cleanup and restart completed")
        except Exception as e:
            logger.error(f"Error during cleanup and restart: {e}")
            # Don't raise - we want to attempt reconnection anyway
            
    async def _force_restart_server(self):
        """Force restart the entire MCP server process."""
        logger.info("Force restarting MCP server due to lock conflict...")
        try:
            # Clean up current connection
            await self.disconnect()
            
            # Clean up any lingering processes
            await self._cleanup_existing_processes()
            
            # Wait a bit longer for file locks to be released
            await asyncio.sleep(5)
            
            # Reconnect
            await self.connect()
            logger.info("MCP server restart successful")
        except Exception as e:
            logger.error(f"Failed to restart MCP server: {e}")
            raise
            
        
    async def disconnect(self):
        """Disconnect from the server using official MCP cleanup."""
        try:
            # Use AsyncExitStack's proper cleanup
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()  # Create new stack for future connections
            self._is_connected = False
            self.session = None
            
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during MCP disconnect: {e}")
            # Reset anyway
            self._exit_stack = AsyncExitStack()
            self._is_connected = False
            self.session = None
            
        # Clean up server process
        if self._server_process and self._server_process.poll() is None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except:
                try:
                    self._server_process.kill()
                except:
                    pass
                    
        self.session = None
        self._stdio_transport = None
        self._server_process = None
        self._last_health_check = 0
        logger.info("Disconnected from MCP server")
        
    @classmethod
    async def cleanup_all_instances(cls):
        """Clean up all MCP client instances."""
        for instance in cls._instances.values():
            await instance.disconnect()
        cls._instances.clear()

async def get_omop_mcp_client() -> SimpleMCPClient:
    """Get or create the global OMOP MCP client with singleton pattern."""
    from ..config import get_config
    config = get_config()
    
    # Create MCP configuration for OMOP database
    omcp_server_path = config.get_omcp_server_path()
    uv_executable = config.get_uv_executable() or "/opt/homebrew/bin/uv"
    
    mcp_config = MCPConfig(
        command=uv_executable,
        args=["run", "python", "src/omcp/main_robust.py"],
        env={
            'DB_TYPE': 'duckdb',
            'DB_PATH': str(omcp_server_path / "synthetic_data" / "synthea.duckdb"),
            'CDM_SCHEMA': 'base',
            'VOCAB_SCHEMA': 'base',
            'OMCP_SERVER_PATH': str(omcp_server_path),
            'UV_EXECUTABLE': uv_executable
        },
        cwd=str(omcp_server_path)
    )
    
    client = SimpleMCPClient.get_or_create_instance(mcp_config)
    if not client.is_connected():
        await client.connect()
    
    return client


async def call_omop_tool(tool_name: str, parameters: dict) -> dict:
    """
    Simple function to call OMOP MCP tools directly.
    
    Args:
        tool_name: Name of the tool (e.g., "Select_Query", "Get_Information_Schema")
        parameters: Parameters for the tool
        
    Returns:
        Dictionary with result or error
    """
    try:
        client = await get_omop_mcp_client()
        result = await client.call_tool(tool_name, parameters)
        return {"result": result}
    except Exception as e:
        logger.error(f"OMOP tool call failed: {e}")
        return {"error": str(e)}