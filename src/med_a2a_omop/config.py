"""
Configuration management for the Medical A2A OMOP system.
Handles environment detection, path discovery, and cross-platform compatibility.
"""

import os
import sys
import shutil
import platform
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import json
from dotenv import load_dotenv
import logging

try:
    from mcp.client.stdio import StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class MedA2AConfig:
    """Centralized configuration management for Medical A2A OMOP system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration with automatic environment detection.
        
        Args:
            config_file: Optional path to custom config file
        """
        self.config_file = config_file or self._find_config_file()
        self.project_root = self._find_project_root()
        
        # Load configuration hierarchy:
        # 1. JSON config file (if exists)
        # 2. Environment variables  
        # 3. Auto-discovery fallbacks
        self.explicit_config = self._load_config_file()
        
        # Load environment variables
        load_dotenv(self.project_root / ".env")
        
        # Initialize configuration
        self._validate_environment()
        
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations."""
        # Check if config file path is specified via environment variable
        env_config_file = os.getenv('MEDA2A_CONFIG_FILE')
        if env_config_file and Path(env_config_file).exists():
            logger.info(f"Using config file from environment: {env_config_file}")
            return Path(env_config_file)
        
        # Store the original working directory from when the process started
        original_cwd = Path(os.environ.get('PWD', os.getcwd()))
        
        possible_locations = [
            Path.cwd() / ".medA2A.config.json",                    # Current working directory
            original_cwd / ".medA2A.config.json",                  # Original directory where command was run
            self._find_project_root() / ".medA2A.config.json",     # Project root
            Path.home() / ".config" / "medA2A" / "config.json",    # User config directory
            Path("/etc/medA2A/config.json"),                       # Linux system-wide
        ]
        
        for location in possible_locations:
            if location.exists():
                logger.info(f"Found config file: {location}")
                return location
        return None
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        
        # Fallback to current working directory
        return Path.cwd()
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from JSON file if it exists."""
        if not self.config_file or not Path(self.config_file).exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from: {self.config_file}")
                return config
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config file {self.config_file}: {e}")
            return {}
        
    def _get_config_value(self, key: str, env_var: str, default: Any = None, discovery_func = None) -> Any:
        """
        Get configuration value using priority hierarchy:
        1. JSON config file
        2. Environment variable
        3. Auto-discovery function
        4. Default value
        """
        # 1. Check JSON config file first - handle nested keys with dot notation
        if "." in key:
            # Handle nested keys like "services.ollama_timeout"
            keys = key.split(".")
            config_value = self.explicit_config
            for k in keys:
                if isinstance(config_value, dict) and k in config_value:
                    config_value = config_value[k]
                else:
                    config_value = None
                    break
            if config_value is not None:
                logger.debug(f"Using config file value for {key}: {config_value}")
                return config_value
        else:
            # Handle simple keys
            if key in self.explicit_config:
                value = self.explicit_config[key]
                logger.debug(f"Using config file value for {key}: {value}")
                return value
        
        # 2. Check environment variable
        env_value = os.getenv(env_var)
        if env_value:
            logger.debug(f"Using environment variable {env_var}: {env_value}")
            return env_value
        
        # 3. Try auto-discovery if function provided
        if discovery_func:
            discovered = discovery_func()
            if discovered:
                logger.debug(f"Auto-discovered {key}: {discovered}")
                return discovered
        
        # 4. Return default
        if default is not None:
            logger.debug(f"Using default value for {key}: {default}")
        return default
    
    def _validate_environment(self):
        """Validate that the environment is properly configured."""
        issues = []
        
        # Check Ollama installation
        if not self.is_ollama_available():
            issues.append("Ollama not available. Install from: https://ollama.ai/")
        
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    

    
    
    def get_omcp_server_path(self) -> Optional[Path]:
        """Get the path to the OMCP server installation."""
        def discover_omcp():
            # 1. First check for git submodule (automatic setup - no config needed!)
            submodule_path = self.project_root / "omcp_server"
            if submodule_path.exists() and (submodule_path / "src" / "omcp" / "main.py").exists():
                return submodule_path

            # 2. Try common locations
            possible_locations = [
                Path.home() / "omcp_server",
                Path.home() / "projects" / "omcp_server",
                Path.home() / "src" / "omcp_server",
                Path("/opt/omcp_server"),
                Path("/usr/local/omcp_server"),
            ]

            for location in possible_locations:
                if location.exists() and (location / "src" / "omcp" / "main.py").exists():
                    return location
            return None
        
        path_str = self._get_config_value(
            "paths.omcp_server_path",
            "OMCP_SERVER_PATH",
            None,
            discover_omcp
        )
        
        if path_str:
            return Path(path_str)
        return None

    def get_vocabulary_path(self) -> Optional[Path]:
        """Get the path to the OMOP vocabulary files."""
        def discover_vocabulary():
            # 1. Try common OMOP vocabulary locations
            possible_locations = [
                Path.home() / "Downloads" / "omop_vocab_current",
                Path.home() / "omop_vocabulary",
                Path.home() / "data" / "omop_vocabulary",
                Path("/opt/omop_vocabulary"),
                Path("/usr/local/omop_vocabulary"),
                # Check if vocabulary is bundled with OMCP server
                self.get_omcp_server_path() / "vocabulary" if self.get_omcp_server_path() else None,
            ]

            for location in possible_locations:
                if location and location.exists() and (location / "CONCEPT.csv").exists():
                    return location
            return None

        path_str = self._get_config_value(
            "paths.vocabulary_path",
            "OMOP_VOCABULARY_PATH",
            None,
            discover_vocabulary
        )

        if path_str:
            return Path(path_str)

        return None

    def get_uv_executable(self) -> Optional[str]:
        """Get the UV package manager executable path."""
        def discover_uv():
            # Check if uv is in PATH
            uv_path = shutil.which("uv")
            if uv_path:
                return uv_path
            
            # Check common installation locations
            possible_locations = [
                Path.home() / ".cargo" / "bin" / "uv",
                Path.home() / ".local" / "bin" / "uv",
                Path("/usr/local/bin/uv"),
                Path("/opt/homebrew/bin/uv"),  # macOS with Homebrew
            ]
            
            for location in possible_locations:
                if location.exists() and location.is_file():
                    return str(location)
            
            return None
        
        return self._get_config_value(
            "paths.uv_executable",
            "UV_EXECUTABLE", 
            None,
            discover_uv
        )
    
    def create_wrapper_script(self) -> Path:
        """Create a wrapper script for running the OMCP server."""
        omcp_path = self.get_omcp_server_path()
        if not omcp_path:
            raise ValueError("OMCP server path not configured")
        
        wrapper_content = f"""#!/bin/bash
# Auto-generated wrapper script for OMCP server
cd "{omcp_path}"
{self.get_uv_executable() or 'uv'} run --project "{omcp_path}" python src/omcp/main.py "$@"
"""
        
        wrapper_path = self.project_root / "omcp_wrapper.sh"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        # Make executable
        import stat
        st = os.stat(wrapper_path)
        os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)
        
        return wrapper_path

    def create_python_wrapper_script(self) -> Path:
        """Create a Python wrapper script for running the OMCP server via MCP."""
        omcp_path = self.get_omcp_server_path()
        if not omcp_path:
            raise ValueError("OMCP server path not configured")

        # Build environment variables for the wrapper
        synthea_db = omcp_path / "synthetic_data" / "synthea.duckdb"
        if synthea_db.exists():
            db_type = "duckdb"
            db_path = str(synthea_db)
        else:
            db_type = os.getenv("DB_TYPE", "duckdb")
            db_path = os.getenv("DB_PATH", str(self.project_root / "omop.duckdb"))

        wrapper_content = f'''#!/usr/bin/env python3
"""Auto-generated Python wrapper for OMCP server MCP integration."""
import subprocess
import sys
import os

def main():
    # Change to OMCP server directory
    os.chdir("{omcp_path}")

    # Set up environment
    env = os.environ.copy()
    env.update({{
        "DB_TYPE": "{db_type}",
        "DB_PATH": "{db_path}",
        "CDM_SCHEMA": "base",
        "VOCAB_SCHEMA": "base",
        "MCP_HOST": "localhost",
        "MCP_PORT": "8080"
    }})

    # Run UV command
    uv_cmd = "{self.get_uv_executable() or 'uv'}"
    cmd = [uv_cmd, "run", "python", "src/omcp/main.py"] + sys.argv[1:]

    try:
        result = subprocess.run(cmd, env=env, check=False)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running OMCP server: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        wrapper_path = self.project_root / "omcp_wrapper_mcp.py"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)

        # Make executable
        import stat
        st = os.stat(wrapper_path)
        os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)

        return wrapper_path

    # =================== SERVICE CONFIGURATION ===================
    
    def get_ollama_url(self) -> str:
        """Get Ollama service URL."""
        # Check config file first
        if "services" in self.explicit_config and "ollama_url" in self.explicit_config["services"]:
            return self.explicit_config["services"]["ollama_url"]
        # Fallback to environment variable
        return os.getenv("OLLAMA_URL", "http://localhost:11434")

    @property
    def ollama_model_name(self) -> str:
        """Get the Ollama model name."""
        # Check config file first
        if "services" in self.explicit_config and "ollama_model_name" in self.explicit_config["services"]:
            return self.explicit_config["services"]["ollama_model_name"]
        # Fallback to environment variable
        return os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama service is available."""
        import httpx
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.get_ollama_url()}/api/version")
                return response.status_code == 200
        except:
            return False
    
    def get_omop_agent_config(self) -> Dict[str, Any]:
        """Get OMOP agent server configuration."""
        config = {}
        
        # Get host
        if "agent_config" in self.explicit_config and "omop_agent_host" in self.explicit_config["agent_config"]:
            config["host"] = self.explicit_config["agent_config"]["omop_agent_host"]
        else:
            config["host"] = os.getenv("OMOP_AGENT_HOST", "127.0.0.1")
        
        # Get port
        if "agent_config" in self.explicit_config and "omop_agent_port" in self.explicit_config["agent_config"]:
            config["port"] = int(self.explicit_config["agent_config"]["omop_agent_port"])
        else:
            config["port"] = int(os.getenv("OMOP_AGENT_PORT", "8002"))
        
        # Build URL
        config["url"] = f"http://{config['host']}:{config['port']}"
        
        return config
    
    # =================== MCP SERVER CONFIGURATION ===================
    
    def get_mcp_servers_config(self) -> Dict[str, Dict[str, Any]]:
        """Get MCP servers configuration for all servers."""
        # Check for explicit mcpServers configuration
        if "mcpServers" in self.explicit_config:
            return self.explicit_config["mcpServers"]
        
        # Fallback to single server configuration for backward compatibility
        try:
            single_server_config = self.get_mcp_server_config()
            return {
                "omop_db_server": {
                    "command": single_server_config["stdio_params"].command,
                    "args": single_server_config["stdio_params"].args,
                    "env": single_server_config["stdio_params"].env
                }
            }
        except Exception:
            # Default configuration if nothing else works
            return {
                "omop_db_server": {
                    "command": "uv",
                    "args": ["run", "python", "src/omcp/main.py"],
                    "env": {
                        "DB_TYPE": "duckdb",
                        "CDM_SCHEMA": "base",
                        "VOCAB_SCHEMA": "base"
                    }
                }
            }
    
    def get_mcp_server_config(self) -> Dict[str, Any]:
        """Get MCP server configuration for the external OMOP server."""
        
        # Get the path to the external OMCP server
        omcp_path = self.get_omcp_server_path()
        if not omcp_path:
            raise ValueError("OMCP server path not configured. Please set 'omcp_server_path' in config or OMCP_SERVER_PATH environment variable")
        
        # Check if the OMCP server main.py exists
        omcp_main = omcp_path / "src" / "omcp" / "main.py"
        if not omcp_main.exists():
            raise ValueError(f"OMCP server main.py not found at {omcp_main}")
        
        # Get UV executable or use python directly
        uv_exec = self.get_uv_executable()
        
        # Determine the command to run the server
        if uv_exec:
            # Use UV to run in the OMCP server's environment
            command = str(uv_exec)
            args = ["run", "python", "src/omcp/main.py"]
        else:
            # Fallback to direct python execution
            command = sys.executable
            args = [str(omcp_main)]
        
        # The working directory should be the OMCP server root
        cwd = str(omcp_path)
        
        # Build environment variables for the external OMCP server
        # The external server expects DB_TYPE and DB_PATH, not DB_CONNECTION_STRING
        env = {}
        
        # Check for the synthea database
        synthea_db = omcp_path / "synthetic_data" / "synthea.duckdb"
        if synthea_db.exists():
            env["DB_TYPE"] = "duckdb"
            env["DB_PATH"] = str(synthea_db)
        else:
            # Fallback to environment variables or defaults
            env["DB_TYPE"] = os.getenv("DB_TYPE", "duckdb")
            env["DB_PATH"] = os.getenv("DB_PATH", str(self.project_root / "omop.duckdb"))
        
        # Add schema configurations
        env["CDM_SCHEMA"] = os.getenv("CDM_SCHEMA", "base")
        env["VOCAB_SCHEMA"] = os.getenv("VOCAB_SCHEMA", "base")
        
        # MCP server host/port (for the external server, though it uses stdio)
        env["MCP_HOST"] = os.getenv("MCP_HOST", "localhost")
        env["MCP_PORT"] = os.getenv("MCP_PORT", "8080")

        # Construct StdioServerParameters for the external server
        stdio_params = StdioServerParameters(
            command=command,
            args=args,
            cwd=cwd,
            env=env
        )
        
        return {
            "name": "omop_db_server",
            "description": "Provides OMOP CDM database access via external OMCP server",
            "medical_speciality": "omop_cdm",
            "stdio_params": stdio_params,
        }
    
    
    
    def check_and_resolve_database_locks(self) -> List[str]:
        """Check for and resolve database file locks."""
        issues = []
        resolved = []
        
        omcp_path = self.get_omcp_server_path()
        if not omcp_path:
            return ["OMCP server path not configured - cannot check database locks"]
        
        # Find database files that might be locked
        db_files = []
        synthea_db = omcp_path / "synthetic_data" / "synthea.duckdb"
        if synthea_db.exists():
            db_files.append(synthea_db)
        
        for db_file in db_files:
            try:
                # Try to find processes using this database file
                result = subprocess.run(
                    ["lsof", str(db_file)], 
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    # Parse lsof output to find PIDs
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    pids = []
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                pids.append(pid)
                            except ValueError:
                                continue
                    
                    if pids:
                        logger.info(f"Found {len(pids)} processes holding locks on {db_file.name}: {pids}")
                        
                        # Try to kill the processes gracefully
                        for pid in pids:
                            try:
                                # Check if process is still running
                                os.kill(pid, 0)  # Signal 0 just checks if process exists
                                
                                # Try graceful termination first
                                logger.info(f"Terminating process {pid} holding database lock...")
                                os.kill(pid, signal.SIGTERM)
                                time.sleep(1)
                                
                                # Check if it's still running
                                try:
                                    os.kill(pid, 0)
                                    # Still running, force kill
                                    logger.warning(f"Force killing stubborn process {pid}...")
                                    os.kill(pid, signal.SIGKILL)
                                    time.sleep(0.5)
                                    resolved.append(f"Killed process {pid} holding lock on {db_file.name}")
                                except ProcessLookupError:
                                    resolved.append(f"Process {pid} terminated gracefully")
                                    
                            except ProcessLookupError:
                                # Process already gone
                                resolved.append(f"Process {pid} was already terminated")
                            except PermissionError:
                                issues.append(f"Permission denied: cannot kill process {pid} holding lock on {db_file.name}")
                            except Exception as e:
                                issues.append(f"Error killing process {pid}: {e}")
                
            except FileNotFoundError:
                # lsof not available, try alternative method
                try:
                    # Try fuser if available
                    result = subprocess.run(
                        ["fuser", str(db_file)], 
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        issues.append(f"Database file {db_file.name} may be locked (fuser found processes)")
                except FileNotFoundError:
                    # Neither lsof nor fuser available, try pattern-based cleanup
                    try:
                        result = subprocess.run(
                            ["pkill", "-f", str(db_file.name)], 
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            resolved.append(f"Killed processes related to {db_file.name}")
                        time.sleep(1)
                    except Exception:
                        pass
            except Exception as e:
                issues.append(f"Error checking locks on {db_file.name}: {e}")
        
        # Also clean up any lingering OMCP processes
        try:
            result = subprocess.run(
                ["pkill", "-f", "src/omcp/main.py"], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                resolved.append("Cleaned up lingering OMCP processes")
                time.sleep(1)
        except Exception:
            pass
        
        if resolved:
            logger.info(f"Database lock cleanup completed: {resolved}")
            # Wait a bit for locks to fully release
            time.sleep(2)
        
        return issues

    def validate_setup(self) -> List[str]:
        """Validate the complete setup and return any issues."""
        issues = []
        
        try:
            # First, check and resolve any database locks
            print("🔒 Checking for database locks...")
            lock_issues = self.check_and_resolve_database_locks()
            if lock_issues:
                issues.extend(lock_issues)
            else:
                print("✅ No database locks found or all resolved")
            
            # Check OMCP server - this is the critical path that must be configured
            omcp_path = self.get_omcp_server_path()
            if not omcp_path:
                if not self.explicit_config and not os.getenv("OMCP_SERVER_PATH"):
                    issues.append("OMCP server path not configured - create a config file with 'med-a2a-setup --generate-config'")
                else:
                    issues.append("OMCP server path configured but invalid or missing")
            elif not (omcp_path / "src" / "omcp" / "main.py").exists():
                issues.append("OMCP server found but main.py missing - check installation")
            
            # Check UV - can be installed if missing
            uv_exec = self.get_uv_executable()
            if not uv_exec:
                issues.append("UV package manager not found - install from https://docs.astral.sh/uv/")
            else:
                # Test UV works
                try:
                    result = subprocess.run([uv_exec, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode != 0:
                        issues.append("UV executable found but not working properly")
                except Exception:
                    issues.append("UV executable found but not accessible")
            
            # Check Ollama - service dependency
            if not self.is_ollama_available():
                issues.append("Ollama service not available - install and start Ollama")
                
        except Exception as e:
            issues.append(f"Configuration validation failed: {e}")
        
        return issues
    
    def get_setup_instructions(self) -> List[str]:
        """Get setup instructions for missing components."""
        instructions = []
        issues = self.validate_setup()
        
        # Check if we need to create a config file
        if not self.explicit_config and not any(os.getenv(var) for var in ["OMCP_SERVER_PATH"]):
            instructions.append("""
🔧 STEP 1: Create Configuration File
   • Run: med-a2a-setup --generate-config
   • Edit the generated .medA2A.config.json file
   • Set the correct paths for your system
   
   Example configuration:
   {
     "omcp_server_path": "/path/to/your/omcp_server"
   }
            """.strip())
        
        for issue in issues:
            if "OMCP server" in issue and "not configured" in issue:
                instructions.append("""
📊 Setup OMCP Server:
   • Clone the OMCP server repository
   • Note the full path to the cloned directory
   • Add to config file: "omcp_server_path": "/full/path/to/omcp_server"
   • OR set environment: export OMCP_SERVER_PATH=/full/path/to/omcp_server
                """.strip())
            
            elif "UV package manager" in issue:
                instructions.append("""
🔧 Install UV Package Manager:
   • Visit: https://docs.astral.sh/uv/getting-started/installation/
   • Or run: curl -LsSf https://astral.sh/uv/install.sh | sh
   • Then restart your terminal
                """.strip())
            
            elif "Ollama" in issue:
                instructions.append("""
🤖 Install Ollama:
   • Visit: https://ollama.ai/
   • Download and install for your platform
   • Run: ollama pull llama3.1:8b
   • Start service: ollama serve
                """.strip())
        
        return instructions

    def generate_sample_config(self, output_path: str = ".medA2A.config.json") -> Path:
        """Generate a sample configuration file that requires user input."""
        
        sample_config = {
            "_comment": "Medical A2A OMOP Configuration File",
            "_description": "Edit the paths below to match your system. All paths must be absolute and valid.",
            "_instructions": [
                "1. Set 'omcp_server_path' to the directory where you cloned the OMCP server",
                "2. Optionally customize other settings below",
                "3. Run 'med-a2a-setup --check' to validate your configuration"
            ],
            
            "paths": {
                "_comment": "REQUIRED: Set these paths to match your system",
                "omcp_server_path": "/PLEASE/EDIT/path/to/omcp_server",
                "uv_executable": "uv"
            },
            
            "services": {
                "_comment": "Optional: Customize service URLs if different from defaults",
                "ollama_url": self.get_ollama_url(),
                "ollama_model_name": self.ollama_model_name,
                "ollama_timeout": 60,
                "mcp_timeout": 30
            },
            
            "agent_config": {
                "_comment": "Optional: Agent server configuration",
                "omop_agent_host": "127.0.0.1",
                "omop_agent_port": 8002
            },
            
            "database": {
                "_comment": "Optional: Database configuration",
                "db_type": "duckdb",
                "cdm_schema": "base",
                "vocab_schema": "base"
            }
        }
        
        config_path = Path(output_path)
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        return config_path
    
    def show_configuration_sources(self) -> Dict[str, Dict[str, str]]:
        """Show where each configuration value is coming from for debugging."""
        sources = {}
        
        # Test each major configuration item
        configs_to_check = [
            ("OMCP Server", ["paths", "omcp_server_path"], "OMCP_SERVER_PATH", self.get_omcp_server_path),
            ("UV Executable", ["paths", "uv_executable"], "UV_EXECUTABLE", self.get_uv_executable),
            ("Vocabulary Path", ["paths", "vocabulary_path"], "OMOP_VOCABULARY_PATH", self.get_vocabulary_path),
            ("Ollama URL", ["services", "ollama_url"], "OLLAMA_URL", self.get_ollama_url),
            ("Ollama Model", ["services", "ollama_model_name"], "OLLAMA_MODEL_NAME", lambda: self.ollama_model_name),
        ]
        
        for name, config_keys, env_var, getter_func in configs_to_check:
            source = "default"
            value = getter_func()
            
            # Check where it came from
            config_value = self.explicit_config
            for key in config_keys:
                if isinstance(config_value, dict) and key in config_value:
                    config_value = config_value[key]
                else:
                    config_value = None
                    break
            
            if config_value is not None:
                source = f"config file: {self.config_file}"
            elif os.getenv(env_var):
                source = f"environment: {env_var}"
            elif value:
                source = "auto-discovery"
            
            sources[name] = {
                "value": str(value) if value else "NOT FOUND",
                "source": source
            }
        
        return sources

    @property
    def ollama_timeout(self) -> float:
        return float(self._get_config_value(
            "services.ollama_timeout", "OLLAMA_TIMEOUT", 120))
    @property
    def mcp_timeout(self) -> float:
        return float(self._get_config_value(
            "services.mcp_timeout", "MCP_TIMEOUT", 90))

    # Agent-specific model configuration
    @property
    def orchestrator_model(self) -> str:
        return self._get_config_value(
            "services.orchestrator_model", "ORCHESTRATOR_MODEL", "llama3.1:8b")
    
    @property
    def semantic_model(self) -> str:
        return self._get_config_value(
            "services.semantic_model", "SEMANTIC_MODEL", "llama3.1:8b")
    
    @property
    def omop_model(self) -> str:
        return self._get_config_value(
            "services.omop_model", "OMOP_MODEL", "gpt-oss:20b")

    # Agent-specific timeout configuration
    @property
    def orchestrator_timeout(self) -> float:
        return float(self._get_config_value(
            "services.orchestrator_timeout", "ORCHESTRATOR_TIMEOUT", 30))
    
    @property
    def semantic_timeout(self) -> float:
        return float(self._get_config_value(
            "services.semantic_timeout", "SEMANTIC_TIMEOUT", 45))
    
    @property
    def omop_timeout(self) -> float:
        return float(self._get_config_value(
            "services.omop_timeout", "OMOP_TIMEOUT", 180))

    @property
    def omop_agent_host(self) -> str:
        """Get the host for the OMOP agent server."""
        return self._get_config_value(
            "agent_config.omop_agent_host", "MEDA2A_OMOP_AGENT_HOST", "127.0.0.1"
        )
    
    @property
    def omop_agent_port(self) -> int:
        """Get OMOP agent server port."""
        return int(self._get_config_value(
            "agent_config.omop_agent_port", "MEDA2A_OMOP_AGENT_PORT", 8003
        ))
    
    @property
    def semantic_agent_host(self) -> str:
        """Get semantic agent server host."""
        return self._get_config_value(
            "agent_config.semantic_agent_host", "MEDA2A_SEMANTIC_AGENT_HOST", "127.0.0.1"
        )
    
    @property
    def semantic_agent_port(self) -> int:
        """Get semantic agent server port."""
        return int(self._get_config_value(
            "agent_config.semantic_agent_port", "MEDA2A_SEMANTIC_AGENT_PORT", 8004
        ))

# Global configuration instance
_config_instance = None

def get_config() -> MedA2AConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = MedA2AConfig()
    return _config_instance 

def get_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()