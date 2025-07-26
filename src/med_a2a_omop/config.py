"""
Configuration management for the Medical A2A OMOP system.
Handles environment detection, path discovery, and cross-platform compatibility.
"""

import os
import sys
import shutil
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import json
from dotenv import load_dotenv
import logging

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
        # 1. Check JSON config file first
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
        
        # Check UV installation
        if not self.get_uv_executable():
            issues.append("UV package manager not found. Install from: https://docs.astral.sh/uv/")
        
        # Check Ollama installation
        if not self.is_ollama_available():
            issues.append("Ollama not available. Install from: https://ollama.ai/")
        
        # Check OMCP server
        if not self.get_omcp_server_path():
            issues.append("OMCP server not found. Set OMCP_SERVER_PATH or place in expected locations.")
        
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    # =================== PATH DISCOVERY ===================
    
    def get_omcp_server_path(self) -> Optional[Path]:
        """Get OMCP server path from explicit configuration only."""
        
        # 1. Check JSON config file first (recommended)
        if "paths" in self.explicit_config and "omcp_server_path" in self.explicit_config["paths"]:
            path_str = self.explicit_config["paths"]["omcp_server_path"]
            path = Path(path_str)
            if path.exists() and (path / "src" / "omcp" / "main.py").exists():
                logger.info(f"Using OMCP server from config file: {path}")
                return path
            else:
                logger.error(f"OMCP server path in config file is invalid: {path_str}")
                return None
        
        # 2. Check environment variable as fallback
        env_path = os.getenv("OMCP_SERVER_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists() and (path / "src" / "omcp" / "main.py").exists():
                logger.info(f"Using OMCP server from environment: {path}")
                return path
            else:
                logger.error(f"OMCP server path in environment variable is invalid: {env_path}")
                return None
        
        # 3. No auto-discovery - require explicit configuration
        logger.warning("OMCP server path not configured. Please set in config file or environment variable.")
        return None

    def get_uv_executable(self) -> Optional[str]:
        """Get UV executable path with limited fallback to PATH only."""
        
        # 1. Check JSON config file first
        if "paths" in self.explicit_config and "uv_executable" in self.explicit_config["paths"]:
            uv_config = self.explicit_config["paths"]["uv_executable"]
            
            # If it's an absolute path, check if it exists
            if Path(uv_config).is_absolute():
                if Path(uv_config).exists():
                    logger.info(f"Using UV from config file: {uv_config}")
                    return uv_config
                else:
                    logger.error(f"UV executable path in config file is invalid: {uv_config}")
                    return None
            else:
                # If it's just a command name, try to find it in PATH
                uv_exec = shutil.which(uv_config)
                if uv_exec:
                    logger.info(f"Using UV from config file (found in PATH): {uv_exec}")
                    return uv_exec
                else:
                    logger.error(f"UV executable '{uv_config}' from config file not found in PATH")
                    return None
        
        # 2. Check environment variable
        env_uv = os.getenv("UV_EXECUTABLE")
        if env_uv:
            if Path(env_uv).exists():
                logger.info(f"Using UV from environment: {env_uv}")
                return env_uv
            else:
                uv_exec = shutil.which(env_uv)
                if uv_exec:
                    logger.info(f"Using UV from environment (found in PATH): {uv_exec}")
                    return uv_exec
        
        # 3. Only check PATH (standard practice) - no system searching
        uv_exec = shutil.which("uv")
        if uv_exec:
            logger.info(f"Found UV in PATH: {uv_exec}")
            return uv_exec
        
        # 4. No deep system searching
        logger.warning("UV executable not found. Please install UV or set path in config file.")
        return None
    
    # =================== SERVICE CONFIGURATION ===================
    
    def get_ollama_url(self) -> str:
        """Get Ollama service URL."""
        # Check config file first
        if "services" in self.explicit_config and "ollama_url" in self.explicit_config["services"]:
            return self.explicit_config["services"]["ollama_url"]
        # Fallback to environment variable
        return os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    def get_ollama_model(self) -> str:
        """Get Ollama model name."""
        # Check config file first
        if "services" in self.explicit_config and "ollama_model" in self.explicit_config["services"]:
            return self.explicit_config["services"]["ollama_model"]
        # Fallback to environment variable
        return os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
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
    
    def get_mcp_server_config(self) -> Dict[str, Any]:
        """Get MCP server configuration for OMCP."""
        omcp_path = self.get_omcp_server_path()
        if not omcp_path:
            raise RuntimeError("OMCP server path not found. Set OMCP_SERVER_PATH environment variable.")
        
        # Create wrapper script path
        wrapper_script = self.project_root / "scripts" / "omcp_wrapper.py"
        
        return {
            "name": "omop_db_server",
            "url": f"stdio://{wrapper_script}",
            "description": "Provides OMOP CDM database access via MCP",
            "medical_speciality": "omop_cdm",
            "working_dir": str(omcp_path),
            "wrapper_script": str(wrapper_script),
            "env": {
                "DB_TYPE": os.getenv("DB_TYPE", "duckdb"),
                "CDM_SCHEMA": os.getenv("CDM_SCHEMA", "base"), 
                "VOCAB_SCHEMA": os.getenv("VOCAB_SCHEMA", "base"),
                "OMCP_SERVER_PATH": str(omcp_path),
                "UV_EXECUTABLE": self.get_uv_executable(),
            }
        }
    
    # =================== SETUP AND VALIDATION ===================
    
    def create_wrapper_script(self) -> Path:
        """Create a cross-platform OMCP wrapper script."""
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        wrapper_path = scripts_dir / "omcp_wrapper.py"
        
        # Generate wrapper script content
        wrapper_content = self._generate_wrapper_script()
        
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        # Make executable on Unix-like systems
        if platform.system() != "Windows":
            os.chmod(wrapper_path, 0o755)
        
        return wrapper_path
    
    def _generate_wrapper_script(self) -> str:
        """Generate cross-platform wrapper script content."""
        uv_executable = self.get_uv_executable()
        omcp_path = self.get_omcp_server_path()
        
        return f'''#!/usr/bin/env python3
"""
Cross-platform OMCP server wrapper script.
Generated automatically by medA2A configuration system.
"""

import os
import sys
import subprocess
import signal
import atexit

# Configuration from medA2A config system
UV_EXECUTABLE = "{uv_executable}"
OMCP_SERVER_PATH = "{omcp_path}"

omcp_process = None

def cleanup_process():
    """Clean up the OMCP server process."""
    global omcp_process
    if omcp_process:
        try:
            omcp_process.terminate()
            omcp_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            omcp_process.kill()
        except:
            pass

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    cleanup_process()
    sys.exit(0)

def main():
    global omcp_process
    
    # Register cleanup and signal handlers
    atexit.register(cleanup_process)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Change to OMCP server directory
        os.chdir(OMCP_SERVER_PATH)
        
        # Execute UV run with proper environment
        cmd = [UV_EXECUTABLE, "run", "python", "src/omcp/main.py"]
        
        omcp_process = subprocess.Popen(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ
        )
        
        return_code = omcp_process.wait()
        sys.exit(return_code)
        
    except Exception as e:
        print(f"Error running OMCP server: {{e}}", file=sys.stderr)
        cleanup_process()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def validate_setup(self) -> List[str]:
        """Validate the complete setup and return any issues."""
        issues = []
        
        try:
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
                "ollama_model": self.get_ollama_model()
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
            ("Ollama URL", ["services", "ollama_url"], "OLLAMA_URL", self.get_ollama_url),
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

# Global configuration instance
_config_instance = None

def get_config() -> MedA2AConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = MedA2AConfig()
    return _config_instance 