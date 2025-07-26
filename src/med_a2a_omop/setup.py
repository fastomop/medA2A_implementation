#!/usr/bin/env python3
"""
Medical A2A OMOP Setup and Validation Tool

This script helps users set up and validate their Medical A2A OMOP environment
across different platforms and configurations.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

def main():
    parser = argparse.ArgumentParser(
        description="Medical A2A OMOP Setup and Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check environment and show status
  python -m med_a2a_omop.setup --check

  # Generate environment configuration (.env file)
  python -m med_a2a_omop.setup --configure

  # Generate JSON configuration file (explicit paths)
  python -m med_a2a_omop.setup --generate-config

  # Show where configuration values are coming from  
  python -m med_a2a_omop.setup --show-sources

  # Show setup instructions for missing components
  python -m med_a2a_omop.setup --help-setup

  # Test the complete system
  python -m med_a2a_omop.setup --test
        """
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check environment configuration and show status'
    )
    
    parser.add_argument(
        '--configure',
        action='store_true', 
        help='Generate environment configuration file'
    )
    
    parser.add_argument(
        '--help-setup',
        action='store_true',
        help='Show setup instructions for missing components'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test the complete system functionality'
    )
    
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to environment file (default: .env)'
    )
    
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate a sample JSON configuration file'
    )
    
    parser.add_argument(
        '--config-file',
        default='.medA2A.config.json',
        help='Path to JSON configuration file (default: .medA2A.config.json)'
    )
    
    parser.add_argument(
        '--show-sources',
        action='store_true',
        help='Show where each configuration value is coming from'
    )
    
    parser.add_argument(
        '--generate-prompts',
        action='store_true',
        help='Generate a sample prompts configuration file'
    )
    
    parser.add_argument(
        '--prompts-file',
        default='.medA2A.prompts.sample.json',
        help='Path to prompts configuration file (default: .medA2A.prompts.sample.json)'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid issues if dependencies aren't installed yet
    try:
        from .config import get_config
    except ImportError as e:
        print("❌ Failed to import configuration system.")
        print(f"   Error: {e}")
        print("   Please install the package first: pip install -e .")
        sys.exit(1)
    
    config = get_config()
    
    if args.check or not any([args.configure, args.help_setup, args.test, args.generate_config, args.show_sources]):
        check_environment(config)
    
    if args.configure:
        configure_environment(config, args.env_file)
    
    if args.generate_config:
        generate_json_config(config, args.config_file)
    
    if args.show_sources:
        show_configuration_sources(config)
    
    if args.generate_prompts:
        generate_prompts_config(args.prompts_file)
    
    if args.help_setup:
        show_setup_instructions(config)
    
    if args.test:
        test_system(config)

def check_environment(config):
    """Check and display environment status."""
    print("🔍 Medical A2A OMOP Environment Check")
    print("=" * 50)
    
    # System info
    import platform
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()
    
    # Component checks
    components = [
        ("UV Package Manager", lambda: config.get_uv_executable()),
        ("Ollama Service", lambda: config.is_ollama_available()),
        ("OMCP Server", lambda: config.get_omcp_server_path()),
    ]
    
    all_good = True
    
    for name, check_func in components:
        try:
            result = check_func()
            if result:
                if name == "Ollama Service":
                    print(f"✅ {name}: Available at {config.get_ollama_url()}")
                else:
                    print(f"✅ {name}: {result}")
            else:
                print(f"❌ {name}: Not found")
                all_good = False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            all_good = False
    
    print()
    
    if all_good:
        print("🎉 All components are properly configured!")
        print("   You can now run: run-med-a2a")
    else:
        print("⚠️  Some components need attention.")
        print("   Run with --help-setup for detailed instructions.")
        
        # Show validation issues
        issues = config.validate_setup()
        if issues:
            print("\n📋 Specific Issues:")
            for issue in issues:
                print(f"   • {issue}")

def configure_environment(config, env_file):
    """Generate environment configuration."""
    print(f"📝 Generating environment configuration: {env_file}")
    
    env_vars = {}
    
    # Auto-detect what we can
    uv_exec = config.get_uv_executable()
    if uv_exec:
        env_vars["UV_EXECUTABLE"] = uv_exec
    
    omcp_path = config.get_omcp_server_path()
    if omcp_path:
        env_vars["OMCP_SERVER_PATH"] = str(omcp_path)
    
    # Add defaults for other settings
    env_vars.update({
        "OLLAMA_URL": config.get_ollama_url(),
        "OLLAMA_MODEL": config.get_ollama_model(),
        "OMOP_AGENT_HOST": "127.0.0.1",
        "OMOP_AGENT_PORT": "8002",
        "DB_TYPE": "duckdb",
        "CDM_SCHEMA": "base",
        "VOCAB_SCHEMA": "base",
    })
    
    # Write environment file
    env_path = Path(env_file)
    with open(env_path, 'w') as f:
        f.write("# Medical A2A OMOP Environment Configuration\n")
        f.write("# Generated automatically - modify as needed\n\n")
        
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    print(f"✅ Environment configuration written to: {env_path.absolute()}")
    print("   Edit this file to customize your setup.")

def generate_json_config(config, config_file):
    """Generate a JSON configuration file."""
    print(f"📄 Generating JSON configuration file: {config_file}")
    
    try:
        config_path = config.generate_sample_config(config_file)
        print(f"✅ JSON configuration written to: {config_path.absolute()}")
        print("   This file provides explicit control over all paths and settings.")
        print("   Values are pre-filled with current discovered settings.")
        print(f"   Edit this file and it will take priority over environment variables and auto-discovery.")
    except Exception as e:
        print(f"❌ Failed to generate configuration file: {e}")

def generate_prompts_config(prompts_file):
    """Generate a sample prompts configuration file."""
    print(f"📝 Generating prompts configuration file: {prompts_file}")
    
    try:
        from .prompts import get_prompts_manager
        prompts_manager = get_prompts_manager()
        config_path = prompts_manager.generate_sample_prompts_config(prompts_file)
        print(f"✅ Prompts configuration written to: {config_path.absolute()}")
        print("   This file allows you to customize system prompts for different agents.")
        print("   Edit this file and set PROMPTS_CONFIG_FILE environment variable to use custom prompts.")
    except Exception as e:
        print(f"❌ Failed to generate prompts configuration file: {e}")

def show_configuration_sources(config):
    """Show where each configuration value is coming from."""
    print("🔍 Configuration Sources")
    print("=" * 50)
    
    try:
        sources = config.show_configuration_sources()
        
        for component, info in sources.items():
            print(f"📊 {component}:")
            print(f"   Value: {info['value']}")
            print(f"   Source: {info['source']}")
            print()
            
        print("Configuration Priority (highest to lowest):")
        print("   1. JSON configuration file")
        print("   2. Environment variables")  
        print("   3. Auto-discovery")
        print("   4. Default values")
        
    except Exception as e:
        print(f"❌ Failed to show configuration sources: {e}")

def show_setup_instructions(config):
    """Show detailed setup instructions."""
    print("📚 Medical A2A OMOP Setup Instructions")
    print("=" * 50)
    
    instructions = config.get_setup_instructions()
    
    if not instructions:
        print("🎉 No setup needed! All components are properly configured.")
        return
    
    for i, instruction in enumerate(instructions, 1):
        print(f"\n{i}. {instruction}")
    
    print(f"\n{'='*50}")
    print("After completing setup, run: python -m med_a2a_omop.setup --check")

def test_system(config):
    """Test the complete system functionality."""
    print("🧪 Testing Medical A2A OMOP System")
    print("=" * 40)
    
    # Basic configuration test
    print("1. Testing configuration...")
    issues = config.validate_setup()
    if issues:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"   • {issue}")
        return
    print("✅ Configuration valid")
    
    # Test wrapper script generation
    print("2. Testing wrapper script generation...")
    try:
        wrapper_path = config.create_wrapper_script()
        print(f"✅ Wrapper script created: {wrapper_path}")
    except Exception as e:
        print(f"❌ Wrapper script creation failed: {e}")
        return
    
    # Test Ollama connection
    print("3. Testing Ollama connection...")
    if config.is_ollama_available():
        print(f"✅ Ollama available at {config.get_ollama_url()}")
    else:
        print(f"❌ Ollama not available at {config.get_ollama_url()}")
        return
    
    print("\n🎉 System test completed successfully!")
    print("   You can now run: run-med-a2a")

if __name__ == "__main__":
    main() 