#!/usr/bin/env python3
"""Auto-generated Python wrapper for OMCP server MCP integration."""
import subprocess
import sys
import os

def main():
    # Change to OMCP server directory
    os.chdir("/Users/k24118093/Documents/omcp_server")

    # Set up environment
    env = os.environ.copy()
    env.update({
        "DB_TYPE": "duckdb",
        "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
        "CDM_SCHEMA": "base",
        "VOCAB_SCHEMA": "base",
        "MCP_HOST": "localhost",
        "MCP_PORT": "8080"
    })

    # Run UV command
    uv_cmd = "/opt/homebrew/bin/uv"
    cmd = [uv_cmd, "run", "python", "src/omcp/main.py"] + sys.argv[1:]

    try:
        result = subprocess.run(cmd, env=env, check=False)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running OMCP server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
