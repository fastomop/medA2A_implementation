#!/usr/bin/env python3
"""Test script to check MCP stdio communication."""

import subprocess
import json
import sys
import os

def test_mcp_stdio():
    """Test MCP server communication via stdio."""
    print("🔍 Testing MCP server via stdio...")
    
    # Start the MCP server process
    process = subprocess.Popen(
        ["uv", "run", "python", "src/omcp/main.py"],
        cwd="/Users/k24118093/Documents/omcp_server",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send an initialize request
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    try:
        # Send the request
        request_json = json.dumps(initialize_request) + "\n"
        print(f"Sending: {request_json.strip()}")
        
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        print(f"Received: {response_line.strip()}")
        
        if response_line:
            response = json.loads(response_line)
            print(f"✅ MCP server responded: {response}")
        else:
            print("❌ No response from MCP server")
            
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    test_mcp_stdio()