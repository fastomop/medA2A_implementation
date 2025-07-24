#!/usr/bin/env python3
"""Debug MCP communication with OMCP server."""

import subprocess
import json
import asyncio

async def test_mcp_communication():
    """Test basic MCP communication."""
    print("🔍 Testing MCP Communication")
    print("=" * 40)
    
    # Start OMCP server
    import os
    env = os.environ.copy()
    env.update({
        "DB_TYPE": "duckdb",
        "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
        "CDM_SCHEMA": "cdm",
        "VOCAB_SCHEMA": "cdm"
    })
    
    process = subprocess.Popen(
        ["uv", "run", "python", "src/omcp/main.py"],
        cwd="/Users/k24118093/Documents/omcp_server",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env
    )
    
    print("✅ Process started")
    
    # Give it time to start
    await asyncio.sleep(1)
    
    # Check if still running
    if process.poll() is not None:
        stderr = process.stderr.read()
        print(f"❌ Process died: {stderr}")
        return
    
    # Test 1: Initialize
    print("\n1️⃣ Testing initialize...")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {
                "name": "test",
                "version": "1.0"
            }
        }
    }
    
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    
    # Read response
    response = await read_json_response(process)
    print(f"Response: {response}")
    
    # Test 2: List tools
    print("\n2️⃣ Testing tools/list...")
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    
    response = await read_json_response(process)
    print(f"Response: {response}")
    
    # Cleanup
    process.terminate()
    process.wait()
    print("\n✅ Test complete")

async def read_json_response(process, timeout=5):
    """Read JSON response from process stdout."""
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        line = await asyncio.get_event_loop().run_in_executor(
            None, process.stdout.readline
        )
        
        if not line:
            break
            
        line = line.strip()
        print(f"Raw output: {line}")
        
        if line and (line.startswith("{") or line.startswith("[")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                print(f"Failed to parse: {line}")
                
        await asyncio.sleep(0.1)
    
    return None

if __name__ == "__main__":
    asyncio.run(test_mcp_communication())