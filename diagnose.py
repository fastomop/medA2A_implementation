#!/usr/bin/env python3
"""Diagnostic script to check the medical A2A implementation setup."""

import subprocess
import time
import httpx
import asyncio
import os
from dotenv import load_dotenv

def check_port(port: int, service_name: str):
    """Check if a port is in use."""
    result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Port {port} is in use (expected for {service_name})")
        print(f"   {result.stdout.strip()}")
        return True
    else:
        print(f"❌ Port {port} is NOT in use ({service_name} may not be running)")
        return False

async def check_http_endpoint(url: str, service_name: str):
    """Check if an HTTP endpoint is accessible."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5.0)
            print(f"✅ {service_name} at {url} responded with status {response.status_code}")
            return True
        except Exception as e:
            print(f"❌ {service_name} at {url} is not accessible: {e}")
            return False

async def check_omop_agent():
    """Check if OMOP agent can be started."""
    print("\n🔍 Checking OMOP Agent...")
    
    # Try to start the OMOP agent
    process = subprocess.Popen(
        ["uv", "run", "python", "-m", "src.med_a2a_omop.run_omop_agent"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("⏳ Waiting 5 seconds for OMOP agent to start...")
    time.sleep(5)
    
    # Check if process is still running
    if process.poll() is None:
        print("✅ OMOP Agent process is running")
        # Check the port
        check_port(8002, "OMOP Agent")
        # Terminate the process
        process.terminate()
        process.wait()
    else:
        print("❌ OMOP Agent process exited")
        stdout, stderr = process.communicate()
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")

async def main():
    """Run all diagnostic checks."""
    print("🏥 Medical A2A Implementation Diagnostic Tool")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    print("\n📋 Environment Variables:")
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
    omop_url = os.getenv("OMOP_AGENT_URL", "http://127.0.0.1:8002")
    print(f"   MCP_SERVER_URL: {mcp_url}")
    print(f"   OMOP_AGENT_URL: {omop_url}")
    
    # Check if MCP server is running
    print("\n🔍 Checking MCP Server...")
    mcp_running = check_port(8080, "MCP Server")
    if mcp_running:
        await check_http_endpoint(mcp_url, "MCP Server")
    
    # Check OMOP agent
    await check_omop_agent()
    
    # Check dependencies
    print("\n📦 Checking Python dependencies...")
    result = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True)
    if "a2a-medical-foundation" in result.stdout:
        print("✅ a2a-medical-foundation is installed")
    else:
        print("❌ a2a-medical-foundation is NOT installed")
    
    print("\n🏁 Diagnostic complete!")
    
    if not mcp_running:
        print("\n⚠️  IMPORTANT: The MCP server is not running!")
        print("   The OMOP agent requires an MCP server to be running at port 8080.")
        print("   Without this, the OMOP agent cannot access the OMOP database.")
        print("\n   To fix this, you need to:")
        print("   1. Set up and start an MCP server that provides OMOP database access")
        print("   2. Ensure it's running on http://localhost:8080")
        print("   3. Or update the MCP_SERVER_URL in .env to point to your MCP server")

if __name__ == "__main__":
    asyncio.run(main())