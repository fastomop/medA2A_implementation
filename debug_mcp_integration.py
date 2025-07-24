#!/usr/bin/env python3
"""
Debug script to test MCP integration issues.
This will help identify why we're getting HTTP Error 503.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import sys
sys.path.insert(0, '/Users/k24118093/Documents/medA2A_implementation/src')
from a2a_medical.integrations.mcp_official import MCPServer, MCPManager

async def test_mcp_connection():
    """Test MCP server connection directly."""
    print("=== TESTING MCP SERVER CONNECTION ===")
    
    # Configure MCP server
    mcp_server = MCPServer(
        name="omop_db_server",
        url="stdio:///Users/k24118093/Documents/omcp_server/src/omcp/main.py",
        description="Provides OMOP CDM database access via MCP",
        medical_speciality="omop_cdm",
        working_dir="/Users/k24118093/Documents/omcp_server",
        env={
            "DB_TYPE": "duckdb",
            "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
            "CDM_SCHEMA": "base",
            "VOCAB_SCHEMA": "base"
        }
    )
    
    print(f"MCP Server Config:")
    print(f"  Name: {mcp_server.name}")
    print(f"  URL: {mcp_server.url}")
    print(f"  Transport: {mcp_server.transport}")
    print(f"  Working Dir: {mcp_server.working_dir}")
    
    # Test MCP Manager
    mcp_manager = MCPManager([mcp_server])
    
    try:
        print("\nAttempting to connect to MCP server...")
        await mcp_manager._connect_all()
        print("✅ MCP connection successful!")
        
        # Get available tools
        tools = await mcp_manager.get_available_tools()
        print(f"\nAvailable MCP Tools:\n{tools}")
        
    except Exception as e:
        print(f"❌ MCP connection failed: {e}")
        import traceback
        traceback.print_exc()

async def test_mcp_tool_call():
    """Test calling an MCP tool."""
    print("\n=== TESTING MCP TOOL CALL ===")
    
    mcp_server = MCPServer(
        name="omop_db_server",
        url="stdio:///Users/k24118093/Documents/omcp_server/src/omcp/main.py",
        description="Provides OMOP CDM database access via MCP",
        medical_speciality="omop_cdm",
        working_dir="/Users/k24118093/Documents/omcp_server",
        env={
            "DB_TYPE": "duckdb",
            "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
            "CDM_SCHEMA": "base",
            "VOCAB_SCHEMA": "base"
        }
    )
    
    mcp_manager = MCPManager([mcp_server])
    
    try:
        await mcp_manager._connect_all()
        
        # Try a simple query
        test_query = "SELECT COUNT(*) FROM base.person"
        print(f"\nExecuting test query: {test_query}")
        
        result = await mcp_manager.call_tool(
            "omop_db_server:Select_Query",
            {"query": test_query}
        )
        
        print(f"✅ Query result: {result}")
        
    except Exception as e:
        print(f"❌ Tool call failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main debugging function."""
    print("=== MCP INTEGRATION DEBUGGING ===\n")
    
    # Test 1: Basic connection
    await test_mcp_connection()
    
    # Test 2: Tool call
    await test_mcp_tool_call()

if __name__ == "__main__":
    asyncio.run(main())