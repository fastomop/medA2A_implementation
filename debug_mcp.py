#!/usr/bin/env python3
"""Debug script to test MCP integration directly."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, '/Users/k24118093/Documents/medical-a2a/src')

async def test_mcp_direct():
    """Test MCP integration directly."""
    print("🔍 Testing MCP Integration Direct")
    print("=" * 40)
    
    # Test 1: Check if MCP SDK is available
    try:
        import mcp
        print("✅ MCP SDK is available")
        print(f"   Version info: {getattr(mcp, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"❌ MCP SDK not available: {e}")
        print("   The system will fall back to the original implementation")
    
    # Test 2: Test the integration
    try:
        from a2a_medical.integrations.mcp_official import MCPServer, MCPManager
        print("✅ MCP Official integration imported successfully")
        
        # Create test server
        server = MCPServer(
            name="omop_db_server",
            url="stdio:///Users/k24118093/Documents/omcp_server/src/omcp/main.py",
            description="OMOP CDM database access via MCP",
            working_dir="/Users/k24118093/Documents/omcp_server",
            env={
                "DB_TYPE": "duckdb",
                "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
                "CDM_SCHEMA": "cdm",
                "VOCAB_SCHEMA": "cdm"
            }
        )
        
        print(f"✅ Server configured: {server.name}")
        print(f"   Transport: {server.transport}")
        
        # Test manager
        manager = MCPManager([server])
        print("✅ Manager created")
        
        # Test connection
        print("\n🔌 Testing connection...")
        await manager._connect_all()
        print("✅ Connection established")
        
        # List tools
        print("\n🛠️  Testing tool discovery...")
        tools = await manager.get_available_tools()
        print(f"Available tools:\n{tools}")
        
        # Test tool call
        if "omop_db_server:Get_Information_Schema" in manager.available_tools:
            print("\n📞 Testing tool call...")
            result = await manager.call_tool("omop_db_server:Get_Information_Schema", {})
            print(f"✅ Tool call successful")
            print(f"   Result type: {type(result)}")
            print(f"   Result preview: {str(result)[:200]}...")
        
    except Exception as e:
        print(f"❌ Error testing MCP integration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            await manager.shutdown()
            print("\n✅ Shutdown complete")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_mcp_direct())