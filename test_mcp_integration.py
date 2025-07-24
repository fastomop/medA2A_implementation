#!/usr/bin/env python3
"""Test the proper MCP integration with OMCP server."""

import asyncio
import logging
import os
from dotenv import load_dotenv

# Make sure we're using the updated medical-a2a framework
import sys
sys.path.insert(0, '/Users/k24118093/Documents/medical-a2a/src')

from a2a_medical.integrations.mcp_official import MCPServer, MCPManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_mcp_integration():
    """Test MCP integration with OMCP server."""
    load_dotenv()
    
    print("🧪 Testing MCP Integration with OMCP Server")
    print("=" * 50)
    
    # Create OMCP server configuration
    omcp_server = MCPServer(
        name="omop_db_server",
        url=os.getenv("MCP_SERVER_URL", "stdio:///Users/k24118093/Documents/omcp_server/src/omcp/main.py"),
        description="OMOP CDM database access via MCP",
        working_dir="/Users/k24118093/Documents/omcp_server",
        env={
            "DB_TYPE": "duckdb",
            "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
            "CDM_SCHEMA": "cdm",
            "VOCAB_SCHEMA": "cdm"
        }
    )
    
    # Create MCP manager
    manager = MCPManager([omcp_server])
    
    try:
        # Connect to servers
        print("\n1️⃣ Connecting to MCP servers...")
        await manager._connect_all()
        print("✅ Connected successfully")
        
        # List available tools
        print("\n2️⃣ Available tools:")
        tools_list = await manager.get_available_tools()
        print(tools_list)
        
        # Test Get_Information_Schema
        print("\n3️⃣ Testing Get_Information_Schema tool...")
        result = await manager.call_tool("omop_db_server:Get_Information_Schema", {})
        print(f"✅ Schema retrieved: {len(result.get('result', '').splitlines())} lines")
        
        # Test Select_Query
        print("\n4️⃣ Testing Select_Query tool...")
        query_result = await manager.call_tool(
            "omop_db_server:Select_Query",
            {"query": "SELECT COUNT(*) as patient_count FROM person"}
        )
        print(f"✅ Query result: {query_result}")
        
        # Test query for hypertension
        print("\n5️⃣ Testing hypertension query...")
        hypertension_query = """
        SELECT COUNT(DISTINCT p.person_id) as patient_count
        FROM person p
        JOIN condition_occurrence co ON p.person_id = co.person_id
        JOIN concept c ON co.condition_concept_id = c.concept_id
        WHERE LOWER(c.concept_name) LIKE '%hypertension%'
        """
        
        hypertension_result = await manager.call_tool(
            "omop_db_server:Select_Query",
            {"query": hypertension_query}
        )
        print(f"✅ Hypertension query result: {hypertension_result}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Shutting down...")
        await manager.shutdown()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    asyncio.run(test_mcp_integration())