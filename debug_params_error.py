#!/usr/bin/env python3
"""
Debug script to isolate the params.params error.
This will help us find where the error is occurring.
"""

import sys
import os
import asyncio
import traceback
import json
from typing import Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import A2A types
from a2a.types import MessageSendParams, Message, TextPart, Role

# Import our agent
from med_a2a_omop.agents.omop_database_agent import OMOPDatabaseAgent
from a2a_medical.integrations.mcp_official import MCPServer

async def test_params_directly():
    """Test creating and using MessageSendParams directly."""
    print("=== Testing MessageSendParams directly ===")
    
    # Create a test message
    test_message = Message(
        messageId="test-123",
        parts=[TextPart(text=json.dumps({"question": "How many patients have hypertension?"}))],
        role=Role.user
    )
    
    # Create MessageSendParams
    params = MessageSendParams(message=test_message)
    
    print(f"Created MessageSendParams: {params}")
    print(f"Message content: {params.message}")
    print(f"Message parts: {params.message.parts}")
    print(f"Message parts[0]: {params.message.parts[0]}")
    print(f"Message parts[0].root: {params.message.parts[0].root}")
    print(f"Message parts[0].root.text: {params.message.parts[0].root.text}")
    
    return params

async def test_agent_directly():
    """Test calling the agent's on_message_send method directly."""
    print("\n=== Testing Agent on_message_send directly ===")
    
    # Create MCP server configuration
    mcp_servers = [MCPServer(
        name="omop_db_server",
        url="stdio:///Users/k24118093/Documents/omcp_server/src/omcp/main.py",
        description="Provides OMOP CDM database access via MCP",
        medical_speciality="omop_cdm",
        working_dir="/Users/k24118093/Documents/omcp_server",
        env={
            "DB_TYPE": "duckdb",
            "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
            "CDM_SCHEMA": "cdm",
            "VOCAB_SCHEMA": "cdm"
        }
    )]
    
    # Create agent
    print("Creating OMOPDatabaseAgent...")
    agent = OMOPDatabaseAgent(
        agent_id="test-agent",
        mcp_servers=mcp_servers
    )
    print("Agent created successfully")
    
    # Create test params
    params = await test_params_directly()
    
    # Try calling the on_message_send method
    print("Calling agent.on_message_send...")
    try:
        result = await agent.on_message_send(params, context=None)
        print(f"Success! Result: {result}")
        return result
    except Exception as e:
        print(f"ERROR in on_message_send: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        traceback.print_exc()
        return None

async def main():
    """Main test function."""
    print("=== DEBUGGING PARAMS.PARAMS ERROR ===")
    
    try:
        # Test MessageSendParams directly
        await test_params_directly()
        
        # Test agent directly
        await test_agent_directly()
        
    except Exception as e:
        print(f"\nMAIN ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())