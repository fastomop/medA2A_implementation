import uvicorn
from a2a.server.apps import A2AStarletteApplication
from dotenv import load_dotenv
import os
import uvicorn
import logging
import asyncio

from a2a.types import AgentCard

from .agents.omop_database_agent import OMOPDatabaseAgent
from a2a_medical.integrations.mcp_official import MCPServer, MCPClient
from mcp.client.stdio import StdioServerParameters

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("a2a_medical").setLevel(logging.DEBUG)
logging.getLogger("mcp").setLevel(logging.DEBUG)

async def main():
    load_dotenv()

    # Configure MCP server with proper settings for stdio transport
    mcp_server_url = os.getenv("MCP_SERVER_URL", "stdio:///Users/k24118093/Documents/omcp_server/src/omcp/main.py")
    
    # Create MCP server configuration with subprocess settings
    mcp_servers = [MCPServer(
        name="omop_db_server",
        description="Provides OMOP CDM database access via MCP",
        medical_speciality="omop_cdm",
        stdio_params=StdioServerParameters(
            command=[os.path.join("/Users/k24118093/Documents/omcp_server", ".venv", "bin", "python")],
            args=[os.path.join("/Users/k24118093/Documents/omcp_server", "src", "omcp", "main.py")],
            cwd="/Users/k24118093/Documents/omcp_server",
            env={
                "DB_TYPE": "duckdb",
                "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
                "CDM_SCHEMA": "base",
                "VOCAB_SCHEMA": "base"
            }
        )
    )]

    omop_agent = OMOPDatabaseAgent(
        agent_id="omop-db-agent-01",
        mcp_servers=mcp_servers
    )

    agent_card = omop_agent.build_agent_card()

    app_instance = A2AStarletteApplication(agent_card=agent_card, http_handler=omop_agent)
    app = app_instance.build(
        rpc_url="/rpc",
        agent_card_url="/.well-known/agent-card"
    )  # Build the actual Starlette ASGI application
    
    # Debug: Print route information
    print(f"Routes registered in app: {[route.path for route in app.routes]}")
    print(f"Agent card URL: {agent_card.url if hasattr(agent_card, 'url') else 'No URL'}")

    host = os.getenv("OMOP_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("OMOP_AGENT_PORT", "8002"))
    print(f"🚀 Starting OMOP Database Agent server at http://{host}:{port}")
    
    # Run Uvicorn in a separate task
    uvicorn_task = asyncio.create_task(asyncio.to_thread(uvicorn.run, app, host=host, port=port))

    # Give Uvicorn a moment to start
    await asyncio.sleep(5)

    # Now attempt to connect to MCP server and log
    try:
        temp_client = MCPClient(mcp_servers[0])
        await temp_client.connect()
        logging.info("OMCP server connected successfully from run_omop_agent.py")
        await temp_client.disconnect()
    except Exception as e:
        logging.error(f"Error connecting to OMCP server from run_omop_agent.py: {e}")
        import traceback
        traceback.print_exc()

    await uvicorn_task # Keep the Uvicorn server running

if __name__ == "__main__":
    asyncio.run(main())
