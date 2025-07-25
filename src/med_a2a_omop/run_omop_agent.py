import uvicorn
from a2a.server.apps import A2AStarletteApplication
from dotenv import load_dotenv
import os
import uvicorn

from a2a.types import AgentCard
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler

from .agents.omop_database_agent import OMOPDatabaseAgent
from a2a_medical.integrations.mcp_official import MCPServer

async def main():
    load_dotenv()

    # Configure MCP server with explicit stdio transport for OMCP server
    # Note: We explicitly use stdio transport here instead of MCP_SERVER_URL env var
    # because the OMOP server runs as a subprocess, not an HTTP server
    # We use a Python wrapper script to call 'uv run' with proper dependencies
    wrapper_script = "/Users/k24118093/Documents/medA2A_implementation/omcp_wrapper.py"
    mcp_server_url = f"stdio://{wrapper_script}"
    
    # Create MCP server configuration with subprocess settings
    mcp_servers = [MCPServer(
        name="omop_db_server",
        url=mcp_server_url,
        description="Provides OMOP CDM database access via MCP",
        medical_speciality="omop_cdm",
        working_dir="/Users/k24118093/Documents/omcp_server",
        env={
            "DB_TYPE": "duckdb",
            "DB_PATH": "/Users/k24118093/Documents/omcp_server/synthetic_data/synthea.duckdb",
            "CDM_SCHEMA": "base",
            "VOCAB_SCHEMA": "base"
        }
    )]

    omop_agent = await OMOPDatabaseAgent.create(
        agent_id="omop-db-agent-01",
        mcp_servers=mcp_servers
    )

    agent_card = omop_agent.build_agent_card()

    app_instance = A2AStarletteApplication(agent_card=agent_card, http_handler=omop_agent)

    app = app_instance.build(agent_card_url="/.well-known/agent-card.json", rpc_url="/rpc")

    config = uvicorn.Config(app, host=os.getenv("OMOP_AGENT_HOST", "127.0.0.1"), port=int(os.getenv("OMOP_AGENT_PORT", "8002")))
    server = uvicorn.Server(config)
    
    print(f"🚀 Starting OMOP Database Agent server at http://{config.host}:{config.port}")
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())