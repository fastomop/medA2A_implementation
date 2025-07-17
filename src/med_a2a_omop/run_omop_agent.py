import uvicorn
from a2a.server.apps import A2AStarletteApplication
from dotenv import load_dotenv
import os
import uvicorn

from a2a.types import AgentCard
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler

from .agents.omop_database_agent import OMOPDatabaseAgent
from a2a_medical.integrations.mcp import MCPServer

def main():
    load_dotenv()

    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
    mcp_servers = [MCPServer(name="omop_db_server", url=mcp_server_url, description="Provides OMOP DB access.")]

    omop_agent = OMOPDatabaseAgent(
        agent_id="omop-db-agent-01",
        mcp_servers=mcp_servers
    )

    agent_card = omop_agent.build_agent_card()
    http_handler = JSONRPCHandler(agent_card=agent_card, request_handler=omop_agent)

    app = A2AStarletteApplication(agent_card=agent_card, http_handler=http_handler)

    host = os.getenv("OMOP_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("OMOP_AGENT_PORT", "8002"))
    print(f"🚀 Starting OMOP Database Agent server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()