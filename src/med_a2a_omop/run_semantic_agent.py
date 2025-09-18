import sys
import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from dotenv import load_dotenv
import logging
import asyncio

# Setup file-based logging
log_file = os.path.join(os.path.dirname(__file__), 'semantic_agent.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w'
)

from a2a.types import AgentCard
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler

from med_a2a_omop.agents.semantic_agent import SemanticAgent
from med_a2a_omop.config import get_config

logger = logging.getLogger(__name__)

async def main():
    """Start the SemanticAgent server."""
    logger.info("Starting SemanticAgent server...")
    
    # Load configuration
    config = get_config()
    
    # Create SemanticAgent (note: SemanticAgent expects omop_agent_client, but for standalone mode we'll pass None)
    # We need to create a dummy client for now - in practice the SemanticAgent works independently
    from a2a.client import A2AClient
    import httpx
    
    # Create a minimal client (not used in standalone mode)
    dummy_client = A2AClient(httpx_client=httpx.AsyncClient(), url="http://localhost:8003/rpc")
    
    semantic_agent = SemanticAgent(
        agent_id="semantic-agent-01",
        omop_agent_client=dummy_client,
        model_name=config.semantic_model
    )
    
    # Build agent card using the inherited method from MedicalAgent (same pattern as OMOP agent)
    agent_card = semantic_agent.build_agent_card()
    app_instance = A2AStarletteApplication(agent_card=agent_card, http_handler=semantic_agent)
    app = app_instance.build(agent_card_url="/.well-known/agent-card.json", rpc_url="/rpc")
    
    # Get server configuration
    host = config.semantic_agent_host
    port = config.semantic_agent_port
    
    logger.info(f"SemanticAgent server starting on {host}:{port}")
    
    # Start the server
    uvicorn_config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
