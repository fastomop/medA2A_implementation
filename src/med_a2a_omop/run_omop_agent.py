import sys
import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from dotenv import load_dotenv
import logging

# Setup file-based logging
log_file = os.path.join(os.path.dirname(__file__), 'omop_agent.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w'
)

from a2a.types import AgentCard
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler

from med_a2a_omop.agents.omop_database_agent import OMOPDatabaseAgent
from med_a2a_omop.config import get_config

async def main():
    load_dotenv()
    
    # Get configuration instance
    config = get_config()
    
    # Validate environment before starting
    issues = config.validate_setup()
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
        
        print("\n📋 Setup instructions:")
        instructions = config.get_setup_instructions()
        for instruction in instructions:
            print(instruction)
            print()
        
        print("Please resolve these issues and try again.")
        return
    
    # Validate configuration
    try:
        omcp_server_path = config.get_omcp_server_path()
        print(f"🏥 OMCP Server: {omcp_server_path}")
        print(f"🔧 Using UV: {config.get_uv_executable() or 'uv'}")
        print(f"📄 Schemas: CDM=base, VOCAB=base")
        
    except RuntimeError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Create MCP server configuration using the expected format
    from med_a2a_omop.integrations.mcp_manager import MCPServerConfig

    # Use the plural method that reads from JSON config
    mcp_servers_config = config.get_mcp_servers_config()
    mcp_servers = []
    
    for server_name, server_config in mcp_servers_config.items():
        mcp_servers.append(MCPServerConfig(
            name=server_name,
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env", {}),
            cwd=server_config.get("cwd")
        ))

    # Convert to the format expected by the agent (MCPServer objects from a2a_medical)
    from a2a_medical.integrations.mcp_official import MCPServer

    # Initialize logger for this module
    logger = logging.getLogger(__name__)
    
    # First, let's inspect the actual MCPServer constructor
    import inspect
    sig = inspect.signature(MCPServer.__init__)
    logger.info(f"MCPServer constructor signature: {sig}")

    # Use the config system to create a proper Python wrapper script for MCP
    wrapper_path = config.create_python_wrapper_script()

    adapted_servers = [MCPServer(
        name=server.name,
        description="OMOP Database MCP Server",
        url=f"stdio://{wrapper_path}",
        args=[],
        env={},
        working_dir=str(config.project_root)
    ) for server in mcp_servers]

    # Create OMOP agent
    omop_agent = await OMOPDatabaseAgent.create(
        agent_id="omop-db-agent-01",
        mcp_servers=adapted_servers,
        ollama_model=config.omop_model
    )

    # Build agent card and application
    agent_card = omop_agent.build_agent_card()
    app_instance = A2AStarletteApplication(agent_card=agent_card, http_handler=omop_agent)
    app = app_instance.build(agent_card_url="/.well-known/agent-card.json", rpc_url="/rpc")

    # Get server configuration
    server_config = config.get_omop_agent_config()
    
    print(f"🚀 Starting OMOP Agent server on {server_config['host']}:{server_config['port']}")
    
    config = uvicorn.Config(
        app, 
        host=server_config['host'], 
        port=server_config['port']
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())