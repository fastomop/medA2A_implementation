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
from .config import get_config

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
    
    # Create wrapper script if needed
    wrapper_script = config.create_wrapper_script()
    print(f"📜 Created OMCP wrapper script: {wrapper_script}")

    # Get MCP server configuration
    try:
        mcp_config = config.get_mcp_server_config()
        mcp_servers = [MCPServer(
            name=mcp_config["name"],
            url=mcp_config["url"],
            description=mcp_config["description"],
            medical_speciality=mcp_config["medical_speciality"],
            working_dir=mcp_config["working_dir"],
            env=mcp_config["env"]
        )]
        
        print(f"🏥 OMCP Server: {mcp_config['working_dir']}")
        print(f"🔧 Using UV: {mcp_config['env']['UV_EXECUTABLE']}")
        print(f"📄 Schemas: CDM={mcp_config['env']['CDM_SCHEMA']}, VOCAB={mcp_config['env']['VOCAB_SCHEMA']}")
        
    except RuntimeError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Create OMOP agent
    omop_agent = await OMOPDatabaseAgent.create(
        agent_id="omop-db-agent-01",
        mcp_servers=mcp_servers,
        ollama_model=config.get_ollama_model()
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