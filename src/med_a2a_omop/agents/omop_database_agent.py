import json
from typing import List, Dict, Any, Optional
from collections.abc import AsyncIterable

from a2a.types import Message, TextPart, Role, SendMessageSuccessResponse, JSONRPCErrorResponse
from a2a_medical.base.agent import MedicalAgent, ProcessedObservation, Action, ActionResult, MentalState
from a2a_medical.integrations.ollama import OllamaReasoningMixin
from a2a_medical.integrations.mcp import MCPDiscoveryMixin, MCPServer
from a2a.types import AgentCard, AgentCapabilities
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.context import ServerCallContext

from ..models.a2a_messages import OMOPQueryRequest, OMOPQueryResponse

class OMOPDatabaseAgent(OllamaReasoningMixin, MCPDiscoveryMixin, MedicalAgent, RequestHandler):
    """
    An intelligent agent that performs text-to-SQL conversion and then calls an 
    external MCP tool to execute the query against the OMOP CDM.
    """
    def __init__(self, agent_id: str, mcp_servers: List[MCPServer], ollama_model: str = "llama3.1:8b"):
        super().__init__(
            agent_id=agent_id,
            agent_type="omop_database_client",
            capabilities=["omop_text_to_sql", "mcp_tool_caller"],
            world_model=None,
            mcp_servers=mcp_servers,
            model_name=ollama_model
        )
        print("[OMOPDatabaseAgent] Starting MCP server discovery...")
        # This needs to be awaited, but __init__ cannot be async. 
        # The MCPManager will attempt discovery when get_available_tools is called later.
        # For immediate testing, we can call it here, but it won't block __init__.
        # asyncio.create_task(self.mcp_manager.discover_servers())
        print("[OMOPDatabaseAgent] MCP server discovery initiated.")

    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Parses the incoming A2A Message to extract the user's question."""
        if isinstance(observation, Message):
            try:
                # The content of the message is expected to be a JSON string
                request_data = json.loads(observation.parts[0].text)
                request = OMOPQueryRequest(**request_data)
                return ProcessedObservation(data=request, timestamp=0, source="a2a_message")
            except (json.JSONDecodeError, IndexError):
                return ProcessedObservation(data=None, error="Invalid message format")
        return ProcessedObservation(data=None, error="Unsupported observation type")

    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        if observation.data:
            state.memory["current_nl_query"] = observation.data.question
        return state

    async def reason(self, state: MentalState) -> Action:
        nl_query = state.memory.get("current_nl_query")
        if not nl_query:
            return Action(action_type="error", parameters={"message": "No natural language query to process."})

        system_prompt = f"""
        You are an expert in OMOP CDM. Your task is to convert a natural language question into a single, executable SQL query. 
        - The database dialect is PostgreSQL.
        - Only output the raw SQL query. Do not add explanations, markdown, or any other text.
        - Key tables include: `person`, `condition_occurrence`, `drug_exposure`, `concept`.
        """
        prompt = f"""Convert the following question to an OMOP SQL query: \"{nl_query}\""""

        ollama_response = await self.ollama_reason(prompt, system_prompt=system_prompt, include_tools=False)
        sql_query = ollama_response.get('message', {}).get('content', '').strip()

        if not sql_query.upper().startswith("SELECT"):
            return Action(action_type="error", parameters={"message": "Failed to generate a valid SQL query."})

        return Action(
            action_type="call_mcp_tool", 
            parameters={
                "tool_id": "omop_db_server:query_omop_database",
                "tool_parameters": {"sql_query": sql_query}
            }
        )

    async def execute(self, action: Action) -> ActionResult:
        if action.action_type == "call_mcp_tool":
            tool_id = action.parameters.get("tool_id")
            tool_params = action.parameters.get("tool_parameters")
            try:
                result = await self.mcp_manager.call_tool(tool_id, tool_params)
                
                # Package the response into our defined Pydantic model
                response_data = OMOPQueryResponse(
                    generated_sql=tool_params.get("sql_query", ""),
                    query_result=result.get("result", [])
                )
                # The ActionResult `data` will be sent back to the Orchestrator
                return ActionResult(success=True, data=response_data.model_dump())
            except Exception as e:
                return ActionResult(success=False, error=f"MCP Tool call failed: {str(e)}")
        
        return ActionResult(success=False, error=f"Unknown action type: {action.action_type}")

    def build_agent_card(self) -> AgentCard:
        return AgentCard(name=self.agent_name, description=self.agent_description, version="1.0.0", url=f"http://localhost:8002/{self.agent_id}", capabilities=AgentCapabilities(streaming=False), skills=[], defaultInputModes=[], defaultOutputModes=[])

    # Implement abstract methods from RequestHandler
    async def on_message_send(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        # Delegate to the agent's PCE cycle
        observation = await self.perceive(params.message)
        state = await self.learn(self.mental_state, observation)
        action = await self.reason(state)
        result = await self.execute(action)
        return result.data # Assuming result.data is the message content

    async def on_message_send_stream(self, params: Any, context: Optional[ServerCallContext] = None) -> AsyncIterable[Any]:
        # Implement streaming logic if needed, otherwise raise NotImplementedError
        raise NotImplementedError("Streaming not implemented for OMOPDatabaseAgent")

    async def on_get_task(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Get task not implemented for OMOPDatabaseAgent")

    async def on_cancel_task(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Cancel task not implemented for OMOPDatabaseAgent")

    async def on_resubscribe_to_task(self, params: Any, context: Optional[ServerCallContext] = None) -> AsyncIterable[Any]:
        raise NotImplementedError("Resubscribe to task not implemented for OMOPDatabaseAgent")

    async def on_get_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Get task push notification config not implemented for OMOPDatabaseAgent")

    async def on_set_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Set task push notification config not implemented for OMOPDatabaseAgent")

    async def on_list_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("List task push notification config not implemented for OMOPDatabaseAgent")

    async def on_delete_task_push_notification_config(self, params: Any, context: Optional[ServerCallContext] = None) -> Any:
        raise NotImplementedError("Delete task push notification config not implemented for OMOPDatabaseAgent")
