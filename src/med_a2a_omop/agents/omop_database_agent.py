import json
import uuid
from typing import List, Dict, Any, Optional
from collections.abc import AsyncIterable

from a2a.types import Message, TextPart, Part, Role, SendMessageSuccessResponse, JSONRPCErrorResponse
from a2a_medical.base.agent import MedicalAgent, ProcessedObservation, Action, ActionResult, MentalState
from a2a_medical.integrations.ollama import OllamaReasoningMixin
from a2a_medical.integrations.mcp_official import MCPDiscoveryMixin, MCPServer
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
                request_data = json.loads(observation.parts[0].root.text)
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
        You are an expert SQL query generator specialized in the OMOP Common Data Model (CDM) v5.4  for healthcare analytics. Your role is to convert natural language questions into accurate, efficient SQL queries that follow OMOP best practices.

### Core OMOP CDM v5.4 Schema

**base.PERSON Table** (Patient Demographics)
- base.person_id (PK): Unique patient identifier
- base.gender_concept_id, base.year_of_birth, base.race_concept_id, base.ethnicity_concept_id 
- All clinical events link to base.PERSON via base.person_id

**base.CONDITION_OCCURRENCE** (Diagnoses/Conditions)
- base.condition_occurrence_id (PK), base.person_id (FK), base.condition_concept_id
- base.condition_start_date, base.condition_end_date (often NULL)
- base.condition_type_concept_id (provenance), base.visit_occurrence_id (FK)
- Use for: diagnoses, symptoms, medical history

**base.DRUG_EXPOSURE** (Medications)
- base.drug_exposure_id (PK), base.person_id (FK), base.drug_concept_id
- base.drug_exposure_start_date, base.drug_exposure_end_date
- base.days_supply, base.quantity, base.refills, base.route_concept_id
- Use for: prescriptions, administrations, dispensings

**base.MEASUREMENT** (Lab Results/Vitals)
- base.measurement_id (PK), base.person_id (FK), base.measurement_concept_id
- base.measurement_date, base.value_as_number, base.value_as_concept_id
- base.unit_concept_id, base.range_low, base.range_high, base.operator_concept_id
- Use for: lab tests, vital signs, clinical measurements

**base.OBSERVATION** (Clinical Observations)
- base.observation_id (PK), base.person_id (FK), base.observation_concept_id
- base.observation_date, base.value_as_number/string/concept_id
- Use for: social history, family history, clinical findings 

**base.VISIT_OCCURRENCE** (Healthcare Encounters)
- base.visit_occurrence_id (PK), base.person_id (FK), base.visit_concept_id
- base.visit_start_date, base.visit_end_date, base.visit_type_concept_id
- Links clinical events to encounters

**base.PROCEDURE_OCCURRENCE** (Medical Procedures)
- base.procedure_occurrence_id (PK), base.person_id (FK), base.procedure_concept_id
- base.procedure_date, base.procedure_type_concept_id, base.quantity

**base.OBSERVATION_PERIOD** (Data Capture Periods)
- base.observation_period_id (PK), base.person_id (FK)
- base.observation_period_start_date, base.observation_period_end_date
- Defines when patient data is reliably captured

### OMOP Vocabulary System

**base.CONCEPT Table**: Central vocabulary reference
- base.concept_id (PK): Unique identifier for medical concepts
- base.concept_name: Human-readable name
- base.domain_id: Clinical domain (Condition, Drug, Procedure, etc.)
- base.standard_concept: 'S' for standard concepts (use these in queries) 
- base.vocabulary_id: Source vocabulary (SNOMED, RxRxNorm, LOINC, etc.)

**base.CONCEPT_ANCESTOR**: Pre-computed concept hierarchies
- base.ancestor_concept_id, base.descendant_concept_id
- Use for finding all diabetes types, drug classes, etc. 

**Key Relationships**:
- Standard concepts only: WHERE base.standard_concept = 'S' 
- Hierarchical queries: JOIN base.concept_ancestor for comprehensive searches
- Concept mapping: Source concepts map to standard concepts 

### SQL Generation Rules

1. **Always use standard concepts**  (base.standard_concept = 'S') in queries
2. **Join patterns**:
   - Clinical events → base.PERSON: JOIN ON base.person_id
   - Events → base.Concepts: JOIN base.concept ON [domain]_concept_id = base.concept.concept_id
   - Hierarchies: JOIN base.concept_ancestor ON base.descendant_concept_id
3. **Date handling**:
   - Use DATE or DATETIME fields as available
   - Filter within base.observation_period for complete capture 
   - Handle NULL end dates appropriately
4. **Common patterns**:
   ```sql
   -- Cohort with condition
   SELECT DISTINCT base.person_id 
   FROM base.condition_occurrence 
   WHERE base.condition_concept_id IN (
     SELECT base.descendant_concept_id 
     FROM base.concept_ancestor 
     WHERE base.ancestor_concept_id = [concept_id]
   )
   
   -- Temporal relationships
   WHERE base.event_date BETWEEN base.start_date AND DATEADD(day, 30, base.start_date)
   
   -- Measurement with units
   WHERE base.measurement_concept_id = [concept_id]
     AND base.value_as_number > [threshold]
     AND base.unit_concept_id = [unit_concept_id]
        """
        prompt = f"""Convert the following question to an OMOP SQL query: \"{nl_query}\""""

        ollama_response = await self.ollama_reason(prompt, system_prompt=system_prompt, include_tools=False)
        sql_query = ollama_response.get('message', {}).get('content', '').strip()

        if not sql_query.upper().startswith("SELECT"):
            return Action(action_type="error", parameters={"message": "Failed to generate a valid SQL query."})

        return Action(
            action_type="call_mcp_tool", 
            parameters={
                "tool_id": "omop_db_server:Select_Query",
                "tool_parameters": {"query": sql_query}
            }
        )

    async def execute(self, action: Action) -> ActionResult:
        if action.action_type == "call_mcp_tool":
            tool_id = action.parameters.get("tool_id")
            tool_params = action.parameters.get("tool_parameters")
            try:
                result = await self.call_mcp_tool(tool_id, tool_params)
                
                # Package the response into our defined Pydantic model
                # The new MCP framework may return result directly or wrapped
                actual_result = result.get("result", result) if isinstance(result, dict) else result
                response_data = OMOPQueryResponse(
                    generated_sql=tool_params.get("query", ""),
                    query_result=actual_result
                )
                # The ActionResult `data` will be sent back to the Orchestrator
                return ActionResult(success=True, data=response_data.model_dump())
            except Exception as e:
                return ActionResult(success=False, error=f"MCP Tool call failed: {str(e)}")
        
        return ActionResult(success=False, error=f"Unknown action type: {action.action_type}")

    def build_agent_card(self) -> AgentCard:
        return AgentCard(name=self.agent_name, description=self.agent_description, version="1.0.0", url=f"http://localhost:8002/{self.agent_id}", capabilities=AgentCapabilities(streaming=False), skills=[], defaultInputModes=[], defaultOutputModes=[])

    # Implement abstract methods from RequestHandler
    async def on_message_send(self, params: Any, context: Optional[ServerCallContext] = None) -> Message:
        # Delegate to the agent's PCE cycle
        observation = await self.perceive(params.message)
        state = await self.learn(self.mental_state, observation)
        action = await self.reason(state)
        result = await self.execute(action)

        if result.success:
            # If successful, return the data as a JSON string in a TextPart
            message_content = json.dumps(result.data)
            return Message(
                kind="message",
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(kind="text", text=message_content))]
            )
        else:
            # If there's an error, return the error message as a JSON string in a TextPart
            error_content = json.dumps({"error": result.error})
            return Message(
                kind="message",
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(kind="text", text=error_content))]
            )

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
