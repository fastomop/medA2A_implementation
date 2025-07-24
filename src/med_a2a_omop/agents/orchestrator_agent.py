
import json
from typing import List, Dict, Any, Optional

from a2a.types import AgentCard, AgentCapabilities

from a2a.client import A2AClient
from a2a.types import Message, TextPart, Role, SendMessageSuccessResponse, JSONRPCErrorResponse
from a2a_medical.base.agent import MedicalAgent, ProcessedObservation, Action, ActionResult, MentalState, WorldModel
from a2a_medical.integrations.ollama import OllamaReasoningMixin

from ..models.a2a_messages import OMOPQueryRequest, OMOPQueryResponse


class OrchestratorWorldModel(WorldModel):
    """Simple world model for orchestrator operations."""
    
    def __init__(self):
        super().__init__()
        self.user_queries: List[str] = []
        self.agent_responses: List[Dict[str, Any]] = []
        
    def update(self, observation: ProcessedObservation) -> None:
        """Update the world model with new observations."""
        if isinstance(observation.data, str):
            self.user_queries.append(observation.data)
        elif isinstance(observation.data, dict):
            self.agent_responses.append(observation.data)
        self.last_updated = observation.timestamp
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Query the world model for information."""
        if "recent_queries" in query:
            return self.user_queries[-5:]
        elif "responses" in query:
            return self.agent_responses[-5:]
        return None
    
    def predict(self, scenario: Dict[str, Any]) -> Any:
        """Make predictions based on the current world model."""
        return {"confidence": 0.8}
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world model state."""
        base_summary = super().get_state_summary()
        base_summary.update({
            "user_queries_count": len(self.user_queries),
            "agent_responses_count": len(self.agent_responses),
        })
        return base_summary
    
    def reset(self) -> None:
        """Reset the world model to initial state."""
        self.user_queries.clear()
        self.agent_responses.clear()
        super().__init__()

class OrchestratorAgent(OllamaReasoningMixin, MedicalAgent):
    """
    The main agent that orchestrates calls to the OMOPDatabaseAgent via A2A.
    """
    def __init__(self, agent_id: str, omop_agent_client: A2AClient, ollama_model: str = "llama3.1:8b"):
        # Create world model for this agent
        world_model = OrchestratorWorldModel()
        
        super().__init__(
            agent_id=agent_id,
            agent_type="orchestrator",
            capabilities=["user_interaction", "agent_orchestration"],
            world_model=world_model,
            model_name=ollama_model,
            agent_name=f"Orchestrator-{agent_id}",
            agent_description="Medical orchestrator agent that coordinates between user and OMOP database agent"
        )
        self.omop_agent_client = omop_agent_client
        self.add_client("omop_database_agent", omop_agent_client)
        self.user_question = ""
        self.final_result = None

    async def perceive(self, observation: Any) -> ProcessedObservation:
        if isinstance(observation, str):
            self.user_question = observation
        elif isinstance(observation, dict) and "generated_sql" in observation and "query_result" in observation:
            # This is the response from the OMOPDatabaseAgent
            self.final_result = OMOPQueryResponse(**observation)
        return ProcessedObservation(data=observation, timestamp=0, source="user_input")

    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        if self.user_question:
            state.memory["user_question"] = self.user_question
        if self.final_result:
            state.memory["final_result"] = self.final_result
        return state

    async def reason(self, state: MentalState) -> Action:
        user_question = state.memory.get("user_question")
        final_result = state.memory.get("final_result")

        if user_question and not final_result:
            print("[Orchestrator] Decided to delegate to OMOP Agent.")
            request_data = OMOPQueryRequest(question=user_question)
            return Action(
                action_type="delegate_to_omop_agent",
                parameters=request_data.model_dump()
            )

        elif final_result:
            system_prompt = "You are a helpful medical assistant. Summarize the following data from a database query into a clear, human-readable answer."
            prompt = f"The user asked: '{user_question}'. The data is: {json.dumps(final_result.model_dump())}"
            
            print("[Orchestrator] Decided to summarize the final result.")
            ollama_response = await self.ollama_reason(prompt, system_prompt=system_prompt)
            summary = ollama_response.get('message', {}).get('content', 'No summary available.')
            return Action(action_type="final_answer", parameters={"summary": summary})

        return Action(action_type="no_action", parameters={})

    async def execute(self, action: Action) -> ActionResult:
        if action.action_type == "delegate_to_omop_agent":
            try:
                print(f"[Orchestrator] Sending request to OMOP Agent: {action.parameters}")
                # This is the core A2A communication step
                response_message = await self.send_message_to_agent(
                    target_agent_id="omop_database_agent",
                    message=json.dumps(action.parameters)
                )
                
                print(f"[Orchestrator] Raw response message: {response_message}")
                
                # Check if response_message is None
                if response_message is None:
                    print("[Orchestrator] Error: Received None response from OMOP Agent")
                    return ActionResult(success=False, error="No response from OMOP Agent - check if server is running")

                if isinstance(response_message.root, SendMessageSuccessResponse):
                    if not response_message.root.result.parts:
                        print("[Orchestrator] Error: Received a response with no parts.")
                        return ActionResult(success=False, error="Response from OMOP agent had no parts.")

                    # The response from the OMOP agent is in the message body
                    response_data = json.loads(response_message.root.result.parts[0].root.text)
                elif isinstance(response_message.root, JSONRPCErrorResponse):
                    print(f"[Orchestrator] Error response from OMOP Agent: {response_message.root.error.message}")
                    return ActionResult(success=False, error=f"OMOP Agent Error: {response_message.root.error.message}")
                else:
                    print(f"[Orchestrator] Unexpected response type from OMOP Agent: {type(response_message.root)}")
                    return ActionResult(success=False, error="Unexpected response type from OMOP Agent.")
                print(f"[Orchestrator] Parsed response data: {response_data}")

                if "error" in response_data:
                    print(f"[Orchestrator] !!! A2A communication failed: {response_data['error']}")
                    return ActionResult(success=False, error=response_data['error'])
                else:
                    omop_response = OMOPQueryResponse(**response_data)
                
                # Update our own state with the result
                self.final_result = omop_response
                return ActionResult(success=True, data=omop_response.model_dump())

            except Exception as e:
                print(f"[Orchestrator] !!! A2A communication failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return ActionResult(success=False, error=f"A2A communication failed: {str(e)}")

        if action.action_type == "final_answer":
            return ActionResult(success=True, data=action.parameters)

        return ActionResult(success=False, error=f"Unknown action type: {action.action_type}")

    def build_agent_card(self) -> AgentCard:
        """Build the agent card for A2A discovery."""
        return AgentCard(
            name=self.agent_name,
            description=self.agent_description,
            version="1.0.0",
            url=f"http://localhost:8001/{self.agent_id}", # Mock URL
            capabilities=AgentCapabilities(streaming=False),
            skills=[],
        )
