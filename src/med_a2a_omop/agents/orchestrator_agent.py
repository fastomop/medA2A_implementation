
import json
import re
from typing import List, Dict, Any, Optional
import logging

from a2a.types import AgentCard, AgentCapabilities

from a2a.client import A2AClient
from a2a.types import Message, TextPart, Role, SendMessageSuccessResponse, JSONRPCErrorResponse
from a2a_medical.base.agent import MedicalAgent, ProcessedObservation, Action, ActionResult, MentalState, WorldModel
from a2a_medical.integrations.ollama import OllamaReasoningMixin

from ..models.a2a_messages import OMOPQueryRequest, OMOPQueryResponse

logger = logging.getLogger(__name__)

class OrchestratorWorldModel(WorldModel):
    """World model for the orchestrator, now with planning capabilities."""
    
    def __init__(self):
        super().__init__()
        self.original_query: Optional[str] = None
        self.plan: Optional[List[str]] = None
        self.executed_steps: List[Dict[str, Any]] = []
        
    def update(self, observation: ProcessedObservation) -> None:
        """Update the world model with new observations."""
        if observation.source == "user_question":
            self.original_query = observation.data
            # Reset plan when a new question is asked
            self.plan = None
            self.executed_steps = []
        elif observation.source == "omop_agent_response":
            # Append sub-task results
            self.executed_steps.append({
                "sub_question": self.plan.pop(0) if self.plan else "Unknown sub-question",
                "result": observation.data
            })
        self.last_updated = observation.timestamp
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Query the world model for information."""
        if "plan" in query:
            return self.plan
        elif "executed_steps" in query:
            return self.executed_steps
        return None
    
    def predict(self, scenario: Dict[str, Any]) -> Any:
        """Make predictions based on the current world model."""
        # This can be enhanced later for more complex predictions.
        if "next_step_success_probability" in scenario:
            return {"confidence": 0.85} # Placeholder confidence
        return None
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world model state."""
        base_summary = super().get_state_summary()
        base_summary.update({
            "original_query": self.original_query,
            "plan_steps_remaining": len(self.plan) if self.plan else 0,
            "executed_steps_count": len(self.executed_steps),
        })
        return base_summary
    
    def reset(self) -> None:
        """Reset the world model to initial state."""
        self.original_query = None
        self.plan = None
        self.executed_steps = []
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

        # Explicitly type hint the world_model for the linter
        self.world_model: OrchestratorWorldModel = world_model

    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Processes incoming data and identifies its source."""
        source = "unknown"
        data = observation
        if isinstance(observation, str):
            source = "user_question"
        elif isinstance(observation, dict) and "generated_sql" in observation:
             # This is a result from the OMOP Agent's execute() method
            source = "omop_agent_response"
        return ProcessedObservation(data=data, timestamp=0, source=source)

    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        """Updates the agent's world model and mental state based on new observations."""
        
        # Ensure world_model exists and is the correct type before using it
        if isinstance(self.world_model, OrchestratorWorldModel):
            # Update the world model first
            self.world_model.update(observation)

            # Populate mental state from the world model for the reason/execute cycle
            state.memory['original_query'] = self.world_model.original_query
            state.memory['plan'] = self.world_model.plan
            state.memory['executed_steps'] = self.world_model.executed_steps
        else:
            # Fallback for basic memory if world model is not set up correctly
            if observation.source == "user_question":
                state.memory["original_query"] = observation.data
        
        return state

    async def reason(self, state: MentalState) -> Action:
        """
        Reasons about the current state to decide the next action:
        1. Generate a plan if none exists.
        2. Execute the next step of an existing plan.
        3. Synthesize a final answer if the plan is complete.
        """
        original_query = state.memory.get('original_query')
        plan = state.memory.get('plan')
        executed_steps = state.memory.get('executed_steps', [])

        if not original_query:
            return Action(action_type="error", parameters={"message": "No user query found."})

        # Scenario 1: No plan exists yet. Generate one.
        if plan is None and not executed_steps:
            print("[Orchestrator] 🧠 Phase 1: Generating a plan...")
            return await self._generate_plan(original_query)

        # Scenario 2: Plan exists and has steps remaining. Execute the next step.
        if plan and len(plan) > 0:
            next_sub_question = plan[0]
            print(f"[Orchestrator] 🏃 Phase 2: Executing next step -> '{next_sub_question}'")
            request_data = OMOPQueryRequest(question=next_sub_question)
            return Action(
                action_type="delegate_to_omop_agent",
                parameters=request_data.model_dump()
            )

        # Scenario 3: Plan is complete. Synthesize the final answer.
        if plan is not None and len(plan) == 0 and executed_steps:
            print("[Orchestrator] 💡 Phase 3: Synthesizing final answer...")
            return await self._synthesize_answer(original_query, executed_steps)

        # Default or error case
        return Action(action_type="error", parameters={"message": "Orchestrator is in an inconsistent state."})

    async def _generate_plan(self, user_question: str) -> Action:
        """Generates a multi-step plan to answer a complex query."""
        system_prompt = """
You are a master medical research planner. Your task is to break down a complex user question into a series of simple, sequential sub-questions. Each sub-question must be answerable with a single, straightforward SQL query.

**CRITICAL RULES for Plan Generation:**
1.  **SIMPLE, FACTUAL QUESTIONS ONLY:** Every step MUST be a simple data retrieval question (e.g., "Count...", "Find...", "List...").
2.  **NO CALCULATIONS OR COMPARISONS:** Do NOT create steps that require math, percentages, or comparing results from other steps. The final synthesis step will handle all calculations.
3.  **BE EFFICIENT:** Do NOT create steps that ask for large amounts of raw data. Be specific.
4.  **PREFER COUNTS OVER LISTS:** Use "Count patients with X" instead of "List all patients with X" whenever possible.
5.  **AVOID EXPENSIVE OPERATIONS:** Do NOT ask for "all unique values", "complete lists", or queries that would return hundreds of rows.

---
**GOOD vs. BAD Plan Examples:**

**User Question:** "Compare hypertension in males vs females over 40"
**GOOD Plan:**
```json
[
    "Count male patients over 40",
    "Count male patients over 40 with hypertension",
    "Count female patients over 40",
    "Count female patients over 40 with hypertension"
]
```
*WHY IT'S GOOD: Each step is a simple count. The final comparison will be done later.*

**BAD Plan:**
```json
[
    "Calculate prevalence of hypertension in males over 40",
    "Calculate prevalence of hypertension in females over 40"
]
```
*WHY IT'S BAD: "Calculate prevalence" is a complex operation, not a simple data retrieval.*

---
**User Question:** "What medications are prescribed to diabetic patients?"
**GOOD Plan:**
```json
[
    "Count patients with diabetes",
    "Count most common medications prescribed to diabetic patients"
]
```
*WHY IT'S GOOD: Focuses on counts and aggregates, not exhaustive lists.*

**BAD Plan:**
```json
[
    "List all medications prescribed to diabetic patients",
    "List all diabetic patients and their medications"
]
```
*WHY IT'S BAD: "List all" operations are expensive and can timeout.*

---
**User Question:** "What is the rarest condition?"
**GOOD Plan:**
```json
[
    "Find the condition with the lowest patient count"
]
```
*WHY IT'S GOOD: Directly asks for the minimum, not a full list to sort through.*

**BAD Plan:**
```json
[
    "List all unique conditions found in patient records",
    "Count patients for each condition"
]
```
*WHY IT'S BAD: "List all unique conditions" is extremely expensive and will timeout.*

---
**User Question:** "How old is the youngest patient?"
**GOOD Plan:**
```json
[
    "Find the minimum age of all patients"
]
```
*WHY IT'S GOOD: Specific, efficient, directly answers the question.*

**BAD Plan:**
```json
[
    "Get all patient records"
]
```
*WHY IT'S BAD: Inefficient. Fetches unnecessary data and will time out. It is also unspecific and unethical to ask for all patient records. Your task is to be specific and ethical.*

---

**Now, generate a plan for the user's question below. Respond ONLY with a JSON list of strings inside a `json` markdown block.**
        """.strip()
        
        prompt = f"User Question: \"{user_question}\""
        
        response = await self.ollama_reason(prompt, system_prompt=system_prompt, include_tools=False)
        response_text = self._extract_summary_from_response(response) # Re-use for text extraction
        
        try:
            # A more robust way to find the JSON block
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if not json_match:
                # Fallback to finding any list-like structure
                json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)

            if json_match:
                plan_str = json_match.group(1)
                plan = json.loads(plan_str)
                if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                    if isinstance(self.world_model, OrchestratorWorldModel):
                        self.world_model.plan = plan
                    return Action(action_type="plan_generated", parameters={"plan": plan})
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode plan JSON: {e}\nRaw response was:\n{response_text}")
            pass # Fallback to error

        logger.error(f"Failed to generate a valid plan from response:\n{response_text}")
        return Action(action_type="error", parameters={"message": "Failed to generate a valid plan."})

    async def _synthesize_answer(self, original_query: str, executed_steps: List[Dict]) -> Action:
        """Synthesizes a final answer from the results of the executed plan."""
        system_prompt = """
You are a helpful and highly knowledgeable Clinical Data Analyst.
Your role is to synthesize the results from a series of sub-queries into a final, comprehensive answer for a clinical user.

**CRITICAL INSTRUCTIONS:**
1.  **Address the Original Question:** Your primary goal is to answer the user's original, complex question.
2.  **Use All the Data:** Incorporate the results from all the executed steps in your summary.
3.  **Show Your Work:** Briefly mention the results of the sub-questions to build a logical narrative.
4.  **Perform Final Calculations:** If the final step requires calculations (e.g., percentages, comparisons), do them.
5.  **Use Clear Formatting:** Use Markdown (bolding, bullet points) for readability.
6.  **Acknowledge Limitations:** If any sub-step failed or returned no data, mention this and how it impacts the final answer.
        """.strip()

        # Format the executed steps for the prompt
        context = f"Original Question: \"{original_query}\"\n\n"
        context += "Here is the data that was collected to answer the question:\n"
        for i, step in enumerate(executed_steps, 1):
            result_data = step['result']
            # Ensure we are working with a serializable dict
            if isinstance(result_data, dict):
                serializable_result = result_data
            elif hasattr(result_data, 'model_dump'):
                serializable_result = result_data.model_dump()
            else:
                serializable_result = str(result_data)
                
            context += f"- Step {i} ('{step['sub_question']}'): {json.dumps(serializable_result)}\n"
            
        prompt = context + "\nSynthesize a final, comprehensive answer based on this data."

        response = await self.ollama_reason(prompt, system_prompt=system_prompt, include_tools=False)
        summary = self._extract_summary_from_response(response)
        
        return Action(action_type="final_answer", parameters={"summary": summary})

    async def execute(self, action: Action) -> ActionResult:
        """Executes the action decided by the reason method."""

        if action.action_type == "plan_generated":
            # The plan is stored. The runner will now loop.
            return ActionResult(success=True, data={"message": "Plan created. Now executing."})

        if action.action_type == "delegate_to_omop_agent":
            try:
                print(f"[Orchestrator]  outgoing to OMOP Agent: {action.parameters}")
                response_message = await self.send_message_to_agent(
                    target_agent_id="omop_database_agent",
                    message=json.dumps(action.parameters)
                )
                
                print(f"[Orchestrator]  incoming from OMOP Agent: {response_message}")
                
                if response_message is None:
                    return ActionResult(success=False, error="No response from OMOP Agent.")

                if isinstance(response_message.root, SendMessageSuccessResponse):
                    response_data = json.loads(response_message.root.result.parts[0].root.text)
                elif isinstance(response_message.root, JSONRPCErrorResponse):
                    return ActionResult(success=False, error=f"OMOP Agent Error: {response_message.root.error.message}")
                else:
                    return ActionResult(success=False, error="Unexpected response type from OMOP Agent.")

                if "error" in response_data:
                    return ActionResult(success=False, error=response_data['error'])
                else:
                    omop_response = OMOPQueryResponse(**response_data)
                
                return ActionResult(success=True, data=omop_response.model_dump())

            except Exception as e:
                import traceback
                traceback.print_exc()
                return ActionResult(success=False, error=f"A2A communication failed: {str(e)}")

        if action.action_type == "final_answer":
            return ActionResult(success=True, data=action.parameters)

        if action.action_type == "error":
            return ActionResult(success=False, error=action.parameters.get("message", "An unknown error occurred in reasoning."))

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
            default_input_modes=[],
            default_output_modes=[],
        )

    def _extract_summary_from_response(self, ollama_response: Any) -> str:
        """Extract summary from Ollama response, handling different response formats."""
        summary = ""
        if isinstance(ollama_response, dict):
            if 'response' in ollama_response:
                summary = ollama_response['response']
            elif 'message' in ollama_response and isinstance(ollama_response['message'], dict):
                summary = ollama_response['message'].get('content', '')
            elif 'message' in ollama_response:
                summary = str(ollama_response['message'])
            elif 'content' in ollama_response:
                summary = ollama_response['content']
            else:
                summary = str(ollama_response)
        else:
            summary = str(ollama_response)
        
        summary = summary.strip()
        
        # Fallback if still empty
        if not summary:
            summary = "No summary available."
            
        return summary

    async def process_query(self, question: str) -> ActionResult:
        """
        The main entry point for the agent to handle a query from start to finish.
        This method contains the full control loop for planning, execution, and synthesis.
        """
        try:
            # 1. Perceive the initial question and update the world model
            observation = await self.perceive(question)
            await self.learn(self.mental_state, observation)

            # 2. This is the agent's main control loop
            max_loops = 10  # Safety break
            for _ in range(max_loops):
                action = await self.reason(self.mental_state)

                if action.action_type == "final_answer":
                    return await self.execute(action)

                if action.action_type == "error":
                    return ActionResult(success=False, error=action.parameters.get("message", "Reasoning error"))

                result = await self.execute(action)
                if not result.success:
                    return result # Propagate the execution error

                # Learn from the result of the executed step
                observation = await self.perceive(result.data)
                await self.learn(self.mental_state, observation)
            
            return ActionResult(success=False, error="Agent exceeded maximum execution loops.")

        except Exception as e:
            logger.error(f"An unhandled exception occurred in process_query: {e}", exc_info=True)
            return ActionResult(success=False, error=f"An internal agent error occurred: {e}")
