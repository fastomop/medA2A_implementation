
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
from ..prompts import get_prompt
from ..config import get_config
from ..prompts_omop_cdm import OMOP_CDM_CONTEXT
from ..cache import get_query_cache

logger = logging.getLogger(__name__)

class OrchestratorWorldModel(WorldModel):
    """World model for the orchestrator, now with planning capabilities and semantic context."""
    
    def __init__(self):
        super().__init__()
        self.original_query: Optional[str] = None
        self.semantic_context: Optional[Dict[str, Any]] = None
        self.plan: Optional[List[str]] = None
        self.executed_steps: List[Dict[str, Any]] = []
        
        # Initialize semantic analysis cache integration
        self.query_cache = get_query_cache()
        logger.info("Orchestrator World Model initialized with semantic caching")
        
    def update(self, observation: ProcessedObservation) -> None:
        """Update the world model with new observations."""
        if observation.source == "user_question":
            self.original_query = observation.data
            # Reset plan when a new question is asked
            self.semantic_context = None
            self.plan = None
            self.executed_steps = []
        elif observation.source == "semantic_agent_response":
            # Store semantic analysis results
            self.semantic_context = observation.data
        elif observation.source == "omop_agent_response":
            # Append sub-task results
            # Mark if this is the final answer step (last item in plan)
            is_final = len(self.plan) == 1 if self.plan else False
            self.executed_steps.append({
                "sub_question": self.plan.pop(0) if self.plan else "Unknown sub-question",
                "result": observation.data,
                "is_final_answer": is_final
            })
        self.last_updated = observation.timestamp
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Query the world model for information."""
        if "plan" in query:
            return self.plan
        elif "executed_steps" in query:
            return self.executed_steps
        elif "semantic_context" in query:
            return self.semantic_context
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
            "has_semantic_context": self.semantic_context is not None,
            "plan_steps_remaining": len(self.plan) if self.plan else 0,
            "executed_steps_count": len(self.executed_steps),
        })
        return base_summary
    
    def reset(self) -> None:
        """Reset the world model to initial state."""
        self.original_query = None
        self.semantic_context = None
        self.plan = None
        self.executed_steps = []
        super().__init__()
    
    # =================== SEMANTIC CACHING METHODS ===================
    
    def check_semantic_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we have cached semantic analysis for this query."""
        try:
            # Look for cached queries with semantic context
            cached_result = self.query_cache.get_cached_result(query)
            if cached_result and cached_result.semantic_context:
                logger.info(f"Semantic cache HIT: Found cached semantic analysis for: {query[:100]}...")
                return cached_result.semantic_context
            
            # Look for similar queries with semantic context
            similar_queries = self.query_cache.find_similar_queries(query, limit=2)
            for similar_query in similar_queries:
                if similar_query.semantic_context:
                    logger.info(f"Semantic cache SIMILAR: Using semantic analysis from similar query")
                    return similar_query.semantic_context
            
            logger.debug(f"Semantic cache MISS: No cached semantic analysis for: {query[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"Semantic cache lookup failed: {e}")
            return None
    
    def get_cached_semantic_analysis(self, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached semantic analysis for a query."""
        try:
            # Check if we have cached semantic analysis for this exact query
            cached_result = self.query_cache.get_cached_result(query)  # FIXED: Use correct method name
            if cached_result and cached_result.semantic_context:
                logger.info(f"Semantic cache HIT: Found cached semantic analysis")
                return cached_result.semantic_context
            
            # Also check similar queries for semantic context
            similar_queries = self.query_cache.find_similar_queries(query, limit=2)
            for similar_query in similar_queries:
                if similar_query.semantic_context:
                    logger.info(f"Semantic cache SIMILAR: Using semantic analysis from similar query")
                    return similar_query.semantic_context
            
            logger.debug(f"Semantic cache MISS: No cached semantic analysis")
            return None
        except Exception as e:
            logger.error(f"Semantic cache lookup failed: {e}")
            return None
    
    def cache_semantic_analysis(self, query: str, semantic_context: Dict[str, Any]):
        """Cache semantic analysis results."""
        try:
            # Cache only semantic analysis (no SQL or results)
            self.query_cache.cache_sql_template(
                query=query,
                sql_query=None,  # No SQL yet, just semantic analysis
                semantic_context=semantic_context,
                success=True,
                confidence_score=1.0,
                tags={"semantic_analysis", "orchestrator"}
            )
            logger.debug(f"Cached semantic analysis for: {query[:100]}...")
        except Exception as e:
            logger.error(f"Failed to cache semantic analysis: {e}")
    
    # REMOVED: check_complete_query_cache - we never cache complete results anymore
    
    def check_sql_template_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we have similar SQL queries that can serve as templates (NO RESULTS)."""
        try:
            # Look for similar queries with SQL that can guide generation
            similar_queries = self.query_cache.find_similar_queries(query, limit=3)
            for similar_query in similar_queries:
                if similar_query.sql_query:
                    logger.info(f"SQL template cache HIT: Found template from similar query")
                    return {
                        "template_sql": similar_query.sql_query,
                        "template_query": similar_query.original_query,
                        "semantic_context": similar_query.semantic_context,
                        "confidence_score": similar_query.confidence_score * 0.7,  # Lower confidence for template
                        "from_template": True,
                        "similarity_match": True
                    }
            
            logger.debug(f"SQL template cache MISS: No template for: {query[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"SQL template cache lookup failed: {e}")
            return None
    
    def cache_sql_template(self, original_query: str, 
                          sql_query: Optional[str] = None,
                          semantic_context: Optional[Dict[str, Any]] = None,
                          response_time: float = 0.0,
                          confidence_score: float = 0.9):
        """Cache only the successful SQL template for future generation assistance."""
        try:
            self.query_cache.cache_sql_template(
                query=original_query,
                sql_query=sql_query,           # Only SQL template (no results)
                semantic_context=semantic_context,
                success=True,
                response_time=response_time,
                confidence_score=confidence_score,
                tags={"sql_template", "orchestrator"}
            )
            logger.info(f"Cached SQL template: {original_query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to cache SQL template: {e}")

class OrchestratorAgent(OllamaReasoningMixin, MedicalAgent):
    """
    The main agent that orchestrates calls to the OMOPDatabaseAgent via A2A.
    """
    def __init__(self, agent_id: str, omop_agent_client: A2AClient, semantic_agent_client: Optional[A2AClient] = None, **kwargs):
        # Create world model for this agent
        world_model = OrchestratorWorldModel()
        
        # Get model from config
        config = get_config()
        model_name = kwargs.pop('model_name', config.ollama_model_name)

        super().__init__(
            agent_id=agent_id,
            agent_type="orchestrator",
            capabilities=["user_interaction", "agent_orchestration", "semantic_enhanced"],
            world_model=world_model,
            model_name=model_name,
            agent_name=f"Orchestrator-{agent_id}",
            agent_description="Medical orchestrator agent that coordinates between user, semantic analysis, and OMOP database agents",
            **kwargs
        )
        self.omop_agent_client = omop_agent_client
        self.semantic_agent_client = semantic_agent_client
        self.add_client("omop_database_agent", omop_agent_client)
        
        if semantic_agent_client:
            self.add_client("semantic_agent", semantic_agent_client)

        # Set agent-specific model and timeout for ollama calls
        self.ollama_model = config.orchestrator_model
        self._ollama_timeout = config.orchestrator_timeout

        # Explicitly type hint the world_model for the linter
        self.world_model: OrchestratorWorldModel = world_model
    
    # Note: Removed _filter_semantic_context_for_subquery and _is_co_occurrence_query methods
    # New approach: Planner focuses on natural language decomposition only.
    # Rich semantic context (concept codes, tables, etc.) gets passed to OMOP agent where it's useful.
    
    def _extract_numeric_from_result(self, result_list):
        """Extract numeric value from query result using the same logic as evaluation."""
        if not result_list or not isinstance(result_list, list) or len(result_list) == 0:
            return None
            
        first_row = result_list[0]
        if not isinstance(first_row, dict):
            return None
        
        # Strategy 1: Try common column names first
        common_names = ['patient_count', 'distinct_patient_count', 'distinct_person_count', 'distinct_patients', 'num_patients', 'num_persons', 'person_count', 'count', 'total', 'value', 'result']
        for key in common_names:
            if key in first_row:
                try:
                    return float(first_row[key]) if isinstance(first_row[key], str) else float(first_row[key])
                except (ValueError, TypeError):
                    continue
        
        # Strategy 2: If single column, use it
        if len(first_row) == 1:
            key, val = next(iter(first_row.items()))
            try:
                return float(val) if isinstance(val, str) else float(val)
            except (ValueError, TypeError):
                pass
        
        # Strategy 3: Find first numeric column 
        for key, val in first_row.items():
            try:
                numeric_val = float(val) if isinstance(val, str) else float(val)
                if numeric_val >= 0:
                    return numeric_val
            except (ValueError, TypeError):
                continue
                
        return None

    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Processes incoming data and identifies its source."""
        source = "unknown"
        data = observation
        if isinstance(observation, str):
            source = "user_question"
        elif isinstance(observation, dict):
            if "generated_sql" in observation:
                # This is a result from the OMOP Agent's execute() method
                source = "omop_agent_response"
            elif "semantic_context" in observation:
                # This is a result from the Semantic Agent's execute() method
                source = "semantic_agent_response"
        return ProcessedObservation(data=data, timestamp=0, source=source)

    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        """Updates the agent's world model and mental state based on new observations."""
        
        # Ensure world_model exists and is the correct type before using it
        if isinstance(self.world_model, OrchestratorWorldModel):
            # Update the world model first
            self.world_model.update(observation)

            # Populate mental state from the world model for the reason/execute cycle
            state.memory['original_query'] = self.world_model.original_query
            state.memory['semantic_context'] = self.world_model.semantic_context
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
        0. Perform semantic analysis if not done yet (new step)
        1. Generate a plan if none exists.
        2. Execute the next step of an existing plan.
        3. Synthesize a final answer if the plan is complete.
        """
        original_query = state.memory.get('original_query')
        semantic_context = state.memory.get('semantic_context')
        plan = state.memory.get('plan')
        executed_steps = state.memory.get('executed_steps', [])

        if not original_query:
            return Action(action_type="error", parameters={"message": "No user query found."})

        # REMOVED: Complete result caching - we only cache SQL templates now

        # Step 0: Get semantic analysis for the ORIGINAL query if not done yet
        if semantic_context is None and self.semantic_agent_client and original_query:
            # Check if we already have it cached
            cached_semantic = self.world_model.get_cached_semantic_analysis(original_query)
            if cached_semantic:
                print("[Orchestrator] ⚡ Using cached semantic analysis for original query")
                semantic_context = cached_semantic
                state.memory['semantic_context'] = semantic_context
            else:
                print("[Orchestrator] 🔍 Getting semantic analysis for original query...")
                return Action(
                    action_type="delegate_to_semantic_agent",
                    parameters={"question": original_query}
                )

        # Scenario 1: No plan exists yet. Generate a simple plan.
        if plan is None and not executed_steps:
            print("[Orchestrator] 🧠 Phase 1: Generating a simple plan...")
            return await self._generate_plan(original_query, semantic_context=semantic_context)

        # Scenario 2: Plan exists and has steps remaining. Execute the next step.
        if plan and len(plan) > 0:
            next_sub_question = plan[0]
            print(f"[Orchestrator] 🏃 Phase 2: Executing next step -> '{next_sub_question}'")
            
            # CRITICAL: Get fresh semantic analysis for each sub-query
            print(f"[Orchestrator] 🧠 SUB-QUERY EXECUTION: Getting independent semantic analysis...")
            print(f"[Orchestrator] 📋 CONTEXT SCOPE: Sub-query will get fresh semantic analysis (no inheritance)")
            sub_query_semantic_result = await self._get_semantic_analysis(next_sub_question)
            
            if sub_query_semantic_result:
                print(f"[Orchestrator] ✅ Fresh semantic context obtained for sub-query")
                # Store the sub-query specific context  
                state.memory[f'semantic_{next_sub_question}'] = sub_query_semantic_result
                self.mental_state.memory[f'semantic_{next_sub_question}'] = sub_query_semantic_result
                
                # Log sub-query classification vs parent
                sub_intent = sub_query_semantic_result.get('query_intent', {})
                parent_intent = semantic_context.get('query_intent', {}) if semantic_context else {}
                
                print(f"[Orchestrator] 📊 SUB-QUERY classified as: {sub_intent.get('type', 'unknown')} ({sub_intent.get('complexity', 'unknown')})")
                print(f"[Orchestrator] 📊 PARENT QUERY was: {parent_intent.get('type', 'unknown')} ({parent_intent.get('complexity', 'unknown')})")
                print(f"[Orchestrator] 📋 CONTEXT SCOPE: Sub-query gets its own classification and concept mapping")
                
                # Execute step with sub-query's own semantic context
                request_data = OMOPQueryRequest(
                    question=next_sub_question,
                    semantic_context=sub_query_semantic_result
                )
            else:
                print(f"[Orchestrator] ⚠️ Failed to get semantic analysis for sub-query, proceeding without context")
                print(f"[Orchestrator] 📋 CONTEXT SCOPE: Sub-query will execute without semantic context")
                request_data = OMOPQueryRequest(question=next_sub_question)
            
            return Action(
                action_type="delegate_to_omop_agent",
                parameters=request_data.model_dump()
            )

        # Scenario 3: Plan is complete. Synthesize the final answer.
        if plan is not None and len(plan) == 0 and executed_steps:
            print("[Orchestrator] 💡 Phase 3: Synthesizing final answer...")
            return await self._synthesize_answer(original_query, executed_steps, semantic_context)

        # Default or error case
        return Action(action_type="error", parameters={"message": "Orchestrator is in an inconsistent state."})

    async def _generate_plan(self, user_question: str, semantic_context: Optional[Dict[str, Any]] = None) -> Action:
        """Generates a plan based on semantic query classification - respects query intent."""
        
        # CRITICAL: Check semantic classification first
        if semantic_context:
            # Handle nested semantic context structure
            actual_context = semantic_context
            if 'semantic_context' in semantic_context:
                # Unwrap nested structure returned by semantic agent
                actual_context = semantic_context['semantic_context']
            
            query_intent = actual_context.get('query_intent', {})
            query_type = query_intent.get('type', 'unknown')
            query_complexity = query_intent.get('complexity', 'unknown')
            
            print(f"[Orchestrator] 🧠 PARENT QUERY classified as: {query_type} ({query_complexity})")
            print(f"[Orchestrator] 📋 CONTEXT SCOPE: Using parent semantic context for planning decisions only")
            
            # Respect semantic classification for planning decisions
            if query_type in ['simple_count', 'co_occurrence', 'temporal_analysis']:
                print(f"[Orchestrator] ✅ Single-operation query detected - no decomposition needed")
                print(f"[Orchestrator] 📋 CONTEXT SCOPE: Parent context will be used for execution")
                return Action(
                    action_type="execute_plan", 
                    parameters={"plan": [user_question]}
                )
            elif query_type == 'hierarchical_search' and query_complexity in ['simple', 'moderate']:
                print(f"[Orchestrator] ✅ Simple hierarchical query - executing as single operation")
                print(f"[Orchestrator] 📋 CONTEXT SCOPE: Parent context will be used for execution")
                return Action(
                    action_type="execute_plan", 
                    parameters={"plan": [user_question]}
                )
            elif query_complexity == 'complex' or 'multi_step' in query_type:
                print(f"[Orchestrator] 🧠 Complex multi-step query detected - proceeding with decomposition")
                print(f"[Orchestrator] 📋 CONTEXT SCOPE: Sub-queries will get independent semantic analysis")
                # Continue with decomposition logic below
            else:
                print(f"[Orchestrator] ✅ Fallback to direct execution")
                print(f"[Orchestrator] 📋 CONTEXT SCOPE: Parent context will be used for execution")
                return Action(
                    action_type="execute_plan", 
                    parameters={"plan": [user_question]}
                )
        
        # Only reach here for complex multi-step queries that need decomposition
        system_prompt = get_prompt("orchestrator", "planner")
        
        print("[Orchestrator] 🧠 Generating plan using natural language decomposition for complex query")
        prompt = f"User Question: \"{user_question}\""
        prompt += "\n\nThis is a complex query that needs decomposition into logical sub-questions."
        prompt += "\nDecompose this medical query into meaningful sub-questions."
        prompt += "\nFocus on the natural language structure and what information is actually needed."
        prompt += "\nThe semantic analysis and medical concept mapping will be handled later."
        
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
                # Clean up common JSON formatting issues
                plan_str = re.sub(r',\s*]', ']', plan_str)  # Remove trailing commas
                plan_str = re.sub(r',\s*}', '}', plan_str)  # Remove trailing commas in objects
                
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

    async def _synthesize_answer(self, original_query: str, executed_steps: List[Dict], semantic_context: Optional[Dict[str, Any]] = None) -> Action:
        """Synthesizes a final answer from the results of the executed plan."""
        system_prompt = get_prompt("orchestrator", "synthesizer")

        # Format the executed steps for the prompt
        context = f"Original Question: \"{original_query}\"\n\n"
        
        if semantic_context:
            context += f"Semantic Analysis:\n{json.dumps(semantic_context, indent=2)}\n\n"
        
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

        # REMOVED: answer_complete handling - we never cache complete results anymore

        if action.action_type == "plan_generated":
            # The plan is stored. The runner will now loop.
            return ActionResult(success=True, data={"message": "Plan created. Now executing."})

        if action.action_type == "delegate_to_semantic_agent":
            try:
                print(f"[Orchestrator] → outgoing to Semantic Agent: {action.parameters}")
                response_message = await self.send_message_to_agent(
                    target_agent_id="semantic_agent",
                    message=json.dumps(action.parameters)
                )
                
                print(f"[Orchestrator] ← incoming from Semantic Agent: {response_message}")
                
                if response_message is None:
                    return ActionResult(success=False, error="No response from Semantic Agent.")

                if isinstance(response_message.root, SendMessageSuccessResponse):
                    response_data = json.loads(response_message.root.result.parts[0].root.text)
                elif isinstance(response_message.root, JSONRPCErrorResponse):
                    return ActionResult(success=False, error=f"Semantic Agent Error: {response_message.root.error.message}")
                else:
                    return ActionResult(success=False, error="Unexpected response type from Semantic Agent.")

                if "error" in response_data:
                    return ActionResult(success=False, error=response_data['error'])
                else:
                    # Store semantic context for the original query
                    if 'semantic_context' in response_data:
                        original_query = action.parameters.get('question', '')
                        
                        # Store in mental state as the main semantic context
                        self.mental_state.memory['semantic_context'] = response_data['semantic_context']
                        
                        # Also cache for future identical queries
                        self.world_model.cache_semantic_analysis(original_query, response_data['semantic_context'])
                        print(f"[Orchestrator] ✅ Cached semantic analysis for original query")
                    
                    return ActionResult(success=True, data=response_data)

            except Exception as e:
                import traceback
                traceback.print_exc()
                return ActionResult(success=False, error=f"Semantic Agent communication failed: {str(e)}")

        if action.action_type == "delegate_to_omop_agent":
            try:
                # Include semantic context in the message to OMOP agent
                enhanced_params = action.parameters.copy()
                
                # Get the full semantic context for this sub-query (simplified approach)
                sub_query = enhanced_params.get('question', '')
                semantic_context_for_query = self.mental_state.memory.get(f'semantic_{sub_query}')
                
                if semantic_context_for_query:
                    # Handle nested semantic context structure
                    if isinstance(semantic_context_for_query, dict) and 'semantic_context' in semantic_context_for_query:
                        # Unwrap nested structure
                        actual_context = semantic_context_for_query['semantic_context']
                        enhanced_params["semantic_context"] = actual_context
                        analysis_type = actual_context.get('analysis_type', 'unknown')
                        num_concepts = len(actual_context.get('medical_concepts', []))
                    else:
                        # Use as-is if not nested
                        enhanced_params["semantic_context"] = semantic_context_for_query
                        analysis_type = semantic_context_for_query.get('analysis_type', 'unknown')
                        num_concepts = len(semantic_context_for_query.get('medical_concepts', []))
                    
                    print(f"[Orchestrator] 🧠 Including full semantic context: {analysis_type} ({num_concepts} concepts)")
                else:
                    print(f"[Orchestrator] ⚠️ No semantic context found for query: {sub_query[:50]}...")
                
                # Include SQL template guidance if available
                if sql_template := self.mental_state.memory.get('sql_template'):
                    enhanced_params["sql_template"] = sql_template
                    print(f"[Orchestrator] 📝 Including SQL template guidance from: {sql_template['template_query'][:50]}...")
                
                print(f"[Orchestrator]  outgoing to OMOP Agent: {enhanced_params}")
                response_message = await self.send_message_to_agent(
                    target_agent_id="omop_database_agent",
                    message=json.dumps(enhanced_params)
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
            # Cache the complete query result for future use
            try:
                original_query = self.mental_state.memory.get('original_query')
                executed_steps = self.mental_state.memory.get('executed_steps', [])
                semantic_context = self.mental_state.memory.get('semantic_context')
                
                # Extract SQL query and final result from executed steps
                sql_query = None
                final_result = None
                
                if executed_steps:
                    # Get the last executed step's SQL and result
                    last_step = executed_steps[-1]
                    step_result = last_step.get('result')
                    
                    # Check if result contains SQL in OMOP agent format
                    if isinstance(step_result, dict):
                        if 'generated_sql' in step_result:
                            sql_query = step_result['generated_sql']
                        if 'query_result' in step_result:
                            final_result = step_result['query_result']
                    
                    # If multiple steps, combine their SQL information
                    if len(executed_steps) > 1:
                        sql_queries = []
                        for step in executed_steps:
                            step_result = step.get('result', {})
                            if isinstance(step_result, dict) and 'generated_sql' in step_result:
                                sql_queries.append(step_result['generated_sql'])
                        
                        if sql_queries:
                            sql_query = "; ".join(sql_queries)
                
                if original_query and sql_query:
                    self.world_model.cache_sql_template(
                        original_query=original_query,
                        sql_query=sql_query,
                        # REMOVED: result=final_result - never cache results
                        semantic_context=semantic_context,
                        response_time=0.0,  # Could be calculated if needed
                        confidence_score=0.9
                    )
                    print(f"[Orchestrator] 💾 Cached SQL template for future query generation")
                    
            except Exception as e:
                logger.error(f"Failed to cache SQL template: {e}")
            
            return ActionResult(success=True, data=action.parameters)

        if action.action_type == "execute_plan":
            """Handle direct execution of simple queries without decomposition."""
            try:
                plan = action.parameters.get("plan", [])
                if not plan:
                    return ActionResult(success=False, error="No plan provided for execution")
                
                # For execute_plan, we execute the single query directly
                query_to_execute = plan[0]
                print(f"[Orchestrator] 🎯 Direct execution: {query_to_execute}")
                
                # Set up proper state management for direct execution
                if isinstance(self.world_model, OrchestratorWorldModel):
                    self.world_model.plan = []  # Mark as complete
                    
                # Use the parent semantic context that was used for classification
                semantic_context = self.mental_state.memory.get('semantic_context')
                
                if semantic_context:
                    # Handle nested semantic context structure
                    actual_context = semantic_context
                    if 'semantic_context' in semantic_context:
                        actual_context = semantic_context['semantic_context']
                        
                    print(f"[Orchestrator] 🧠 Using parent semantic context for direct execution")
                    request_data = OMOPQueryRequest(
                        question=query_to_execute,
                        semantic_context=actual_context  # Use unwrapped context
                    )
                else:
                    print(f"[Orchestrator] ⚠️ No semantic context available for direct execution")
                    request_data = OMOPQueryRequest(question=query_to_execute)
                
                # Execute directly through OMOP agent
                omop_result = await self.execute(Action(
                    action_type="delegate_to_omop_agent",
                    parameters=request_data.model_dump()
                ))
                
                if omop_result.success:
                    # Update world model with the executed step
                    executed_step = {
                        'sub_question': query_to_execute,
                        'result': omop_result.data
                    }
                    
                    if isinstance(self.world_model, OrchestratorWorldModel):
                        self.world_model.executed_steps = [executed_step]
                    
                    self.mental_state.memory['executed_steps'] = [executed_step]
                    self.mental_state.memory['plan'] = []  # Empty plan = complete
                    
                    print(f"[Orchestrator] ✅ Direct execution completed, ready for synthesis")
                    
                return omop_result
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ActionResult(success=False, error=f"Execute plan failed: {str(e)}")

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
            # CRITICAL: Reset world model state for each new query to prevent state corruption
            if isinstance(self.world_model, OrchestratorWorldModel):
                self.world_model.reset()
                print(f"[Orchestrator] 🔄 Reset world model state for new query")
            
            # 1. Perceive the initial question and update the world model  
            # Note: No more hardcoded enhancement - semantic agent will handle analysis
            observation = await self.perceive(question)
            await self.learn(self.mental_state, observation)

            # 2. This is the agent's main control loop
            max_loops = 10  # Safety break
            for _ in range(max_loops):
                action = await self.reason(self.mental_state)

                if action.action_type == "final_answer":
                    return await self.execute(action)
                
                # REMOVED: answer_complete handling - we never cache complete results

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
