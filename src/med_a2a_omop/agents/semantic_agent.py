"""
Semantic Agent for Medical A2A OMOP System.

This agent handles semantic analysis and medical terminology standardization:
- Extracts semantic meaning from user queries
- Converts medical abbreviations and synonyms to standardized terms
- Maps user terms to OMOP vocabulary concepts
- Provides structured semantic context for SQL generation
"""

import json
import re
import time
from typing import List, Dict, Any, Optional
import logging

from a2a.types import AgentCard, AgentCapabilities
from a2a.client import A2AClient
from a2a_medical.base.agent import MedicalAgent, ProcessedObservation, Action, ActionResult, MentalState, WorldModel
from a2a_medical.integrations.ollama import OllamaReasoningMixin

from ..models.a2a_messages import SemanticAnalysisRequest, SemanticAnalysisResponse
from ..prompts import get_prompt
from ..config import get_config
from ..vocabulary_fast import get_fast_vocabulary, FastMatch

logger = logging.getLogger(__name__)

class SemanticWorldModel(WorldModel):
    """
    Enhanced semantic world model that provides structured semantic analysis
    and learns SQL patterns for semantic attributes.
    """
    
    def __init__(self):
        super().__init__()
        self.original_query: Optional[str] = None
        self.extracted_terms: List[Dict[str, Any]] = []
        self.omop_mappings: List[Dict[str, Any]] = []
        self.semantic_context: Optional[Dict[str, Any]] = None
        
        # New: SQL pattern learning system
        self.semantic_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.successful_sql_templates: Dict[str, Dict[str, Any]] = {}
        self.semantic_cache: Dict[str, Dict[str, Any]] = {}
        
        # CRITICAL: Concept code learning system
        self.drug_concept_codes: Dict[str, str] = {}  # drug_name -> concept_code
        self.concept_mappings: Dict[str, List[Dict[str, Any]]] = {}  # concept_code -> related concepts
        self.learned_sql_patterns: Dict[str, Dict[str, Any]] = {}  # pattern_signature -> successful SQL template
        
    def update(self, observation: ProcessedObservation) -> None:
        """Update the world model with new semantic observations."""
        if observation.source == "user_query":
            self.original_query = observation.data
            # Reset semantic state for new query
            self.extracted_terms = []
            self.omop_mappings = []
            self.semantic_context = None
        elif observation.source == "semantic_analysis":
            self.semantic_context = observation.data
        self.last_updated = observation.timestamp
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Query the semantic world model for information."""
        if "semantic_context" in query:
            return self.semantic_context
        elif "extracted_terms" in query:
            return self.extracted_terms
        elif "omop_mappings" in query:
            return self.omop_mappings
        return None
    
    def predict(self, scenario: Dict[str, Any]) -> Any:
        """Predict semantic analysis outcomes."""
        if "term_mapping_confidence" in scenario:
            # Return confidence based on number of successful mappings
            if self.omop_mappings:
                return {"confidence": min(0.9, len(self.omop_mappings) / 5)}
            return {"confidence": 0.5}
        return None
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the semantic world model state."""
        base_summary = super().get_state_summary()
        base_summary.update({
            "original_query": self.original_query,
            "extracted_terms_count": len(self.extracted_terms),
            "omop_mappings_count": len(self.omop_mappings),
            "has_semantic_context": self.semantic_context is not None,
        })
        return base_summary
    
    def reset(self) -> None:
        """Reset the semantic world model to initial state."""
        self.original_query = None
        self.extracted_terms = []
        self.omop_mappings = []
        self.semantic_context = None
        super().__init__()
    
    def get_semantic_signature(self, semantic_context: Dict[str, Any]) -> str:
        """Create a signature for semantic patterns to enable pattern matching."""
        if not semantic_context:
            return "unknown"
        
        # Create signature from semantic attributes
        domains = sorted(semantic_context.get('omop_domains', []))
        analysis_type = semantic_context.get('analysis_type', 'unknown')
        has_temporal = bool(semantic_context.get('temporal_relationships'))
        
        return f"{'+'.join(domains)}:{analysis_type}:temporal={has_temporal}"
    
    def learn_sql_pattern(self, semantic_context: Dict[str, Any], sql_query: str, success: bool):
        """Learn SQL patterns for specific semantic attributes."""
        if not success or not semantic_context:
            return
            
        signature = self.get_semantic_signature(semantic_context)
        
        if signature not in self.semantic_patterns:
            self.semantic_patterns[signature] = []
        
        pattern_entry = {
            "sql_template": sql_query,
            "semantic_context": semantic_context,
            "success_count": 1,
            "timestamp": time.time()
        }
        
        # Check if we already have a similar pattern
        for existing in self.semantic_patterns[signature]:
            if self._sql_similarity(existing["sql_template"], sql_query) > 0.8:
                existing["success_count"] += 1
                return
        
        self.semantic_patterns[signature].append(pattern_entry)
        logger.info(f"Learned new SQL pattern for signature: {signature}")
    
    def get_similar_patterns(self, semantic_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get similar successful SQL patterns for given semantic context."""
        signature = self.get_semantic_signature(semantic_context)
        
        # Direct match
        if signature in self.semantic_patterns:
            return sorted(
                self.semantic_patterns[signature], 
                key=lambda x: x["success_count"], 
                reverse=True
            )[:3]  # Top 3 patterns
        
        # Fuzzy match for similar signatures
        similar_patterns = []
        for sig, patterns in self.semantic_patterns.items():
            if self._signature_similarity(signature, sig) > 0.6:
                similar_patterns.extend(patterns)
        
        return sorted(similar_patterns, key=lambda x: x["success_count"], reverse=True)[:2]
    
    def learn_concept_code(self, drug_name: str, concept_code: str, success: bool = True):
        """Learn concept code mappings from successful queries."""
        if success and drug_name and concept_code:
            normalized_name = drug_name.lower().strip()
            self.drug_concept_codes[normalized_name] = concept_code
            logger.info(f"Learned concept code: '{normalized_name}' -> '{concept_code}'")
    
    def get_concept_code_suggestion(self, drug_name: str) -> Optional[str]:
        """Get learned concept code for a drug name (no hallucinations)."""
        # Safety check: only return codes if we have real learned data
        if not self.drug_concept_codes:
            return None
            
        normalized_name = drug_name.lower().strip()
        
        # Direct match
        if normalized_name in self.drug_concept_codes:
            code = self.drug_concept_codes[normalized_name]
            # Don't return obviously invalid codes
            if code and code != 'unknown' and code.isdigit():
                return code
        
        # Fuzzy match (more restrictive)
        for learned_name, code in self.drug_concept_codes.items():
            if code and code != 'unknown' and code.isdigit():
                if (normalized_name in learned_name or learned_name in normalized_name) and len(normalized_name) > 3:
                    return code
        
        return None
    
    def learn_sql_pattern_from_dataset(self, sql: str, drug_terms: List[str]):
        """Learn patterns from dataset SQL queries."""
        try:
            # Extract concept codes from SQL
            import re
            concept_codes = re.findall(r"concept_code = '(\w+)'", sql)
            
            # Map drug terms to concept codes
            for i, drug_term in enumerate(drug_terms):
                if i < len(concept_codes):
                    self.learn_concept_code(drug_term, concept_codes[i])
                    
        except Exception as e:
            logger.warning(f"Failed to learn from dataset SQL: {e}")
    
    def _sql_similarity(self, sql1: str, sql2: str) -> float:
        """Calculate similarity between two SQL queries (simple approach)."""
        sql1_norm = ' '.join(sql1.lower().split())
        sql2_norm = ' '.join(sql2.lower().split())
        
        # Simple token-based similarity
        tokens1 = set(sql1_norm.split())
        tokens2 = set(sql2_norm.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between semantic signatures."""
        parts1 = sig1.split(':')
        parts2 = sig2.split(':')
        
        if len(parts1) != len(parts2):
            return 0.0
        
        matches = sum(1 for p1, p2 in zip(parts1, parts2) if p1 == p2)
        return matches / len(parts1)

class SemanticAgent(OllamaReasoningMixin, MedicalAgent):
    """
    Semantic Agent that handles medical terminology analysis and OMOP concept mapping.
    
    This agent sits between the orchestrator and OMOP database agent to:
    1. Analyze user queries for medical terms and concepts
    2. Standardize medical abbreviations and synonyms
    3. Map terms to OMOP vocabulary concepts
    4. Provide structured semantic context for better SQL generation
    """
    
    def __init__(self, agent_id: str, omop_agent_client: A2AClient, **kwargs):
        # Create semantic world model
        world_model = SemanticWorldModel()
        
        # Get model from config
        config = get_config()
        model_name = kwargs.pop('model_name', config.ollama_model_name)

        super().__init__(
            agent_id=agent_id,
            agent_type="semantic_analyzer",
            capabilities=["semantic_analysis", "medical_terminology", "omop_mapping"],
            world_model=world_model,
            model_name=model_name,
            agent_name=f"Semantic-{agent_id}",
            agent_description="Medical semantic analysis agent for OMOP concept mapping and terminology standardization",
            **kwargs
        )
        self.omop_agent_client = omop_agent_client
        self.add_client("omop_database_agent", omop_agent_client)
        
        # Note: We'll override on_message_send instead of registering a handler
        # because the orchestrator uses send_message_to_agent without a specific message type

        # Set agent-specific model and timeout for ollama calls
        self.ollama_model = config.semantic_model
        self._ollama_timeout = config.semantic_timeout

        # Explicitly type hint the world_model
        self.world_model: SemanticWorldModel = world_model
        
        # Initialize fast vocabulary index lazily
        self.vocabulary_index = None
        self._vocabulary_initialized = False
        logger.info("Semantic agent initialized (fast vocabulary will load on first use)")

    async def perceive(self, observation: Any) -> ProcessedObservation:
        """Process incoming data and identify its source."""
        source = "unknown"
        data = observation
        
        if isinstance(observation, str):
            source = "user_query"
        elif isinstance(observation, dict):
            if "semantic_analysis" in observation:
                source = "semantic_analysis"
            elif "omop_concepts" in observation:
                source = "omop_mapping"
        
        return ProcessedObservation(data=data, timestamp=0, source=source)

    async def learn(self, state: MentalState, observation: ProcessedObservation) -> MentalState:
        """Update the agent's world model and mental state based on observations."""
        
        if isinstance(self.world_model, SemanticWorldModel):
            # Update the world model first
            self.world_model.update(observation)

            # Populate mental state from the world model
            state.memory['original_query'] = self.world_model.original_query
            state.memory['extracted_terms'] = self.world_model.extracted_terms
            state.memory['omop_mappings'] = self.world_model.omop_mappings
            state.memory['semantic_context'] = self.world_model.semantic_context
        else:
            # Fallback for basic memory
            if observation.source == "user_query":
                state.memory["original_query"] = observation.data
        
        return state

    async def reason(self, state: MentalState) -> Action:
        """
        Reason about the current state to decide the next semantic analysis action.
        """
        original_query = state.memory.get('original_query')
        semantic_context = state.memory.get('semantic_context')

        if not original_query:
            return Action(action_type="error", parameters={"message": "No user query found for semantic analysis."})

        # If we don't have semantic context yet, analyze the query
        if semantic_context is None:
            print("[Semantic Agent] 🧠 Analyzing query semantics...")
            return await self._analyze_semantics(original_query)

        # If we have semantic context, we're done
        return Action(action_type="semantic_complete", parameters={"semantic_context": semantic_context})

    async def _analyze_semantics(self, user_query: str) -> Action:
        """
        Enhanced semantic analysis that reasons about medical concepts
        and determines appropriate OMOP domains and query types.
        """
        # Check cache first
        if user_query in self.world_model.semantic_cache:
            cached_analysis = self.world_model.semantic_cache[user_query]
            print("[Semantic Agent] ⚡ Cache HIT for semantic analysis")
            
            # CRITICAL: Always enhance cached results too (in case enhancement was added after caching)
            self._enhance_with_concept_codes(cached_analysis)
            return Action(
                action_type="semantic_analysis_complete",
                parameters={"semantic_context": cached_analysis}
            )
        
        system_prompt = get_prompt("semantic_agent", "analyzer")
        
        prompt = f"""
Analyze this medical query using semantic reasoning (not pattern matching):
"{user_query}"

Think step by step:

1. MEDICAL CONCEPT IDENTIFICATION:
   - What medical concepts are mentioned? (conditions, drugs, procedures, measurements)
   - Are these abbreviations that need expansion? (HTN→Hypertension, T2DM→Type 2 Diabetes)
   - What OMOP domains do these belong to? (Condition, Drug, Measurement, Person, etc.)

2. SEMANTIC RELATIONSHIP ANALYSIS:
   - What is the relationship between concepts? (co-occurrence, causation, comparison)
   - Are there temporal constraints? (within X days, before/after, concurrent)
   - Is this about multiple entities together or separately?

3. QUERY INTENT UNDERSTANDING:
   - What does the user want to know? (count, prevalence, comparison, trend)
   - What type of analysis is needed? (descriptive stats, co-occurrence, interaction)
   - Query complexity: simple count vs. multi-concept analysis vs. temporal patterns
   - CRITICAL: If query mentions drug families/classes (like "diabetes medications", "antibiotics", "any X drugs"), classify as hierarchical_search

4. OMOP MAPPING:
   - Which OMOP tables would contain this information?
   - What joins would be needed?
   - Are there any domain-specific considerations?

IMPORTANT: For drug names, PRESERVE ORIGINAL SPECIFICITY when possible. 
If the original term includes specific dosage, formulation, or route information, maintain that detail in the standardized_term.
Only fall back to generic drug names if the specific formulation cannot be found in vocabulary.

QUERY CLASSIFICATION EXAMPLES:
- "Count patients taking amlodipine 2.5 MG" → simple_count (specific drug)
- "Count patients taking diabetes medications" → hierarchical_search (drug family) 
- "Count patients taking both X and Y" → co_occurrence (multiple drugs)
- "Count patients taking X within 30 days of Y" → temporal_analysis (time constraints)

Respond with VALID JSON (no trailing commas, proper quotes):
```json
{{
  "medical_concepts": [
    {{
      "original_term": "exact term from query",
      "core_drug_name": "main drug ingredient (for drugs only)",
      "standardized_term": "preserve original specificity - use exact term if specific, or generic if original was generic",
      "concept_type": "condition|drug|procedure|measurement|demographic",
      "omop_domain": "Condition|Drug|Procedure|Measurement|Person|Observation",
      "search_strategy": "exact|contains|flexible",
      "confidence": 0.9
    }}
  ],
  "temporal_relationships": [
    {{
      "type": "within_period|concurrent|sequential|age_constraint",
      "constraint": "specific constraint (e.g., '365 days', 'same_visit')",
      "entities_involved": ["entity1", "entity2"],
      "description": "clear description of temporal requirement"
    }}
  ],
  "analysis_type": "count|prevalence|co_occurrence|demographics|trend|comparison|interaction",
  "query_intent": {{
    "type": "simple_count|hierarchical_search|co_occurrence|temporal_analysis|demographic_analysis",
    "complexity": "simple|moderate|complex",
    "sql_strategy": "direct_lookup|hierarchy_search|multi_table_join|temporal_join",
    "reasoning": "brief explanation of classification"
  }},
  "omop_domains": ["primary domains needed"],
  "suggested_tables": ["likely OMOP tables to use"],
  "semantic_signature": "brief signature for pattern learning"
}}
```

Be precise about medical terminology and OMOP domain classification.
        """.strip()
        
        try:
            response = await self.ollama_reason(prompt, system_prompt=system_prompt, include_tools=False)
            response_text = self._extract_summary_from_response(response)
            
            # Extract JSON from response with better error handling
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if not json_match:
                # Try to find JSON without markdown
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                
                # Clean up common JSON issues
                # Remove trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                # Fix unquoted keys (simple cases)
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                # Remove any BOM or special characters
                json_str = json_str.replace('\ufeff', '').replace('\u200b', '')
                
                try:
                    semantic_analysis = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode failed, attempting repair: {e}")
                    # Try one more time with aggressive cleanup
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                    semantic_analysis = json.loads(json_str)
                
                # CRITICAL: Enhance with learned concept codes
                self._enhance_with_concept_codes(semantic_analysis)
                
                # Store in world model and cache
                if isinstance(self.world_model, SemanticWorldModel):
                    self.world_model.semantic_context = semantic_analysis
                    self.world_model.semantic_cache[user_query] = semantic_analysis
                    print(f"[Semantic Agent] ✅ Cached semantic analysis for: {user_query[:100]}...")
                
                return Action(
                    action_type="semantic_analysis_complete",
                    parameters={"semantic_context": semantic_analysis}
                )
            else:
                logger.error(f"Failed to extract JSON from semantic analysis: {response_text}")
                return Action(
                    action_type="error",
                    parameters={"message": "Failed to parse semantic analysis response"}
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode semantic analysis JSON: {e}")
            return Action(
                action_type="error",
                parameters={"message": f"Invalid JSON in semantic analysis: {e}"}
            )
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return Action(
                action_type="error",
                parameters={"message": f"Semantic analysis error: {e}"}
            )
    
    def _enhance_with_concept_codes(self, semantic_analysis: Dict[str, Any]):
        """Enhance semantic analysis with OMOP vocabulary concept codes using smart selection."""
        try:
            if 'medical_concepts' in semantic_analysis:
                for concept in semantic_analysis['medical_concepts']:
                    original_term = concept.get('original_term', '')
                    concept_type = concept.get('concept_type', '')
                    
                    # Search vocabulary for matching concepts using smart strategy
                    best_match = self._find_best_concept_match(original_term, concept_type, concept)
                    
                    if best_match:
                        concept['rxnorm_concept_code'] = best_match.concept.concept_code
                        concept['vocabulary_id'] = best_match.concept.vocabulary_id
                        concept['concept_id'] = best_match.concept.concept_id
                        concept['search_strategy'] = 'vocabulary_lookup'
                        concept['match_confidence'] = best_match.confidence
                        concept['matched_name'] = best_match.concept.concept_name
                        
                        print(f"[Semantic Agent] 🎯 Enhanced '{original_term}' with concept:")
                        print(f"  - Code: {best_match.concept.concept_code}")
                        print(f"  - Name: {best_match.concept.concept_name}")
                        print(f"  - Vocabulary: {best_match.concept.vocabulary_id}")
                        print(f"  - Confidence: {best_match.confidence:.2f}")
                    else:
                        # Fallback to core drug name and variations
                        if concept_type == 'drug':
                            # Try different variations of the drug name
                            drug_variations = [
                                concept.get('core_drug_name', ''),
                                concept.get('standardized_term', ''),
                                original_term.split()[0] if original_term else '',  # First word
                            ]
                            
                            fallback_matches = None
                            for variation in drug_variations:
                                if variation and len(variation) > 2:
                                    fallback_matches = self._search_vocabulary_for_concept(variation, concept_type)
                                    if fallback_matches:
                                        print(f"[Semantic Agent] 🔄 Found '{original_term}' via fallback search: '{variation}'")
                                        # Use smart selection for fallback matches too
                                        best_fallback = self._select_best_drug_concept(fallback_matches)
                                        if best_fallback:
                                            concept['rxnorm_concept_code'] = best_fallback.concept.concept_code
                                            concept['vocabulary_id'] = best_fallback.concept.vocabulary_id
                                            concept['concept_id'] = best_fallback.concept.concept_id
                                            concept['search_strategy'] = 'vocabulary_fallback'
                                            concept['match_confidence'] = best_fallback.confidence
                                            concept['matched_name'] = best_fallback.concept.concept_name
                                            break
                            
                            # Last resort: check learned patterns (but don't hallucinate)
                            if 'rxnorm_concept_code' not in concept and isinstance(self.world_model, SemanticWorldModel):
                                core_drug = concept.get('core_drug_name') or concept.get('standardized_term')
                                if core_drug:
                                    learned_code = self.world_model.get_concept_code_suggestion(core_drug)
                                    if learned_code and learned_code != 'unknown':  # Prevent hallucinations
                                        concept['rxnorm_concept_code'] = learned_code
                                        concept['search_strategy'] = 'learned_verified'
                                        print(f"[Semantic Agent] 📚 Enhanced '{core_drug}' with verified learned code: {learned_code}")
                        
                        if 'rxnorm_concept_code' not in concept:
                            # Enhanced fallback: provide structured search guidance for OMOP agent
                            core_name = concept.get('core_drug_name') or concept.get('standardized_term') or original_term
                            
                            concept['search_strategy'] = 'database_lookup'
                            concept['db_search_terms'] = [
                                original_term.lower(),  # Try exact original first
                                core_name.lower(),      # Then core drug name
                                original_term.split()[0].lower() if original_term else ''  # First word fallback
                            ]
                            concept['search_guidance'] = {
                                'primary_term': original_term,
                                'fallback_terms': [core_name, original_term.split()[0] if original_term else ''],
                                'vocabulary_filter': 'RxNorm',
                                'domain_filter': 'Drug',
                                'suggested_like_patterns': [
                                    f"concept_name ILIKE '%{original_term.lower()}%'",
                                    f"concept_name ILIKE '%{core_name.lower()}%'"
                                ]
                            }
                            print(f"[Semantic Agent] 🔄 No vocabulary match for '{original_term}' → delegating to OMOP agent database lookup")
        except Exception as e:
            logger.warning(f"Failed to enhance with concept codes: {e}")
    
    def _ensure_vocabulary_loaded(self):
        """Ensure fast vocabulary is loaded (lazy initialization)."""
        if not self._vocabulary_initialized:
            logger.info("Loading fast OMOP vocabulary for first use...")
            self.vocabulary_index = get_fast_vocabulary()
            self._vocabulary_initialized = True
            logger.info("✅ Fast OMOP vocabulary loaded")
    
    def _find_best_concept_match(self, original_term: str, concept_type: str, concept_data: Dict[str, Any]) -> Optional['FastMatch']:
        """Find the best concept match preserving original term specificity when available."""
        try:
            # Ensure vocabulary is loaded
            self._ensure_vocabulary_loaded()
            
            if not self.vocabulary_index:
                return None
            
            if concept_type == 'drug':
                # PRIORITY 1: Try exact match for original term first (preserve specificity)
                original_matches = self.vocabulary_index.get_drug_concepts(original_term.strip(), limit=5)
                
                if original_matches:
                    # Check if we have a high-confidence exact match
                    exact_match = None
                    for match in original_matches:
                        if match.confidence >= 0.9:
                            # Found high-confidence match for original term - use it!
                            print(f"[Semantic Agent] 🎯 Found exact match for '{original_term}': {match.concept.concept_name} (confidence: {match.confidence:.3f})")
                            return match
                    
                    # If no high-confidence exact match, apply smart selection to original term matches
                    exact_match = self._select_best_drug_concept(original_matches, prefer_specificity=True)
                    if exact_match:
                        print(f"[Semantic Agent] ✅ Selected best match for original term '{original_term}': {exact_match.concept.concept_name}")
                        return exact_match
                
                # PRIORITY 2: Fallback to normalized terms if original didn't yield good matches
                fallback_terms = [
                    concept_data.get('standardized_term', ''),  # LLM's standardization
                    concept_data.get('core_drug_name', '')     # Core drug ingredient
                ]
                
                fallback_matches = []
                for term in fallback_terms:
                    if term and len(term.strip()) > 2 and term.lower() != original_term.lower():
                        matches = self.vocabulary_index.get_drug_concepts(term.strip(), limit=5)
                        fallback_matches.extend(matches)
                
                if fallback_matches:
                    print(f"[Semantic Agent] 🔄 Using fallback search for '{original_term}' -> normalized terms")
                    return self._select_best_drug_concept(fallback_matches, prefer_specificity=False)
                
            elif concept_type == 'condition':
                matches = self.vocabulary_index.get_condition_concepts(original_term, limit=3)
                return matches[0] if matches else None
            else:
                # General search with domain filtering
                domain_map = {
                    'procedure': 'Procedure',
                    'measurement': 'Measurement', 
                    'observation': 'Observation',
                    'device': 'Device'
                }
                domain = domain_map.get(concept_type)
                matches = self.vocabulary_index.search_concepts(
                    query=original_term,
                    domain_filter=domain,
                    limit=3
                )
                return matches[0] if matches else None
                
            return None
        except Exception as e:
            logger.warning(f"Failed to find concept match for '{original_term}': {e}")
            return None
    
    def _select_best_drug_concept(self, matches: List['FastMatch'], prefer_specificity: bool = False) -> Optional['FastMatch']:
        """Select the best drug concept, with option to prefer specificity over generality."""
        if not matches:
            return None
            
        # Score matches based on clinical preference
        scored_matches = []
        for match in matches:
            base_score = match.confidence
            concept_name = match.concept.concept_name.lower()
            concept_code = match.concept.concept_code
            
            # Start with base confidence
            score = base_score
            
            if prefer_specificity:
                # When preserving specificity, prefer detailed formulations
                # Strong preference for non-NDA specific but still detailed concepts
                if not concept_name.startswith('nda'):
                    score += 8.0
                else:
                    score -= 3.0  # Moderate penalty for NDA-specific
                
                # Prefer concepts with dosage/formulation information when preserving specificity
                dosage_terms = ['mg', 'ml', 'tablet', 'capsule', 'injection', 'inhaler', 'auto-injector']
                if any(term in concept_name for term in dosage_terms):
                    score += 5.0  # Reward specificity
                
                # Prefer clinical drugs that include formulation details
                if 'clinical drug' in concept_name:
                    score += 6.0
                elif 'branded drug' in concept_name:
                    score += 3.0
                
                # Don't penalize complex formulations when preserving specificity
                complex_formulation_terms = [
                    'auto-injector', 'prefilled', 'cartridge', 'pen', 
                    'syringe', 'inhaler', 'patch', 'extended release'
                ]
                if any(term in concept_name for term in complex_formulation_terms):
                    score += 2.0  # Reward specific formulations
                
            else:
                # When preferring generality, use original logic
                # Strong preference for non-NDA specific concepts (more generic)
                if not concept_name.startswith('nda'):
                    score += 10.0
                else:
                    score -= 5.0  # Penalize NDA-specific
                
                # Prefer clinical drugs over branded drugs
                if 'clinical drug' in concept_name:
                    score += 5.0
                elif 'branded drug' in concept_name:
                    score += 2.0
                
                # Prefer generic ingredient forms over complex formulations
                complex_formulation_terms = [
                    'auto-injector', 'prefilled', 'cartridge', 'pen', 
                    'syringe', 'inhaler', 'patch', 'extended release'
                ]
                if not any(term in concept_name for term in complex_formulation_terms):
                    score += 3.0
                
                # Avoid very specific dosage forms unless necessary
                if any(term in concept_name for term in ['mg/ml', 'mcg/ml', 'units/ml']):
                    score -= 0.5
            
            # Common scoring for both modes
            # Prefer shorter, simpler names (but less penalty when preserving specificity)
            if len(concept_name) < 50:
                score += 2.0
            elif len(concept_name) > 80:
                penalty = -0.5 if prefer_specificity else -1.0
                score += penalty
            
            # Prefer standard vocabularies
            if match.concept.vocabulary_id == 'RxNorm':
                score += 1.0
            
            # Prefer concepts with standard ingredient names
            standard_ingredients = [
                'epinephrine', 'loratadine', 'lisinopril', 'amlodipine',
                'metformin', 'ibuprofen', 'acetaminophen', 'aspirin'
            ]
            if any(ingredient in concept_name for ingredient in standard_ingredients):
                score += 2.0
            
            scored_matches.append((score, match))
        
        # Sort by score (highest first) and return the best
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        best_match = scored_matches[0][1]
        best_score = scored_matches[0][0]
        
        mode = "specificity-preserving" if prefer_specificity else "generality-preferring"
        logger.info(f"Selected '{best_match.concept.concept_name}' (code: {best_match.concept.concept_code}) "
                   f"with score {best_score:.2f} from {len(matches)} options using {mode} mode")
        
        # Log all candidates for debugging
        if len(matches) > 1:
            logger.debug("All candidates:")
            for score, match in scored_matches[:5]:  # Top 5
                logger.debug(f"  Score {score:.2f}: {match.concept.concept_name} ({match.concept.concept_code})")
        
        return best_match

    def _search_vocabulary_for_concept(self, term: str, concept_type: str) -> List['FastMatch']:
        """Search the fast vocabulary for matching concepts."""
        try:
            # Ensure vocabulary is loaded
            self._ensure_vocabulary_loaded()
            
            if not self.vocabulary_index:
                return []
            
            if concept_type == 'drug':
                return self.vocabulary_index.get_drug_concepts(term, limit=3)
            elif concept_type == 'condition':
                return self.vocabulary_index.get_condition_concepts(term, limit=3)
            else:
                # General search with domain filtering
                domain_map = {
                    'procedure': 'Procedure',
                    'measurement': 'Measurement',
                    'observation': 'Observation',
                    'device': 'Device'
                }
                domain = domain_map.get(concept_type)
                return self.vocabulary_index.search_concepts(
                    query=term,
                    domain_filter=domain,
                    limit=3
                )
        except Exception as e:
            logger.warning(f"Failed to search vocabulary for '{term}': {e}")
            return []

    async def execute(self, action: Action) -> ActionResult:
        """Execute the action decided by the reason method."""
        
        if action.action_type == "semantic_analysis_complete":
            return ActionResult(success=True, data=action.parameters)
        
        if action.action_type == "semantic_complete":
            return ActionResult(success=True, data=action.parameters)
        
        if action.action_type == "error":
            return ActionResult(
                success=False, 
                error=action.parameters.get("message", "Semantic analysis error")
            )

        return ActionResult(
            success=False, 
            error=f"Unknown semantic action type: {action.action_type}"
        )

    async def _handle_semantic_message(self, message_data):
        """
        Handle incoming semantic analysis messages from other agents.
        """
        import json
        
        try:
            logger.info(f"SemanticAgent _handle_semantic_message called with data type: {type(message_data)}")
            logger.info(f"SemanticAgent _handle_semantic_message data: {message_data}")
            
            # Extract the question from the message data
            question = None
            if isinstance(message_data, dict):
                question = message_data.get('question')
            elif isinstance(message_data, str):
                try:
                    parsed_data = json.loads(message_data)
                    if isinstance(parsed_data, dict):
                        question = parsed_data.get('question')
                except json.JSONDecodeError:
                    question = message_data
            
            if not question:
                return {"error": "No question found in message", "success": False}
            
            # Process the query using the semantic analysis pipeline
            result = await self.process_query(question)
            
            # Debug logging
            logger.info(f"SemanticAgent process_query result type: {type(result)}")
            logger.info(f"SemanticAgent process_query result: {result}")
            
            # Format the response as JSON
            if hasattr(result, 'success') and result.success:
                # result.data contains the action parameters
                semantic_context = {}
                if isinstance(result.data, dict):
                    semantic_context = result.data.get("semantic_context", result.data)
                
                return {
                    "semantic_context": semantic_context,
                    "success": True
                }
            elif hasattr(result, 'success') and not result.success:
                return {
                    "error": result.error,
                    "success": False
                }
            else:
                # Handle unexpected result type
                return {
                    "error": f"Unexpected result type: {type(result)}, value: {result}",
                    "success": False
                }
            
        except Exception as e:
            logger.error(f"SemanticAgent message handling error: {e}")
            return {"error": f"Semantic analysis failed: {str(e)}", "success": False}

    async def on_message_send(self, params, context=None):
        """
        Handle incoming messages from other agents (like the orchestrator).
        Extract the question from the message and perform semantic analysis.
        """
        import json
        from a2a.types import Message, TextPart, Role
        
        try:
            logger.info(f"SemanticAgent on_message_send called with params type: {type(params)}")
            logger.info(f"SemanticAgent on_message_send params: {params}")
            
            # Extract the message text from params
            message_text = None
            if hasattr(params, 'message') and hasattr(params.message, 'parts'):
                for part in params.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        message_text = part.root.text
                        break
            
            if not message_text:
                message_text = str(params.message) if hasattr(params, 'message') else str(params)
            
            # Use the existing handler logic
            result = await self._handle_semantic_message(message_text)
            
            # Return a proper Message response
            response_text = json.dumps(result)
            
            import uuid
            return Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=response_text)]
            )
            
        except Exception as e:
            logger.error(f"SemanticAgent on_message_send error: {e}", exc_info=True)
            error_response = {"error": f"Message handling failed: {str(e)}", "success": False}
            response_text = json.dumps(error_response)
            
            import uuid
            return Message(
                messageId=str(uuid.uuid4()),
                role=Role.agent,
                parts=[TextPart(text=response_text)]
            )

    def build_agent_card(self) -> AgentCard:
        """Build the agent card for A2A discovery."""
        return AgentCard(
            name=self.agent_name,
            description=self.agent_description,
            version="1.0.0",
            url=f"http://localhost:8002/{self.agent_id}",
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
        
        if not summary:
            summary = "No semantic analysis available."
            
        return summary

    async def process_query(self, question: str) -> ActionResult:
        """
        Main entry point for semantic analysis of a query.
        """
        try:
            # 1. Perceive the initial question and update the world model
            observation = await self.perceive(question)
            await self.learn(self.mental_state, observation)

            # 2. Run the semantic analysis
            max_loops = 3  # Safety break
            for _ in range(max_loops):
                action = await self.reason(self.mental_state)

                if action.action_type in ["semantic_complete", "semantic_analysis_complete"]:
                    return await self.execute(action)

                if action.action_type == "error":
                    return ActionResult(
                        success=False, 
                        error=action.parameters.get("message", "Semantic reasoning error")
                    )

                result = await self.execute(action)
                if not result.success:
                    return result

                # Learn from the result
                observation = await self.perceive(result.data)
                await self.learn(self.mental_state, observation)
            
            return ActionResult(
                success=False, 
                error="Semantic agent exceeded maximum execution loops."
            )

        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}", exc_info=True)
            return ActionResult(
                success=False, 
                error=f"Semantic analysis internal error: {e}"
            )