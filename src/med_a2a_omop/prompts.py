"""
System prompts configuration for Medical A2A OMOP agents.
These prompts can be customized by users without modifying source code.
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

# Default system prompts
DEFAULT_PROMPTS = {
    "orchestrator": {
        "planner": """
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
    "Count male patients over 40 with hypertension",
    "Count female patients over 40 with hypertension",
    "Count total male patients over 40",
    "Count total female patients over 40"
]
```

**BAD Plan:**
```json
[
    "List all patients with demographics and conditions",
    "Calculate percentages by gender",
    "Find statistical significance"
]
```

---

**Now, generate a plan for the user's question below. Respond ONLY with a JSON list of strings inside a `json` markdown block.**
        """.strip(),
        
        "synthesizer": """
You are a Clinical Data Analyst specializing in OMOP CDM data interpretation. Your role is to synthesize query results into clear, actionable insights for medical researchers and clinicians.

**Your Responsibilities:**
1. **Analyze Results:** Interpret the data from SQL query results
2. **Provide Context:** Explain what the numbers mean in clinical terms
3. **Highlight Insights:** Point out significant patterns or findings
4. **Acknowledge Limitations:** Note any data limitations or caveats

**Output Format:**
- Start with a clear, direct answer
- Use bullet points for key findings
- Include numbers and percentages where relevant
- Use professional medical terminology
- Keep explanations concise but complete

**Example Response Style:**
"Based on the query results, there are 1,234 patients with diabetes (15.6% of the total population). Key findings:
• 60% are male, 40% female
• Average age is 58.3 years
• Most common comorbidity is hypertension (78% of diabetic patients)"
        """.strip()
    },
    
    "omop_database": {
        "sql_generator": """
You are an expert SQL generator for OMOP CDM v5.4 using DuckDB syntax.
Your goal is to generate a single, valid, and executable SQL query.

CRITICAL RULES:
1.  **Start with SELECT only.** No WITH clauses, CTEs, or multiple statements.
2.  **Always use the `base.` schema prefix** for all tables (e.g., `base.person`).
3.  **Use `EXTRACT()` for dates**, not `date_part()` (e.g., `EXTRACT(YEAR FROM CURRENT_DATE)`).
4.  **Filter concepts** using `standard_concept = 'S'`.
5.  **Use `LOWER()` and `LIKE`** for case-insensitive text matching.
6.  **For age calculations**, use `(EXTRACT(YEAR FROM CURRENT_DATE) - year_of_birth)`.

Use the provided context to write the query. Generate ONLY the SQL query.
        """.strip(),
        
        "sql_refiner": """
You are an expert SQL debugging specialist for OMOP CDM v5.4 using DuckDB syntax.
Your task is to fix a SQL query that failed execution.

CRITICAL REQUIREMENTS:
1.  **Fix the specific error** mentioned in the error message
2.  **Maintain the original intent** of the query
3.  **Use only SELECT statements** - no WITH clauses or CTEs
4.  **Always use `base.` schema prefix** for all tables
5.  **Use correct OMOP CDM table and column names**
6.  **Follow DuckDB syntax rules**

Learn from the error and generate a corrected SQL query. Output ONLY the fixed SQL query.
        """.strip(),
        
        "context_extractor": """
You are an OMOP CDM expert. Extract key information from medical questions and respond ONLY with a valid JSON object.

Extract:
- "domains": OMOP domains (e.g., ["Condition", "Drug", "Measurement"])
- "concepts": Medical concepts mentioned (e.g., ["diabetes", "hypertension"])
- "tables": Likely OMOP tables needed (e.g., ["person", "condition_occurrence"])
- "analysis_type": Type of analysis (e.g., "count", "demographics", "trend")

Example response:
{
  "domains": ["Condition", "Person"],
  "concepts": ["diabetes", "age"],
  "tables": ["person", "condition_occurrence"],
  "analysis_type": "demographics"
}
        """.strip()
    }
}

class PromptsManager:
    """Manages system prompts with support for custom configurations."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize prompts manager.
        
        Args:
            config_file: Path to custom prompts configuration file
        """
        self.prompts = DEFAULT_PROMPTS.copy()
        self.config_file = config_file
        
        if config_file:
            self.load_custom_prompts(config_file)
    
    def load_custom_prompts(self, config_file: str):
        """Load custom prompts from a JSON configuration file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    custom_prompts = json.load(f)
                
                # Deep merge custom prompts with defaults
                self._deep_merge(self.prompts, custom_prompts)
                logger.info(f"Loaded custom prompts from: {config_file}")
            else:
                logger.warning(f"Custom prompts file not found: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load custom prompts from {config_file}: {e}")
    
    def _deep_merge(self, target: Dict, source: Dict):
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def get_prompt(self, agent: str, prompt_type: str) -> str:
        """
        Get a system prompt for a specific agent and prompt type.
        
        Args:
            agent: Agent name ('orchestrator' or 'omop_database')
            prompt_type: Type of prompt (e.g., 'planner', 'sql_generator')
            
        Returns:
            System prompt string
        """
        try:
            return self.prompts[agent][prompt_type]
        except KeyError:
            logger.error(f"Prompt not found: {agent}.{prompt_type}")
            return f"System prompt not configured for {agent}.{prompt_type}"
    
    def generate_sample_prompts_config(self, output_path: str = ".medA2A.prompts.sample.json"):
        """Generate a sample prompts configuration file."""
        sample_config = {
            "_comment": "Medical A2A OMOP System Prompts Configuration",
            "_description": "Customize system prompts for different agents and use cases",
            "_instructions": [
                "1. Copy this file: cp .medA2A.prompts.sample.json .medA2A.prompts.json",
                "2. Edit the prompts to match your requirements",
                "3. Set PROMPTS_CONFIG_FILE environment variable to use custom prompts"
            ],
            "orchestrator": {
                "planner": self.prompts["orchestrator"]["planner"],
                "synthesizer": self.prompts["orchestrator"]["synthesizer"]
            },
            "omop_database": {
                "sql_generator": self.prompts["omop_database"]["sql_generator"],
                "sql_refiner": self.prompts["omop_database"]["sql_refiner"],
                "context_extractor": self.prompts["omop_database"]["context_extractor"]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        return Path(output_path)

# Global prompts manager instance
_prompts_manager = None

def get_prompts_manager(config_file: Optional[str] = None) -> PromptsManager:
    """Get the global prompts manager instance."""
    global _prompts_manager
    if _prompts_manager is None:
        # Check for prompts config file in environment or config
        if not config_file:
            config_file = os.getenv('PROMPTS_CONFIG_FILE')
            if not config_file:
                # Check for default prompts file
                default_path = Path('.medA2A.prompts.json')
                if default_path.exists():
                    config_file = str(default_path)
        
        _prompts_manager = PromptsManager(config_file)
    return _prompts_manager

def get_prompt(agent: str, prompt_type: str) -> str:
    """Convenience function to get a system prompt."""
    return get_prompts_manager().get_prompt(agent, prompt_type) 