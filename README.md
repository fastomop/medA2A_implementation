# FastOMOP /Medical A2A OMOP - Intelligent Healthcare Data Query System

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OMOP CDM v5.4](https://img.shields.io/badge/OMOP_CDM-v5.4-green.svg)](https://ohdsi.github.io/CommonDataModel/)

> **An intelligent multi-agent framework for natural language querying of OMOP Common Data Model (CDM) healthcare databases using agents and semantic analysis.**

**DISCLAIMER: This system is under active development. For bug reports, contact k24118093@kcl.ac.uk**

## **Key Features**

### **Intelligent Multi-Agent Architecture**
- **Orchestrator Agent**: Coordinates workflow and manages complex queries
- **Semantic Agent**: Handles medical terminology and concept mapping
- **OMOP Database Agent**: Specialized text-to-SQL conversion with OMOP expertise
- **Mixed Model Strategy**: Optimized LLM selection for different tasks
- **A2A Medical Framework**: Built on the [A2A Medical Foundation Framework](https://github.com/fastomop/omcp_a2a/tree/feature/medical-a2a-framework)

### **Multiple Interaction Modes**
- **Interactive CLI**: Real-time question-answer sessions
- **Command Line**: Single or multiple question processing
- **Batch Processing**: Process questions from files (text/JSON)
- **Programmatic API**: Python integration for applications
- **Evaluation Framework**: Built-in system performance evaluation

### **Advanced Capabilities**
- **OMOP CDM v5.4 Expert Knowledge**: Complete understanding of healthcare data standards
- **Vocabulary Integration**: Fast RxNorm, SNOMED, and CPT4 concept mapping
- **Semantic Analysis**: Medical terminology standardization and enhancement
- **MCP Integration**: Secure database communication via Model Context Protocol
- **Intelligent Caching**: SQL template learning and semantic pattern recognition
- **Multi-Level Caching**: Query patterns, semantic analysis, vocabulary, and SQL templates
- **Learning System**: Adapts from successful queries and concept mappings

## **Quick Start**

### Prerequisites
- Python 3.13+
- [UV package manager](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.ai/) installed and running with required models
- Git (for cloning with submodules)
- OMOP Vocabulary files (see vocabulary setup below)

### Installation

```bash
# Clone the repository with OMCP server 
git clone --recursive https://github.com/fastomop/medA2A_implementation.git
cd medA2A_implementation
git submodule update --init --recursive

# Install dependencies using uv (recommended)
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install -e .

# That's it! The OMCP server is automatically included as a submodule
```

**For existing repositories:**
```bash
# If you already cloned without --recursive, get the submodule:
git submodule update --init --recursive

# To update submodule to latest version:
git submodule update --remote omcp_server
```

### Required Ollama Models

Install the required models for optimal performance:
```bash
# Install required models
ollama pull llama3.1:8b          # For orchestrator (fast planning)
ollama pull gpt-oss:20b          # For semantic analysis and SQL generation (if available)

# Start Ollama service
ollama serve
```

### Configuration

1. **Generate configuration file:**
```bash
med-a2a-setup --generate-config
```

2. **Edit configuration (optional):**
```json
{
  "services": {
    "ollama_url": "http://localhost:11434",
    "orchestrator_model": "llama3.1:8b",
    "semantic_model": "gpt-oss:20b",
    "omop_model": "gpt-oss:20b"
  }
}
```

3. **Validate setup:**
```bash
med-a2a-setup --check
```

**Note:** The OMCP server path is automatically detected from the submodule - no manual configuration needed!

### Vocabulary Setup

1. **Download OMOP Vocabulary** (required for medical concept mapping):
   - Visit [OHDSI Athena](https://athena.ohdsi.org/)
   - Download vocabulary files (RxNorm, SNOMED, CPT4, etc.)
   - Extract to one of these locations:
     - `~/Downloads/omop_vocab_current/` (auto-detected)
     - `~/omop_vocabulary/` (auto-detected)
     - Custom path via configuration

2. **Configure vocabulary path (optional):**
```json
{
  "paths": {
    "vocabulary_path": "/path/to/your/omop_vocabulary"
  }
}
```

3. **Environment variable (alternative):**
```bash
export OMOP_VOCABULARY_PATH=/path/to/omop_vocabulary
```

### Basic Usage

```bash
# Interactive mode (default)
run-med-a2a

# Single question
run-med-a2a -q "How many patients have hypertension?"

# Multiple questions
run-med-a2a -q "How many patients have diabetes?" -q "What drugs are prescribed for hypertension?"

# Batch processing from file
run-med-a2a --batch example_questions.txt --output results.json

# System evaluation
med-a2a-eval --limit 10

# Get help
run-med-a2a --help
```

## **Usage Examples**

### Interactive Mode
```bash
$ run-med-a2a

🎯 Interactive Medical A2A OMOP Query Interface
============================================================
Enter your medical questions (type 'quit', 'exit', or press Ctrl+C to stop)
Type 'help' for available commands
============================================================

❓ Your question: How many patients have hypertension?

--- ✔️ Answer ---
Based on the query results, there are 0 patients with hypertension in the database.

--- 📝 Generated SQL ---
SELECT COUNT(DISTINCT p.person_id) as patient_count
FROM base.person p
JOIN base.condition_occurrence co ON p.person_id = co.person_id
JOIN base.concept c ON co.condition_concept_id = c.concept_id
WHERE c.standard_concept = 'S'
  AND c.domain_id = 'Condition'
  AND c.concept_name = 'Hypertension'
```

### Programmatic API
```python
import asyncio
from med_a2a_omop.runner import MedA2AAPI

async def main():
    async with MedA2AAPI() as api:
        # Single question
        result = await api.ask("How many patients have diabetes?")
        print(f"Answer: {result['answer']}")
        
        # Multiple questions
        questions = [
            "How many patients have hypertension?",
            "What is the average age of patients with diabetes?"
        ]
        results = await api.ask_multiple(questions)
        
        for result in results:
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}\n")

asyncio.run(main())
```

### Batch Processing
```bash
# From text file (one question per line)
$ cat medical_questions.txt
How many patients have hypertension?
What is the average age of patients with diabetes?
How many patients are taking metformin?

$ run-med-a2a --batch medical_questions.txt --output results.json
📁 Loaded 3 questions from medical_questions.txt
🔄 Processing 3 questions...
✅ Processed 3 questions
💾 Results saved to results.json
```

### System Evaluation

1. **Prepare evaluation dataset:**
   - Create a JSON file with questions and expected results
   - See `example_questions.json` for format reference

2. **Run evaluation:**
```bash
# Run evaluation with your dataset
med-a2a-eval --dataset your_evaluation_data.json

# Limit to first 10 questions for testing
med-a2a-eval --dataset your_evaluation_data.json --limit 10

# Specify custom output directory
med-a2a-eval --dataset your_evaluation_data.json --output my_results/

# Verbose output
med-a2a-eval --dataset your_evaluation_data.json --verbose
```

3. **View results:**
```bash
# Show evaluation results
cat evaluation_results/evaluation_results_*.json

# View human-readable report
cat evaluation_results/evaluation_report_*.txt
```

## **System Architecture**

### Multi-Agent Workflow
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Orchestrator    │───▶│ Semantic Agent  │
│  (CLI/API/File) │    │     Agent        │    │   (Terminology) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ OMOP Database    │◀───│ Vocabulary &    │
                       │     Agent        │    │ Concept Mapping │
                       │  (Text-to-SQL)   │    └─────────────────┘
                       └──────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ OMOP Database   │
                       │ (via MCP Server)│
                       └─────────────────┘
```


## **Command Line Options**

| Command | Description | Example |
|---------|-------------|---------|
| `run-med-a2a` | Start interactive mode | `run-med-a2a` |
| `run-med-a2a -q "question"` | Ask specific question | `-q "How many patients have diabetes?"` |
| `run-med-a2a --batch file.txt` | Process questions from file | `--batch questions.txt` |
| `run-med-a2a --output results.json` | Save results to file | `--output results.json` |
| `med-a2a-eval --dataset <file>` | Run system evaluation | `med-a2a-eval --dataset data.json --limit 10` |
| `med-a2a-setup --check` | Validate configuration | `med-a2a-setup --check` |

## **Project Structure**

```
medA2A_implementation/
├── src/
│   └── med_a2a_omop/
│       ├── agents/
│       │   ├── orchestrator_agent.py     # Workflow coordination
│       │   ├── semantic_agent.py         # Medical terminology & concepts
│       │   └── omop_database_agent.py    # Text-to-SQL conversion
│       ├── models/
│       │   └── a2a_messages.py          # Message schemas
│       ├── vocabulary_fast.py           # Fast OMOP vocabulary integration
│       ├── config.py                    # Configuration management
│       ├── runner.py                    # Main application interface
│       ├── evaluate_system.py           # System evaluation framework
│       └── setup.py                     # Setup and validation
├── omcp_server/                         # OMCP server submodule (automatic!)
│   ├── src/omcp/
│   │   ├── main_robust.py              # Robust server with DB lock prevention
│   │   ├── db_robust.py                # Database connection with retry logic
│   │   └── main.py                     
│   └── pyproject.toml
├── .medA2A.config.json                  # Configuration file
├── example_questions.txt                # Sample questions
├── evaluation_results/                  # Evaluation output
└── README.md                           # This file
```

## **Configuration**

### Configuration File (.medA2A.config.json)
```json
{
  "services": {
    "ollama_url": "http://localhost:11434",
    "orchestrator_model": "llama3.1:8b",
    "orchestrator_timeout": 30,
    "semantic_model": "gpt-oss:20b",
    "semantic_timeout": 180,
    "omop_model": "gpt-oss:20b",
    "omop_timeout": 240,
    "mcp_timeout": 10
  },
  "agent_config": {
    "omop_agent_host": "127.0.0.1",
    "omop_agent_port": 8003
  },
  "database": {
    "db_type": "duckdb",
    "cdm_schema": "base",
    "vocab_schema": "base"
  }
}
```

**Simplified Setup:** The OMCP server is now automatically detected from the git submodule, so no manual path configuration is needed!

### Environment Variables (Alternative)
```bash
# Optional services (OMCP server path auto-detected from submodule)
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b

# Vocabulary path (optional - auto-detected if in standard locations)
export OMOP_VOCABULARY_PATH=/path/to/omop_vocabulary

# Agent configuration
export OMOP_AGENT_HOST=127.0.0.1
export OMOP_AGENT_PORT=8003
```

## **Troubleshooting**

### Common Issues

**Configuration Validation**
```bash
# Check system configuration
med-a2a-setup --check

# Generate new configuration
med-a2a-setup --generate-config
```

**Agent Connection Issues**
```bash
# Check if services are running
❌ OMOP Agent server failed to become ready!
# Solution: Check logs and ensure no port conflicts
```

**Database Connection Issues**
```bash
# Check MCP server configuration
# Ensure database path is correct in .medA2A.config.json
# Verify OMCP server is properly configured
```

**Query Generation Issues**
```bash
# The system learns from failures automatically
[OMOPDatabaseAgent] Attempting to refine SQL (attempt 2 of 10)
# The world model adapts and improves subsequent queries
```

### Getting Help
1. **Interactive Help**: Type `help` in interactive mode
2. **Configuration Check**: Run `med-a2a-setup --check`
3. **Evaluation**: Create test dataset and run `med-a2a-eval --dataset test.json --limit 5`
4. **Logs**: Check console output for detailed error messages

## **Performance & Evaluation**

### Current Performance Metrics
- **Accuracy**: ~84% exact matches
- **Success Rate**: 100% (no failed queries)
- **Average Response Time**: ~49 seconds
- **Mixed Model Strategy**: Optimized for speed vs accuracy balance

### Evaluation Framework
```bash
# Run evaluation with your dataset
med-a2a-eval --dataset evaluation_data.json --limit 10

# View results
cat evaluation_results/evaluation_results_*.json
```

## **Contributing**

### Development Setup
```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
python -m pytest

# Format code
black src/
isort src/
```

### Adding New Features
1. **New Analysis Types**: Extend the world model in `omop_database_agent.py`
2. **New Interaction Modes**: Enhance `runner.py`
3. **New Agents**: Follow the pattern in `agents/`

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

- **OHDSI Community**: For the OMOP Common Data Model standard
- **A2A Protocol**: For agent communication framework
- **A2A Medical Foundation Framework**: The underlying framework for medical agent systems
- **Model Context Protocol**: For secure database integration
- **Ollama**: For local LLM capabilities

## **Support**

For questions, issues, or contributions:
1. Check the configuration with `med-a2a-setup --check`
2. Run evaluation to test system: `med-a2a-eval --limit 5`
3. Review console output for detailed error messages
4. Open an issue for bugs or feature requests

---