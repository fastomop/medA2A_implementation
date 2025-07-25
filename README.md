# 🏥 Medical A2A OMOP - Intelligent Healthcare Data Query System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OMOP CDM v5.4](https://img.shields.io/badge/OMOP_CDM-v5.4-green.svg)](https://ohdsi.github.io/CommonDataModel/)

> **An intelligent multi-agent framework for natural language querying of OMOP Common Data Model (CDM) healthcare databases.**

Transform complex medical questions into precise SQL queries using advanced AI agents, comprehensive OMOP CDM knowledge, and multiple interaction modes for maximum flexibility.

## 🌟 **Key Features**

### 🧠 **Intelligent Query Generation**
- **Natural Language Processing**: Ask questions in plain English
- **OMOP CDM v5.4 Expert Knowledge**: Complete understanding of healthcare data standards
- **Iterative Learning**: System improves from database feedback and learns schema patterns
- **World Model**: Comprehensive knowledge base of OMOP tables, relationships, and best practices

### 🎯 **Multiple Interaction Modes**
- **🖥️ Interactive CLI**: Real-time question-answer sessions with help system
- **⚡ Command Line**: Direct single or multiple question processing
- **📁 Batch Processing**: Process multiple questions from files (text/JSON)
- **🔧 Programmatic API**: Python integration for applications
- **📊 JSON Output**: Machine-readable results for data pipelines

### 🏗️ **Multi-Agent Architecture**
- **Orchestrator Agent**: Coordinates workflow and summarizes results
- **OMOP Database Agent**: Specialized in text-to-SQL conversion and database interaction
- **MCP Integration**: Secure database communication via Model Context Protocol


## 🚀 **Quick Start**

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd medA2A_implementation

# Install dependencies using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### Basic Usage

```bash
# Interactive mode (default) - Start asking questions immediately
run-med-a2a

# Single question
run-med-a2a -q "How many patients have hypertension?"

# Multiple questions
run-med-a2a -q "How many patients have diabetes?" -q "What drugs are prescribed for hypertension?"

# Batch processing from file
run-med-a2a --batch example_questions.txt --output results.json

# Get help and examples
run-med-a2a --examples
run-med-a2a --help
```

## 📖 **Usage Examples**

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
from src.med_a2a_omop.runner import MedA2AAPI

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

## 🏗️ **Architecture**

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Orchestrator    │───▶│ OMOP Database   │
│  (CLI/API/File) │    │     Agent        │    │     Agent       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────┐       ┌─────────────────┐
                       │ Summarization│       │ Text-to-SQL +   │
                       │   & Results  │       │ MCP Integration │
                       └──────────────┘       └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ OMOP Database   │
                                              │ (via MCP Server)│
                                              └─────────────────┘
```

### Key Technologies
- **🐍 Python 3.11+**: Modern async/await patterns
- **🤖 Ollama**: Local LLM for text-to-SQL conversion
- **📡 A2A Protocol**: Agent-to-Agent communication
- **🔗 MCP (Model Context Protocol)**: Secure database communication
- **🦆 DuckDB**: High-performance analytical database
- **📊 OMOP CDM v5.4**: Healthcare data standardization

## 🎯 **Command Line Options**

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--question` | `-q` | Ask specific question(s) | `-q "How many patients have diabetes?"` |
| `--batch` | | Process questions from file | `--batch questions.txt` |
| `--output` | `-o` | Save results to JSON file | `--output results.json` |
| `--json` | | Output in JSON format | `--json` |
| `--interactive` | `-i` | Start interactive mode | `--interactive` |
| `--examples` | | Show example questions | `--examples` |
| `--help` | `-h` | Show help message | `--help` |

## 📁 **Project Structure**

```
medA2A_implementation/
├── src/
│   └── med_a2a_omop/
│       ├── agents/
│       │   ├── omop_database_agent.py    # OMOP-specialized agent with world model
│       │   └── orchestrator_agent.py     # Workflow coordination
│       ├── models/
│       │   └── a2a_messages.py          # Message schemas
│       ├── runner.py                    # Main application with multiple interfaces
│       └── run_omop_agent.py           # OMOP agent server
├── omcp_wrapper.py                     # MCP server wrapper script
├── example_questions.txt               # Sample questions (text format)
├── example_questions.json              # Sample questions (JSON format)
├── example_api_usage.py               # API usage examples
├── USAGE_GUIDE.md                     # Comprehensive usage documentation
├── pyproject.toml                     # Project configuration
└── README.md                          # This file
```

## 🧠 **OMOP CDM World Model**

The system includes a comprehensive world model with:

### **📚 Complete OMOP CDM v5.4 Knowledge**
- **Standard Tables**: Person, Condition, Drug, Measurement, Observation, etc.
- **Vocabulary Integration**: SNOMED, ICD-10, RxNorm, LOINC
- **Relationships**: Foreign keys, business rules, common join patterns
- **Domain Logic**: Clinical domains and concept hierarchies

### **🎯 Intelligent Query Templates**
- **Patient Counting**: Count distinct patients with conditions
- **Drug Analysis**: Medication usage and prescription patterns  
- **Measurement Analysis**: Lab results and vital signs
- **Comorbidity Analysis**: Multiple condition combinations

### **📈 Adaptive Learning**
- **Schema Discovery**: Automatically explores actual database structure
- **Error Learning**: Learns from failed queries and database feedback
- **Pattern Recognition**: Identifies successful query patterns
- **Iterative Refinement**: Up to 10 attempts with progressive learning

## 🔧 **Configuration**

### Environment Variables
Create a `.env` file in the project root:

```bash
# OMOP Agent Configuration
OMOP_AGENT_URL=http://localhost:8002

# Database Configuration (handled by MCP server)
DB_TYPE=duckdb
DB_PATH=/path/to/your/omop/database.duckdb
CDM_SCHEMA=base
VOCAB_SCHEMA=base

# Ollama Configuration
OLLAMA_MODEL=llama3.1:8b
OLLAMA_URL=http://localhost:11434
```

### Dependencies
Key dependencies are managed in `pyproject.toml`:
- `a2a-medical-foundation`: Core medical agent framework
- `ollama`: LLM integration
- `fastapi`: Web framework for agents
- `httpx`: HTTP client for agent communication
- `uvicorn`: ASGI server

## 🚨 **Troubleshooting**

### Common Issues

**Database Locks**
```bash
# The system automatically handles database locks, but if issues persist:
🔍 Checking for existing database locks...
✅ No existing database locks found
```

**Agent Connection Issues**
```bash
# Check if all services are running:
❌ OMOP Agent server failed to become ready!
# Solution: Check logs and ensure no port conflicts
```

**Query Generation Issues**
```bash
# The system learns from failures:
[OMOPDatabaseAgent] Attempting to refine SQL (attempt 2 of 10)
# The world model will adapt and improve subsequent queries
```

### Getting Help
1. **Interactive Help**: Type `help` in interactive mode
2. **Examples**: Run `run-med-a2a --examples`
3. **Documentation**: See `USAGE_GUIDE.md` for comprehensive documentation
4. **API Examples**: Check `example_api_usage.py`

## 🤝 **Contributing**

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
1. **New Query Types**: Extend the world model in `omop_database_agent.py`
2. **New Interaction Modes**: Enhance `runner.py`
3. **New Agents**: Follow the pattern in `agents/`

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **OHDSI Community**: For the OMOP Common Data Model standard
- **A2A Protocol**: For agent communication framework
- **Model Context Protocol**: For secure database integration
- **Ollama**: For local LLM capabilities

## 📞 **Support**

For questions, issues, or contributions:
1. Check the [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed documentation
2. Review example files for implementation patterns
3. Open an issue for bugs or feature requests

---

**🎯 Ready to transform your healthcare data queries? Start with `run-med-a2a` and experience intelligent medical data analysis!** 