# Medical A2A OMOP Usage Guide

## Overview

The Medical A2A OMOP system now supports multiple interaction modes to accommodate different use cases:

1. **Interactive CLI** - Real-time question-and-answer interface
2. **Command Line Questions** - Direct question processing via arguments
3. **Batch Processing** - Process multiple questions from files
4. **Programmatic API** - Python integration for other applications
5. **JSON Output** - Machine-readable results

## 🚀 Quick Start

### Default Interactive Mode
```bash
# Start interactive session
run-med-a2a

# Or explicitly
python -m med_a2a_omop.runner --interactive
```

### Single Question
```bash
# Ask one question
run-med-a2a -q "How many patients have hypertension?"

# With JSON output
run-med-a2a --json -q "How many patients have diabetes?"
```

## 📋 All Usage Modes

### 1. Interactive CLI Mode

**Start the interactive interface:**
```bash
run-med-a2a
```

**Features:**
- Real-time question input
- Built-in help system (`help` command)
- Example questions (`examples` command)
- Shows both answers and generated SQL
- Graceful exit with `quit`, `exit`, or Ctrl+C

**Example session:**
```
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

❓ Your question: quit
👋 Goodbye!
```

### 2. Command Line Questions

**Single question:**
```bash
run-med-a2a -q "How many patients have diabetes?"
```

**Multiple questions:**
```bash
run-med-a2a -q "How many patients have hypertension?" -q "What drugs are used for diabetes?"
```

**With output file:**
```bash
run-med-a2a -q "How many patients have diabetes?" --output results.json
```

### 3. Batch Processing

**From text file (one question per line):**
```bash
run-med-a2a --batch example_questions.txt
```

**From JSON file:**
```bash
run-med-a2a --batch example_questions.json --output batch_results.json
```

**Example text file (`example_questions.txt`):**
```
How many patients have hypertension?
What is the average age of patients with diabetes?
How many patients are taking metformin?
```

**Example JSON file (`example_questions.json`):**
```json
{
  "questions": [
    "How many patients have hypertension?",
    "What is the average age of patients with diabetes?",
    "How many patients are taking metformin?"
  ]
}
```

### 4. JSON Output Mode

**Get machine-readable results:**
```bash
# Single question with JSON output
run-med-a2a --json -q "How many patients have hypertension?"

# Batch processing with JSON output
run-med-a2a --json --batch questions.txt
```

**Example JSON output:**
```json
{
  "question": "How many patients have hypertension?",
  "success": true,
  "timestamp": 1640995200.0,
  "generated_sql": "SELECT COUNT(DISTINCT p.person_id) as patient_count...",
  "query_result": [{"patient_count": "0"}],
  "answer": "Based on the query results, there are 0 patients with hypertension in the database."
}
```

### 5. Programmatic API

**Use in Python applications:**

```python
import asyncio
from src.med_a2a_omop.runner import MedA2AAPI

async def main():
    # Method 1: Context manager (recommended)
    async with MedA2AAPI() as api:
        result = await api.ask("How many patients have diabetes?")
        print(f"Answer: {result['answer']}")
        
        # Multiple questions
        questions = ["How many patients have hypertension?", "What drugs are used for diabetes?"]
        results = await api.ask_multiple(questions)
        
    # Method 2: Manual management
    api = MedA2AAPI()
    try:
        await api.initialize()
        result = await api.ask("How many patients have heart disease?")
    finally:
        await api.cleanup()

asyncio.run(main())
```

## 🎯 Command Line Options

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--question` | `-q` | Ask specific question(s) | `-q "How many patients have diabetes?"` |
| `--batch` | | Process questions from file | `--batch questions.txt` |
| `--output` | `-o` | Save results to file | `--output results.json` |
| `--json` | | Output in JSON format | `--json` |
| `--interactive` | `-i` | Start interactive mode | `--interactive` |
| `--examples` | | Show example questions | `--examples` |
| `--help` | `-h` | Show help message | `--help` |

## 📊 Output Formats

### Human-Readable Output
- Clear question and answer format
- Shows generated SQL queries
- Progress indicators for batch processing
- Error messages with context

### JSON Output
- Machine-readable format
- Includes metadata (timestamps, success status)
- Complete query results and generated SQL
- Suitable for integration with other tools

## 🔧 Advanced Usage

### Environment Variables
- `OMOP_AGENT_URL`: Override default agent URL
- Standard `.env` file support

### File Formats
- **Text files**: One question per line
- **JSON files**: `{"questions": ["question1", "question2"]}`
- **Output files**: Always JSON format with complete metadata

### Error Handling
- Graceful error recovery
- Detailed error messages
- Proper cleanup on interruption
- Database lock management

## 💡 Best Practices

1. **Interactive Mode**: Best for exploration and learning
2. **Batch Processing**: Efficient for multiple related questions
3. **Programmatic API**: Integration with larger applications
4. **JSON Output**: When results need to be processed by other tools

## 🚨 Common Issues

### Database Locks
The system automatically handles database locks and cleanup. If you encounter lock issues, the system will:
- Detect existing locks on startup
- Clean up processes holding locks
- Provide detailed error messages

### Question Complexity
- Start with simple questions to understand the data
- Use the `examples` command for inspiration
- The system learns from failures and improves over time

## 📝 Example Questions

Use these examples to get started:

- `"How many patients have hypertension?"`
- `"What is the average age of patients with diabetes?"`
- `"How many patients are taking metformin?"`
- `"What are the most common conditions in the database?"`
- `"How many patients have both diabetes and hypertension?"`
- `"What is the gender distribution of patients with heart disease?"`
- `"How many lab tests were performed last year?"`
- `"What medications are prescribed most frequently?"` 