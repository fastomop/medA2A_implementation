# 🚀 Medical A2A OMOP Quick Start Guide

Get your Medical A2A OMOP system running quickly with explicit configuration.

## 📋 Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed
- **Git** for cloning repositories
- **OMCP Server** cloned and set up
- **OMOP CDM Database** in DuckDB format
- Internet connection for downloading dependencies

## 🔧 Step 1: Install the System

```bash
# Clone the repository
git clone [YOUR_REPO_URL] medA2A_implementation
cd medA2A_implementation

# Install the package
pip install -e .
```

## ⚙️ Step 2: Create Configuration File

```bash
# Option 1: Generate a configuration template
med-a2a-setup --generate-config

# Option 2: Copy the sample configuration
cp .medA2A.config.sample.json .medA2A.config.json

# Both create .medA2A.config.json - you MUST edit this file
```

## ✏️ Step 3: Edit Configuration (REQUIRED)

Open `.medA2A.config.json` and set the correct paths:

```json
{
  "paths": {
    "omcp_server_path": "/full/path/to/your/omcp_server"
  }
}
```

**Important:** 
- Use **absolute paths** only
- Ensure the OMCP server directory contains `src/omcp/main.py`
- The OMCP server will handle its own database configuration

## 🔍 Step 4: Validate Configuration

```bash
# Check if your configuration is correct
med-a2a-setup --check

# See exactly where settings are coming from
med-a2a-setup --show-sources
```

## 📦 Step 5: Install Missing Dependencies

If validation fails, install missing components:

### UV Package Manager
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Ollama (LLM Service)
```bash
# Install Ollama from https://ollama.ai/
# Then pull the required model:
ollama pull llama3.1:8b

# Start the service:
ollama serve
```

## ✅ Step 6: Test Installation

```bash
# Test the complete system
med-a2a-setup --test
```

## 🏃 Step 7: Run the System

```bash
# Interactive mode (default)
run-med-a2a

# Single question
run-med-a2a -q "How many patients have diabetes?"

# Batch processing
run-med-a2a --batch questions.txt

# API mode
run-med-a2a --json -q "Count total patients"
```

## 🛠️ Configuration Options

### Full Configuration Example
```json
{
  "_description": "Complete configuration example",
  "paths": {
    "omcp_server_path": "/home/user/omcp_server",
    "uv_executable": "/usr/local/bin/uv"
  },
  "services": {
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.1:8b"
  },
  "agent_config": {
    "omop_agent_host": "127.0.0.1",
    "omop_agent_port": 8002
  },
  "database": {
    "db_type": "duckdb",
    "cdm_schema": "base",
    "vocab_schema": "base"
  }
}
```

### Environment Variables (Alternative)
If you prefer environment variables over the config file:

```bash
export OMCP_SERVER_PATH=/path/to/omcp_server
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b
```

**Note:** Config file settings take priority over environment variables.

## 🆘 Troubleshooting

### Common Issues

**"OMCP server path not configured"**
```bash
# Check your config file
cat .medA2A.config.json

# Ensure the path exists and contains src/omcp/main.py
ls /your/path/to/omcp_server/src/omcp/main.py
```

**"UV package manager not found"**
```bash
# Install UV and restart terminal
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**"Ollama service not available"**
```bash
# Start Ollama service
ollama serve

# Check if model is downloaded
ollama list
ollama pull llama3.1:8b
```

### Debug Commands

```bash
# Check configuration sources
med-a2a-setup --show-sources

# Validate all settings
med-a2a-setup --check

# Test system functionality
med-a2a-setup --test
```

## 🎯 Example Queries

Once configured and running, try these queries:

```bash
# Patient demographics
run-med-a2a -q "How many patients are in the database?"
run-med-a2a -q "What is the average age of patients?"

# Medical conditions  
run-med-a2a -q "How many patients have hypertension?"
run-med-a2a -q "What is the most common condition?"

# Complex analysis
run-med-a2a -q "Compare diabetes prevalence between male and female patients"
```

---

🎯 **Key Points:**
- **Configuration is required** - the system will not search your filesystem
- **Use absolute paths** in your configuration file
- **Validate before running** with `med-a2a-setup --check`
- **Config file beats environment variables** for explicit control 