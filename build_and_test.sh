#!/bin/bash
# build_and_test.sh - Clean build without problematic commands

cat src/med_a2a_omop/agents/omop_database_agent.py

set -e

echo "🧹 Cleaning processes..."
# Simple process cleanup without lsof
ps aux | grep -E "(omcp|main_robust|uvicorn)" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

echo "🧹 Cleaning Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "🧹 Removing old virtual environment..."
rm -rf .venv

echo "🔨 Creating fresh environment..."
uv venv

echo "📦 Activating and installing dependencies..."
source .venv/bin/activate
uv sync

echo "📦 Installing local package..."
uv pip install -e . --force-reinstall

echo "🧪 Verifying updated MCP parsing code is installed..."
python -c "
try:
    from med_a2a_omop.agents.omop_database_agent import OMOPDatabaseAgent
    import inspect
    source = inspect.getsource(OMOPDatabaseAgent._extract_data_from_parsed_result)
    if 'Found structured data in' in source:
        print('✅ Updated MCP parsing code confirmed in installed package')
    else:
        print('❌ Updated MCP parsing code NOT found - old code still loaded')
        exit(1)
except Exception as e:
    print(f'❌ Error checking updated parsing code: {e}')
    exit(1)
"

echo "🚀 Environment ready!"
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  uv run med-a2a-eval --limit 1 --verbose"
