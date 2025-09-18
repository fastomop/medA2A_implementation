#!/bin/bash
# comprehensive_cleanup.sh - Clean up all medA2A processes and locks

echo "🧹 Starting comprehensive cleanup of medA2A processes..."

# Kill processes by pattern
echo "🛑 Terminating processes..."
pkill -f "src/omcp/main.py" 2>/dev/null && echo "✅ Killed omcp/main.py processes" || true
pkill -f "src/omcp/main_robust.py" 2>/dev/null && echo "✅ Killed omcp/main_robust.py processes" || true  
pkill -f "med-a2a-eval" 2>/dev/null && echo "✅ Killed med-a2a-eval processes" || true
pkill -f "run_omop_agent" 2>/dev/null && echo "✅ Killed run_omop_agent processes" || true
pkill -f "uvicorn.*8003" 2>/dev/null && echo "✅ Killed uvicorn processes on port 8003" || true

# Wait a moment for graceful shutdown
sleep 2

# Force kill any remaining processes
echo "⚡ Force killing any remaining processes..."
ps aux | grep -E "(omcp|main_robust|uvicorn.*8003|med-a2a)" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true

# Clean up Python cache
echo "🧹 Cleaning Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Check for remaining processes
echo "🔍 Checking for remaining processes..."
remaining=$(ps aux | grep -E "(omcp|main_robust|uvicorn.*8003|med-a2a)" | grep -v grep | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "✅ All processes cleaned up successfully"
else
    echo "⚠️ Warning: $remaining processes may still be running"
    ps aux | grep -E "(omcp|main_robust|uvicorn.*8003|med-a2a)" | grep -v grep
fi

echo "✅ Comprehensive cleanup completed!"