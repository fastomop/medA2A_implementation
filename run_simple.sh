#!/bin/bash
# Simplified script to run just the medical A2A implementation
# The MCP server should now be launched as a subprocess by the framework

echo "🏥 Starting Medical A2A Implementation (with integrated MCP server)"
echo "=================================================================="

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Shutting down..."
    # Kill all child processes
    pkill -P $$
    wait
    echo "✅ All processes stopped"
}

# Set up trap to call cleanup on script exit
trap cleanup EXIT INT TERM

# Start the medical A2A implementation
echo "🚀 Starting Medical A2A Implementation..."
cd /Users/k24118093/Documents/medA2A_implementation
uv run run-med-a2a

echo -e "\n✅ Medical A2A Implementation completed"