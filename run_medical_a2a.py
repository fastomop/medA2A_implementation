#!/usr/bin/env python3
"""
Simplified runner for the medical A2A implementation.
The framework now handles MCP server launching automatically.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from med_a2a_omop.runner import ApplicationWrapper

async def main():
    """Main entry point for medical A2A system."""
    print("🏥 Starting Medical A2A Implementation")
    print("=" * 50)
    print("ℹ️  The framework will automatically:")
    print("   • Launch OMCP MCP server as subprocess")
    print("   • Start OMOP Database Agent")
    print("   • Initialize orchestrator")
    print("   • Process the question: 'How many patients have hypertension?'")
    print()
    
    try:
        # Create and run the application
        app = ApplicationWrapper()
        await app.run_main_workflow()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())