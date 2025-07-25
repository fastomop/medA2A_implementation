#!/usr/bin/env python3
"""
Wrapper script to run OMCP server via uv run.
This is needed because the MCP SDK can only execute Python scripts directly.
Includes proper cleanup to ensure database locks are released.
"""

import subprocess
import sys
import os
import signal
import atexit
import time

# Global variable to track the subprocess
omcp_process = None

def cleanup_process():
    """Cleanup function to ensure the OMCP process is terminated and database locks are freed."""
    global omcp_process
    if omcp_process and omcp_process.poll() is None:
        print("🧹 Cleaning up OMCP server process...", file=sys.stderr)
        try:
            # Send SIGTERM first for graceful shutdown
            omcp_process.terminate()
            # Wait up to 5 seconds for graceful shutdown
            omcp_process.wait(timeout=5)
            print("✅ OMCP server terminated gracefully", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("⚠️ Forcing OMCP server shutdown...", file=sys.stderr)
            # Force kill if graceful shutdown fails
            omcp_process.kill()
            try:
                omcp_process.wait(timeout=2)
                print("✅ OMCP server force-killed", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print("❌ Failed to kill OMCP server", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error during cleanup: {e}", file=sys.stderr)

def signal_handler(signum, frame):
    """Handle signals by cleaning up and exiting."""
    print(f"🛑 Received signal {signum}, shutting down...", file=sys.stderr)
    cleanup_process()
    sys.exit(0)

def main():
    global omcp_process
    
    # Register cleanup function to run on exit
    atexit.register(cleanup_process)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    try:
        # Change to the OMCP server directory
        omcp_dir = "/Users/k24118093/Documents/omcp_server"
        os.chdir(omcp_dir)
        
        # Execute uv run python src/omcp/main.py
        cmd = ["/opt/homebrew/bin/uv", "run", "python", "src/omcp/main.py"]
        
        print("🚀 Starting OMCP server with proper cleanup...", file=sys.stderr)
        
        # Start the process
        omcp_process = subprocess.Popen(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=os.environ
        )
        
        # Wait for the process to complete
        return_code = omcp_process.wait()
        
        print(f"📋 OMCP server exited with code {return_code}", file=sys.stderr)
        sys.exit(return_code)
        
    except KeyboardInterrupt:
        print("🛑 Interrupted by user", file=sys.stderr)
        cleanup_process()
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"❌ Error running OMCP server: {e}", file=sys.stderr)
        cleanup_process()
        sys.exit(1)

if __name__ == "__main__":
    main() 