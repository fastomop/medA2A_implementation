

import asyncio
import subprocess
import time
import os
import sys
import signal
import atexit
import argparse
import json
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import httpx

from a2a.client import A2AClient
from med_a2a_omop.agents.orchestrator_agent import OrchestratorAgent
from a2a_medical.base.agent import ActionResult

# Keep the original ApplicationWrapper for backward compatibility
class ApplicationWrapper:
    """Manages the lifecycle of our multi-agent application with proper cleanup."""
    
    def __init__(self):
        # Import config here to avoid circular imports
        from .config import get_config
        self.config = get_config()
        
        load_dotenv()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.omop_agent_process = None
        self._shutdown_requested = False
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)

    def cleanup_all(self):
        """Comprehensive cleanup of all processes and resources."""
        if self._shutdown_requested:
            return  # Avoid double cleanup
        self._shutdown_requested = True
        
        print("\n🧹 Starting comprehensive cleanup...")
        self.stop_background_services()
        
        # Additional cleanup: kill any remaining OMCP processes
        try:
            result = subprocess.run(
                ["pkill", "-f", "src/omcp/main.py"], 
                capture_output=True, 
                timeout=5
            )
            if result.returncode == 0:
                print("✅ Cleaned up any remaining OMCP processes")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # pkill might not be available or no processes found
        
        print("✅ Comprehensive cleanup completed")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.cleanup_all()
        sys.exit(0)

    def cleanup_existing_locks(self):
        """Clean up any existing database locks before starting."""
        print("🔍 Checking for existing database locks...")
        try:
            # Find processes that might be holding the database lock
            result = subprocess.run(
                ["pgrep", "-f", "synthea.duckdb"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"🧹 Found {len(pids)} processes potentially holding database locks")
                
                for pid in pids:
                    try:
                        pid = int(pid.strip())
                        print(f"🛑 Terminating process {pid}...")
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(1)  # Give it time to clean up
                        
                        # Check if it's still running
                        try:
                            os.kill(pid, 0)  # This will raise an exception if process is dead
                            print(f"⚠️ Force killing process {pid}...")
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            print(f"✅ Process {pid} terminated successfully")
                            
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        print(f"⚠️ Could not terminate process {pid}: {e}")
                        
            else:
                print("✅ No existing database locks found")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️ Could not check for existing locks (pgrep not available)")
        except Exception as e:
            print(f"⚠️ Error checking for locks: {e}")

    async def start_background_services(self):
        """Starts the OMOP Database Agent server as a background process."""
        # The command now directly uses the installed script for the OMOP agent runner
        command = [sys.executable, "-m", "med_a2a_omop.run_omop_agent"]
        
        # Set up environment to pass config file location
        env = os.environ.copy()
        if self.config.config_file:
            env['MEDA2A_CONFIG_FILE'] = str(self.config.config_file)
        
        print("🚀 Starting background OMOP Agent server...")
        self.omop_agent_process = subprocess.Popen(
            command,
            cwd=self.project_root,
            env=env,  # Pass the environment with config file path
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,  # Decode stdout/stderr as text
            bufsize=1,  # Line-buffered
            universal_newlines=True # Ensure consistent newline handling
        )
        print(f"✅ OMOP Agent server started in background (PID: {self.omop_agent_process.pid})")
        
        # Wait for the server to be ready
        server_ready = False
        for attempt in range(30): # Try for 30 seconds
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://127.0.0.1:8002/.well-known/agent-card.json")
                    if response.status_code == 200: # Or whatever indicates readiness
                        server_ready = True
                        break
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)

        if not server_ready:
            # Read any remaining output from the process
            stdout_output = ""
            stderr_output = ""
            if self.omop_agent_process and self.omop_agent_process.stdout:
                try:
                    stdout_output = self.omop_agent_process.stdout.read()
                except ValueError: # Raised if stream is closed
                    pass
            if self.omop_agent_process and self.omop_agent_process.stderr:
                try:
                    stderr_output = self.omop_agent_process.stderr.read()
                except ValueError: # Raised if stream is closed
                    pass

            print(f"❌ OMOP Agent server failed to become ready! Exit Code: {self.omop_agent_process.returncode if self.omop_agent_process else 'N/A'}")
            if stdout_output:
                print(f"[OMOP Agent STDOUT]:\n{stdout_output}")
            if stderr_output:
                print(f"[OMOP Agent STDERR]:\n{stderr_output}")
            raise RuntimeError("OMOP Agent server failed to start")
        else:
            print("✅ OMOP Agent server is running")

    async def _stream_subprocess_output(self):
        """Streams output from the subprocess to the console."""
        print("[OMOP Agent Live Output]:")
        if self.omop_agent_process and self.omop_agent_process.stdout:
            while True:
                line = await asyncio.to_thread(self.omop_agent_process.stdout.readline) # Use to_thread for blocking read
                if not line:
                    break
                print(f"    {line.strip()}")
        print("[OMOP Agent Output Stream Ended]")

    def stop_background_services(self):
        """Ensures the background server is cleanly terminated with enhanced cleanup."""
        if self.omop_agent_process:
            print(f"\n🛑 Stopping background OMOP Agent server (PID: {self.omop_agent_process.pid})...")
            try:
                # First, try graceful termination
                self.omop_agent_process.terminate()
                self.omop_agent_process.wait(timeout=10)  # Increased timeout for cleanup
                print("✅ Server stopped cleanly.")
            except subprocess.TimeoutExpired:
                print("⚠️ Server did not terminate in time, forcing shutdown.")
                self.omop_agent_process.kill()
                try:
                    self.omop_agent_process.wait(timeout=5)
                    print("✅ Server force-killed successfully.")
                except subprocess.TimeoutExpired:
                    print("❌ Failed to kill server process.")
            except Exception as e:
                print(f"❌ Error stopping server: {e}")

class MedA2AInterface(ApplicationWrapper):
    """
    Enhanced interface for the Medical A2A OMOP system supporting multiple interaction modes:
    - Interactive CLI
    - Batch processing
    - Programmatic API
    """
    
    def __init__(self):
        super().__init__()
        
        self.orchestrator = None
        self.omop_client = None
        
    async def initialize(self):
        """Initialize the system and start background services."""
        print("🚀 Initializing Medical A2A OMOP System...")
        
        # Validate environment before starting
        issues = self.config.validate_setup()
        if issues:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            
            print("\n📋 Setup Instructions:")
            instructions = self.config.get_setup_instructions()
            for instruction in instructions:
                print(instruction)
                print()
            
            raise RuntimeError("Environment not properly configured. Please follow setup instructions above.")
        
        print("✅ Environment validation passed")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Clean up any existing database locks
        self.cleanup_existing_locks()
        
        # Start background services
        await self.start_background_services()
        
        # Initialize orchestrator
        omop_agent_config = self.config.get_omop_agent_config()
        omop_agent_url = f"{omop_agent_config['url'].rstrip('/')}/rpc"

        print(f"[DEBUG] Connecting to OMOP Agent at: {omop_agent_url}")
        self.omop_client = A2AClient(httpx_client=httpx.AsyncClient(timeout=60.0), url=omop_agent_url)
        
        self.orchestrator = OrchestratorAgent(
            agent_id="orchestrator-01",
            omop_agent_client=self.omop_client,
            ollama_model=self.config.get_ollama_model()
        )
        
        print("✅ System initialized successfully!")
    
    async def ask_single_question(self, question: str) -> Dict[str, Any]:
        """
        Processes a single question by delegating entirely to the orchestrator agent's
        internal control loop.
        """
        if not self.orchestrator:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        print(f"\n--- ❓ Processing Question: {question} ---")
        
        try:
            # Delegate the entire process to the agent's control loop
            final_result = await self.orchestrator.process_query(question)

            # Format and return the final result
            return self._format_final_result(question, final_result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._format_final_result(question, ActionResult(success=False, error=str(e)))

    def _format_final_result(self, question: str, final_result: ActionResult) -> Dict[str, Any]:
        """Formats the final ActionResult into a consistent dictionary."""
        result_data = {
            "question": question,
            "success": final_result.success,
            "timestamp": time.time()
        }
        
        if final_result.success and final_result.data:
            if isinstance(final_result.data, dict):
                result_data.update(final_result.data)
                result_data["answer"] = final_result.data.get('summary', str(final_result.data))
            else:
                result_data["answer"] = str(final_result.data)
        else:
            result_data["error"] = final_result.error if hasattr(final_result, 'error') else 'Unknown error'
            result_data["answer"] = f"An error occurred: {result_data['error']}"
        
        return result_data
    
    async def ask_multiple_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in sequence.
        
        Args:
            questions: List of natural language medical questions
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        print(f"\n🔄 Processing {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing question...")
            result = await self.ask_single_question(question)
            results.append(result)
            
            # Brief pause between questions to avoid overwhelming the system
            if i < len(questions):
                await asyncio.sleep(1)
        
        return results
    
    async def interactive_mode(self):
        """
        Interactive command-line interface for asking questions.
        """
        print("\n🎯 Interactive Medical A2A OMOP Query Interface")
        print("=" * 60)
        print("Enter your medical questions (type 'quit', 'exit', or press Ctrl+C to stop)")
        print("Type 'help' for available commands")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if question.lower() == 'examples':
                    self._show_examples()
                    continue
                
                # Process the question
                result = await self.ask_single_question(question)
                
                # Display result
                print(f"\n--- ✔️ Answer ---")
                print(result["answer"])
                
                if result["success"] and "generated_sql" in result:
                    print(f"\n--- 📝 Generated SQL ---")
                    print(result["generated_sql"])
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\n📚 Available Commands:")
        print("  help      - Show this help message")
        print("  examples  - Show example questions")
        print("  quit/exit - Exit the program")
        print("\n💡 Tips:")
        print("  - Ask questions about patient counts, conditions, medications, etc.")
        print("  - Use natural language - the system will convert to SQL")
        print("  - Examples: 'How many patients have diabetes?', 'What drugs are prescribed for hypertension?'")
    
    def _show_examples(self):
        """Show example questions."""
        print("\n📋 Example Questions:")
        print("  • How many patients have hypertension?")
        print("  • What is the average age of patients with diabetes?")
        print("  • How many patients are taking metformin?")
        print("  • What are the most common conditions in the database?")
        print("  • How many patients have both diabetes and hypertension?")
        print("  • What is the gender distribution of patients with heart disease?")
    
    async def batch_from_file(self, file_path: str, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process questions from a file.
        
        Args:
            file_path: Path to file containing questions (one per line or JSON)
            output_file: Optional path to save results
            
        Returns:
            List of result dictionaries
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    questions = data
                elif isinstance(data, dict) and 'questions' in data:
                    questions = data['questions']
                else:
                    raise ValueError("Invalid JSON format")
            except json.JSONDecodeError:
                # Treat as plain text file (one question per line)
                questions = [line.strip() for line in content.split('\n') if line.strip()]
            
            print(f"📁 Loaded {len(questions)} questions from {file_path}")
            
            # Process questions
            results = await self.ask_multiple_questions(questions)
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Results saved to {output_file}")
            
            return results
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {e}")

async def main_async():
    """Enhanced main function with multiple interaction modes."""
    parser = argparse.ArgumentParser(
        description="Medical A2A OMOP Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python -m med_a2a_omop.runner

  # Single question
  python -m med_a2a_omop.runner -q "How many patients have diabetes?"

  # Multiple questions
  python -m med_a2a_omop.runner -q "How many patients have diabetes?" -q "What drugs are used for hypertension?"

  # Batch from file
  python -m med_a2a_omop.runner --batch questions.txt --output results.json

  # Programmatic mode (returns JSON)
  python -m med_a2a_omop.runner --json -q "How many patients have hypertension?"
        """
    )
    
    parser.add_argument(
        '-q', '--question',
        action='append',
        help='Ask a specific question (can be used multiple times)'
    )
    
    parser.add_argument(
        '--batch',
        help='Process questions from a file (one per line or JSON format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Save results to a file (JSON format)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode (default if no other options)'
    )
    
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Show example questions and exit'
    )
    
    args = parser.parse_args()
    
    # Show examples and exit
    if args.examples:
        print("\n📋 Example Medical Questions:")
        print("  • How many patients have hypertension?")
        print("  • What is the average age of patients with diabetes?")
        print("  • How many patients are taking metformin?")
        print("  • What are the most common conditions in the database?")
        print("  • How many patients have both diabetes and hypertension?")
        print("  • What is the gender distribution of patients with heart disease?")
        print("  • How many lab tests were performed last year?")
        print("  • What medications are prescribed most frequently?")
        return
    
    # Initialize the system
    interface = MedA2AInterface()
    
    try:
        await interface.initialize()
        
        # Determine mode based on arguments
        if args.batch:
            # Batch processing mode
            print(f"📁 Batch processing from file: {args.batch}")
            results = await interface.batch_from_file(args.batch, args.output)
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                print(f"\n✅ Processed {len(results)} questions")
                for i, result in enumerate(results, 1):
                    print(f"\n[{i}] {result['question']}")
                    print(f"    Answer: {result['answer']}")
                    
        elif args.question:
            # Single or multiple question mode
            if len(args.question) == 1:
                # Single question
                result = await interface.ask_single_question(args.question[0])
                
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"\n--- ✔️ Answer ---")
                    print(result["answer"])
                    if result["success"] and "generated_sql" in result:
                        print(f"\n--- 📝 Generated SQL ---")
                        print(result["generated_sql"])
            else:
                # Multiple questions
                results = await interface.ask_multiple_questions(args.question)
                
                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print(f"\n✅ Processed {len(results)} questions")
                    for i, result in enumerate(results, 1):
                        print(f"\n[{i}] {result['question']}")
                        print(f"    Answer: {result['answer']}")
                
                # Save results if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"💾 Results saved to {args.output}")
                    
        else:
            # Interactive mode (default)
            await interface.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"❌ Error: {e}")
    finally:
        interface.cleanup_all()

def main():
    """The main synchronous entry point for the application script."""
    try:
        asyncio.run(main_async())
    except (Exception, KeyboardInterrupt) as e:
        print(f"An error occurred during shutdown: {e}")

# Programmatic API for external use
class MedA2AAPI:
    """
    Programmatic API for the Medical A2A OMOP system.
    Use this class to integrate with other Python applications.
    """
    
    def __init__(self):
        self.interface: Optional[MedA2AInterface] = None
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the system."""
        if self._initialized:
            return
            
        self.interface = MedA2AInterface()
        await self.interface.initialize()
        self._initialized = True
    
    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a single question.
        
        Args:
            question: Natural language medical question
            
        Returns:
            Dictionary with result
        """
        if not self._initialized or not self.interface:
            await self.initialize()
        
        assert self.interface is not None
        return await self.interface.ask_single_question(question)
    
    async def ask_multiple(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Ask multiple questions.
        
        Args:
            questions: List of natural language medical questions
            
        Returns:
            List of result dictionaries
        """
        if not self._initialized or not self.interface:
            await self.initialize()
        
        assert self.interface is not None
        return await self.interface.ask_multiple_questions(questions)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.interface:
            self.interface.cleanup_all()
        self._initialized = False

if __name__ == "__main__":
    main()

