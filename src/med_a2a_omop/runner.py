

import asyncio
import subprocess
import time
import os
import sys
from dotenv import load_dotenv
import httpx

from a2a.client import A2AClient
from med_a2a_omop.agents.orchestrator_agent import OrchestratorAgent

class ApplicationWrapper:
    """Manages the lifecycle of our multi-agent application."""

    def __init__(self):
        load_dotenv()
        self.omop_agent_process = None
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    async def start_background_services(self):
        """Starts the OMOP Database Agent server as a background process."""
        # The command now directly uses the installed script for the OMOP agent runner
        command = [sys.executable, "-m", "med_a2a_omop.run_omop_agent"]
        
        print("🚀 Starting background OMOP Agent server...")
        self.omop_agent_process = subprocess.Popen(
            command,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr separately
            text=True,  # Decode stdout/stderr as text
            bufsize=1,  # Line-buffered
            universal_newlines=True # Ensure consistent newline handling
        )
        print(f"✅ OMOP Agent server started in background (PID: {self.omop_agent_process.pid})")
        
        # Wait for the server to be ready
        server_ready = False
        for _ in range(30): # Try for 30 seconds
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://127.0.0.1:8002/rpc")
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
            try:
                stdout_output = self.omop_agent_process.stdout.read()
                stderr_output = self.omop_agent_process.stderr.read()
            except ValueError: # Raised if stream is closed
                pass

            print(f"❌ OMOP Agent server failed to become ready! Exit Code: {self.omop_agent_process.returncode}")
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
        """Ensures the background server is cleanly terminated."""
        if self.omop_agent_process:
            print(f"\n🛑 Stopping background OMOP Agent server (PID: {self.omop_agent_process.pid})...")
            self.omop_agent_process.terminate()
            try:
                self.omop_agent_process.wait(timeout=5)
                print("✅ Server stopped cleanly.")
            except subprocess.TimeoutExpired:
                print("⚠️ Server did not terminate in time, forcing shutdown.")
                self.omop_agent_process.kill()

    async def run_main_workflow(self):
        """Runs the main orchestrator logic."""
        # Start streaming subprocess output as a background task
        if self.omop_agent_process and self.omop_agent_process.poll() is None:
            asyncio.create_task(self._stream_subprocess_output())

        # Ensure the URL includes the /rpc endpoint
        omop_agent_base_url = os.getenv("OMOP_AGENT_URL", "http://localhost:8002")
        if not omop_agent_base_url.endswith("/rpc"):
            omop_agent_url = f"{omop_agent_base_url.rstrip('/')}/rpc"
        else:
            omop_agent_url = omop_agent_base_url

        print(f"[DEBUG] Connecting to OMOP Agent at: {omop_agent_url}")
        omop_client = A2AClient(httpx_client=httpx.AsyncClient(), url=omop_agent_url)

        orchestrator = OrchestratorAgent(
            agent_id="orchestrator-01",
            omop_agent_client=omop_client
        )

        user_question = "How many patients have hypertension?"
        print(f"\n--- ❓ User Question: {user_question} ---")

        observation1 = await orchestrator.perceive(user_question)
        state1 = await orchestrator.learn(orchestrator.mental_state, observation1)
        action1 = await orchestrator.reason(state1)
        
        result1 = await orchestrator.execute(action1)
        print(f"[Orchestrator] Received response from OMOP Agent.")

        observation2 = await orchestrator.perceive(result1.data)
        state2 = await orchestrator.learn(orchestrator.mental_state, observation2)
        action2 = await orchestrator.reason(state2)

        final_result = await orchestrator.execute(action2)

        print("\n--- ✔️ Final Answer ---")
        print(final_result.data['summary'])

    async def main():
        wrapper = ApplicationWrapper()
        try:
            await wrapper.start_background_services()
            await wrapper.run_main_workflow()
        except (Exception, KeyboardInterrupt) as e:
            print(f"\nAn error occurred: {e}")
        finally:
            wrapper.stop_background_services()

if __name__ == "__main__":
    main()

