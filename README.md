
# Medical A2A OMOP Framework

A multi-agent framework for medical Q&A on OMOP data. This project provides a sophisticated architecture for posing complex medical questions in natural language and retrieving answers from an OMOP database.

## How it Works

The framework is built around a multi-agent system based on the `a2a-medical-foundation` library. It consists of two primary agents:

1.  **Orchestrator Agent**: This agent is the main entry point for user interaction. It receives a user's question, coordinates with other agents to gather the necessary information, and synthesizes a final answer.
2.  **OMOP Database Agent**: This agent acts as an interface to the OMOP database. It is responsible for translating the orchestrator's requests into database queries, executing them, and returning the results.

The `runner.py` script starts the OMOP Database Agent as a background service and then uses the Orchestrator Agent to process a sample question.

## Project Structure

```
medA2A_implementation/
├── src/
│   └── med_a2a_omop/
│       ├── agents/
│       │   ├── orchestrator_agent.py
│       │   └── omop_database_agent.py
│       ├── runner.py
│       └── run_omop_agent.py
├── pyproject.toml
└── README.md
```

-   `src/med_a2a_omop/agents/`: Contains the core logic for the different agents.
-   `src/med_a2a_omop/runner.py`: The main application entry point that orchestrates the agents.
-   `src/med_a2a_omop/run_omop_agent.py`: A script to run the OMOP Database Agent as a standalone server.
-   `pyproject.toml`: Project metadata and dependencies.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd medA2A_implementation
    ```

2.  **Create a virtual environment and install dependencies:**

    This project uses `uv` for package management.

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install uv
    uv pip install -e .
    ```

    This will install all the dependencies listed in `pyproject.toml`, including the editable install of the current project.

## Configuration

The application uses a `.env` file for configuration. Create a `.env` file in the project root and add the following variables:

```
OMOP_AGENT_URL=http://localhost:8002
OMOP_AGENT_HOST=127.0.0.1
OMOP_AGENT_PORT=8002
MCP_SERVER_URL=http://localhost:8080
```

## Usage

To run the application, use the command-line executable created by the project:

```bash
run-med-a2a
```

This will:
1.  Start the OMOP Database Agent in the background.
2.  Run the main workflow in `runner.py`, which asks the sample question: "How many patients have hypertension?"
3.  Print the final answer to the console.
4.  Stop the background agent. 