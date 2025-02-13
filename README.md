# reactive-ai-agent

A custom reactive AI Agent framework that allow for creating reactive agents to carry out tasks using tools.

## Overview

The main purpose of this project is to create a custom AI Agent Framework that allows AI Agents driven by Large Language Models (LLMs) to make real-time decisions and take action to solve real-world tasks. Key features include:

- **Model Providers**: Currently Supports `Ollama` for open-source models (local) or `Groq` fast cloud-based models.
- **Agent Reflection**: The agent has the ability to reflect on its previous actions, improve as it iterates, and grade itself until it arrives at a final result.
- **Tool Integration**: Agents can take tools as ordinary Python functions and use a `@tool()` decorator to transform these functions into function definitions that the language model can understand.

## Installation Instructions

To install and set up this project locally, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/projectname.git
   cd projectname
   ```

2. Install dependencies using Poetry:

   ```sh
   poetry install
   ```

3. Configure your environment by setting up necessary variables in `.env`.

## Usage Details

To run the application and explore its functionalities, follow these steps:

1. Run the main application script with:

   ```sh
   python main.py
   ```

2. Explore additional functionalities through sub-modules like `agents`, `model_providers`, and `tools`.

3. For detailed usage of specific modules, refer to their respective documentation or source code.

## Running Tests

To ensure the project is functioning correctly, you can run the following commands:

1. Navigate to the root directory of the project.
2. Install dependencies using `poetry install`.
3. Run the tests with:
   ```sh
   poetry run pytest
   ```

If there are specific test scripts or configurations, please refer to the `.env` file and any additional documentation provided.

## Modules Description

- **agents**: Contains classes and functions related to AI agents.

- **model_providers**: Provides interfaces to different machine learning models.

  - `ModelProvider`: An abstract base class or interface for providing access to various ML models. Specific methods include:
    - `initialize(model_name: str) -> None`: Initialize the specified model.
    - `generate_response(prompt: str) -> str`: Generate a response to the given prompt.

- **prompts**: Stores prompt templates used in the application.

  - `PromptTemplate`: A class that defines a template for generating prompts used by AI agents. Key methods include:
    - `__init__(template_path: str) -> None`: Initializes the prompt template from a file path.
    - `generate_prompt(context: dict) -> str`: Generates a prompt based on the provided context.

- **tools**: Includes utility scripts and functions.
  - `UtilityFunction`: A collection of utility functions that provide common operations or services. Key methods include:
    - `__init__() -> None`: Initializes the utility function with logging setup.
    - `log_info(message: str) -> None`: Logs an info message.
    - `handle_file(file_path: str) -> bool`: Handles a file operation and returns success status.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
