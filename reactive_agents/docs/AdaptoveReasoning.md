## 🧠 Master Prompt: Refactor AI Agent Framework for Adaptive Reasoning

You're an expert AI Coding Agent responsible for improving and extending an existing Python-based AI agent framework. The goal is to refactor the current codebase to support **dynamic, intelligent agent behavior** using adaptive reasoning strategies, contextual task classification, and flexible tool usage — with minimal complexity and high modularity.

---

### 🔧 Overall Goals

Refactor and extend the current agent framework to:

1. **Support dynamic reasoning strategies**

   - Enable agents to switch between planning, reactive, reflective, and goal-driven modes based on task type and progress.
   - Implement a default loop like **Reflect → Decide → Act**, with optional planning.

2. **Add task classification at runtime**

   - Introduce a lightweight module that classifies tasks into categories (e.g., simple lookup, multi-step reasoning, tool-driven, creative generation).
   - Use classification to inform the reasoning style selected by the agent.

3. **Allow automatic reasoning strategy pivots**

   - Allow agents to change reasoning strategies mid-run based on progress, failures, tool feedback, or reflection.

4. **Refactor tool management for broader model support**

   - Enable tool invocation through **intent blocks** or natural-language outputs (e.g., `<tool_request>` tags).
   - Decouple tool availability from model capabilities — let the framework orchestrate all tool calls.
   - Include a simple registry and router for tool execution and fallback handling.

5. **Improve context summarization and message pruning**

   - Add support for incremental assistant-side summaries after each loop or tool call.
   - Allow summary injection into the message history to maintain progress across long tasks without token overload.

6. **Keep the framework simple, composable, and developer-friendly**

   - No hardcoded loops or chains — agent behavior should be modular and declarative.
   - Allow easy creation of custom agent templates by swapping out reasoning strategies or tools.

---

### 📦 Specific Modules to Implement or Modify

- `reasoning_engine.py`

  - Handles the agent's main control loop (e.g., Reflect → Decide → Act).
  - Supports multiple reasoning styles: `reflective`, `plan_execute_reflect`, `reactive`, `self_ask`, etc.
  - Auto-switching behavior based on agent feedback or progress.

- `task_classifier.py`

  - A lightweight module that analyzes the initial task prompt.
  - Outputs labels like `["multi_step", "tool_required", "creative"]`.

- `tool_manager.py`

  - Routes tool intent outputs to real tool calls.
  - Supports non-tool-native models by parsing structured intent outputs like `<tool_request>` blocks.
  - Includes error handling and fallback behavior.

- `agent_context.py`

  - Handles assistant-side summaries.
  - Supports incremental state summarization, memory compression, and history pruning.
  - Optionally injects updated summaries into assistant messages.

- `agent_template.py`

  - Allows developers to register custom agents with reasoning modes, toolsets, memory config, etc.
  - Example:

    ```python
    ResearchAgent = AgentTemplate(
      reasoning="reflective",
      tools=[web, summarize, file_reader],
      memory="summary"
    )
    ```

---

### 🧠 Design Philosophy

- **Adaptivity first**: The agent should evolve its thinking mid-task.
- **Simplicity for devs**: All configuration should be simple and declarative.
- **Tool-agnostic orchestration**: Tools should work regardless of model type.
- **Reasoning awareness**: Agents should "know how to think," not just "do."

---

### 🛠 Instructions

- Only add or refactor what is necessary to support the above features.
- Preserve existing framework structure where possible, but improve modularity.
- Keep class and function naming clear and semantic.
- Write clean, well-documented Python.
- Add docstrings and comments to key reasoning components.

---

### ✅ Deliverables

- Refactored or new Python modules implementing the above.
- Optional: a README update or usage example showing how to define and run an agent using the new system.
