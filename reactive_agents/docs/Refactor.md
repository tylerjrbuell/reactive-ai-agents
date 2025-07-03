# 🧠 Reactive Agent Framework — Refactor & Design Specification

A high-level design spec and implementation guide for refactoring and extending a lightweight, Python-based agent framework built primarily for **local models (via OLAMA)**, but capable of supporting any LLM with tool usage and adaptive reasoning.

---

## 🎯 Framework Vision

Create a Laravel-like development framework for **building reactive AI agents** that:

* Are **task- and context-aware**, adapting their reasoning style automatically.
* Can **coordinate with other agents** via a common communication protocol.
* Support **natural language-based configuration** to simplify developer onboarding.
* Prioritize **local-first design**, with optional remote or hybrid model use.
* Offer a plug-and-play system for tools, workflows, memory, and reasoning strategies.

---

## 🔁 Core Agent Architecture

### ✅ Modular Reasoning Loop (Reflect → Decide → Act)

The default control loop supports:

* Reflection over current task state
* Deciding the next best action (tool call, generation, plan)
* Executing that action and feeding back results

Other strategies available (and swappable):

* Plan–Execute–Reflect
* Self-Ask
* Reactive (no plan, pure prompt–response)
* Goal-driven (e.g., GAF: Goal → Action → Feedback)

### 🧠 Dynamic Reasoning Style Switching

Agents can pivot between reasoning strategies mid-task, based on:

* Task complexity
* Tool result feedback
* Self-evaluated stagnation
* Task classification metadata

---

## 🗂️ Task Classification System

A lightweight classifier (`task_classifier.py`) will label tasks at runtime:

* `simple_lookup`
* `tool_required`
* `creative_generation`
* `multi_step`
* `agent_collaboration`
* `external_context_required`

Classification is used to:

* Inform tool usage
* Select initial reasoning style
* Determine if the task should be delegated to another agent

---

## 🛠️ Tool Management

### 🔌 Tool Invocation Support

* Supports both tool-native models and tool-agnostic local models
* Models emit **intent blocks** like:

  ```xml
  <tool_request>
    <tool>web_search</tool>
    <input>Latest AI alignment papers</input>
  </tool_request>
  ```
* `tool_manager.py` handles parsing, routing, execution, and fallback

### ↺ Tool Registry & Routing

Tools are defined declaratively in the agent template:

```python
tools = [
  {"name": "web_search", "fn": web_search, "fallback": fallback_search},
  {"name": "run_code", "fn": code_executor}
]
```

---

## 🧠 Natural Language Config System

Agents can be **created and configured via natural language**, allowing:

* Dynamic instantiation with no hardcoded YAML or code blocks
* Prompt-based configuration parsing:

  > “Create an agent that can analyze PDFs, summarize research, and collaborate with another agent using shared memory.”

This would generate:

```python
AgentConfig(
  tools=["pdf_reader", "summarize"],
  reasoning="plan_execute_reflect",
  communication_protocol="a2a"
)
```

---

## 🧐 Agent-to-Agent (A2A) Collaboration

Inspired by Google's **A2A protocol**, this framework will support:

* A standard message schema for inter-agent communication
* Shared task queues or delegation protocols
* Optional broadcast/response models for swarm coordination

### A2A Features

* Send/receive structured requests between agents
* Ask another agent for help, tools, or perspectives
* Shared memory or document chains (eventually peer-to-peer)

---

## 📚 Workflow System

Agents can be linked in **workflow graphs** that describe sequences, branching, or parallel execution:

* Declarative DAG-style structure
* Auto-routed task handoff
* Shared context or results between nodes
* Example:

```python
workflow = [
  {"agent": "PlannerAgent", "output_to": "WorkerAgent"},
  {"agent": "WorkerAgent", "output_to": "QA_Agent"},
  {"agent": "QA_Agent", "output_to": "SummarizerAgent"}
]
```

---

## 🧠 Context & Memory Management

### Summary Memory

* After each loop or tool call, agents summarize task progress
* Inject assistant summary messages to preserve context without overflow

### Message Pruning

* Old steps replaced with updated summaries
* Important interactions retained based on salience scoring or tags

---

## 🛡️ Developer API & Framework Style

Modeled after Laravel-style design principles:

* **Declarative, clean API**
* Modular, composable components
* Extensible templates for reasoning modes, tools, memory, workflows
* CLI or Web UI for agent creation via prompt

---

## ✅ Deliverables

* Updated agent control loop
* Task classifier module
* Tool intent routing system
* Natural language config parser
* A2A messaging layer (base version)
* Workflow orchestrator
* Summary-based context management
* Example agents and workflows
* Dev guide for defining new agents via prompt or config
