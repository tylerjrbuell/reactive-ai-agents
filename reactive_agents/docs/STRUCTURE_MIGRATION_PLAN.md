# 🚀 Structure Migration Plan

## 🎯 Migration Strategy: Current → Proposed Structure

This document outlines the exact steps to migrate from the current flat structure to the proposed Laravel-inspired hierarchical organization.

---

## 📊 Current vs. Proposed Mapping

### Directory Mappings

| Current Location          | New Location                    | Reasoning                                |
| ------------------------- | ------------------------------- | ---------------------------------------- |
| `agents/`                 | `app/agents/` + `app/builders/` | Separate agent definitions from builders |
| `components/`             | `core/` (multiple dirs)         | Split by functionality                   |
| `reasoning/`              | `core/reasoning/`               | Core framework component                 |
| `tools/`                  | `core/tools/`                   | Core framework component                 |
| `memory/`                 | `core/memory/`                  | Core framework component                 |
| `components/event_bus.py` | `core/events/`                  | Dedicated event system                   |
| `model_providers/`        | `providers/llm/`                | External integrations                    |
| `cli/`                    | `console/`                      | Laravel-style naming                     |
| `config/`                 | `config/`                       | Centralized configuration                |
| `common/types/`           | `core/types/`                   | Core type system                         |
| `communication/`          | `app/communication/`            | Application-level feature                |
| `workflows/`              | `app/workflows/`                | Application-level feature                |
| `context/`                | `core/engine/`                  | Part of execution engine                 |
| `loggers/`                | `utils/logging.py`              | Utility function                         |
| `prompts/`                | `core/reasoning/prompts/`       | Part of reasoning system                 |
| `examples/`               | `examples/`                     | Stays the same, but reorganized          |
| `tests/`                  | `testing/` + root `tests/`      | Enhanced testing system                  |

---

## 🛠️ Phase 1: Core Restructure (Week 1-2)

### Step 1: Create New Directory Structure

```bash
# Create main directories
mkdir -p reactive_agents/app/{agents,builders,workflows,communication}
mkdir -p reactive_agents/core/{engine,reasoning,memory,events,tools,types}
mkdir -p reactive_agents/providers/{llm,storage,external}
mkdir -p reactive_agents/console/{commands,output,stubs}
mkdir -p reactive_agents/plugins/{interfaces,examples}
mkdir -p reactive_agents/utils
mkdir -p reactive_agents/testing/{fixtures,mocks,factories}
mkdir -p reactive_agents/storage/{agents,workflows,logs,memory,cache}

# Create subdirectories
mkdir -p reactive_agents/app/agents/{factories,templates}
mkdir -p reactive_agents/app/workflows/{nodes,templates}
mkdir -p reactive_agents/app/communication/channels
mkdir -p reactive_agents/core/reasoning/{strategies,prompts}
mkdir -p reactive_agents/core/events/middleware
mkdir -p reactive_agents/core/tools/processors
mkdir -p reactive_agents/console/commands/{make,agent,config,db}
mkdir -p reactive_agents/config/{templates,schema}
mkdir -p reactive_agents/examples/{quickstart,advanced,integrations}
```

### Step 2: Move and Refactor Core Components

#### A. Agent System Migration

```bash
# Current: agents/ → New: app/agents/ + app/builders/
mv reactive_agents/agents/base.py reactive_agents/app/agents/
mv reactive_agents/agents/reactive_agent_v2.py reactive_agents/app/agents/reactive_agent.py
mv reactive_agents/agents/builders.py reactive_agents/app/builders/agent_builder.py
```

**Update imports in `app/agents/reactive_agent.py`:**

```python
# Change:
from reactive_agents.components.execution_engine import AgentExecutionEngine
# To:
from reactive_agents.core.engine.execution_engine import AgentExecutionEngine

# Change:
from reactive_agents.components.memory_manager import MemoryManager
# To:
from reactive_agents.core.memory.memory_manager import MemoryManager
```

#### B. Core Framework Migration

```bash
# Move reasoning system
mv reactive_agents/reasoning/* reactive_agents/core/reasoning/strategies/
mv reactive_agents/prompts/* reactive_agents/core/reasoning/prompts/

# Move memory system
mv reactive_agents/components/memory_manager.py reactive_agents/core/memory/memory_manager.py
mv reactive_agents/components/vector_memory_manager.py reactive_agents/core/memory/vector_memory.py

# Move event system
mv reactive_agents/components/event_bus.py reactive_agents/core/events/event_bus.py

# Move tool system
mv reactive_agents/components/tool_manager.py reactive_agents/core/tools/tool_manager.py
mv reactive_agents/tools/* reactive_agents/core/tools/

# Move execution engine
mv reactive_agents/components/execution_engine.py reactive_agents/core/engine/execution_engine.py
mv reactive_agents/components/task_executor.py reactive_agents/core/engine/task_executor.py
mv reactive_agents/context/* reactive_agents/core/engine/

# Move types
mv reactive_agents/common/types/* reactive_agents/core/types/
```

#### C. Provider System Migration

```bash
# Move model providers
mv reactive_agents/model_providers/* reactive_agents/providers/llm/

# Move external integrations
mv reactive_agents/communication/a2a_official_bridge.py reactive_agents/providers/external/a2a_sdk.py
```

#### D. Application Layer Migration

```bash
# Move workflows
mv reactive_agents/workflows/* reactive_agents/app/workflows/

# Move communication
mv reactive_agents/communication/* reactive_agents/app/communication/
```

### Step 3: Create New Framework Components

#### A. Plugin System (`plugins/plugin_manager.py`)

```python
"""
Plugin Management System

Provides hot-loading and management of framework extensions.
"""

from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
import importlib
import inspect
from pathlib import Path

class PluginInterface(ABC):
    """Base interface for all plugins"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the plugin"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass

class PluginManager:
    """Manages framework plugins and extensions"""

    def __init__(self):
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_paths: List[Path] = []

    def add_plugin_path(self, path: Path) -> None:
        """Add a directory to search for plugins"""
        self.plugin_paths.append(path)

    async def load_plugin(self, plugin_name: str) -> PluginInterface:
        """Load and initialize a plugin"""
        # Implementation for plugin loading
        pass

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a loaded plugin by name"""
        return self.loaded_plugins.get(name)

    async def unload_plugin(self, name: str) -> None:
        """Unload a plugin"""
        if name in self.loaded_plugins:
            await self.loaded_plugins[name].cleanup()
            del self.loaded_plugins[name]
```

#### B. Enhanced CLI System (`console/cli.py`)

```python
"""
Enhanced CLI System

Laravel Artisan-inspired command line interface.
"""

import asyncio
import sys
from typing import Dict, Type, List
from pathlib import Path

from reactive_agents.console.commands.base import Command
from reactive_agents.console.output.colors import ConsoleColors

class ReactiveCLI:
    """Main CLI application"""

    def __init__(self):
        self.commands: Dict[str, Type[Command]] = {}
        self.colors = ConsoleColors()
        self._register_default_commands()

    def _register_default_commands(self):
        """Register all built-in commands"""
        from reactive_agents.console.commands.make.agent import MakeAgentCommand
        from reactive_agents.console.commands.agent.run import RunAgentCommand
        # ... register all commands

    def register_command(self, name: str, command_class: Type[Command]):
        """Register a custom command"""
        self.commands[name] = command_class

    async def run(self, args: List[str] = None):
        """Execute CLI with given arguments"""
        if args is None:
            args = sys.argv[1:]

        if not args:
            self.show_help()
            return

        command_name = args[0]
        if command_name not in self.commands:
            self.colors.error(f"Unknown command: {command_name}")
            return

        command = self.commands[command_name]()
        await command.execute(args[1:])
```

### Step 4: Update Import System

#### A. Create new `__init__.py` files with proper exports

**`reactive_agents/__init__.py`** (Main package init):

```python
"""
Reactive Agents Framework

A Laravel-inspired framework for building reactive AI agents.
"""

# Public API exports
from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.app.builders.agent_builder import ReactiveAgentBuilder
from reactive_agents.app.workflows.orchestrator import WorkflowOrchestrator
from reactive_agents.config.settings import Settings

# Backward compatibility
from reactive_agents.app.agents.reactive_agent import ReactiveAgent as ReactAgent

__version__ = "2.0.0"
__all__ = [
    "ReactiveAgent",
    "ReactiveAgentBuilder",
    "WorkflowOrchestrator",
    "Settings"
]
```

**`reactive_agents/app/__init__.py`**:

```python
"""
Application Layer

User-facing components for building agents and workflows.
"""

from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.app.builders.agent_builder import ReactiveAgentBuilder
from reactive_agents.app.workflows.orchestrator import WorkflowOrchestrator

__all__ = ["ReactiveAgent", "ReactiveAgentBuilder", "WorkflowOrchestrator"]
```

**`reactive_agents/core/__init__.py`**:

```python
"""
Framework Core

Internal framework components and utilities.
"""

from reactive_agents.core.engine.execution_engine import ExecutionEngine
from reactive_agents.core.reasoning.strategy_manager import StrategyManager
from reactive_agents.core.memory.memory_manager import MemoryManager
from reactive_agents.core.events.event_bus import EventBus

__all__ = ["ExecutionEngine", "StrategyManager", "MemoryManager", "EventBus"]
```

---

## 🔄 Phase 2: Enhanced Features (Week 3-4)

### Step 1: Implement Plugin Architecture

```python
# Create plugin interfaces
# reactive_agents/plugins/interfaces/strategy_plugin.py
from reactive_agents.plugins.plugin_manager import PluginInterface
from reactive_agents.core.reasoning.base import ReasoningStrategies

class StrategyPlugin(PluginInterface):
    """Interface for reasoning strategy plugins"""

    @abstractmethod
    def get_strategies(self) -> List[Type[ReasoningStrategies]]:
        """Return strategy classes provided by this plugin"""
        pass
```

### Step 2: Enhanced Configuration System

```python
# reactive_agents/config/settings.py
from pydantic import BaseSettings
from typing import Dict, Any, Optional

class Settings(BaseSettings):
    """Global framework settings"""

    # Core settings
    debug: bool = False
    log_level: str = "INFO"

    # Agent defaults
    default_model: str = "ollama:qwen2:7b"
    default_strategy: str = "reflect_decide_act"

    # Storage settings
    storage_path: str = "storage"
    memory_backend: str = "json"  # json, chromadb, redis

    # Plugin settings
    plugin_paths: List[str] = ["plugins"]
    auto_load_plugins: bool = True

    class Config:
        env_prefix = "REACTIVE_"
        env_file = ".env"
```

### Step 3: Testing Framework

```python
# reactive_agents/testing/test_case.py
import asyncio
import unittest
from typing import Any, Dict, Optional

from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.testing.mocks.mock_llm import MockLLMProvider

class ReactiveTestCase(unittest.TestCase):
    """Base test case for framework testing"""

    def setUp(self):
        """Set up test environment"""
        self.mock_llm = MockLLMProvider()

    async def create_test_agent(self, **kwargs) -> ReactiveAgent:
        """Create an agent for testing"""
        from reactive_agents.app.builders.agent_builder import ReactiveAgentBuilder

        builder = ReactiveAgentBuilder() \
            .with_name("TestAgent") \
            .with_llm_provider(self.mock_llm)

        for key, value in kwargs.items():
            if hasattr(builder, f"with_{key}"):
                getattr(builder, f"with_{key}")(value)

        return await builder.build()
```

---

## 🧪 Phase 3: Validation & Testing (Week 5)

### Migration Validation Checklist

- [ ] **All imports work correctly**

  - [ ] Public API imports unchanged
  - [ ] Internal imports updated
  - [ ] Backward compatibility maintained

- [ ] **Core functionality preserved**

  - [ ] Agent creation works
  - [ ] Reasoning strategies functional
  - [ ] Tool execution operational
  - [ ] Memory system working

- [ ] **New features operational**

  - [ ] Plugin system functional
  - [ ] Enhanced CLI working
  - [ ] Configuration system operational
  - [ ] Testing framework usable

- [ ] **Performance maintained**
  - [ ] No significant slowdowns
  - [ ] Memory usage reasonable
  - [ ] Import times acceptable

### Test Migration Script

```python
#!/usr/bin/env python3
"""
Migration validation script

Tests that all functionality works after restructure.
"""

import asyncio
import sys
from pathlib import Path

async def test_basic_agent_creation():
    """Test basic agent creation still works"""
    try:
        from reactive_agents import ReactiveAgentBuilder

        agent = await ReactiveAgentBuilder() \
            .with_name("Migration Test Agent") \
            .with_model("ollama:qwen2:7b") \
            .build()

        print("✅ Agent creation works")
        await agent.close()
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

async def test_plugin_system():
    """Test plugin system functionality"""
    try:
        from reactive_agents.plugins.plugin_manager import PluginManager

        manager = PluginManager()
        print("✅ Plugin manager works")
        return True
    except Exception as e:
        print(f"❌ Plugin system failed: {e}")
        return False

async def test_cli_system():
    """Test CLI system functionality"""
    try:
        from reactive_agents.console.cli import ReactiveCLI

        cli = ReactiveCLI()
        print("✅ CLI system works")
        return True
    except Exception as e:
        print(f"❌ CLI system failed: {e}")
        return False

async def main():
    """Run all migration tests"""
    print("🧪 Testing migration...")

    tests = [
        test_basic_agent_creation,
        test_plugin_system,
        test_cli_system,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    if all(results):
        print("\n🎉 Migration successful! All tests passed.")
        return 0
    else:
        print(f"\n❌ Migration issues found. {sum(results)}/{len(results)} tests passed.")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

---

## 📋 Migration Timeline

| Week | Phase             | Activities                                         | Deliverables                             |
| ---- | ----------------- | -------------------------------------------------- | ---------------------------------------- |
| 1-2  | Core Restructure  | Move files, update imports, maintain compatibility | New directory structure, working imports |
| 3-4  | Enhanced Features | Plugin system, enhanced CLI, configuration         | Plugin architecture, enhanced CLI        |
| 5    | Validation        | Testing, performance checks, documentation         | Migration validation, updated docs       |

---

## 🎯 Success Criteria

### Functional Requirements

- [ ] All existing APIs continue to work
- [ ] No breaking changes for end users
- [ ] Plugin system operational
- [ ] Enhanced CLI functional

### Quality Requirements

- [ ] Import time < 2 seconds
- [ ] Memory usage increase < 20%
- [ ] All tests passing
- [ ] Documentation updated

### Developer Experience

- [ ] Intuitive directory navigation
- [ ] Clear separation of concerns
- [ ] Easy plugin development
- [ ] Improved debugging experience

---

This migration plan ensures a smooth transition from the current structure to the proposed Laravel-inspired organization while maintaining full backward compatibility and adding powerful new features.
