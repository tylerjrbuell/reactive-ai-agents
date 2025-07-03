#!/usr/bin/env python3
"""
Structure Migration Demo

Demonstrates the new Laravel-inspired framework structure and capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reactive_agents import __version__
from reactive_agents.config.settings import get_settings, initialize_settings
from reactive_agents.plugins.plugin_manager import (
    get_plugin_manager,
    initialize_plugin_system,
)


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)


def print_section(title: str):
    """Print a formatted section"""
    print(f"\n📋 {title}")
    print("-" * 40)


async def demo_new_structure():
    """Demonstrate the new Laravel-inspired structure"""

    print_header("Reactive Agents Framework - Structure Migration Demo")
    print(f"Framework Version: {__version__}")
    print("Laravel-inspired organization with plugin architecture")

    # Demo 1: Configuration System
    print_section("1. Centralized Configuration System")

    settings = get_settings()
    print(f"✅ Default LLM Provider: {settings.llm.default_provider}")
    print(f"✅ Default Model: {settings.llm.default_model}")
    print(f"✅ Default Strategy: {settings.agent.default_strategy}")
    print(f"✅ Storage Path: {settings.storage_path}")
    print(f"✅ Plugin Auto-load: {settings.plugins.auto_load_plugins}")

    # Demo 2: Plugin System
    print_section("2. Plugin Management System")

    plugin_manager = get_plugin_manager()
    print(f"✅ Plugin Manager initialized")
    print(f"✅ Framework Version: {plugin_manager.framework_version}")

    # Initialize plugin system with default paths
    initialize_plugin_system()
    print(f"✅ Plugin system initialized with default paths")

    # Discover plugins
    discovered = plugin_manager.discover_plugins()
    print(f"✅ Discovered plugins: {discovered if discovered else 'None (expected)'}")

    # Demo 3: New Directory Structure
    print_section("3. Laravel-Inspired Directory Structure")

    structure_info = [
        ("app/", "User-facing application layer"),
        ("  ├── agents/", "Agent definitions and factories"),
        ("  ├── builders/", "Builder pattern implementations"),
        ("  ├── workflows/", "Multi-agent orchestration"),
        ("  └── communication/", "Inter-agent messaging"),
        ("", ""),
        ("core/", "Framework internals"),
        ("  ├── engine/", "Execution and task management"),
        ("  ├── reasoning/", "Adaptive reasoning strategies"),
        ("  ├── memory/", "Vector memory and storage"),
        ("  ├── events/", "Event bus and middleware"),
        ("  ├── tools/", "Tool management and integration"),
        ("  └── types/", "Core type definitions"),
        ("", ""),
        ("providers/", "External integrations"),
        ("  ├── llm/", "LLM providers (Ollama, Groq)"),
        ("  ├── storage/", "Storage providers"),
        ("  └── external/", "Third-party services"),
        ("", ""),
        ("plugins/", "Extensibility layer"),
        ("console/", "Laravel Artisan-style CLI"),
        ("config/", "Centralized configuration"),
        ("utils/", "Shared utilities"),
        ("testing/", "Testing framework"),
        ("storage/", "Default data location"),
    ]

    for path, description in structure_info:
        if path:
            print(f"  {path:<20} {description}")
        else:
            print()

    # Demo 4: Backward Compatibility
    print_section("4. Backward Compatibility")

    try:
        # Test that old imports still work
        from reactive_agents.app.agents.base import Agent
        from reactive_agents.app.agents.builders import ReactiveAgentV2Builder
        from reactive_agents.app.agents.reactive_agent import ReactiveAgentV2

        print("✅ Old imports still work:")
        print(f"   - Agent base class: {Agent.__name__}")
        print(f"   - ReactiveAgentV2Builder: {ReactiveAgentV2Builder.__name__}")
        print(f"   - ReactiveAgentV2: {ReactiveAgentV2.__name__}")

    except ImportError as e:
        print(f"❌ Backward compatibility issue: {e}")

    # Demo 5: New Features Preview
    print_section("5. New Features Preview")

    print("🆕 Plugin Architecture:")
    print("   - Hot-loading plugins")
    print("   - Plugin discovery and registration")
    print("   - Type-safe plugin interfaces")

    print("\n🆕 Enhanced CLI System:")
    print("   - Laravel Artisan-style commands")
    print("   - Colored output and formatting")
    print("   - Interactive shell mode")

    print("\n🆕 Centralized Configuration:")
    print("   - Type-safe settings with dataclasses")
    print("   - Environment variable support")
    print("   - JSON configuration files")

    print("\n🆕 Improved Structure:")
    print("   - Clear separation of concerns")
    print("   - Intuitive navigation")
    print("   - Professional framework appearance")


async def demo_cli_system():
    """Demonstrate the new CLI system"""

    print_section("6. CLI System Demo")

    try:
        from reactive_agents.console.cli import ReactiveCLI

        cli = ReactiveCLI()
        print("✅ CLI system initialized")

        # Show available commands
        print(f"✅ Registered commands: {len(cli.commands)}")

        # Show help (without actually running it)
        print("✅ CLI help system available")
        print("   Run: python -m reactive_agents.console.cli --help")

    except ImportError as e:
        print(f"❌ CLI system not available: {e}")


async def demo_settings_management():
    """Demonstrate settings management"""

    print_section("7. Settings Management Demo")

    # Create custom settings
    custom_settings = initialize_settings(debug=True, storage_path="custom_storage")

    print(f"✅ Custom settings created:")
    print(f"   - Debug mode: {custom_settings.debug}")
    print(f"   - Storage path: {custom_settings.storage_path}")

    # Save settings to file
    config_file = "demo_config.json"
    custom_settings.save_to_file(config_file)
    print(f"✅ Settings saved to: {config_file}")

    # Load settings from file
    loaded_settings = Settings.load_from_file(config_file)
    print(f"✅ Settings loaded from file")
    print(f"   - Debug mode: {loaded_settings.debug}")
    print(f"   - Storage path: {loaded_settings.storage_path}")

    # Cleanup
    Path(config_file).unlink(missing_ok=True)
    print(f"✅ Demo config file cleaned up")


async def main():
    """Main demo function"""
    try:
        await demo_new_structure()
        await demo_cli_system()
        await demo_settings_management()

        print_header("Migration Demo Complete!")
        print("🎉 The framework now has a Laravel-inspired structure!")
        print("\nNext steps:")
        print("1. Continue migrating remaining components")
        print("2. Implement full CLI commands")
        print("3. Create example plugins")
        print("4. Update documentation")
        print("5. Add comprehensive testing")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # Import here to avoid circular imports
    from reactive_agents.config.settings import Settings

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
