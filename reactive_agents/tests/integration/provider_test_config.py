"""
Configuration management for provider consistency testing.

This module provides centralized configuration for testing all model providers
with their specific parameters, capabilities, and constraints.
"""

import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class ProviderCapability(Enum):
    """Capabilities that providers may or may not support."""

    TOOL_CALLING = "tool_calling"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"
    SYSTEM_MESSAGES = "system_messages"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    LARGE_CONTEXT = "large_context"  # >32k tokens
    LOW_LATENCY = "low_latency"  # <1s typical response


class TestComplexity(Enum):
    """Test complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    STRESS = "stress"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    capabilities: Set[ProviderCapability] = field(default_factory=set)
    recommended_params: Dict[str, Any] = field(default_factory=dict)
    cost_per_1k_tokens: Optional[float] = None
    typical_latency_ms: Optional[int] = None
    reliability_score: float = 1.0  # 0.0 to 1.0
    notes: str = ""


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    models: Dict[str, ModelConfig]
    default_params: Dict[str, Any] = field(default_factory=dict)
    required_env_vars: List[str] = field(default_factory=list)
    capabilities: Set[ProviderCapability] = field(default_factory=set)
    rate_limits: Dict[str, int] = field(default_factory=dict)  # requests per minute
    endpoint_url: Optional[str] = None
    auth_method: str = "api_key"
    timeout_seconds: int = 120
    retry_attempts: int = 3
    cost_tier: str = "unknown"  # free, paid, enterprise


@dataclass
class TestScenarioConfig:
    """Enhanced test scenario configuration."""

    name: str
    description: str
    task: str
    complexity: TestComplexity
    required_capabilities: Set[ProviderCapability] = field(default_factory=set)
    expected_tools: List[str] = field(default_factory=list)
    reasoning_strategy: str = "reactive"
    max_iterations: int = 10
    timeout_seconds: int = 120
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    provider_specific_adjustments: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    cost_budget_tokens: Optional[int] = None
    performance_targets: Dict[str, float] = field(default_factory=dict)


class ProviderTestConfiguration:
    """Centralized configuration for provider testing."""

    def __init__(self):
        """Initialize the configuration with all provider details."""
        self.providers = self._initialize_providers()
        self.test_scenarios = self._initialize_test_scenarios()
        self.global_config = self._initialize_global_config()

    def _initialize_providers(self) -> Dict[str, ProviderConfig]:
        """Initialize all provider configurations."""
        return {
            "ollama": ProviderConfig(
                name="ollama",
                models={
                    "cogito:14b": ModelConfig(
                        name="cogito:14b",
                        context_window=8192,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                        },
                        recommended_params={
                            "temperature": 0.1,
                            "num_ctx": 4000,
                            "num_gpu": 256,
                            "num_thread": 8,
                        },
                        cost_per_1k_tokens=0.0,  # Free local model
                        typical_latency_ms=2000,
                        reliability_score=0.85,
                        notes="Local model, performance depends on hardware",
                    ),
                    "qwen3": ModelConfig(
                        name="qwen3",
                        context_window=128000,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                        },
                        recommended_params={
                            "temperature": 0.1,
                            "num_ctx": 8000,
                            "num_gpu": 256,
                        },
                        cost_per_1k_tokens=0.0,
                        typical_latency_ms=1500,
                        reliability_score=0.9,
                    ),
                    "qwen2.5:14b": ModelConfig(
                        name="qwen2.5:14b",
                        context_window=32768,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                        },
                        recommended_params={"temperature": 0.1, "num_ctx": 4000},
                        cost_per_1k_tokens=0.0,
                        typical_latency_ms=1800,
                        reliability_score=0.85,
                    ),
                },
                default_params={"temperature": 0.1, "num_ctx": 4000, "stream": False},
                required_env_vars=[],  # No API key required for local
                capabilities={
                    ProviderCapability.TOOL_CALLING,
                    ProviderCapability.SYSTEM_MESSAGES,
                    ProviderCapability.STREAMING,
                    ProviderCapability.LARGE_CONTEXT,
                    ProviderCapability.LOW_LATENCY,
                },
                endpoint_url="http://localhost:11434",
                auth_method="none",
                timeout_seconds=180,
                cost_tier="free",
            ),
            "openai": ProviderConfig(
                name="openai",
                models={
                    "gpt-4o-mini": ModelConfig(
                        name="gpt-4o-mini",
                        max_tokens=4000,
                        context_window=128000,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.JSON_MODE,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.FUNCTION_CALLING,
                            ProviderCapability.LARGE_CONTEXT,
                            ProviderCapability.LOW_LATENCY,
                        },
                        recommended_params={
                            "temperature": 0.1,
                            "max_tokens": 2000,
                            "top_p": 1.0,
                        },
                        cost_per_1k_tokens=0.00015,  # Input cost
                        typical_latency_ms=800,
                        reliability_score=0.95,
                    ),
                    "gpt-3.5-turbo": ModelConfig(
                        name="gpt-3.5-turbo",
                        max_tokens=4000,
                        context_window=16385,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.JSON_MODE,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.FUNCTION_CALLING,
                            ProviderCapability.LOW_LATENCY,
                        },
                        recommended_params={"temperature": 0.1, "max_tokens": 2000},
                        cost_per_1k_tokens=0.0005,
                        typical_latency_ms=600,
                        reliability_score=0.92,
                    ),
                },
                default_params={"temperature": 0.1, "max_tokens": 2000, "top_p": 1.0},
                required_env_vars=["OPENAI_API_KEY"],
                capabilities={
                    ProviderCapability.TOOL_CALLING,
                    ProviderCapability.JSON_MODE,
                    ProviderCapability.SYSTEM_MESSAGES,
                    ProviderCapability.STREAMING,
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.LARGE_CONTEXT,
                    ProviderCapability.LOW_LATENCY,
                },
                rate_limits={"requests_per_minute": 3000, "tokens_per_minute": 250000},
                endpoint_url="https://api.openai.com/v1",
                timeout_seconds=60,
                cost_tier="paid",
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                models={
                    "claude-3-5-sonnet-latest": ModelConfig(
                        name="claude-3-5-sonnet-latest",
                        max_tokens=4000,
                        context_window=200000,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                            ProviderCapability.LOW_LATENCY,
                        },
                        recommended_params={"temperature": 0.1, "max_tokens": 2000},
                        cost_per_1k_tokens=0.00025,
                        typical_latency_ms=1000,
                        reliability_score=0.93,
                    ),
                    "claude-3-7-sonnet-latest": ModelConfig(
                        name="claude-3-7-sonnet-latest",
                        max_tokens=4000,
                        context_window=200000,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                        },
                        recommended_params={"temperature": 0.1, "max_tokens": 2000},
                        cost_per_1k_tokens=0.003,
                        typical_latency_ms=1500,
                        reliability_score=0.95,
                    ),
                },
                default_params={"temperature": 0.1, "max_tokens": 2000},
                required_env_vars=["ANTHROPIC_API_KEY"],
                capabilities={
                    ProviderCapability.TOOL_CALLING,
                    ProviderCapability.SYSTEM_MESSAGES,
                    ProviderCapability.STREAMING,
                    ProviderCapability.LARGE_CONTEXT,
                },
                rate_limits={"requests_per_minute": 1000, "tokens_per_minute": 40000},
                endpoint_url="https://api.anthropic.com",
                timeout_seconds=90,
                cost_tier="paid",
            ),
            "groq": ProviderConfig(
                name="groq",
                models={
                    "llama-3.3-70b-versatile": ModelConfig(
                        name="llama-3.3-70b-versatile",
                        max_tokens=8000,
                        context_window=131072,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.JSON_MODE,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                            ProviderCapability.LOW_LATENCY,
                        },
                        recommended_params={"temperature": 0.1, "max_tokens": 2000},
                        cost_per_1k_tokens=0.00005,
                        typical_latency_ms=300,  # Very fast
                        reliability_score=0.88,
                    ),
                    "moonshotai/kimi-k2-instruct": ModelConfig(
                        name="moonshotai/kimi-k2-instruct",
                        max_tokens=8192,
                        context_window=8192,
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.JSON_MODE,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LOW_LATENCY,
                        },
                        recommended_params={"temperature": 0.1, "max_tokens": 2000},
                        cost_per_1k_tokens=0.00005,
                        typical_latency_ms=300,
                        reliability_score=0.88,
                    ),
                },
                default_params={"temperature": 0.1, "max_tokens": 2000},
                required_env_vars=["GROQ_API_KEY"],
                capabilities={
                    ProviderCapability.TOOL_CALLING,
                    ProviderCapability.JSON_MODE,
                    ProviderCapability.SYSTEM_MESSAGES,
                    ProviderCapability.STREAMING,
                    ProviderCapability.LOW_LATENCY,
                },
                rate_limits={"requests_per_minute": 30, "tokens_per_minute": 6000},
                endpoint_url="https://api.groq.com",
                timeout_seconds=30,
                cost_tier="paid",
            ),
            "google": ProviderConfig(
                name="google",
                models={
                    "gemini-1.5-flash": ModelConfig(
                        name="gemini-1.5-flash",
                        context_window=1048576,  # 1M tokens
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                            ProviderCapability.VISION,
                            ProviderCapability.LOW_LATENCY,
                        },
                        recommended_params={"temperature": 0.1},
                        cost_per_1k_tokens=0.000075,
                        typical_latency_ms=1200,
                        reliability_score=0.90,
                    ),
                    "gemini-2.5-flash": ModelConfig(
                        name="gemini-2.5-flash",
                        context_window=2097152,  # 2M tokens
                        capabilities={
                            ProviderCapability.TOOL_CALLING,
                            ProviderCapability.SYSTEM_MESSAGES,
                            ProviderCapability.STREAMING,
                            ProviderCapability.LARGE_CONTEXT,
                            ProviderCapability.VISION,
                        },
                        recommended_params={"temperature": 0.1},
                        cost_per_1k_tokens=0.00125,
                        typical_latency_ms=2000,
                        reliability_score=0.93,
                    ),
                },
                default_params={"temperature": 0.1},
                required_env_vars=["GOOGLE_API_KEY"],
                capabilities={
                    ProviderCapability.TOOL_CALLING,
                    ProviderCapability.SYSTEM_MESSAGES,
                    ProviderCapability.STREAMING,
                    ProviderCapability.LARGE_CONTEXT,
                    ProviderCapability.VISION,
                },
                rate_limits={"requests_per_minute": 360, "tokens_per_minute": 30000},
                endpoint_url="https://generativelanguage.googleapis.com",
                timeout_seconds=120,
                cost_tier="paid",
            ),
        }

    def _initialize_test_scenarios(self) -> Dict[str, TestScenarioConfig]:
        """Initialize comprehensive test scenarios."""
        return {
            "basic_tool_call": TestScenarioConfig(
                name="basic_tool_call",
                description="Test basic tool calling capability",
                task="Use the test_tool_simple to process the text 'Hello World'",
                complexity=TestComplexity.SIMPLE,
                required_capabilities={ProviderCapability.TOOL_CALLING},
                expected_tools=["test_tool_simple"],
                reasoning_strategy="reactive",
                max_iterations=3,
                timeout_seconds=60,
                validation_criteria={
                    "should_contain": ["Processed:", "Hello World"],
                    "min_tool_calls": 1,
                    "max_tool_calls": 2,
                },
                performance_targets={
                    "max_execution_time": 10.0,
                    "min_quality_score": 0.8,
                },
            ),
            "multi_step_math": TestScenarioConfig(
                name="multi_step_math",
                description="Test multi-step mathematical reasoning with tools",
                task="Use test_tool_math to calculate 25 + 17, then multiply the result by 3",
                complexity=TestComplexity.MODERATE,
                required_capabilities={ProviderCapability.TOOL_CALLING},
                expected_tools=["test_tool_math"],
                reasoning_strategy="reflect_decide_act",
                max_iterations=8,
                timeout_seconds=90,
                validation_criteria={
                    "should_contain": ["42", "126"],
                    "min_tool_calls": 2,
                    "max_tool_calls": 4,
                },
                provider_specific_adjustments={
                    "ollama": {"max_iterations": 10, "timeout_seconds": 120},
                    "groq": {"max_iterations": 6},  # Faster model
                },
                performance_targets={
                    "max_execution_time": 15.0,
                    "min_quality_score": 0.85,
                },
            ),
            "json_processing": TestScenarioConfig(
                name="json_processing",
                description="Test JSON data processing capabilities",
                task="Use test_tool_json to process this data: {'name': 'test', 'value': 123}",
                complexity=TestComplexity.SIMPLE,
                required_capabilities={ProviderCapability.TOOL_CALLING},
                expected_tools=["test_tool_json"],
                reasoning_strategy="reactive",
                max_iterations=5,
                timeout_seconds=60,
                validation_criteria={
                    "should_contain": ["processed", "true", "test", "123"],
                    "min_tool_calls": 1,
                    "max_tool_calls": 2,
                },
            ),
            "complex_workflow": TestScenarioConfig(
                name="complex_workflow",
                description="Test complex multi-tool workflow execution",
                task="First use test_tool_simple with 'Start', then use test_tool_math to add 10 and 5, finally use test_tool_json with the result",
                complexity=TestComplexity.COMPLEX,
                required_capabilities={ProviderCapability.TOOL_CALLING},
                expected_tools=["test_tool_simple", "test_tool_math", "test_tool_json"],
                reasoning_strategy="plan_execute_reflect",
                max_iterations=15,
                timeout_seconds=180,
                validation_criteria={
                    "should_contain": ["Processed: Start", "15"],
                    "min_tool_calls": 3,
                    "max_tool_calls": 6,
                },
                provider_specific_adjustments={
                    "ollama": {"timeout_seconds": 240, "max_iterations": 20}
                },
                performance_targets={
                    "max_execution_time": 30.0,
                    "min_quality_score": 0.75,
                },
            ),
            "adaptive_reasoning": TestScenarioConfig(
                name="adaptive_reasoning",
                description="Test adaptive reasoning strategy selection",
                task="Determine the best approach to solve: What is 8 * 7, then add 6 to it?",
                complexity=TestComplexity.MODERATE,
                required_capabilities={ProviderCapability.TOOL_CALLING},
                expected_tools=["test_tool_math"],
                reasoning_strategy="adaptive",
                max_iterations=12,
                timeout_seconds=120,
                validation_criteria={
                    "should_contain": ["56", "62"],
                    "min_tool_calls": 2,
                    "max_tool_calls": 4,
                },
            ),
            "error_recovery": TestScenarioConfig(
                name="error_recovery",
                description="Test error handling and recovery capabilities",
                task="Use test_tool_math with invalid parameters, then recover and complete the task correctly",
                complexity=TestComplexity.COMPLEX,
                required_capabilities={ProviderCapability.TOOL_CALLING},
                expected_tools=["test_tool_math"],
                reasoning_strategy="reflect_decide_act",
                max_iterations=15,
                timeout_seconds=150,
                validation_criteria={
                    "should_contain": ["error", "recovered", "correct"],
                    "min_tool_calls": 2,
                },
            ),
            "high_iteration_stress": TestScenarioConfig(
                name="high_iteration_stress",
                description="Stress test with high iteration count",
                task="Count from 1 to 10 using test_tool_simple, processing each number individually",
                complexity=TestComplexity.STRESS,
                required_capabilities={ProviderCapability.TOOL_CALLING},
                expected_tools=["test_tool_simple"],
                reasoning_strategy="reactive",
                max_iterations=25,
                timeout_seconds=300,
                validation_criteria={
                    "should_contain": ["1", "5", "10"],
                    "min_tool_calls": 8,
                },
                cost_budget_tokens=5000,
            ),
        }

    def _initialize_global_config(self) -> Dict[str, Any]:
        """Initialize global test configuration."""
        return {
            "default_timeout": 120,
            "max_retries": 3,
            "retry_delay": 1.0,
            "parallel_execution": False,  # Set to True for faster testing
            "collect_detailed_metrics": True,
            "save_conversation_logs": False,  # Enable for debugging
            "cost_tracking": True,
            "performance_profiling": True,
            "fail_fast": False,  # Continue testing even if some providers fail
            "mock_mode_default": True,  # Use mocks by default unless specified
            "log_level": "INFO",
            "output_formats": ["json", "markdown", "csv"],
            "comparison_metrics": [
                "execution_time",
                "success_rate",
                "quality_score",
                "cost_efficiency",
                "reliability",
            ],
        }

    def get_providers_for_scenario(self, scenario_name: str) -> List[str]:
        """Get list of providers that support a given scenario."""
        if scenario_name not in self.test_scenarios:
            return []

        scenario = self.test_scenarios[scenario_name]
        compatible_providers = []

        for provider_name, provider_config in self.providers.items():
            # Check if provider has required capabilities
            if scenario.required_capabilities.issubset(provider_config.capabilities):
                compatible_providers.append(provider_name)

        return compatible_providers

    def get_recommended_models(
        self, provider_name: str, max_cost: Optional[float] = None
    ) -> List[str]:
        """Get recommended models for a provider within cost constraints."""
        if provider_name not in self.providers:
            return []

        provider = self.providers[provider_name]
        models = []

        for model_name, model_config in provider.models.items():
            if max_cost is None or (model_config.cost_per_1k_tokens or 0) <= max_cost:
                models.append(model_name)

        # Sort by reliability score and cost
        models.sort(
            key=lambda m: (
                -provider.models[m].reliability_score,
                provider.models[m].cost_per_1k_tokens or 0,
            )
        )

        return models

    def get_adjusted_params(
        self, provider_name: str, model_name: str, scenario_name: str
    ) -> Dict[str, Any]:
        """Get provider/model/scenario specific parameters."""
        params = {}

        # Start with provider defaults
        if provider_name in self.providers:
            params.update(self.providers[provider_name].default_params)

        # Add model recommendations
        if (
            provider_name in self.providers
            and model_name in self.providers[provider_name].models
        ):
            model_params = (
                self.providers[provider_name].models[model_name].recommended_params
            )
            params.update(model_params)

        # Apply scenario-specific adjustments
        if scenario_name in self.test_scenarios:
            scenario = self.test_scenarios[scenario_name]
            if provider_name in scenario.provider_specific_adjustments:
                adjustments = scenario.provider_specific_adjustments[provider_name]
                params.update(adjustments)

        return params

    def validate_environment(self, provider_names: List[str]) -> Dict[str, bool]:
        """Validate environment setup for given providers."""
        validation_results = {}

        for provider_name in provider_names:
            if provider_name not in self.providers:
                validation_results[provider_name] = False
                continue

            provider = self.providers[provider_name]
            is_valid = True

            # Check required environment variables
            for env_var in provider.required_env_vars:
                if not os.getenv(env_var):
                    is_valid = False
                    break

            validation_results[provider_name] = is_valid

        return validation_results

    def estimate_test_cost(
        self, provider_names: List[str], scenario_names: List[str]
    ) -> Dict[str, float]:
        """Estimate testing costs for given providers and scenarios."""
        cost_estimates = {}

        for provider_name in provider_names:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]
            total_cost = 0.0

            for model_name, model_config in provider.models.items():
                model_cost = 0.0

                for scenario_name in scenario_names:
                    if scenario_name not in self.test_scenarios:
                        continue

                    scenario = self.test_scenarios[scenario_name]

                    # Estimate tokens based on scenario complexity
                    estimated_tokens = {
                        TestComplexity.SIMPLE: 500,
                        TestComplexity.MODERATE: 1500,
                        TestComplexity.COMPLEX: 3000,
                        TestComplexity.STRESS: 5000,
                    }.get(scenario.complexity, 1000)

                    if scenario.cost_budget_tokens:
                        estimated_tokens = min(
                            estimated_tokens, scenario.cost_budget_tokens
                        )

                    if model_config.cost_per_1k_tokens:
                        scenario_cost = (
                            estimated_tokens / 1000
                        ) * model_config.cost_per_1k_tokens
                        model_cost += scenario_cost

                total_cost += model_cost

            cost_estimates[provider_name] = total_cost

        return cost_estimates

    def export_config(self, filepath: str) -> None:
        """Export configuration to JSON file."""
        config_data = {
            "providers": {
                name: {
                    "name": config.name,
                    "models": {
                        model_name: {
                            "name": model.name,
                            "max_tokens": model.max_tokens,
                            "context_window": model.context_window,
                            "capabilities": [cap.value for cap in model.capabilities],
                            "recommended_params": model.recommended_params,
                            "cost_per_1k_tokens": model.cost_per_1k_tokens,
                            "typical_latency_ms": model.typical_latency_ms,
                            "reliability_score": model.reliability_score,
                            "notes": model.notes,
                        }
                        for model_name, model in config.models.items()
                    },
                    "default_params": config.default_params,
                    "required_env_vars": config.required_env_vars,
                    "capabilities": [cap.value for cap in config.capabilities],
                    "rate_limits": config.rate_limits,
                    "endpoint_url": config.endpoint_url,
                    "auth_method": config.auth_method,
                    "timeout_seconds": config.timeout_seconds,
                    "retry_attempts": config.retry_attempts,
                    "cost_tier": config.cost_tier,
                }
                for name, config in self.providers.items()
            },
            "test_scenarios": {
                name: {
                    "name": scenario.name,
                    "description": scenario.description,
                    "task": scenario.task,
                    "complexity": scenario.complexity.value,
                    "required_capabilities": [
                        cap.value for cap in scenario.required_capabilities
                    ],
                    "expected_tools": scenario.expected_tools,
                    "reasoning_strategy": scenario.reasoning_strategy,
                    "max_iterations": scenario.max_iterations,
                    "timeout_seconds": scenario.timeout_seconds,
                    "validation_criteria": scenario.validation_criteria,
                    "provider_specific_adjustments": scenario.provider_specific_adjustments,
                    "cost_budget_tokens": scenario.cost_budget_tokens,
                    "performance_targets": scenario.performance_targets,
                }
                for name, scenario in self.test_scenarios.items()
            },
            "global_config": self.global_config,
        }

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)


# Singleton instance
_config_instance = None


def get_test_config() -> ProviderTestConfiguration:
    """Get the singleton test configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ProviderTestConfiguration()
    return _config_instance


if __name__ == "__main__":
    # Example usage and validation
    config = get_test_config()

    print("Provider Test Configuration")
    print("=" * 40)

    print(f"Available providers: {list(config.providers.keys())}")
    print(f"Available scenarios: {list(config.test_scenarios.keys())}")

    # Test environment validation
    env_status = config.validate_environment(["ollama", "openai"])
    print(f"Environment validation: {env_status}")

    # Test cost estimation
    costs = config.estimate_test_cost(
        ["ollama", "openai"], ["basic_tool_call", "multi_step_math"]
    )
    print(f"Estimated costs: {costs}")

    # Export configuration
    config.export_config("provider_test_config.json")
    print("Configuration exported to provider_test_config.json")
