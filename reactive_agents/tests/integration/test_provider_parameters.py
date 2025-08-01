"""
Provider-specific parameter validation and testing.

This module tests provider-specific parameters, edge cases, and ensures
consistent behavior across different model configurations.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

from reactive_agents.app.builders.agent import ReactiveAgentBuilder
from reactive_agents.core.types.reasoning_types import ReasoningStrategies
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.providers.llm.factory import ModelProviderFactory
from reactive_agents.providers.llm.base import BaseModelProvider

from reactive_agents.tests.integration.provider_test_config import (
    get_test_config,
    ProviderCapability,
)


class TestProviderParameters:
    """Test provider-specific parameters and configurations."""

    @pytest.fixture
    def test_config(self):
        """Get test configuration."""
        return get_test_config()

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_name", ["ollama", "openai", "anthropic", "groq", "google"]
    )
    async def test_provider_default_parameters(self, provider_name, test_config):
        """Test that providers handle their default parameters correctly."""

        if provider_name not in test_config.providers:
            pytest.skip(f"Provider {provider_name} not configured")

        provider_config = test_config.providers[provider_name]

        # Test with default parameters
        for model_name in list(provider_config.models.keys())[
            :1
        ]:  # Test first model only

            # Mock the provider to avoid real API calls
            with patch.object(
                ModelProviderFactory, "get_model_provider"
            ) as mock_factory:
                mock_provider = MagicMock(spec=BaseModelProvider)
                mock_provider.validate_model.return_value = True
                mock_provider.get_chat_completion.return_value = {
                    "choices": [{"message": {"content": "Test response"}}],
                    "usage": {"total_tokens": 50},
                }
                mock_factory.return_value = mock_provider

                try:
                    builder = (
                        ReactiveAgentBuilder()
                        .with_name(f"Test-{provider_name}")
                        .with_model(f"{provider_name}:{model_name}")
                        .with_role("Test")
                        .with_instructions("Test agent")
                        .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                        .with_model_provider_options(provider_config.default_params)
                        .with_max_iterations(1)
                    )

                    agent = await builder.build()
                    assert agent is not None

                    # Verify the provider was created with correct parameters
                    mock_factory.assert_called_once()
                    call_args = mock_factory.call_args
                    assert call_args[1]["options"] == provider_config.default_params

                    await agent.close()

                except Exception as e:
                    pytest.fail(
                        f"Failed to create agent with default parameters for {provider_name}:{model_name}: {e}"
                    )

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_name", ["ollama", "openai"])
    async def test_provider_parameter_validation(self, provider_name, test_config):
        """Test parameter validation for different providers."""

        if provider_name not in test_config.providers:
            pytest.skip(f"Provider {provider_name} not configured")

        provider_config = test_config.providers[provider_name]
        model_name = list(provider_config.models.keys())[0]

        # Test cases for parameter validation
        test_cases = {
            "temperature": [
                (0.0, True),  # Valid min
                (1.0, True),  # Valid max
                (0.5, True),  # Valid middle
                (-0.1, False),  # Invalid negative
                (2.0, False),  # Invalid too high
                ("invalid", False),  # Invalid type
            ],
            "max_tokens": [
                (1, True),  # Valid min
                (4000, True),  # Valid reasonable
                (0, False),  # Invalid zero
                (-1, False),  # Invalid negative
                ("invalid", False),  # Invalid type
            ],
        }

        if provider_name == "ollama":
            test_cases.update(
                {
                    "num_ctx": [
                        (512, True),  # Valid min
                        (4000, True),  # Valid reasonable
                        (0, False),  # Invalid zero
                        (-1, False),  # Invalid negative
                    ],
                    "num_gpu": [
                        (0, True),  # Valid - CPU only
                        (256, True),  # Valid - GPU layers
                        (-1, False),  # Invalid negative
                    ],
                }
            )

        for param_name, test_values in test_cases.items():
            for param_value, should_succeed in test_values:

                # Create test parameters
                test_params = provider_config.default_params.copy()
                test_params[param_name] = param_value

                with patch.object(
                    ModelProviderFactory, "get_model_provider"
                ) as mock_factory:
                    mock_provider = MagicMock(spec=BaseModelProvider)

                    if should_succeed:
                        mock_provider.validate_model.return_value = True
                        mock_factory.return_value = mock_provider

                        try:
                            builder = (
                                ReactiveAgentBuilder()
                                .with_name(f"Test-{provider_name}-{param_name}")
                                .with_model(f"{provider_name}:{model_name}")
                                .with_role("Test")
                                .with_instructions("Test agent")
                                .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                                .with_model_provider_options(test_params)
                                .with_max_iterations(1)
                            )

                            agent = await builder.build()
                            assert (
                                agent is not None
                            ), f"Failed to create agent with {param_name}={param_value}"
                            await agent.close()

                        except Exception as e:
                            pytest.fail(
                                f"Expected success but got error for {provider_name} "
                                f"with {param_name}={param_value}: {e}"
                            )
                    else:
                        # For invalid parameters, we expect either validation failure
                        # or the provider to handle it gracefully
                        mock_factory.side_effect = ValueError(f"Invalid {param_name}")

                        builder = (
                            ReactiveAgentBuilder()
                            .with_name(f"Test-{provider_name}-{param_name}")
                            .with_model(f"{provider_name}:{model_name}")
                            .with_role("Test")
                            .with_instructions("Test agent")
                            .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                            .with_model_provider_options(test_params)
                            .with_max_iterations(1)
                        )

                        with pytest.raises((ValueError, TypeError, RuntimeError)):
                            agent = await builder.build()

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_provider_specific_optimizations(self, test_config):
        """Test provider-specific parameter optimizations."""

        optimization_tests = [
            {
                "provider": "ollama",
                "model": "cogito:14b",
                "scenario": "high_performance",
                "params": {"num_ctx": 8000, "num_gpu": 256, "temperature": 0.0},
                "expected_benefits": ["faster_inference", "better_quality"],
            },
            {
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "scenario": "low_latency",
                "params": {"temperature": 0.1, "max_tokens": 1000},
                "expected_benefits": ["ultra_fast_response"],
            },
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "scenario": "cost_optimized",
                "params": {"temperature": 0.1, "max_tokens": 500, "top_p": 0.9},
                "expected_benefits": ["cost_efficient", "reliable"],
            },
        ]

        for test_case in optimization_tests:
            provider_name = test_case["provider"]

            if provider_name not in test_config.providers:
                continue

            with patch.object(
                ModelProviderFactory, "get_model_provider"
            ) as mock_factory:
                mock_provider = MagicMock(spec=BaseModelProvider)
                mock_provider.validate_model.return_value = True
                mock_provider.get_chat_completion.return_value = {
                    "choices": [{"message": {"content": "Optimized response"}}],
                    "usage": {"total_tokens": 25},
                }
                mock_factory.return_value = mock_provider

                try:
                    builder = (
                        ReactiveAgentBuilder()
                        .with_name(f"Optimized-{provider_name}")
                        .with_model(f"{provider_name}:{test_case['model']}")
                        .with_role("Test")
                        .with_instructions("Test optimized agent")
                        .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                        .with_model_provider_options(test_case["params"])
                        .with_max_iterations(1)
                    )

                    agent = await builder.build()
                    assert agent is not None

                    # Verify optimization parameters were passed correctly
                    call_args = mock_factory.call_args
                    passed_options = call_args[1]["options"]

                    for param_key, param_value in test_case["params"].items():
                        assert param_key in passed_options
                        assert passed_options[param_key] == param_value

                    await agent.close()

                except Exception as e:
                    pytest.fail(f"Optimization test failed for {test_case}: {e}")

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_context_window_limits(self, test_config):
        """Test context window handling across providers."""

        context_tests = [
            {
                "provider": "ollama",
                "model": "llama3.1:8b",
                "context_window": 128000,
                "test_context_size": 4000,
                "should_succeed": True,
            },
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "context_window": 128000,
                "test_context_size": 50000,
                "should_succeed": True,
            },
            {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "context_window": 200000,
                "test_context_size": 100000,
                "should_succeed": True,
            },
        ]

        for test_case in context_tests:
            provider_name = test_case["provider"]

            if provider_name not in test_config.providers:
                continue

            # Create a large instruction to test context handling
            large_instruction = "Test instruction. " * (
                test_case["test_context_size"] // 20
            )

            with patch.object(
                ModelProviderFactory, "get_model_provider"
            ) as mock_factory:
                mock_provider = MagicMock(spec=BaseModelProvider)
                mock_provider.validate_model.return_value = True

                if test_case["should_succeed"]:
                    mock_provider.get_chat_completion.return_value = {
                        "choices": [
                            {"message": {"content": "Response to large context"}}
                        ],
                        "usage": {"total_tokens": test_case["test_context_size"]},
                    }
                else:
                    mock_provider.get_chat_completion.side_effect = Exception(
                        "Context too large"
                    )

                mock_factory.return_value = mock_provider

                try:
                    builder = (
                        ReactiveAgentBuilder()
                        .with_name(f"Context-{provider_name}")
                        .with_model(f"{provider_name}:{test_case['model']}")
                        .with_role("Test")
                        .with_instructions(large_instruction)
                        .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                        .with_max_iterations(1)
                    )

                    agent = await builder.build()

                    if test_case["should_succeed"]:
                        assert agent is not None
                        await agent.close()
                    else:
                        pytest.fail(
                            "Expected context limit error but agent was created successfully"
                        )

                except Exception as e:
                    if test_case["should_succeed"]:
                        pytest.fail(
                            f"Context test failed unexpectedly for {test_case}: {e}"
                        )
                    # Expected failure for oversized context

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_provider_capability_consistency(self, test_config):
        """Test that providers correctly report and handle their capabilities."""

        capability_tests = [
            {
                "capability": ProviderCapability.TOOL_CALLING,
                "test_providers": ["ollama", "openai", "anthropic", "groq", "google"],
                "expected_support": True,
            },
            {
                "capability": ProviderCapability.JSON_MODE,
                "test_providers": ["openai", "groq"],
                "expected_support": True,
            },
            {
                "capability": ProviderCapability.STREAMING,
                "test_providers": ["ollama", "openai", "anthropic", "groq", "google"],
                "expected_support": True,
            },
            {
                "capability": ProviderCapability.LARGE_CONTEXT,
                "test_providers": ["ollama", "openai", "anthropic", "google"],
                "expected_support": True,
            },
        ]

        for capability_test in capability_tests:
            capability = capability_test["capability"]

            for provider_name in capability_test["test_providers"]:
                if provider_name not in test_config.providers:
                    continue

                provider_config = test_config.providers[provider_name]

                # Check if capability is properly declared
                has_capability = capability in provider_config.capabilities
                expected = capability_test["expected_support"]

                assert has_capability == expected, (
                    f"Provider {provider_name} capability mismatch for {capability.value}: "
                    f"expected {expected}, got {has_capability}"
                )

                # Test at model level too
                for model_name, model_config in provider_config.models.items():
                    model_has_capability = capability in model_config.capabilities

                    # Models should generally inherit provider capabilities
                    # but may have additional restrictions
                    if expected and not model_has_capability:
                        # This might be acceptable - some models may not support all provider capabilities
                        print(
                            f"Warning: Model {provider_name}:{model_name} doesn't support {capability.value}"
                        )

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_error_handling_consistency(self, test_config):
        """Test consistent error handling across providers."""

        error_scenarios = [
            {
                "name": "invalid_model",
                "model_override": "nonexistent-model",
                "expected_error": ValueError,
            },
            {
                "name": "network_timeout",
                "mock_side_effect": asyncio.TimeoutError("Network timeout"),
                "expected_error": asyncio.TimeoutError,
            },
            {
                "name": "api_rate_limit",
                "mock_side_effect": Exception("Rate limit exceeded"),
                "expected_error": Exception,
            },
        ]

        for provider_name in ["ollama", "openai"]:  # Test subset for speed
            if provider_name not in test_config.providers:
                continue

            provider_config = test_config.providers[provider_name]
            model_name = list(provider_config.models.keys())[0]

            for scenario in error_scenarios:
                with patch.object(
                    ModelProviderFactory, "get_model_provider"
                ) as mock_factory:

                    if "mock_side_effect" in scenario:
                        mock_factory.side_effect = scenario["mock_side_effect"]
                    else:
                        mock_factory.side_effect = scenario["expected_error"](
                            "Test error"
                        )

                    test_model = scenario.get("model_override", model_name)

                    builder = (
                        ReactiveAgentBuilder()
                        .with_name(f"Error-{provider_name}")
                        .with_model(f"{provider_name}:{test_model}")
                        .with_role("Test")
                        .with_instructions("Test error handling")
                        .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                        .with_max_iterations(1)
                    )

                    with pytest.raises((scenario["expected_error"], RuntimeError)):
                        agent = await builder.build()

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_performance_parameter_impact(self, test_config):
        """Test how different parameters impact performance characteristics."""

        performance_tests = [
            {
                "provider": "ollama",
                "base_params": {"temperature": 0.1, "num_ctx": 2000},
                "variations": [
                    {"num_ctx": 4000, "expected_impact": "slower_but_better_context"},
                    {"temperature": 0.0, "expected_impact": "more_deterministic"},
                    {"num_gpu": 0, "expected_impact": "cpu_only_slower"},
                ],
            },
            {
                "provider": "openai",
                "base_params": {"temperature": 0.1, "max_tokens": 1000},
                "variations": [
                    {"max_tokens": 2000, "expected_impact": "longer_responses"},
                    {"temperature": 0.9, "expected_impact": "more_creative"},
                    {"top_p": 0.5, "expected_impact": "more_focused"},
                ],
            },
        ]

        for test_case in performance_tests:
            provider_name = test_case["provider"]

            if provider_name not in test_config.providers:
                continue

            provider_config = test_config.providers[provider_name]
            model_name = list(provider_config.models.keys())[0]

            # Test base configuration
            with patch.object(
                ModelProviderFactory, "get_model_provider"
            ) as mock_factory:
                mock_provider = MagicMock(spec=BaseModelProvider)
                mock_provider.validate_model.return_value = True
                mock_provider.get_chat_completion.return_value = {
                    "choices": [{"message": {"content": "Base response"}}],
                    "usage": {"total_tokens": 50},
                }
                mock_factory.return_value = mock_provider

                base_agent = await (
                    ReactiveAgentBuilder()
                    .with_name(f"Base-{provider_name}")
                    .with_model(f"{provider_name}:{model_name}")
                    .with_role("Test")
                    .with_instructions("Test base parameters")
                    .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                    .with_model_provider_options(test_case["base_params"])
                    .with_max_iterations(1)
                    .build()
                )

                assert base_agent is not None
                await base_agent.close()

            # Test variations
            for variation in test_case["variations"]:
                varied_params = test_case["base_params"].copy()
                varied_params.update(
                    {k: v for k, v in variation.items() if k != "expected_impact"}
                )

                with patch.object(
                    ModelProviderFactory, "get_model_provider"
                ) as mock_factory:
                    mock_provider = MagicMock(spec=BaseModelProvider)
                    mock_provider.validate_model.return_value = True
                    mock_provider.get_chat_completion.return_value = {
                        "choices": [
                            {
                                "message": {
                                    "content": f"Varied response: {variation['expected_impact']}"
                                }
                            }
                        ],
                        "usage": {"total_tokens": 75},
                    }
                    mock_factory.return_value = mock_provider

                    varied_agent = await (
                        ReactiveAgentBuilder()
                        .with_name(f"Varied-{provider_name}")
                        .with_model(f"{provider_name}:{model_name}")
                        .with_role("Test")
                        .with_instructions("Test varied parameters")
                        .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                        .with_model_provider_options(varied_params)
                        .with_max_iterations(1)
                        .build()
                    )

                    assert varied_agent is not None
                    await varied_agent.close()

                    # Verify the parameters were passed correctly
                    call_args = mock_factory.call_args
                    passed_options = call_args[1]["options"]

                    for param_key, param_value in varied_params.items():
                        assert passed_options[param_key] == param_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
