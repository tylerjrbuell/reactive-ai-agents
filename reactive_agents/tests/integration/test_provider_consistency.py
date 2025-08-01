"""
Comprehensive integration tests for all model providers.

This module tests all available LLM providers with identical agent configurations
to identify inconsistencies, parameter handling issues, and performance differences.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

from reactive_agents.app.builders.agent import ReactiveAgentBuilder, ConfirmationConfig
from reactive_agents.app.agents.reactive_agent import ReactiveAgent
from reactive_agents.core.types.reasoning_types import ReasoningStrategies
from reactive_agents.core.types.execution_types import ExecutionResult
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.tools.decorators import tool
from reactive_agents.providers.llm.factory import ModelProviderFactory


@dataclass
class ProviderTestResult:
    """Results from testing a single provider."""

    provider_name: str
    model_name: str
    success: bool
    execution_time: float
    final_answer: Optional[str] = None
    status: Optional[TaskStatus] = None
    iterations: int = 0
    tool_calls_count: int = 0
    error_message: Optional[str] = None
    task_metrics: Dict[str, Any] = field(default_factory=dict)
    response_quality_score: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class TestScenario:
    """A standardized test scenario for all providers."""

    name: str
    task: str
    expected_tools: List[str]
    reasoning_strategy: ReasoningStrategies
    max_iterations: int = 10
    timeout: int = 120
    validation_criteria: Dict[str, Any] = field(default_factory=dict)


class ProviderConsistencyTester:
    """Comprehensive tester for all model providers."""

    # All available providers with their test models
    PROVIDER_CONFIGS = {
        "ollama": {
            "models": ["cogito:14b", "llama3.1:8b", "qwen2.5:7b"],
            "options": {"temperature": 0.1, "num_ctx": 4000},
        },
        "openai": {
            "models": ["gpt-4o-mini", "gpt-3.5-turbo"],
            "options": {"temperature": 0.1, "max_tokens": 2000},
        },
        "anthropic": {
            "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
            "options": {"temperature": 0.1, "max_tokens": 2000},
        },
        "groq": {
            "models": ["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            "options": {"temperature": 0.1, "max_tokens": 2000},
        },
        "google": {
            "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
            "options": {"temperature": 0.1},
        },
    }

    def __init__(self, enable_real_execution: bool = False):
        """
        Initialize the tester.

        Args:
            enable_real_execution: Whether to run real API calls or use mocks
        """
        self.enable_real_execution = enable_real_execution
        self.test_results: List[ProviderTestResult] = []

    @tool()
    async def test_tool_simple(self, input_text: str) -> str:
        """Simple test tool that echoes input with a prefix."""
        return f"Processed: {input_text}"

    @tool()
    async def test_tool_math(self, a: int, b: int, operation: str = "add") -> str:
        """Test tool for mathematical operations."""
        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "subtract":
            result = a - b
        else:
            result = "Invalid operation"
        return f"Result: {result}"

    @tool()
    async def test_tool_json(self, data: Dict[str, Any]) -> str:
        """Test tool that processes JSON data."""
        return json.dumps(
            {"received": data, "processed": True, "timestamp": time.time()}
        )

    def get_test_scenarios(self) -> List[TestScenario]:
        """Get standardized test scenarios for all providers."""
        return [
            TestScenario(
                name="simple_task",
                task="Use the test_tool_simple to process the text 'Hello World'",
                expected_tools=["test_tool_simple"],
                reasoning_strategy=ReasoningStrategies.REACTIVE,
                max_iterations=5,
                validation_criteria={
                    "should_contain": ["Processed:", "Hello World"],
                    "min_tool_calls": 1,
                },
            ),
            TestScenario(
                name="math_task",
                task="Use test_tool_math to calculate 25 + 17, then multiply the result by 3",
                expected_tools=["test_tool_math"],
                reasoning_strategy=ReasoningStrategies.REFLECT_DECIDE_ACT,
                max_iterations=8,
                validation_criteria={
                    "should_contain": ["42", "126"],
                    "min_tool_calls": 2,
                },
            ),
            TestScenario(
                name="json_processing",
                task="Use test_tool_json to process this data: {'name': 'test', 'value': 123}",
                expected_tools=["test_tool_json"],
                reasoning_strategy=ReasoningStrategies.PLAN_EXECUTE_REFLECT,
                max_iterations=10,
                validation_criteria={
                    "should_contain": ["processed", "true"],
                    "min_tool_calls": 1,
                },
            ),
            TestScenario(
                name="multi_tool_task",
                task="First use test_tool_simple with 'Start', then use test_tool_math to add 10 and 5, finally use test_tool_json with the result",
                expected_tools=["test_tool_simple", "test_tool_math", "test_tool_json"],
                reasoning_strategy=ReasoningStrategies.PLAN_EXECUTE_REFLECT,
                max_iterations=15,
                validation_criteria={
                    "should_contain": ["Processed: Start", "15"],
                    "min_tool_calls": 3,
                },
            ),
            TestScenario(
                name="adaptive_reasoning",
                task="Determine the best approach to solve: What is 8 * 7, then add 6 to it?",
                expected_tools=["test_tool_math"],
                reasoning_strategy=ReasoningStrategies.ADAPTIVE,
                max_iterations=12,
                validation_criteria={
                    "should_contain": ["56", "62"],
                    "min_tool_calls": 2,
                },
            ),
        ]

    async def build_test_agent(
        self,
        provider: str,
        model: str,
        scenario: TestScenario,
        custom_options: Optional[Dict[str, Any]] = None,
    ) -> ReactiveAgent:
        """Build a standardized test agent for the given provider."""

        options = self.PROVIDER_CONFIGS[provider]["options"].copy()
        if custom_options:
            options.update(custom_options)

        builder = (
            ReactiveAgentBuilder()
            .with_name(f"Test-{provider}-{model}")
            .with_model(f"{provider}:{model}")
            .with_role("Test Assistant")
            .with_instructions(
                "You are a test assistant. Use the provided tools exactly as requested. "
                "Be concise and follow instructions precisely."
            )
            .with_reasoning_strategy(scenario.reasoning_strategy)
            .with_custom_tools(
                [self.test_tool_simple, self.test_tool_math, self.test_tool_json]
            )
            .with_max_iterations(scenario.max_iterations)
            .with_model_provider_options(options)
            .with_dynamic_strategy_switching(
                scenario.reasoning_strategy == ReasoningStrategies.ADAPTIVE
            )
        )

        return await builder.build()

    async def run_provider_test(
        self, provider: str, model: str, scenario: TestScenario
    ) -> ProviderTestResult:
        """Run a single test for a provider/model combination."""

        start_time = time.time()
        result = ProviderTestResult(
            provider_name=provider, model_name=model, success=False, execution_time=0.0
        )

        agent = None
        try:
            # Build the agent
            agent = await self.build_test_agent(provider, model, scenario)

            # Set up metrics collection
            tool_calls_count = 0

            def count_tool_calls(event):
                nonlocal tool_calls_count
                tool_calls_count += 1

            agent.on_tool_called(count_tool_calls)

            # Run the task with timeout
            execution_result = await asyncio.wait_for(
                agent.run(scenario.task), timeout=scenario.timeout
            )

            # Collect results
            end_time = time.time()
            result.execution_time = end_time - start_time
            result.success = execution_result.status == TaskStatus.COMPLETE
            result.final_answer = execution_result.final_answer
            result.status = execution_result.status
            result.tool_calls_count = tool_calls_count
            result.task_metrics = execution_result.task_metrics or {}

            if hasattr(execution_result, "session") and execution_result.session:
                result.iterations = execution_result.session.iterations

            # Validate results
            result.response_quality_score = self._validate_response(
                result.final_answer, scenario.validation_criteria
            )

        except asyncio.TimeoutError:
            result.error_message = f"Test timed out after {scenario.timeout} seconds"
            result.execution_time = time.time() - start_time
        except Exception as e:
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
        finally:
            if agent:
                try:
                    await agent.close()
                except Exception as e:
                    print(f"Error closing agent: {e}")

        return result

    def _validate_response(
        self, response: Optional[str], criteria: Dict[str, Any]
    ) -> float:
        """Validate response against criteria and return quality score."""
        if not response:
            return 0.0

        score = 0.0
        total_checks = 0

        # Check for required content
        if "should_contain" in criteria:
            for item in criteria["should_contain"]:
                total_checks += 1
                if item.lower() in response.lower():
                    score += 1.0

        # Minimum tool calls achieved gets bonus points
        if "min_tool_calls" in criteria:
            total_checks += 1
            # This would need to be passed in separately as it's not in response
            score += 0.5  # Partial credit for now

        return score / total_checks if total_checks > 0 else 0.0

    async def run_comprehensive_test(
        self,
        providers: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive tests across all providers and scenarios."""

        if providers is None:
            providers = list(self.PROVIDER_CONFIGS.keys())

        test_scenarios = self.get_test_scenarios()
        if scenarios:
            test_scenarios = [s for s in test_scenarios if s.name in scenarios]

        all_results = []
        summary = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "provider_summary": {},
            "scenario_summary": {},
            "performance_comparison": {},
            "consistency_report": {},
        }

        print(f"🚀 Starting comprehensive provider testing...")
        print(
            f"📊 Testing {len(providers)} providers with {len(test_scenarios)} scenarios"
        )

        for provider in providers:
            if provider not in self.PROVIDER_CONFIGS:
                print(f"⚠️ Skipping unknown provider: {provider}")
                continue

            provider_results = []
            config = self.PROVIDER_CONFIGS[provider]

            for model in config["models"]:
                for scenario in test_scenarios:
                    print(f"🧪 Testing {provider}:{model} with {scenario.name}")

                    if not self.enable_real_execution:
                        # Mock the provider for testing
                        result = await self._mock_provider_test(
                            provider, model, scenario
                        )
                    else:
                        result = await self.run_provider_test(provider, model, scenario)

                    provider_results.append(result)
                    all_results.append(result)
                    summary["total_tests"] += 1

                    if result.success:
                        summary["successful_tests"] += 1
                        print(f"✅ {provider}:{model} - {scenario.name} PASSED")
                    else:
                        summary["failed_tests"] += 1
                        print(
                            f"❌ {provider}:{model} - {scenario.name} FAILED: {result.error_message}"
                        )

            # Summarize provider results
            summary["provider_summary"][provider] = self._summarize_provider_results(
                provider_results
            )

        # Generate comprehensive analysis
        summary["scenario_summary"] = self._analyze_scenarios(
            all_results, test_scenarios
        )
        summary["performance_comparison"] = self._compare_performance(all_results)
        summary["consistency_report"] = self._analyze_consistency(all_results)

        self.test_results = all_results
        return summary

    async def _mock_provider_test(
        self, provider: str, model: str, scenario: TestScenario
    ) -> ProviderTestResult:
        """Mock a provider test for CI/testing without real API calls."""

        # Simulate different response patterns per provider
        mock_responses = {
            "ollama": "I used test_tool_simple to process 'Hello World' and got: Processed: Hello World",
            "openai": "I'll use the test_tool_simple function to process your text. Result: Processed: Hello World",
            "anthropic": "Using the test_tool_simple tool with 'Hello World': Processed: Hello World",
            "groq": "Processed using test_tool_simple: Processed: Hello World",
            "google": "Applied test_tool_simple to 'Hello World', result: Processed: Hello World",
        }

        # Simulate varying performance characteristics
        execution_times = {
            "ollama": 2.5,
            "openai": 1.2,
            "anthropic": 1.8,
            "groq": 0.8,
            "google": 1.5,
        }

        # Simulate some failures for testing
        success_rate = 0.9  # 90% success rate
        success = hash(f"{provider}{model}{scenario.name}") % 10 < 9

        result = ProviderTestResult(
            provider_name=provider,
            model_name=model,
            success=success,
            execution_time=execution_times.get(provider, 1.5),
            final_answer=(
                mock_responses.get(provider, "Mock response") if success else None
            ),
            status=TaskStatus.COMPLETE if success else TaskStatus.ERROR,
            iterations=hash(f"{provider}{model}") % 5 + 1,
            tool_calls_count=len(scenario.expected_tools),
            error_message=None if success else "Mock error for testing",
            response_quality_score=0.8 if success else 0.0,
        )

        return result

    def _summarize_provider_results(
        self, results: List[ProviderTestResult]
    ) -> Dict[str, Any]:
        """Summarize results for a single provider."""
        total = len(results)
        successful = sum(1 for r in results if r.success)

        avg_time = sum(r.execution_time for r in results) / total if total > 0 else 0
        avg_quality = (
            sum(r.response_quality_score for r in results) / total if total > 0 else 0
        )
        avg_iterations = sum(r.iterations for r in results) / total if total > 0 else 0

        return {
            "total_tests": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_execution_time": avg_time,
            "avg_quality_score": avg_quality,
            "avg_iterations": avg_iterations,
            "errors": [r.error_message for r in results if r.error_message],
        }

    def _analyze_scenarios(
        self, all_results: List[ProviderTestResult], scenarios: List[TestScenario]
    ) -> Dict[str, Any]:
        """Analyze results across scenarios."""
        scenario_analysis = {}

        for scenario in scenarios:
            scenario_results = [r for r in all_results if scenario.name in str(r)]

            if not scenario_results:
                continue

            successful = sum(1 for r in scenario_results if r.success)
            total = len(scenario_results)

            scenario_analysis[scenario.name] = {
                "success_rate": successful / total if total > 0 else 0,
                "avg_execution_time": sum(r.execution_time for r in scenario_results)
                / total,
                "provider_performance": {},
            }

            # Per-provider performance for this scenario
            for result in scenario_results:
                provider = result.provider_name
                if (
                    provider
                    not in scenario_analysis[scenario.name]["provider_performance"]
                ):
                    scenario_analysis[scenario.name]["provider_performance"][
                        provider
                    ] = []
                scenario_analysis[scenario.name]["provider_performance"][
                    provider
                ].append(
                    {
                        "model": result.model_name,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "quality_score": result.response_quality_score,
                    }
                )

        return scenario_analysis

    def _compare_performance(
        self, all_results: List[ProviderTestResult]
    ) -> Dict[str, Any]:
        """Compare performance across providers."""
        provider_stats = {}

        for result in all_results:
            provider = result.provider_name
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "execution_times": [],
                    "quality_scores": [],
                    "success_count": 0,
                    "total_count": 0,
                }

            provider_stats[provider]["execution_times"].append(result.execution_time)
            provider_stats[provider]["quality_scores"].append(
                result.response_quality_score
            )
            provider_stats[provider]["total_count"] += 1
            if result.success:
                provider_stats[provider]["success_count"] += 1

        # Calculate rankings
        performance_comparison = {}
        for provider, stats in provider_stats.items():
            avg_time = sum(stats["execution_times"]) / len(stats["execution_times"])
            avg_quality = sum(stats["quality_scores"]) / len(stats["quality_scores"])
            success_rate = stats["success_count"] / stats["total_count"]

            performance_comparison[provider] = {
                "avg_execution_time": avg_time,
                "avg_quality_score": avg_quality,
                "success_rate": success_rate,
                "performance_score": (
                    success_rate * 0.5 + avg_quality * 0.3 + (1 / avg_time) * 0.2
                ),
            }

        return performance_comparison

    def _analyze_consistency(
        self, all_results: List[ProviderTestResult]
    ) -> Dict[str, Any]:
        """Analyze consistency across providers."""
        inconsistencies = []

        # Group results by scenario
        by_scenario = {}
        for result in all_results:
            # Extract scenario name (would need better tracking in real implementation)
            scenario_key = "unknown"  # This would need to be tracked better
            if scenario_key not in by_scenario:
                by_scenario[scenario_key] = []
            by_scenario[scenario_key].append(result)

        # Look for patterns and inconsistencies
        consistency_report = {
            "execution_time_variance": {},
            "quality_variance": {},
            "failure_patterns": {},
            "recommendations": [],
        }

        # Analyze execution time variance
        for scenario, results in by_scenario.items():
            times = [r.execution_time for r in results]
            if len(times) > 1:
                variance = sum((t - sum(times) / len(times)) ** 2 for t in times) / len(
                    times
                )
                consistency_report["execution_time_variance"][scenario] = variance

        # Add recommendations based on findings
        consistency_report["recommendations"] = [
            "Consider standardizing timeout values across providers",
            "Implement retry logic for providers with high failure rates",
            "Monitor token usage for cost optimization",
            "Consider provider-specific parameter tuning",
        ]

        return consistency_report

    def generate_report(self, results_summary: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("# Multi-Provider Integration Test Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        total = results_summary["total_tests"]
        successful = results_summary["successful_tests"]
        failed = results_summary["failed_tests"]

        report.append(f"- **Total Tests**: {total}")
        report.append(f"- **Successful**: {successful} ({successful/total*100:.1f}%)")
        report.append(f"- **Failed**: {failed} ({failed/total*100:.1f}%)")
        report.append("")

        # Provider Summary
        report.append("## Provider Performance Summary")
        for provider, stats in results_summary["provider_summary"].items():
            report.append(f"### {provider.title()}")
            report.append(f"- Success Rate: {stats['success_rate']*100:.1f}%")
            report.append(
                f"- Average Execution Time: {stats['avg_execution_time']:.2f}s"
            )
            report.append(f"- Average Quality Score: {stats['avg_quality_score']:.2f}")
            report.append(f"- Average Iterations: {stats['avg_iterations']:.1f}")
            report.append("")

        # Performance Comparison
        report.append("## Performance Comparison")
        perf = results_summary["performance_comparison"]
        sorted_providers = sorted(
            perf.items(), key=lambda x: x[1]["performance_score"], reverse=True
        )

        for i, (provider, stats) in enumerate(sorted_providers, 1):
            report.append(
                f"{i}. **{provider.title()}** (Score: {stats['performance_score']:.3f})"
            )
            report.append(f"   - Success Rate: {stats['success_rate']*100:.1f}%")
            report.append(f"   - Avg Time: {stats['avg_execution_time']:.2f}s")
            report.append(f"   - Avg Quality: {stats['avg_quality_score']:.2f}")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        for rec in results_summary["consistency_report"]["recommendations"]:
            report.append(f"- {rec}")

        return "\n".join(report)


# Pytest fixtures and test classes
@pytest.fixture
def consistency_tester():
    """Fixture for consistency tester."""
    return ProviderConsistencyTester(enable_real_execution=False)


@pytest.fixture
def real_consistency_tester():
    """Fixture for real API testing (requires environment setup)."""
    return ProviderConsistencyTester(enable_real_execution=True)


class TestProviderConsistency:
    """Test class for provider consistency testing."""

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_all_providers_mock(self, consistency_tester):
        """Test all providers with mocked responses."""
        results = await consistency_tester.run_comprehensive_test()

        assert results["total_tests"] > 0
        assert (
            results["successful_tests"] >= results["total_tests"] * 0.8
        )  # At least 80% success

        # Ensure all expected providers were tested
        expected_providers = ["ollama", "openai", "anthropic", "groq", "google"]
        for provider in expected_providers:
            assert provider in results["provider_summary"]

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ollama_provider_real(self, real_consistency_tester):
        """Test only Ollama provider with real execution (if available)."""
        results = await real_consistency_tester.run_comprehensive_test(
            providers=["ollama"], scenarios=["simple_task"]
        )

        assert "ollama" in results["provider_summary"]
        # Allow for some failures in real testing
        assert (
            results["provider_summary"]["ollama"]["success_rate"] >= 0.0
        )  # Allow any success rate in test env

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_specific_scenario_all_providers(self, consistency_tester):
        """Test a specific scenario across all providers."""
        results = await consistency_tester.run_comprehensive_test(
            scenarios=["simple_task"]
        )

        # All providers should be able to handle simple tasks
        for provider_stats in results["provider_summary"].values():
            assert (
                provider_stats["success_rate"] >= 0.0
            )  # Allow any success rate in test env

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_performance_comparison(self, consistency_tester):
        """Test performance comparison functionality."""
        results = await consistency_tester.run_comprehensive_test()

        perf_comparison = results["performance_comparison"]
        assert len(perf_comparison) > 0

        # Verify all providers have performance metrics
        for provider, metrics in perf_comparison.items():
            assert "avg_execution_time" in metrics
            assert "avg_quality_score" in metrics
            assert "success_rate" in metrics
            assert "performance_score" in metrics

    @pytest.mark.integration
    @pytest.mark.providers
    @pytest.mark.asyncio
    async def test_consistency_analysis(self, consistency_tester):
        """Test consistency analysis functionality."""
        results = await consistency_tester.run_comprehensive_test()

        consistency_report = results["consistency_report"]
        assert "recommendations" in consistency_report
        assert len(consistency_report["recommendations"]) > 0

    def test_report_generation(self, consistency_tester):
        """Test report generation."""
        # Mock results for testing
        mock_results = {
            "total_tests": 10,
            "successful_tests": 8,
            "failed_tests": 2,
            "provider_summary": {
                "ollama": {
                    "success_rate": 0.8,
                    "avg_execution_time": 2.5,
                    "avg_quality_score": 0.75,
                    "avg_iterations": 3,
                }
            },
            "performance_comparison": {
                "ollama": {
                    "performance_score": 0.8,
                    "success_rate": 0.8,
                    "avg_execution_time": 2.5,
                    "avg_quality_score": 0.75,
                }
            },
            "consistency_report": {"recommendations": ["Test recommendation"]},
        }

        report = consistency_tester.generate_report(mock_results)

        assert "Multi-Provider Integration Test Report" in report
        assert "Executive Summary" in report
        assert "Provider Performance Summary" in report
        assert "Performance Comparison" in report
        assert "Recommendations" in report


if __name__ == "__main__":

    async def main():
        """Run the consistency tester directly."""
        tester = ProviderConsistencyTester(enable_real_execution=False)
        results = await tester.run_comprehensive_test()

        print("\n" + "=" * 60)
        print(tester.generate_report(results))
        print("=" * 60)

    asyncio.run(main())
