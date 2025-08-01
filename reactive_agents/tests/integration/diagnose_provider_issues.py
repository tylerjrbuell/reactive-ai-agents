#!/usr/bin/env python3
"""
Provider Issue Diagnosis Script

This script runs focused tests to identify specific issues with each provider
that need to be addressed for production readiness.
"""

import asyncio
import sys
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from reactive_agents.app.builders.agent import ReactiveAgentBuilder
from reactive_agents.core.types.reasoning_types import ReasoningStrategies
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.tools.decorators import tool
from reactive_agents.tests.integration.provider_test_config import get_test_config


@tool()
async def simple_test_tool(message: str) -> str:
    """
    Process and format a text message for testing tool functionality.

    This tool takes a text message as input and returns it with a 'Processed: ' prefix
    to verify that tool calling and parameter passing are working correctly.

    Args:
        message: The text message to process (required, must be a non-empty string)

    Returns:
        A formatted string with 'Processed: ' prefix followed by the original message

    Example:
        Input: "Hello World"
        Output: "Processed: Hello World"
    """
    return f"Processed: {message}"


@tool()
async def math_test_tool(a: int, b: int, operation: str = "add") -> str:
    """Math test tool for calculations."""
    if operation == "add":
        result = a + b
    elif operation == "multiply":
        result = a * b
    elif operation == "subtract":
        result = a - b
    else:
        return "Invalid operation"
    return f"Result: {a} {operation} {b} = {result}"


class ProviderIssueDiagnoser:
    """Diagnose specific issues with each provider."""

    def __init__(self):
        self.config = get_test_config()
        self.issues_found = {}
        self.successful_tests = {}
        self.timeout = 30  # Default timeout
        self.max_iterations = 3  # Default max iterations

    async def diagnose_all_providers(
        self, providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run diagnosis on all providers or specific providers.

        Args:
            providers: List of provider names to test. If None, tests all providers.
        """

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_providers": 0,
            "provider_results": {},
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Determine which providers to test
        providers_to_test = (
            providers if providers else list(self.config.providers.keys())
        )

        # Validate provider names
        invalid_providers = [
            p for p in providers_to_test if p not in self.config.providers
        ]
        if invalid_providers:
            available_providers = list(self.config.providers.keys())
            print(f"⚠️ Warning: Unknown providers: {invalid_providers}")
            print(f"Available providers: {available_providers}")
            providers_to_test = [
                p for p in providers_to_test if p in self.config.providers
            ]

        if not providers_to_test:
            print("❌ No valid providers to test!")
            return results

        print(
            f"🎯 Testing {len(providers_to_test)} provider(s): {', '.join(providers_to_test)}"
        )
        print()

        for provider_name in providers_to_test:
            print(f"\n🔍 Diagnosing {provider_name.upper()} Provider")
            print("=" * 50)

            provider_results = await self.diagnose_provider(provider_name)
            results["provider_results"][provider_name] = provider_results
            results["total_providers"] += 1

            # Collect critical issues
            if provider_results["critical_issues"]:
                results["critical_issues"].extend(
                    [
                        f"{provider_name}: {issue}"
                        for issue in provider_results["critical_issues"]
                    ]
                )

            # Collect warnings
            if provider_results["warnings"]:
                results["warnings"].extend(
                    [
                        f"{provider_name}: {warning}"
                        for warning in provider_results["warnings"]
                    ]
                )

        # Generate overall recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    async def diagnose_provider(self, provider_name: str) -> Dict[str, Any]:
        """Diagnose issues with a specific provider."""

        if provider_name not in self.config.providers:
            return {"error": f"Provider {provider_name} not configured"}

        provider_config = self.config.providers[provider_name]
        results = {
            "provider_name": provider_name,
            "models_tested": [],
            "successful_tests": 0,
            "failed_tests": 0,
            "critical_issues": [],
            "warnings": [],
            "performance_notes": [],
            "parameter_issues": [],
            "capability_gaps": [],
        }

        # Test each model for this provider
        for model_name in list(provider_config.models.keys())[
            :2
        ]:  # Limit to 2 models per provider
            print(f"  🧪 Testing {provider_name}:{model_name}")

            model_results = await self._test_provider_model(provider_name, model_name)
            results["models_tested"].append(
                {"model": model_name, "results": model_results}
            )

            if model_results["success"]:
                results["successful_tests"] += 1
                print(f"    ✅ Basic functionality: OK")
                print(
                    f"       - Execution time: {model_results['execution_time']:.2f}s"
                )
                print(f"       - Tool calls: {model_results['tool_calls']}")
                print(f"       - Iterations: {model_results['iterations_used']}")
                if model_results["final_answer"]:
                    answer_preview = (
                        model_results["final_answer"][:100] + "..."
                        if len(model_results["final_answer"]) > 100
                        else model_results["final_answer"]
                    )
                    print(f"       - Final answer: {answer_preview}")
            else:
                results["failed_tests"] += 1
                print(f"    ❌ Basic functionality: FAILED - {model_results['error']}")
                print(
                    f"       - Execution time: {model_results['execution_time']:.2f}s"
                )
                print(f"       - Tool calls: {model_results['tool_calls']}")
                print(f"       - Iterations: {model_results['iterations_used']}")
                if model_results["tool_results"]:
                    print(f"       - Tool results:")
                    for tool_result in model_results["tool_results"]:
                        status = "✅" if tool_result["success"] else "❌"
                        print(
                            f"         {status} {tool_result['name']}: {tool_result['params']} → {tool_result['result']}"
                        )
                results["critical_issues"].append(
                    f"Model {model_name}: {model_results['error']}"
                )

            # Check for specific issues
            await self._check_parameter_issues(provider_name, model_name, results)
            await self._check_capability_issues(provider_name, model_name, results)

        return results

    async def _test_provider_model(
        self, provider_name: str, model_name: str
    ) -> Dict[str, Any]:
        """Test basic functionality of a provider/model combination."""

        test_result = {
            "success": False,
            "error": None,
            "execution_time": 0.0,
            "response_length": 0,
            "tool_calls": 0,
            "final_answer": None,
            "tool_results": [],
            "iterations_used": 0,
        }

        try:
            # Try to build and run a simple agent
            builder = (
                ReactiveAgentBuilder()
                .with_name(f"Diagnosis-{provider_name}-{model_name}")
                .with_model(f"{provider_name}:{model_name}")
                .with_role("Test Agent")
                .with_instructions(
                    "You are a test agent. Use tools as requested and be concise."
                )
                .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                .with_custom_tools([simple_test_tool])
                .with_max_iterations(self.max_iterations)
            )

            # Add provider-specific parameters
            if provider_name in self.config.providers:
                recommended_params = self.config.get_adjusted_params(
                    provider_name, model_name, "simple_task"
                )
                builder = builder.with_model_provider_options(recommended_params)

            start_time = asyncio.get_event_loop().time()

            agent = await builder.build()

            # Run a simple task
            result = await asyncio.wait_for(
                agent.run("Use simple_test_tool to process the message 'Hello World'"),
                timeout=float(self.timeout),
            )

            end_time = asyncio.get_event_loop().time()

            test_result["success"] = result.status == TaskStatus.COMPLETE
            test_result["execution_time"] = end_time - start_time
            test_result["response_length"] = len(result.final_answer or "")
            test_result["final_answer"] = result.final_answer
            test_result["iterations_used"] = getattr(result, "iterations", 0)

            # Get tool usage information from agent context
            if hasattr(agent, "context") and hasattr(agent.context, "tool_manager"):
                tool_history = (
                    agent.context.tool_manager.tool_history
                    if agent.context.tool_manager
                    else []
                )
                test_result["tool_calls"] = len(tool_history)
                test_result["tool_results"] = [
                    {
                        "name": entry["name"],
                        "params": entry["params"],
                        "result": (
                            entry["result"][:200] + "..."
                            if len(str(entry["result"])) > 200
                            else entry["result"]
                        ),
                        "success": not entry.get("error", False),
                    }
                    for entry in tool_history
                ]

            if not test_result["success"]:
                test_result["error"] = f"Task failed with status: {result.status}"

            await agent.close()

        except asyncio.TimeoutError:
            test_result["error"] = f"Task timed out after {self.timeout} seconds"
        except Exception as e:
            test_result["error"] = f"Exception: {str(e)}"

        return test_result

    async def _check_parameter_issues(
        self, provider_name: str, model_name: str, results: Dict[str, Any]
    ):
        """Check for parameter-related issues."""

        provider_config = self.config.providers[provider_name]
        model_config = provider_config.models[model_name]

        # Check for common parameter issues
        param_issues = []

        # Temperature validation
        if "temperature" in model_config.recommended_params:
            temp = model_config.recommended_params["temperature"]
            if temp < 0 or temp > 1:
                param_issues.append(f"Invalid temperature: {temp} (should be 0-1)")

        # Context window issues
        if hasattr(model_config, "context_window") and model_config.context_window:
            if model_config.context_window < 2000:
                param_issues.append(
                    f"Small context window: {model_config.context_window} tokens"
                )

        # Provider-specific checks
        if provider_name == "ollama":
            if "num_ctx" in model_config.recommended_params:
                num_ctx = model_config.recommended_params["num_ctx"]
                if num_ctx > model_config.context_window:
                    param_issues.append(
                        f"num_ctx ({num_ctx}) exceeds context_window ({model_config.context_window})"
                    )

        elif provider_name == "openai":
            if "max_tokens" in model_config.recommended_params:
                max_tokens = model_config.recommended_params["max_tokens"]
                if max_tokens > 4000:  # Reasonable limit for most use cases
                    results["warnings"].append(f"High max_tokens setting: {max_tokens}")

        results["parameter_issues"].extend(param_issues)

    async def _check_capability_issues(
        self, provider_name: str, model_name: str, results: Dict[str, Any]
    ):
        """Check for capability-related issues."""

        provider_config = self.config.providers[provider_name]
        model_config = provider_config.models[model_name]

        capability_gaps = []

        # Check critical capabilities
        from reactive_agents.tests.integration.provider_test_config import (
            ProviderCapability,
        )

        critical_capabilities = [
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.SYSTEM_MESSAGES,
        ]

        for capability in critical_capabilities:
            if capability not in model_config.capabilities:
                capability_gaps.append(f"Missing {capability.value} support")

        # Check for performance expectations
        if (
            hasattr(model_config, "typical_latency_ms")
            and model_config.typical_latency_ms
        ):
            if model_config.typical_latency_ms > 5000:  # > 5 seconds
                results["performance_notes"].append(
                    f"High latency expected: {model_config.typical_latency_ms}ms"
                )

        results["capability_gaps"].extend(capability_gaps)

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnosis results."""

        recommendations = []

        # Critical issues
        if results["critical_issues"]:
            recommendations.append(
                f"🚨 CRITICAL: {len(results['critical_issues'])} providers have critical issues requiring immediate attention"
            )

        # Performance recommendations
        slow_providers = []
        for provider_name, provider_results in results["provider_results"].items():
            if provider_results.get("performance_notes"):
                slow_providers.append(provider_name)

        if slow_providers:
            recommendations.append(
                f"⚡ PERFORMANCE: Consider optimizing or providing alternatives for slower providers: {', '.join(slow_providers)}"
            )

        # Parameter recommendations
        param_issues_count = sum(
            len(provider_results.get("parameter_issues", []))
            for provider_results in results["provider_results"].values()
        )

        if param_issues_count > 0:
            recommendations.append(
                f"🔧 PARAMETERS: {param_issues_count} parameter configuration issues found across providers"
            )

        # Capability recommendations
        capability_gaps_count = sum(
            len(provider_results.get("capability_gaps", []))
            for provider_results in results["provider_results"].values()
        )

        if capability_gaps_count > 0:
            recommendations.append(
                f"🎯 CAPABILITIES: {capability_gaps_count} capability gaps identified - consider feature parity improvements"
            )

        # General recommendations
        recommendations.extend(
            [
                "📊 MONITORING: Implement real-time provider health monitoring",
                "🔄 RETRY: Add intelligent retry logic for failed requests",
                "💰 COST: Implement cost tracking and budget alerts",
                "🧪 TESTING: Schedule regular provider integration testing",
            ]
        )

        return recommendations

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive diagnosis report."""

        report_lines = [
            "# Provider Issue Diagnosis Report",
            f"Generated: {results['timestamp']}",
            f"Providers Tested: {results['total_providers']}",
            "",
            "## Executive Summary",
        ]

        # Critical issues summary
        if results["critical_issues"]:
            report_lines.extend(
                [f"🚨 **Critical Issues Found**: {len(results['critical_issues'])}", ""]
            )
            for issue in results["critical_issues"]:
                report_lines.append(f"- {issue}")
            report_lines.append("")
        else:
            report_lines.extend(["✅ **No Critical Issues Found**", ""])

        # Provider-by-provider analysis
        report_lines.extend(["## Provider Analysis", ""])

        for provider_name, provider_results in results["provider_results"].items():
            report_lines.extend(
                [
                    f"### {provider_name.upper()}",
                    f"- Models Tested: {len(provider_results['models_tested'])}",
                    f"- Success Rate: {provider_results['successful_tests']}/{provider_results['successful_tests'] + provider_results['failed_tests']}",
                    "",
                ]
            )

            if provider_results["critical_issues"]:
                report_lines.append("**Critical Issues:**")
                for issue in provider_results["critical_issues"]:
                    report_lines.append(f"- {issue}")
                report_lines.append("")

            if provider_results["parameter_issues"]:
                report_lines.append("**Parameter Issues:**")
                for issue in provider_results["parameter_issues"]:
                    report_lines.append(f"- {issue}")
                report_lines.append("")

            if provider_results["capability_gaps"]:
                report_lines.append("**Capability Gaps:**")
                for gap in provider_results["capability_gaps"]:
                    report_lines.append(f"- {gap}")
                report_lines.append("")

        # Recommendations
        if results["recommendations"]:
            report_lines.extend(["## Recommendations", ""])
            for rec in results["recommendations"]:
                report_lines.append(f"- {rec}")

        return "\n".join(report_lines)


async def main():
    """Run provider diagnosis."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Diagnose specific issues with provider implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python diagnose_provider_issues.py                    # Test all providers
  python diagnose_provider_issues.py --providers anthropic groq  # Test specific providers
  python diagnose_provider_issues.py -p ollama          # Test single provider
  python diagnose_provider_issues.py --list-providers   # Show available providers
""",
    )

    parser.add_argument(
        "-p",
        "--providers",
        nargs="*",
        help="Specific providers to test (space-separated). If not specified, tests all providers.",
    )

    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List all available providers and exit",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each test (default: 30)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum iterations per test (default: 3)",
    )

    args = parser.parse_args()

    # Handle list providers option
    if args.list_providers:
        config = get_test_config()
        print("📋 Available providers:")
        for provider_name, provider_config in config.providers.items():
            models = list(provider_config.models.keys())
            print(
                f"  • {provider_name}: {len(models)} models ({', '.join(models[:2])}{'...' if len(models) > 2 else ''})"
            )
        return 0

    print("🏥 Reactive Agents Provider Issue Diagnosis")
    print("=" * 60)
    print("This script will diagnose specific issues with each provider")
    print("that need to be addressed for production readiness.")
    print("")

    diagnoser = ProviderIssueDiagnoser()

    # Update diagnoser with custom settings if provided
    diagnoser.timeout = args.timeout
    diagnoser.max_iterations = args.max_iterations

    try:
        results = await diagnoser.diagnose_all_providers(args.providers)

        # Generate and display report
        report = diagnoser.generate_report(results)
        print("\n" + "=" * 60)
        print("DIAGNOSIS REPORT")
        print("=" * 60)
        print(report)
        print("=" * 60)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"provider_diagnosis_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📁 Detailed results saved to: {results_file}")

        # Return success/failure code
        critical_issues_count = len(results.get("critical_issues", []))
        if critical_issues_count > 0:
            print(
                f"\n⚠️  {critical_issues_count} critical issues found - requires attention"
            )
            return 1
        else:
            print(f"\n✅ No critical issues found - providers are functioning well")
            return 0

    except Exception as e:
        print(f"\n❌ Diagnosis failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
