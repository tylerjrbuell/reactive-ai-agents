"""
Performance benchmarking and measurement system for provider consistency testing.

This module provides comprehensive performance measurement, analysis, and
comparison capabilities across all model providers.
"""

import time
import asyncio
import statistics
import json
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import csv
from datetime import datetime, timedelta
import threading

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore
from concurrent.futures import ThreadPoolExecutor

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    pd = None  # type: ignore
from pathlib import Path

from .provider_test_config import get_test_config, ProviderCapability, TestComplexity


class MetricType(Enum):
    """Types of performance metrics."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    QUALITY_SCORE = "quality_score"
    COST_EFFICIENCY = "cost_efficiency"
    RESOURCE_USAGE = "resource_usage"
    RELIABILITY = "reliability"
    TOKEN_EFFICIENCY = "token_efficiency"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a test run."""

    # Timing metrics
    total_execution_time: float = 0.0
    first_response_time: float = 0.0  # Time to first token/response
    average_response_time: float = 0.0
    tool_call_latency: float = 0.0

    # Throughput metrics
    tokens_per_second: float = 0.0
    tool_calls_per_second: float = 0.0
    iterations_per_second: float = 0.0

    # Quality metrics
    task_completion_rate: float = 0.0  # 0-1
    response_accuracy: float = 0.0  # 0-1
    tool_usage_efficiency: float = 0.0  # 0-1
    reasoning_coherence: float = 0.0  # 0-1

    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    network_requests: int = 0

    # Cost metrics
    estimated_cost_usd: float = 0.0
    cost_per_success: float = 0.0

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Error metrics
    error_count: int = 0
    timeout_count: int = 0
    retry_count: int = 0

    # Additional metadata
    timestamp: datetime = field(default_factory=datetime.now)
    provider_name: str = ""
    model_name: str = ""
    scenario_name: str = ""
    test_configuration: Dict[str, Any] = field(default_factory=dict)


class ResourceMonitor:
    """Monitor system resource usage during test execution."""

    def __init__(self):
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 0.5):
        """Start resource monitoring."""
        if not PSUTIL_AVAILABLE:
            print("Warning: psutil not available, resource monitoring disabled")
            return

        self.monitoring = True
        self.measurements = []
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,)
        )
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return summary statistics."""
        if not PSUTIL_AVAILABLE:
            return {"peak_memory_mb": 0.0, "avg_cpu_percent": 0.0}

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        if not self.measurements:
            return {"peak_memory_mb": 0.0, "avg_cpu_percent": 0.0}

        memory_values = [m["memory_mb"] for m in self.measurements]
        cpu_values = [m["cpu_percent"] for m in self.measurements]

        return {
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": statistics.mean(memory_values),
            "avg_cpu_percent": statistics.mean(cpu_values),
            "max_cpu_percent": max(cpu_values),
            "measurement_count": len(self.measurements),
        }

    def _monitor_loop(self, interval: float):
        """Internal monitoring loop."""
        if not PSUTIL_AVAILABLE or psutil is None:
            return

        process = psutil.Process()

        while self.monitoring:
            try:
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()

                self.measurements.append(
                    {
                        "timestamp": time.time(),
                        "memory_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
                        "cpu_percent": cpu_percent,
                    }
                )

                time.sleep(interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore
                break
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                break


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking system."""

    def __init__(self):
        self.config = get_test_config()
        self.results_history: List[PerformanceMetrics] = []
        self.resource_monitor = ResourceMonitor()

    async def benchmark_provider_model(
        self,
        provider_name: str,
        model_name: str,
        scenario_name: str,
        agent_builder_func,
        iterations: int = 3,
    ) -> List[PerformanceMetrics]:
        """Benchmark a specific provider/model combination with multiple iterations."""

        results = []

        for iteration in range(iterations):
            print(
                f"🏃 Benchmark iteration {iteration + 1}/{iterations} for {provider_name}:{model_name}"
            )

            # Single benchmark run
            metrics = await self._single_benchmark_run(
                provider_name, model_name, scenario_name, agent_builder_func
            )

            results.append(metrics)

            # Small delay between iterations
            if iteration < iterations - 1:
                await asyncio.sleep(2.0)

        return results

    async def _single_benchmark_run(
        self,
        provider_name: str,
        model_name: str,
        scenario_name: str,
        agent_builder_func,
    ) -> PerformanceMetrics:
        """Run a single benchmark test and collect comprehensive metrics."""

        metrics = PerformanceMetrics(
            provider_name=provider_name,
            model_name=model_name,
            scenario_name=scenario_name,
        )

        # Get scenario configuration
        scenario = self.config.test_scenarios.get(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        # Initialize timing
        start_time = time.time()
        first_response_time = None
        tool_call_times = []
        network_requests = 0

        agent = None
        try:
            # Build agent
            agent_build_start = time.time()
            agent = await agent_builder_func()
            agent_build_time = time.time() - agent_build_start

            # Set up event handlers for detailed metrics
            def on_tool_called(event_data):
                nonlocal network_requests
                network_requests += 1
                tool_start_time = time.time()
                # Store for latency calculation (simplified)

            def on_tool_completed(event_data):
                tool_end_time = time.time()
                # Calculate tool latency (simplified)

            def on_first_response(event_data):
                nonlocal first_response_time
                if first_response_time is None:
                    first_response_time = time.time() - start_time

            # Register event handlers
            agent.on_tool_called(on_tool_called)
            agent.on_tool_completed(on_tool_completed)
            agent.on_iteration_started(
                on_first_response
            )  # Use as proxy for first response

            # Execute the task
            execution_start = time.time()
            result = await agent.run(scenario.task)
            execution_end = time.time()

            # Calculate basic timing metrics
            metrics.total_execution_time = execution_end - start_time
            metrics.first_response_time = first_response_time or 0.0
            metrics.network_requests = network_requests

            # Calculate success metrics
            metrics.task_completion_rate = (
                1.0 if result.status.value == "completed" else 0.0
            )

            # Extract token information if available
            if hasattr(result, "task_metrics") and result.task_metrics:
                token_data = result.task_metrics.get("token_usage", {})
                metrics.input_tokens = token_data.get("input_tokens", 0)
                metrics.output_tokens = token_data.get("output_tokens", 0)
                metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

            # Calculate throughput metrics
            if metrics.total_execution_time > 0:
                metrics.tokens_per_second = (
                    metrics.total_tokens / metrics.total_execution_time
                )
                if hasattr(result, "session") and result.session:
                    metrics.iterations_per_second = (
                        result.session.iterations / metrics.total_execution_time
                    )

            # Calculate quality metrics
            metrics.response_accuracy = self._calculate_response_accuracy(
                result.final_answer, scenario.validation_criteria
            )
            metrics.reasoning_coherence = self._assess_reasoning_coherence(
                result.final_answer
            )

            # Calculate cost metrics
            if provider_name in self.config.providers:
                provider_config = self.config.providers[provider_name]
                if model_name in provider_config.models:
                    model_config = provider_config.models[model_name]
                    if model_config.cost_per_1k_tokens:
                        metrics.estimated_cost_usd = (
                            metrics.total_tokens
                            / 1000.0
                            * model_config.cost_per_1k_tokens
                        )
                        if metrics.task_completion_rate > 0:
                            metrics.cost_per_success = (
                                metrics.estimated_cost_usd
                                / metrics.task_completion_rate
                            )

        except asyncio.TimeoutError:
            metrics.timeout_count = 1
            metrics.total_execution_time = time.time() - start_time
        except Exception as e:
            metrics.error_count = 1
            metrics.total_execution_time = time.time() - start_time
            print(f"Benchmark error: {e}")
        finally:
            # Stop resource monitoring
            resource_stats = self.resource_monitor.stop_monitoring()
            metrics.peak_memory_mb = resource_stats.get("peak_memory_mb", 0.0)
            metrics.avg_cpu_percent = resource_stats.get("avg_cpu_percent", 0.0)

            # Clean up agent
            if agent:
                try:
                    await agent.close()
                except Exception as e:
                    print(f"Error closing agent: {e}")

        # Store configuration details
        metrics.test_configuration = {
            "scenario_complexity": scenario.complexity.value,
            "max_iterations": scenario.max_iterations,
            "timeout_seconds": scenario.timeout_seconds,
            "reasoning_strategy": scenario.reasoning_strategy,
        }

        return metrics

    def _calculate_response_accuracy(
        self, response: Optional[str], validation_criteria: Dict[str, Any]
    ) -> float:
        """Calculate response accuracy based on validation criteria."""
        if not response:
            return 0.0

        score = 0.0
        total_checks = 0

        # Check required content
        if "should_contain" in validation_criteria:
            for item in validation_criteria["should_contain"]:
                total_checks += 1
                if item.lower() in response.lower():
                    score += 1.0

        # Check minimum quality thresholds
        if "min_length" in validation_criteria:
            total_checks += 1
            if len(response) >= validation_criteria["min_length"]:
                score += 1.0

        return score / total_checks if total_checks > 0 else 0.0

    def _assess_reasoning_coherence(self, response: Optional[str]) -> float:
        """Assess the coherence and structure of the reasoning (simplified)."""
        if not response:
            return 0.0

        # Simple heuristics for reasoning coherence
        score = 0.0

        # Check for logical structure indicators
        structure_indicators = [
            "first",
            "then",
            "next",
            "finally",
            "because",
            "therefore",
            "however",
            "furthermore",
            "in conclusion",
        ]

        found_indicators = sum(
            1
            for indicator in structure_indicators
            if indicator.lower() in response.lower()
        )

        # Normalize by response length and expected structure
        if len(response) > 50:  # Reasonable response length
            score += 0.3

        if found_indicators > 0:
            score += min(0.7, found_indicators * 0.2)

        return min(1.0, score)


class PerformanceComparator:
    """Compare performance across providers and generate insights."""

    def __init__(self):
        self.config = get_test_config()

    def compare_providers(
        self, results: List[PerformanceMetrics], group_by: str = "provider_name"
    ) -> Dict[str, Any]:
        """Compare performance across providers or models."""

        # Group results
        grouped_results = {}
        for result in results:
            key = getattr(result, group_by, "unknown")
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)

        # Calculate aggregate statistics
        comparison_data = {}
        for group_key, group_results in grouped_results.items():
            comparison_data[group_key] = self._calculate_aggregate_stats(group_results)

        # Generate rankings
        rankings = self._generate_rankings(comparison_data)

        # Identify outliers and patterns
        insights = self._analyze_patterns(comparison_data, group_by)

        return {
            "comparison_data": comparison_data,
            "rankings": rankings,
            "insights": insights,
            "summary": self._generate_comparison_summary(comparison_data),
        }

    def _calculate_aggregate_stats(
        self, results: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Calculate aggregate statistics for a group of results."""

        if not results:
            return {}

        # Extract metrics
        execution_times = [r.total_execution_time for r in results]
        success_rates = [r.task_completion_rate for r in results]
        accuracy_scores = [r.response_accuracy for r in results]
        costs = [r.estimated_cost_usd for r in results if r.estimated_cost_usd > 0]
        token_counts = [r.total_tokens for r in results if r.total_tokens > 0]

        stats = {
            "total_tests": len(results),
            "execution_time": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "std_dev": (
                    statistics.stdev(execution_times)
                    if len(execution_times) > 1
                    else 0.0
                ),
                "min": min(execution_times),
                "max": max(execution_times),
            },
            "success_rate": {
                "mean": statistics.mean(success_rates),
                "total_successes": sum(success_rates),
            },
            "accuracy": {
                "mean": statistics.mean(accuracy_scores),
                "median": statistics.median(accuracy_scores),
            },
            "cost": {
                "mean": statistics.mean(costs) if costs else 0.0,
                "total": sum(costs) if costs else 0.0,
                "per_success": (
                    sum(costs) / sum(success_rates)
                    if costs and sum(success_rates) > 0
                    else 0.0
                ),
            },
            "tokens": {
                "mean": statistics.mean(token_counts) if token_counts else 0.0,
                "total": sum(token_counts) if token_counts else 0.0,
            },
            "reliability": {
                "error_rate": sum(r.error_count for r in results) / len(results),
                "timeout_rate": sum(r.timeout_count for r in results) / len(results),
            },
        }

        return stats

    def _generate_rankings(
        self, comparison_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Generate rankings for different performance aspects."""

        rankings = {}

        # Performance rankings
        ranking_criteria = [
            (
                "execution_time",
                lambda x: 1 / x["execution_time"]["mean"],
                "Speed (lower is better)",
            ),
            ("success_rate", lambda x: x["success_rate"]["mean"], "Success Rate"),
            ("accuracy", lambda x: x["accuracy"]["mean"], "Accuracy"),
            (
                "cost_efficiency",
                lambda x: 1 / (x["cost"]["per_success"] + 0.001),
                "Cost Efficiency",
            ),
            (
                "reliability",
                lambda x: 1
                - (x["reliability"]["error_rate"] + x["reliability"]["timeout_rate"]),
                "Reliability",
            ),
        ]

        for criterion_name, score_func, description in ranking_criteria:
            try:
                scored_items = [
                    (provider, score_func(stats))
                    for provider, stats in comparison_data.items()
                    if stats  # Only include non-empty stats
                ]

                # Sort by score (descending)
                scored_items.sort(key=lambda x: x[1], reverse=True)

                rankings[criterion_name] = {
                    "description": description,
                    "ranking": scored_items,
                }
            except Exception as e:
                print(f"Error calculating ranking for {criterion_name}: {e}")
                rankings[criterion_name] = {"description": description, "ranking": []}

        return rankings

    def _analyze_patterns(
        self, comparison_data: Dict[str, Dict[str, Any]], group_by: str
    ) -> List[str]:
        """Analyze patterns and generate insights."""

        insights = []

        if not comparison_data:
            return ["No data available for analysis"]

        # Performance variance analysis
        execution_times = [
            stats["execution_time"]["mean"]
            for stats in comparison_data.values()
            if stats
        ]

        if len(execution_times) > 1:
            time_variance = statistics.stdev(execution_times)
            mean_time = statistics.mean(execution_times)

            if time_variance / mean_time > 0.5:  # High variance
                insights.append(
                    f"High performance variance detected across {group_by}s "
                    f"(CV: {time_variance/mean_time:.2f})"
                )

        # Success rate analysis
        success_rates = [
            stats["success_rate"]["mean"] for stats in comparison_data.values() if stats
        ]

        if success_rates:
            min_success = min(success_rates)
            max_success = max(success_rates)

            if max_success - min_success > 0.3:  # >30% difference
                insights.append(
                    f"Significant success rate differences: "
                    f"{min_success:.1%} to {max_success:.1%}"
                )

        # Cost efficiency analysis
        costs = [
            stats["cost"]["per_success"]
            for stats in comparison_data.values()
            if stats and stats["cost"]["per_success"] > 0
        ]

        if len(costs) > 1:
            cost_range = max(costs) - min(costs)
            if cost_range > min(costs):  # Cost difference > cheapest option
                insights.append(
                    f"Large cost differences detected: "
                    f"${min(costs):.4f} to ${max(costs):.4f} per success"
                )

        # Reliability patterns
        error_rates = [
            stats["reliability"]["error_rate"]
            for stats in comparison_data.values()
            if stats
        ]

        if error_rates and max(error_rates) > 0.1:  # >10% error rate
            problematic_providers = [
                provider
                for provider, stats in comparison_data.items()
                if stats and stats["reliability"]["error_rate"] > 0.1
            ]
            insights.append(
                f"High error rates detected for: {', '.join(problematic_providers)}"
            )

        return insights

    def _generate_comparison_summary(
        self, comparison_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a high-level comparison summary."""

        if not comparison_data:
            return {"total_providers": 0, "overall_performance": "No data"}

        # Overall statistics
        all_execution_times = []
        all_success_rates = []
        all_costs = []

        for stats in comparison_data.values():
            if stats:
                all_execution_times.append(stats["execution_time"]["mean"])
                all_success_rates.append(stats["success_rate"]["mean"])
                if stats["cost"]["per_success"] > 0:
                    all_costs.append(stats["cost"]["per_success"])

        summary = {
            "total_providers": len(comparison_data),
            "avg_execution_time": (
                statistics.mean(all_execution_times) if all_execution_times else 0.0
            ),
            "avg_success_rate": (
                statistics.mean(all_success_rates) if all_success_rates else 0.0
            ),
            "avg_cost_per_success": statistics.mean(all_costs) if all_costs else 0.0,
        }

        # Determine overall performance grade
        avg_success = summary["avg_success_rate"]
        if avg_success >= 0.95:
            summary["overall_performance"] = "Excellent"
        elif avg_success >= 0.85:
            summary["overall_performance"] = "Good"
        elif avg_success >= 0.70:
            summary["overall_performance"] = "Fair"
        else:
            summary["overall_performance"] = "Needs Improvement"

        return summary


class PerformanceReporter:
    """Generate comprehensive performance reports."""

    def __init__(self):
        self.config = get_test_config()

    def generate_comprehensive_report(
        self, results: List[PerformanceMetrics], output_dir: str = "performance_reports"
    ) -> Dict[str, str]:
        """Generate comprehensive performance reports in multiple formats."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        generated_files = {}

        # Generate JSON report
        json_file = output_path / f"performance_report_{timestamp}.json"
        self._generate_json_report(results, json_file)
        generated_files["json"] = str(json_file)

        # Generate CSV report
        csv_file = output_path / f"performance_data_{timestamp}.csv"
        self._generate_csv_report(results, csv_file)
        generated_files["csv"] = str(csv_file)

        # Generate markdown report
        md_file = output_path / f"performance_report_{timestamp}.md"
        self._generate_markdown_report(results, md_file)
        generated_files["markdown"] = str(md_file)

        # Generate performance charts
        charts_dir = output_path / f"charts_{timestamp}"
        charts_dir.mkdir(exist_ok=True)
        chart_files = self._generate_performance_charts(results, charts_dir)
        generated_files["charts"] = chart_files

        return generated_files

    def _generate_json_report(
        self, results: List[PerformanceMetrics], output_file: Path
    ):
        """Generate JSON format report."""

        # Convert results to serializable format
        json_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tests": len(results),
                "test_duration": "calculated_from_timestamps",
            },
            "test_results": [],
        }

        for result in results:
            result_data = {
                "provider_name": result.provider_name,
                "model_name": result.model_name,
                "scenario_name": result.scenario_name,
                "timestamp": result.timestamp.isoformat(),
                "performance_metrics": {
                    "execution_time": {
                        "total": result.total_execution_time,
                        "first_response": result.first_response_time,
                        "average_response": result.average_response_time,
                    },
                    "throughput": {
                        "tokens_per_second": result.tokens_per_second,
                        "tool_calls_per_second": result.tool_calls_per_second,
                        "iterations_per_second": result.iterations_per_second,
                    },
                    "quality": {
                        "completion_rate": result.task_completion_rate,
                        "accuracy": result.response_accuracy,
                        "coherence": result.reasoning_coherence,
                    },
                    "cost": {
                        "estimated_cost_usd": result.estimated_cost_usd,
                        "cost_per_success": result.cost_per_success,
                    },
                    "tokens": {
                        "input": result.input_tokens,
                        "output": result.output_tokens,
                        "total": result.total_tokens,
                    },
                    "reliability": {
                        "error_count": result.error_count,
                        "timeout_count": result.timeout_count,
                        "retry_count": result.retry_count,
                    },
                    "resources": {
                        "peak_memory_mb": result.peak_memory_mb,
                        "avg_cpu_percent": result.avg_cpu_percent,
                        "network_requests": result.network_requests,
                    },
                },
                "test_configuration": result.test_configuration,
            }
            json_data["test_results"].append(result_data)

        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)

    def _generate_csv_report(
        self, results: List[PerformanceMetrics], output_file: Path
    ):
        """Generate CSV format report for data analysis."""

        fieldnames = [
            "provider_name",
            "model_name",
            "scenario_name",
            "timestamp",
            "total_execution_time",
            "first_response_time",
            "tokens_per_second",
            "task_completion_rate",
            "response_accuracy",
            "reasoning_coherence",
            "estimated_cost_usd",
            "cost_per_success",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "error_count",
            "timeout_count",
            "peak_memory_mb",
            "avg_cpu_percent",
            "network_requests",
        ]

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "provider_name": result.provider_name,
                    "model_name": result.model_name,
                    "scenario_name": result.scenario_name,
                    "timestamp": result.timestamp.isoformat(),
                    "total_execution_time": result.total_execution_time,
                    "first_response_time": result.first_response_time,
                    "tokens_per_second": result.tokens_per_second,
                    "task_completion_rate": result.task_completion_rate,
                    "response_accuracy": result.response_accuracy,
                    "reasoning_coherence": result.reasoning_coherence,
                    "estimated_cost_usd": result.estimated_cost_usd,
                    "cost_per_success": result.cost_per_success,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                    "error_count": result.error_count,
                    "timeout_count": result.timeout_count,
                    "peak_memory_mb": result.peak_memory_mb,
                    "avg_cpu_percent": result.avg_cpu_percent,
                    "network_requests": result.network_requests,
                }
                writer.writerow(row)

    def _generate_markdown_report(
        self, results: List[PerformanceMetrics], output_file: Path
    ):
        """Generate markdown format report."""

        comparator = PerformanceComparator()
        comparison = comparator.compare_providers(results)

        report_lines = [
            "# Performance Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- **Total Tests**: {len(results)}",
            f"- **Providers Tested**: {len(set(r.provider_name for r in results))}",
            f"- **Models Tested**: {len(set(f'{r.provider_name}:{r.model_name}' for r in results))}",
            f"- **Scenarios Tested**: {len(set(r.scenario_name for r in results))}",
            "",
            f"**Overall Performance**: {comparison['summary']['overall_performance']}",
            f"**Average Success Rate**: {comparison['summary']['avg_success_rate']:.1%}",
            f"**Average Execution Time**: {comparison['summary']['avg_execution_time']:.2f}s",
            "",
        ]

        # Provider comparison table
        report_lines.extend(
            [
                "## Provider Performance Comparison",
                "",
                "| Provider | Success Rate | Avg Time (s) | Accuracy | Cost/Success | Reliability |",
                "|----------|--------------|--------------|----------|--------------|-------------|",
            ]
        )

        for provider, stats in comparison["comparison_data"].items():
            if stats:
                report_lines.append(
                    f"| {provider} | {stats['success_rate']['mean']:.1%} | "
                    f"{stats['execution_time']['mean']:.2f} | "
                    f"{stats['accuracy']['mean']:.2f} | "
                    f"${stats['cost']['per_success']:.4f} | "
                    f"{1-(stats['reliability']['error_rate']+stats['reliability']['timeout_rate']):.1%} |"
                )

        report_lines.extend(["", "## Performance Rankings", ""])

        # Rankings
        for criterion, ranking_data in comparison["rankings"].items():
            report_lines.extend([f"### {ranking_data['description']}", ""])

            for rank, (provider, score) in enumerate(ranking_data["ranking"], 1):
                report_lines.append(f"{rank}. **{provider}** (Score: {score:.3f})")

            report_lines.append("")

        # Insights
        if comparison["insights"]:
            report_lines.extend(["## Key Insights", ""])

            for insight in comparison["insights"]:
                report_lines.append(f"- {insight}")

            report_lines.append("")

        # Write markdown file
        with open(output_file, "w") as f:
            f.write("\n".join(report_lines))

    def _generate_performance_charts(
        self, results: List[PerformanceMetrics], output_dir: Path
    ) -> List[str]:
        """Generate performance visualization charts."""

        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib/pandas not available, skipping chart generation")
            return []

        chart_files = []

        # Convert to DataFrame for easier plotting
        df_data = []
        for result in results:
            df_data.append(
                {
                    "provider": result.provider_name,
                    "model": result.model_name,
                    "scenario": result.scenario_name,
                    "execution_time": result.total_execution_time,
                    "success_rate": result.task_completion_rate,
                    "accuracy": result.response_accuracy,
                    "cost": result.estimated_cost_usd,
                    "tokens": result.total_tokens,
                }
            )

        # Type check for pandas availability
        if pd is None:
            print("Warning: pandas not available, skipping chart generation")
            return []

        df = pd.DataFrame(df_data)

        # Type check for matplotlib availability
        if plt is None:
            print("Warning: matplotlib not available, skipping chart generation")
            return []

        # Execution time comparison
        plt.figure(figsize=(12, 6))
        provider_times = df.groupby("provider")["execution_time"].mean()
        provider_times.plot(kind="bar")
        plt.title("Average Execution Time by Provider")
        plt.xlabel("Provider")
        plt.ylabel("Execution Time (s)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        chart_file = output_dir / "execution_time_comparison.png"
        plt.savefig(chart_file)
        plt.close()
        chart_files.append(str(chart_file))

        # Success rate comparison
        plt.figure(figsize=(12, 6))
        provider_success = df.groupby("provider")["success_rate"].mean()
        provider_success.plot(kind="bar", color="green", alpha=0.7)
        plt.title("Success Rate by Provider")
        plt.xlabel("Provider")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.tight_layout()

        chart_file = output_dir / "success_rate_comparison.png"
        plt.savefig(chart_file)
        plt.close()
        chart_files.append(str(chart_file))

        # Cost vs Performance scatter
        if df["cost"].sum() > 0:  # Only if cost data available
            plt.figure(figsize=(10, 8))
            for provider in df["provider"].unique():
                provider_data = df[df["provider"] == provider]
                plt.scatter(
                    provider_data["cost"],
                    provider_data["success_rate"],
                    label=provider,
                    alpha=0.7,
                    s=60,
                )

            plt.xlabel("Cost (USD)")
            plt.ylabel("Success Rate")
            plt.title("Cost vs Performance")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            chart_file = output_dir / "cost_vs_performance.png"
            plt.savefig(chart_file)
            plt.close()
            chart_files.append(str(chart_file))

        return chart_files


if __name__ == "__main__":
    # Example usage
    print("Performance Benchmarking System initialized")
    print("Use this module to benchmark and compare provider performance")

    # Example of how to use the benchmarker
    benchmarker = PerformanceBenchmarker()
    comparator = PerformanceComparator()
    reporter = PerformanceReporter()

    print("Ready for performance testing!")
