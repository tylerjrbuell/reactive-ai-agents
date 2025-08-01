#!/usr/bin/env python3
"""
Comprehensive provider testing runner script.

This script provides an easy way to run the complete provider consistency
test suite with various options and configurations.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from reactive_agents.tests.integration.test_provider_consistency import ProviderConsistencyTester
from reactive_agents.tests.integration.performance_benchmarking import PerformanceBenchmarker, PerformanceComparator, PerformanceReporter
from reactive_agents.tests.integration.provider_test_config import get_test_config


async def run_consistency_tests(args):
    """Run provider consistency tests."""
    print("🧪 Running Provider Consistency Tests")
    print("=" * 50)
    
    tester = ProviderConsistencyTester(enable_real_execution=args.real_execution)
    
    # Filter providers and scenarios based on arguments
    providers = args.providers if args.providers else None
    scenarios = args.scenarios if args.scenarios else None
    
    results = await tester.run_comprehensive_test(
        providers=providers,
        scenarios=scenarios
    )
    
    # Generate and display report
    report = tester.generate_report(results)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    
    # Save detailed results if requested
    if args.save_results:
        output_file = f"provider_test_results_{args.timestamp}.json"
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📁 Detailed results saved to: {output_file}")
    
    return results


async def run_performance_benchmarks(args):
    """Run performance benchmarking tests."""
    print("⚡ Running Performance Benchmarks")
    print("=" * 50)
    
    benchmarker = PerformanceBenchmarker()
    comparator = PerformanceComparator()
    reporter = PerformanceReporter()
    
    config = get_test_config()
    all_results = []
    
    # Test specified providers or all available
    test_providers = args.providers if args.providers else list(config.providers.keys())
    
    for provider_name in test_providers:
        if provider_name not in config.providers:
            print(f"⚠️ Skipping unknown provider: {provider_name}")
            continue
        
        provider_config = config.providers[provider_name]
        
        # Test first model only for speed, or all if requested
        models_to_test = list(provider_config.models.keys())
        if not args.all_models:
            models_to_test = models_to_test[:1]
        
        for model_name in models_to_test:
            print(f"🏃 Benchmarking {provider_name}:{model_name}")
            
            # Create a mock agent builder function
            async def mock_agent_builder():
                from reactive_agents.app.builders.agent import ReactiveAgentBuilder
                from reactive_agents.core.types.reasoning_types import ReasoningStrategies
                from unittest.mock import MagicMock, patch
                from reactive_agents.providers.llm.factory import ModelProviderFactory
                from reactive_agents.providers.llm.base import BaseModelProvider
                
                with patch.object(ModelProviderFactory, 'get_model_provider') as mock_factory:
                    mock_provider = MagicMock(spec=BaseModelProvider)
                    mock_provider.validate_model.return_value = True
                    mock_provider.get_chat_completion.return_value = {
                        "choices": [{"message": {"content": "Benchmark response"}}],
                        "usage": {"total_tokens": 100}
                    }
                    mock_factory.return_value = mock_provider
                    
                    return await (
                        ReactiveAgentBuilder()
                        .with_name(f"Benchmark-{provider_name}-{model_name}")
                        .with_model(f"{provider_name}:{model_name}")
                        .with_role("Benchmark Agent")
                        .with_instructions("Benchmarking agent performance")
                        .with_reasoning_strategy(ReasoningStrategies.REACTIVE)
                        .with_max_iterations(3)
                        .build()
                    )
            
            # Run benchmark with fewer iterations for speed
            iterations = args.benchmark_iterations if args.benchmark_iterations else 1
            scenario_name = "simple_task"  # Use simple scenario for benchmarking
            
            try:
                benchmark_results = await benchmarker.benchmark_provider_model(
                    provider_name, model_name, scenario_name, mock_agent_builder, iterations
                )
                all_results.extend(benchmark_results)
                
                # Print quick summary
                avg_time = sum(r.total_execution_time for r in benchmark_results) / len(benchmark_results)
                print(f"  ⏱️ Average execution time: {avg_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Benchmark failed: {e}")
    
    if all_results:
        # Generate comparison and reports
        comparison = comparator.compare_providers(all_results)
        
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)
        
        for provider, stats in comparison["comparison_data"].items():
            if stats:
                print(f"\n{provider.upper()}:")
                print(f"  Success Rate: {stats['success_rate']['mean']:.1%}")
                print(f"  Avg Time: {stats['execution_time']['mean']:.2f}s")
                print(f"  Avg Quality: {stats['accuracy']['mean']:.2f}")
        
        # Generate detailed reports if requested
        if args.save_results:
            report_files = reporter.generate_comprehensive_report(
                all_results, f"performance_reports_{args.timestamp}"
            )
            print(f"\n📊 Performance reports generated:")
            for format_type, filepath in report_files.items():
                if isinstance(filepath, list):
                    print(f"  {format_type}: {len(filepath)} files")
                else:
                    print(f"  {format_type}: {filepath}")
    
    return all_results


async def run_parameter_validation(args):
    """Run parameter validation tests using pytest."""
    print("🔧 Running Parameter Validation Tests")
    print("=" * 50)
    
    import subprocess
    import sys
    
    # Build pytest command
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "reactive_agents/tests/integration/test_provider_parameters.py",
        "-v", "-x"  # Verbose and stop on first failure
    ]
    
    if args.providers:
        # Add provider filter (would need custom pytest plugin to support this properly)
        pytest_cmd.extend(["-k", " or ".join(args.providers)])
    
    # Run pytest
    result = subprocess.run(pytest_cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Provider Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --consistency                    # Run consistency tests (mocked)
  %(prog)s --consistency --real             # Run with real API calls
  %(prog)s --performance --providers ollama # Benchmark only Ollama
  %(prog)s --parameters --providers openai  # Validate OpenAI parameters
  %(prog)s --all --save                     # Run everything and save results
        """
    )
    
    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--consistency", action="store_true", 
                           help="Run provider consistency tests")
    test_group.add_argument("--performance", action="store_true",
                           help="Run performance benchmarking tests")
    test_group.add_argument("--parameters", action="store_true",
                           help="Run parameter validation tests")
    test_group.add_argument("--all", action="store_true",
                           help="Run all test types")
    
    # Filtering options
    parser.add_argument("--providers", nargs="+", 
                       choices=["ollama", "openai", "anthropic", "groq", "google"],
                       help="Specific providers to test")
    parser.add_argument("--scenarios", nargs="+",
                       choices=["simple_task", "math_task", "json_processing", 
                               "multi_tool_task", "adaptive_reasoning"],
                       help="Specific scenarios to test")
    
    # Execution options
    parser.add_argument("--real", dest="real_execution", action="store_true",
                       help="Use real API calls instead of mocks")
    parser.add_argument("--all-models", action="store_true",
                       help="Test all models for each provider (slower)")
    parser.add_argument("--benchmark-iterations", type=int, default=1,
                       help="Number of benchmark iterations per test")
    
    # Output options
    parser.add_argument("--save", dest="save_results", action="store_true",
                       help="Save detailed results to files")
    parser.add_argument("--output-dir", default="test_results",
                       help="Directory for output files")
    
    args = parser.parse_args()
    
    # Add timestamp for file naming
    from datetime import datetime
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Default to consistency tests if no specific test type selected
    if not any([args.consistency, args.performance, args.parameters, args.all]):
        args.consistency = True
    
    async def run_tests():
        """Run the selected tests."""
        results = {}
        
        try:
            if args.all or args.consistency:
                print("Starting consistency tests...")
                results['consistency'] = await run_consistency_tests(args)
                print("✅ Consistency tests completed\n")
            
            if args.all or args.performance:
                print("Starting performance benchmarks...")
                results['performance'] = await run_performance_benchmarks(args)
                print("✅ Performance benchmarks completed\n")
            
            if args.all or args.parameters:
                print("Starting parameter validation...")
                results['parameters'] = await run_parameter_validation(args)
                print("✅ Parameter validation completed\n")
            
            print("🎉 All selected tests completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Test execution failed: {e}")
            sys.exit(1)
        
        return results
    
    # Run the tests
    results = asyncio.run(run_tests())
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    for test_type, result in results.items():
        if test_type == 'parameters':
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_type.upper()}: {status}")
        elif isinstance(result, dict) and 'total_tests' in result:
            success_rate = result['successful_tests'] / result['total_tests'] * 100
            print(f"{test_type.upper()}: {result['successful_tests']}/{result['total_tests']} ({success_rate:.1f}%)")
        else:
            print(f"{test_type.upper()}: Completed")


if __name__ == "__main__":
    main()