#!/usr/bin/env python3
"""Quick test of the provider consistency system."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

async def main():
    """Test the provider consistency system."""
    try:
        from reactive_agents.tests.integration.test_provider_consistency import ProviderConsistencyTester
        
        print("🧪 Testing Provider Consistency System")
        print("=" * 50)
        
        # Create tester with mock execution
        tester = ProviderConsistencyTester(enable_real_execution=False)
        
        # Run a limited test
        results = await tester.run_comprehensive_test(
            providers=['ollama', 'openai'], 
            scenarios=['simple_task']
        )
        
        # Generate report
        report = tester.generate_report(results)
        
        print("\n" + "=" * 60)
        print("TEST REPORT")
        print("=" * 60)
        print(report)
        print("=" * 60)
        
        print(f"\n✅ Provider consistency system test completed successfully!")
        print(f"📊 Tested {results['total_tests']} combinations")
        print(f"✅ {results['successful_tests']} passed, ❌ {results['failed_tests']} failed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)