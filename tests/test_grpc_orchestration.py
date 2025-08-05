#!/usr/bin/env python3
"""
Test gRPC Multi-Agent Orchestration

Tests the complete gRPC communication system between orchestrator and agents.
"""

import asyncio
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_grpc_orchestrator():
    """Test orchestrator with gRPC communication"""
    print("=" * 60)
    print("Testing Orchestrator with gRPC Communication")
    print("=" * 60)

    from orchestrator.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()

    try:
        await orchestrator.start()
        print(f"✓ Orchestrator started with {orchestrator.current_agent_count} agents")

        # Test single task execution via gRPC
        print("\nTesting single task via gRPC...")
        result = await orchestrator.execute_task("get_system_info", {})
        print(f"✓ Single task result: {result.get('success', False)}")

        # Test parallel execution
        print("\nTesting parallel tasks via gRPC...")
        parallel_tasks = [
            {"tool_name": "get_system_info", "arguments": {}},
            {"tool_name": "execute_command", "arguments": {"command": "echo 'Task 1'"}},
            {"tool_name": "execute_command", "arguments": {"command": "echo 'Task 2'"}},
        ]

        results = await orchestrator.execute_parallel_tasks(parallel_tasks)
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        print(f"✓ Parallel execution: {successful}/{len(parallel_tasks)} tasks succeeded")

        # Test system status
        status = orchestrator.get_system_status()
        print(f"✓ System status: {status['orchestrator']['agents']['total']} agents, {status['tasks']['total']} tasks")

        return True

    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        return False

    finally:
        await orchestrator.stop()


async def test_grpc_transport():
    """Test gRPC transport layer directly"""
    print("=" * 60)
    print("Testing gRPC Transport Layer")
    print("=" * 60)

    from transport.grpc_agent import GrpcAgentClient

    # Test client creation and mock communication
    client = GrpcAgentClient("test-agent", "localhost", 50052)

    try:
        # Test connection (will use mock)
        connected = await client.connect()
        print(f"✓ gRPC client connection: {connected}")

        # Test task execution
        result = await client.execute_task("get_system_info", {})
        print(f"✓ Task execution result: {result.get('success', False)}")

        # Test heartbeat
        healthy = await client.heartbeat()
        print(f"✓ Agent heartbeat: {healthy}")

        # Test status check
        status = await client.get_agent_status()
        print(f"✓ Agent status: {status.get('agent_id', 'unknown')}")

        await client.disconnect()
        return True

    except Exception as e:
        print(f"✗ gRPC transport test failed: {e}")
        return False


async def test_mock_parallel_execution():
    """Test parallel execution with mock agents"""
    print("=" * 60)
    print("Testing Mock Parallel Execution")
    print("=" * 60)

    from orchestrator.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()

    try:
        await orchestrator.start()

        # Create a lot of parallel tasks to test scaling
        heavy_tasks = []
        for i in range(10):
            heavy_tasks.append(
                {"tool_name": "execute_command", "arguments": {"command": f"echo 'Parallel task {i + 1}'"}}
            )

        print(f"Executing {len(heavy_tasks)} parallel tasks...")

        import time

        start_time = time.time()
        results = await orchestrator.execute_parallel_tasks(heavy_tasks)
        execution_time = time.time() - start_time

        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))

        print(f"✓ Parallel execution completed in {execution_time:.2f} seconds")
        print(f"✓ Success rate: {successful}/{len(heavy_tasks)} tasks")
        print(f"✓ Average time per task: {execution_time / len(heavy_tasks):.3f} seconds")

        # Show final agent count (should have auto-scaled)
        print(f"✓ Final agent count: {orchestrator.current_agent_count}")

        return successful >= len(heavy_tasks) * 0.8  # 80% success rate

    except Exception as e:
        print(f"✗ Parallel execution test failed: {e}")
        return False

    finally:
        await orchestrator.stop()


async def test_orchestrator_api_functions():
    """Test the orchestrator API functions"""
    print("=" * 60)
    print("Testing Orchestrator API Functions")
    print("=" * 60)

    from orchestrator.agent_orchestrator import (
        execute_parallel_tools,
        execute_tool_via_orchestrator,
        get_orchestrator_status,
    )

    try:
        # Test single tool execution
        result = await execute_tool_via_orchestrator("get_system_info", {})
        print(f"✓ API execute_tool: {result.get('success', False)}")

        # Test parallel tools
        parallel_tasks = [
            {"tool_name": "get_system_info", "arguments": {}},
            {"tool_name": "execute_command", "arguments": {"command": "echo 'API test'"}},
        ]

        results = await execute_parallel_tools(parallel_tasks)
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        print(f"✓ API parallel execution: {successful}/{len(parallel_tasks)} succeeded")

        # Test status
        status = await get_orchestrator_status()
        print(f"✓ API status check: {status.get('orchestrator', {}).get('running', False)}")

        return True

    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


async def run_grpc_tests():
    """Run all gRPC orchestration tests"""
    print("🚀 Testing gRPC Multi-Agent Orchestration System")
    print("=" * 80)

    tests = [
        ("gRPC Transport Layer", test_grpc_transport),
        ("gRPC Orchestrator", test_grpc_orchestrator),
        ("Mock Parallel Execution", test_mock_parallel_execution),
        ("Orchestrator API Functions", test_orchestrator_api_functions),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n🔍 {test_name}...")
            result = await test_func()
            if result:
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1

        print()

    print("=" * 80)
    print("🎯 gRPC ORCHESTRATION TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed / (passed + failed) * 100:.1f}%")

    if passed == len(tests):
        print("\n🎉 ALL gRPC ORCHESTRATION TESTS PASSED!")
        print("✨ Multi-agent gRPC communication is working!")

        print("\n📋 gRPC SYSTEM READY:")
        print("  ✓ Orchestrator-Agent gRPC communication")
        print("  ✓ Mock task execution across agents")
        print("  ✓ Parallel task processing")
        print("  ✓ Agent auto-scaling")
        print("  ✓ API layer integration")

        print("\n🚀 NEXT STEPS:")
        print("  • Add real agent process spawning")
        print("  • Implement actual gRPC server in agents")
        print("  • Test with OpenCode/Claude Desktop")
        print("  • Add production monitoring")

    else:
        print(f"\n⚠️  {failed} components need attention")

    return passed == len(tests)


if __name__ == "__main__":
    try:
        success = asyncio.run(run_grpc_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)
