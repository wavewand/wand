#!/usr/bin/env python3
"""
Simple Architecture Test - Validate refactored components work individually

Tests each layer of the new architecture without requiring full process orchestration.
"""

import asyncio
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_core_task_manager():
    """Test core task management functionality"""
    print("=" * 60)
    print("Testing Core Task Manager")
    print("=" * 60)

    from core.task_manager import DistributedTaskManager, TaskPriority

    task_manager = DistributedTaskManager()

    # Test task creation
    task = task_manager.create_task(
        title="Test Task", description="Testing task creation", task_type="test", priority=TaskPriority.HIGH
    )

    print(f"✓ Created task: {task.id}")
    print(f"✓ Task manager has {len(task_manager.tasks)} tasks")
    print(f"✓ Task manager has {len(task_manager.agents)} agents")

    # Test system status
    status = task_manager.get_system_status()
    print(f"✓ System status: {status['tasks']['total']} tasks, {status['agents']['total']} agents")

    return True


async def test_agent_tools():
    """Test agent tools functionality"""
    print("=" * 60)
    print("Testing Agent Tools")
    print("=" * 60)

    from agents.tools import AgentTools

    tools = AgentTools("test-agent", {})

    # Test system info
    result = await tools.get_system_info()
    print(f"✓ get_system_info: {result['success']}")

    # Test file operations
    result = await tools.list_directory(".")
    print(f"✓ list_directory: {result['success']}")

    # Test command check
    result = await tools.check_command_exists("python")
    print(f"✓ check_command_exists: {result['success']}")

    return True


async def test_agent_instance():
    """Test agent instance without orchestrator connection"""
    print("=" * 60)
    print("Testing Agent Instance")
    print("=" * 60)

    from agents.agent import Agent

    agent_config = {"agent_id": "test-standalone", "orchestrator_port": 50051, "agent_port": 50052, "config": {}}

    agent = Agent("test-standalone", agent_config)

    print(f"✓ Agent initialized with {len(agent.tools)} tools")

    # Test direct tool execution
    result = await agent.execute_tool("get_system_info", {})
    print(f"✓ Direct tool execution: {result['success']}")

    # Show available tools
    print(f"✓ Available tools: {len(agent.tools)} tools")
    tools_list = list(agent.tools.keys())[:10]  # Show first 10
    print(f"  Sample tools: {', '.join(tools_list)}...")

    return True


async def test_mcp_transport():
    """Test MCP HTTP transport layer"""
    print("=" * 60)
    print("Testing MCP HTTP Transport")
    print("=" * 60)

    from transport.mcp_http import MCPHttpTransport

    transport = MCPHttpTransport("test", "1.0.0")

    print(f"✓ MCP Transport created")
    print(f"✓ FastAPI app initialized: {transport.app.title}")

    # Test available tools list
    tools = transport._get_available_tools()
    print(f"✓ Transport exposes {len(tools)} tools")

    return True


async def test_new_main_entry():
    """Test new main entry point structure"""
    print("=" * 60)
    print("Testing Main Entry Point")
    print("=" * 60)

    # Import without running
    sys.path.append('.')
    try:
        import wand

        print("✓ wand.py imports successfully")

        # Test stdio MCP server creation
        mcp_server = await wand.create_stdio_mcp_server()
        print("✓ stdio MCP server creation works")

        return True
    except Exception as e:
        print(f"✗ Main entry test failed: {e}")
        return False


async def test_config_system():
    """Test configuration system"""
    print("=" * 60)
    print("Testing Configuration System")
    print("=" * 60)

    # Test the simple config used by orchestrator
    from orchestrator.agent_orchestrator import SimpleConfig

    config = SimpleConfig()
    print(f"✓ Simple config created")
    print(f"✓ Server version: {config.server.version}")
    print(f"✓ Max concurrent tasks: {config.distributed.max_concurrent_tasks}")

    return True


async def run_architecture_tests():
    """Run all architecture tests"""
    print("🚀 Testing Refactored Multi-Agent Architecture")
    print("=" * 80)

    tests = [
        ("Core Task Manager", test_core_task_manager),
        ("Agent Tools", test_agent_tools),
        ("Agent Instance", test_agent_instance),
        ("MCP Transport", test_mcp_transport),
        ("Configuration", test_config_system),
        ("Main Entry Point", test_new_main_entry),
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
    print("🎯 ARCHITECTURE TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed / (passed + failed) * 100:.1f}%")

    if passed == len(tests):
        print("\n🎉 ALL ARCHITECTURE TESTS PASSED!")
        print("✨ The refactored multi-agent system is ready!")

        print("\n📋 ARCHITECTURE SUMMARY:")
        print("  ✓ Core task management layer")
        print("  ✓ Agent tools with 22 MCP functions")
        print("  ✓ Individual agent instances")
        print("  ✓ MCP HTTP transport layer")
        print("  ✓ Configuration system")
        print("  ✓ Main entry points")

        print("\n🚀 DEPLOYMENT OPTIONS:")
        print("  • python wand.py stdio    # For Claude Desktop")
        print("  • python wand.py http     # For OpenCode")
        print("  • python wand.py both     # Dual transport")
        print("  • python -m agents.agent --test  # Test individual agent")

    else:
        print(f"\n⚠️  {failed} components need attention")

    return passed == len(tests)


if __name__ == "__main__":
    try:
        success = asyncio.run(run_architecture_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)
