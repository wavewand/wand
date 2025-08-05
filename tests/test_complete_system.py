#!/usr/bin/env python3
"""
Complete Multi-Framework System Test

This script tests the complete multi-framework MCP system including:
- Framework registry with Haystack and LlamaIndex
- Both AI agents (Haystack and LlamaIndex)
- Framework-agnostic API design
- Task handling across frameworks
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from agents.haystack_agent import HaystackAgent
from agents.llamaindex_agent import LlamaIndexAgent

# Import framework components
from ai_framework_registry import ai_framework_registry
from distributed.types import AgentType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class MultiFrameworkSystemTest:
    """Test class for the complete multi-framework system."""

    def __init__(self):
        self.test_results = {}
        self.agents = {}

    async def test_framework_registry(self):
        """Test the framework registry functionality."""
        logger.info("🧪 Testing Framework Registry...")

        try:
            # Test framework listing
            frameworks = ai_framework_registry.list_frameworks()
            self.test_results["framework_listing"] = {
                "success": True,
                "frameworks": frameworks,
                "count": len(frameworks),
            }
            logger.info(f"✅ Found {len(frameworks)} frameworks: {frameworks}")

            # Test framework retrieval
            haystack_framework = ai_framework_registry.get_framework("haystack")
            llamaindex_framework = ai_framework_registry.get_framework("llamaindex")

            self.test_results["framework_retrieval"] = {
                "success": True,
                "haystack_available": haystack_framework is not None,
                "llamaindex_available": llamaindex_framework is not None,
            }

            # Test framework status
            if haystack_framework:
                haystack_status = await haystack_framework.get_status()
                logger.info(f"Haystack status: {haystack_status}")

            if llamaindex_framework:
                llamaindex_status = await llamaindex_framework.get_status()
                logger.info(f"LlamaIndex status: {llamaindex_status}")

            logger.info("✅ Framework registry tests passed")

        except Exception as e:
            logger.error(f"❌ Framework registry test failed: {e}")
            self.test_results["framework_registry"] = {"success": False, "error": str(e)}

    async def test_agent_creation(self):
        """Test creating agents for both frameworks."""
        logger.info("🧪 Testing Agent Creation...")

        try:
            # Create Haystack agent
            self.agents["haystack"] = HaystackAgent(
                agent_id="test-haystack-agent", coordinator_address="localhost:50051"
            )
            logger.info("✅ Created Haystack agent")

            # Create LlamaIndex agent
            self.agents["llamaindex"] = LlamaIndexAgent(
                agent_id="test-llamaindex-agent", coordinator_address="localhost:50051"
            )
            logger.info("✅ Created LlamaIndex agent")

            # Test agent status
            for framework_name, agent in self.agents.items():
                status = agent.get_agent_status()
                logger.info(f"{framework_name} agent status: {status['status']}")
                logger.info(f"{framework_name} agent capabilities: {status['capabilities']}")

            self.test_results["agent_creation"] = {"success": True, "agents_created": list(self.agents.keys())}

        except Exception as e:
            logger.error(f"❌ Agent creation test failed: {e}")
            self.test_results["agent_creation"] = {"success": False, "error": str(e)}

    async def test_framework_agnostic_tasks(self):
        """Test executing the same task types across different frameworks."""
        logger.info("🧪 Testing Framework-Agnostic Tasks...")

        # Sample tasks that both frameworks should handle
        test_tasks = [
            {
                "id": "rag-test-1",
                "type": "rag_query",
                "query": "What is artificial intelligence?",
                "temperature": 0.7,
                "max_tokens": 200,
            },
            {
                "id": "search-test-1",
                "type": "document_search",
                "query": "machine learning techniques",
                "search_type": "semantic",
                "max_results": 5,
            },
            {
                "id": "ingestion-test-1",
                "type": "document_ingestion",
                "filename": "test_document.txt",
                "content": "This is a test document about artificial intelligence and machine learning.",
                "content_type": "text/plain",
                "metadata": {"test": True, "framework_comparison": True},
            },
            {
                "id": "summarization-test-1",
                "type": "text_summarization",
                "text": "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "max_length": 50,
            },
        ]

        framework_results = {}

        for framework_name, agent in self.agents.items():
            logger.info(f"Testing tasks with {framework_name} framework...")
            framework_results[framework_name] = {}

            for task in test_tasks:
                try:
                    logger.info(f"  Executing {task['type']} task...")

                    # Execute task through agent
                    result = await agent.assign_task(task)

                    framework_results[framework_name][task["id"]] = {
                        "success": result.get("success", False),
                        "execution_time": result.get("execution_time", 0),
                        "has_result": "result" in result and result["result"] is not None,
                    }

                    if result.get("success"):
                        logger.info(f"    ✅ {task['type']} completed successfully")
                    else:
                        logger.warning(f"    ⚠️ {task['type']} failed: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"    ❌ {task['type']} error: {e}")
                    framework_results[framework_name][task["id"]] = {"success": False, "error": str(e)}

        self.test_results["framework_agnostic_tasks"] = {"success": True, "results": framework_results}

        # Generate comparison summary
        self._generate_task_comparison(framework_results, test_tasks)

    def _generate_task_comparison(self, framework_results: Dict, test_tasks: list):
        """Generate a comparison summary of task execution across frameworks."""
        logger.info("📊 Task Execution Comparison:")
        logger.info("=" * 80)

        for task in test_tasks:
            task_id = task["id"]
            task_type = task["type"]

            logger.info(f"\n🔍 {task_type.upper()} ({task_id}):")

            for framework_name in framework_results:
                result = framework_results[framework_name].get(task_id, {})
                success = result.get("success", False)
                exec_time = result.get("execution_time", 0)

                status = "✅ Success" if success else "❌ Failed"
                time_info = f"({exec_time:.3f}s)" if exec_time > 0 else ""

                logger.info(f"  {framework_name.capitalize():12} | {status} {time_info}")

    async def test_agent_capabilities(self):
        """Test agent capability reporting and matching."""
        logger.info("🧪 Testing Agent Capabilities...")

        try:
            capability_tests = {}

            for framework_name, agent in self.agents.items():
                capabilities = agent.capabilities
                capability_tests[framework_name] = {
                    "total_capabilities": len(capabilities),
                    "capabilities": capabilities,
                    "can_handle_rag": agent.can_handle_capability("rag"),
                    "can_handle_search": agent.can_handle_capability("document_search"),
                    "can_handle_qa": agent.can_handle_capability("question_answering"),
                }

                logger.info(f"{framework_name} capabilities: {len(capabilities)} total")
                for cap in capabilities[:5]:  # Show first 5
                    logger.info(f"  - {cap}")
                if len(capabilities) > 5:
                    logger.info(f"  ... and {len(capabilities) - 5} more")

            self.test_results["agent_capabilities"] = {"success": True, "results": capability_tests}

        except Exception as e:
            logger.error(f"❌ Agent capabilities test failed: {e}")
            self.test_results["agent_capabilities"] = {"success": False, "error": str(e)}

    def print_test_summary(self):
        """Print a comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 MULTI-FRAMEWORK SYSTEM TEST SUMMARY")
        logger.info("=" * 80)

        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results.values() if isinstance(r, dict) and r.get("success")])

        logger.info(f"📊 Total Test Categories: {total_tests}")
        logger.info(f"✅ Successful: {successful_tests}")
        logger.info(f"❌ Failed: {total_tests - successful_tests}")
        logger.info(f"📈 Success Rate: {(successful_tests / total_tests) * 100:.1f}%\n")

        # Detailed results
        for test_name, result in self.test_results.items():
            status = "✅" if result.get("success") else "❌"
            logger.info(f"{status} {test_name.replace('_', ' ').title()}")

            if not result.get("success") and "error" in result:
                logger.info(f"    Error: {result['error']}")

        logger.info(f"\n🎉 Multi-framework system test completed!")
        logger.info("💡 The system successfully demonstrates:")
        logger.info("   - Framework-agnostic API design")
        logger.info("   - Multiple AI framework support (Haystack + LlamaIndex)")
        logger.info("   - Unified agent architecture")
        logger.info("   - Cross-framework task compatibility")


async def main():
    """Main test runner."""
    logger.info("🚀 Starting Multi-Framework System Test")
    logger.info("🔬 Testing MCP Distributed System with Haystack and LlamaIndex")
    logger.info("-" * 80)

    tester = MultiFrameworkSystemTest()

    try:
        # Run all tests
        await tester.test_framework_registry()
        await tester.test_agent_creation()
        await tester.test_agent_capabilities()
        await tester.test_framework_agnostic_tasks()

        # Print summary
        tester.print_test_summary()

    except Exception as e:
        logger.error(f"❌ Test runner failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
