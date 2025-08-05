#!/usr/bin/env python3
"""
Multi-Framework API Test Script

Tests the framework-agnostic REST API with both Haystack and LlamaIndex frameworks.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiFrameworkAPITester:
    """Tests the multi-framework API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = {}

        # Test data
        self.test_document_content = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create
        intelligent machines that can perform tasks that typically require human intelligence.
        These tasks include visual perception, speech recognition, decision-making, and
        translation between languages.

        Machine Learning is a subset of AI that provides systems the ability to automatically
        learn and improve from experience without being explicitly programmed. Deep Learning
        is a subset of machine learning that uses neural networks with multiple layers.

        Natural Language Processing (NLP) is another important area of AI that focuses on
        the interaction between computers and humans through natural language.
        """

        self.test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "What is the difference between ML and Deep Learning?",
            "How does NLP work?",
        ]

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def test_framework_availability(self):
        """Test which frameworks are available."""
        logger.info("🧪 Testing framework availability...")

        try:
            url = f"{self.base_url}/api/v1/ai/frameworks"
            headers = {"x-api-key": "test-key"}  # Mock API key

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_results["framework_availability"] = {
                        "success": True,
                        "frameworks": data.get("frameworks", []),
                        "total": data.get("total", 0),
                    }

                    logger.info(f"✅ Available frameworks: {len(data.get('frameworks', []))}")
                    for framework in data.get("frameworks", []):
                        logger.info(f"   📦 {framework['name']}: {'✅' if framework['available'] else '❌'}")
                else:
                    self.test_results["framework_availability"] = {"success": False, "error": f"HTTP {response.status}"}
                    logger.error(f"❌ Framework availability check failed: HTTP {response.status}")

        except Exception as e:
            logger.error(f"❌ Framework availability test failed: {e}")
            self.test_results["framework_availability"] = {"success": False, "error": str(e)}

    async def test_document_ingestion(self, framework: str):
        """Test document ingestion for a specific framework."""
        logger.info(f"🧪 Testing document ingestion with {framework}...")

        try:
            url = f"{self.base_url}/api/v1/documents"
            headers = {"x-api-key": "test-key", "Content-Type": "application/json"}

            payload = {
                "filename": f"test_document_{framework}.txt",
                "content": self.test_document_content,
                "framework": framework,
                "content_type": "text/plain",
                "metadata": {
                    "test_case": "multi_framework_api_test",
                    "framework": framework,
                    "created_at": datetime.now().isoformat(),
                },
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 201:
                    data = await response.json()
                    self.test_results[f"document_ingestion_{framework}"] = {
                        "success": True,
                        "document_id": data.get("document_id"),
                        "framework": framework,
                    }
                    logger.info(f"✅ Document ingested successfully with {framework}")
                    logger.info(f"   📄 Document ID: {data.get('document_id')}")
                else:
                    error_data = await response.text()
                    self.test_results[f"document_ingestion_{framework}"] = {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_data}",
                    }
                    logger.error(f"❌ Document ingestion failed with {framework}: HTTP {response.status}")

        except Exception as e:
            logger.error(f"❌ Document ingestion test failed for {framework}: {e}")
            self.test_results[f"document_ingestion_{framework}"] = {"success": False, "error": str(e)}

    async def test_rag_queries(self, framework: str):
        """Test RAG queries for a specific framework."""
        logger.info(f"🧪 Testing RAG queries with {framework}...")

        rag_results = []

        for i, query in enumerate(self.test_queries):
            try:
                url = f"{self.base_url}/api/v1/rag"
                headers = {"x-api-key": "test-key", "Content-Type": "application/json"}

                payload = {
                    "query": query,
                    "framework": framework,
                    "pipeline_id": "default",
                    "temperature": 0.7,
                    "max_tokens": 200,
                    "framework_config": {"test_mode": True},
                }

                async with self.session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()

                        result = {
                            "query": query,
                            "success": data.get("success", False),
                            "answer": data.get("answer", "")[:200] + "..."
                            if len(data.get("answer", "")) > 200
                            else data.get("answer", ""),
                            "sources_count": len(data.get("sources", [])),
                            "execution_time_ms": data.get("execution_time_ms", 0),
                        }

                        rag_results.append(result)

                        if data.get("success"):
                            logger.info(f"✅ RAG Query {i + 1}/{len(self.test_queries)} succeeded with {framework}")
                            logger.info(f"   ❓ Query: {query}")
                            logger.info(f"   💬 Answer: {result['answer']}")
                            logger.info(
                                f"   📊 Sources: {result['sources_count']}, Time: {result['execution_time_ms']}ms"
                            )
                        else:
                            logger.warning(f"⚠️ RAG Query {i + 1} failed: {data.get('error', 'Unknown error')}")
                    else:
                        error_data = await response.text()
                        rag_results.append(
                            {"query": query, "success": False, "error": f"HTTP {response.status}: {error_data}"}
                        )
                        logger.error(f"❌ RAG Query {i + 1} failed with HTTP {response.status}")

            except Exception as e:
                logger.error(f"❌ RAG Query {i + 1} failed for {framework}: {e}")
                rag_results.append({"query": query, "success": False, "error": str(e)})

        self.test_results[f"rag_queries_{framework}"] = {
            "total_queries": len(self.test_queries),
            "successful_queries": len([r for r in rag_results if r.get("success")]),
            "results": rag_results,
        }

    async def test_document_search(self, framework: str):
        """Test document search for a specific framework."""
        logger.info(f"🧪 Testing document search with {framework}...")

        try:
            url = f"{self.base_url}/api/v1/search"
            headers = {"x-api-key": "test-key", "Content-Type": "application/json"}

            payload = {
                "query": "machine learning artificial intelligence",
                "framework": framework,
                "search_type": "semantic",
                "max_results": 5,
                "framework_config": {"test_mode": True},
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()

                    self.test_results[f"document_search_{framework}"] = {
                        "success": data.get("success", False),
                        "documents_found": len(data.get("documents", [])),
                        "search_type": data.get("search_type"),
                        "framework": framework,
                    }

                    if data.get("success"):
                        logger.info(f"✅ Document search succeeded with {framework}")
                        logger.info(f"   🔍 Found {len(data.get('documents', []))} documents")
                        logger.info(f"   🎯 Search type: {data.get('search_type')}")
                    else:
                        logger.warning(f"⚠️ Document search failed: {data.get('error', 'Unknown error')}")
                else:
                    error_data = await response.text()
                    self.test_results[f"document_search_{framework}"] = {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_data}",
                    }
                    logger.error(f"❌ Document search failed with {framework}: HTTP {response.status}")

        except Exception as e:
            logger.error(f"❌ Document search test failed for {framework}: {e}")
            self.test_results[f"document_search_{framework}"] = {"success": False, "error": str(e)}

    async def test_text_summarization(self, framework: str):
        """Test text summarization for a specific framework."""
        logger.info(f"🧪 Testing text summarization with {framework}...")

        try:
            url = f"{self.base_url}/api/v1/summarize"
            headers = {"x-api-key": "test-key", "Content-Type": "application/json"}

            payload = {
                "text": self.test_document_content,
                "framework": framework,
                "max_length": 100,
                "framework_config": {"test_mode": True},
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()

                    self.test_results[f"summarization_{framework}"] = {
                        "success": data.get("success", False),
                        "original_length": data.get("original_length", 0),
                        "summary_length": len(data.get("summary", "")),
                        "compression_ratio": data.get("compression_ratio", 0),
                        "framework": framework,
                    }

                    if data.get("success"):
                        logger.info(f"✅ Text summarization succeeded with {framework}")
                        logger.info(f"   📏 Original: {data.get('original_length')} chars")
                        logger.info(f"   📄 Summary: {len(data.get('summary', ''))} chars")
                        logger.info(f"   📊 Compression: {data.get('compression_ratio', 0):.2%}")
                        logger.info(f"   💬 Summary: {data.get('summary', '')[:150]}...")
                    else:
                        logger.warning(f"⚠️ Text summarization failed: {data.get('error', 'Unknown error')}")
                else:
                    error_data = await response.text()
                    self.test_results[f"summarization_{framework}"] = {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_data}",
                    }
                    logger.error(f"❌ Text summarization failed with {framework}: HTTP {response.status}")

        except Exception as e:
            logger.error(f"❌ Text summarization test failed for {framework}: {e}")
            self.test_results[f"summarization_{framework}"] = {"success": False, "error": str(e)}

    async def test_pipeline_listing(self, framework: str):
        """Test pipeline listing for a specific framework."""
        logger.info(f"🧪 Testing pipeline listing with {framework}...")

        try:
            url = f"{self.base_url}/api/v1/pipelines"
            headers = {"x-api-key": "test-key"}

            params = {"framework": framework}

            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    self.test_results[f"pipeline_listing_{framework}"] = {
                        "success": data.get("success", False),
                        "pipelines_count": len(data.get("pipelines", [])),
                        "framework": framework,
                    }

                    if data.get("success"):
                        logger.info(f"✅ Pipeline listing succeeded with {framework}")
                        logger.info(f"   🔧 Found {len(data.get('pipelines', []))} pipelines")
                        for pipeline in data.get("pipelines", [])[:3]:  # Show first 3
                            logger.info(
                                f"      • {pipeline.get('name', 'Unknown')} ({pipeline.get('type', 'Unknown')})"
                            )
                    else:
                        logger.warning(f"⚠️ Pipeline listing failed: {data.get('error', 'Unknown error')}")
                else:
                    error_data = await response.text()
                    self.test_results[f"pipeline_listing_{framework}"] = {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_data}",
                    }
                    logger.error(f"❌ Pipeline listing failed with {framework}: HTTP {response.status}")

        except Exception as e:
            logger.error(f"❌ Pipeline listing test failed for {framework}: {e}")
            self.test_results[f"pipeline_listing_{framework}"] = {"success": False, "error": str(e)}

    async def run_comprehensive_test(self):
        """Run comprehensive tests for all available frameworks."""
        logger.info("🚀 Starting comprehensive multi-framework API tests...")

        # Test framework availability first
        await self.test_framework_availability()

        # Get available frameworks
        frameworks_to_test = []
        if self.test_results.get("framework_availability", {}).get("success"):
            available_frameworks = self.test_results["framework_availability"]["frameworks"]
            frameworks_to_test = [f["name"] for f in available_frameworks if f.get("available")]

        if not frameworks_to_test:
            logger.warning("⚠️ No frameworks available for testing")
            frameworks_to_test = ["haystack", "llamaindex"]  # Test anyway

        logger.info(f"🎯 Testing frameworks: {', '.join(frameworks_to_test)}")

        # Test each framework
        for framework in frameworks_to_test:
            logger.info(f"\n📋 Testing {framework.upper()} Framework:")
            logger.info("=" * 50)

            # Run all tests for this framework
            await self.test_document_ingestion(framework)
            await asyncio.sleep(1)  # Small delay between tests

            await self.test_rag_queries(framework)
            await asyncio.sleep(1)

            await self.test_document_search(framework)
            await asyncio.sleep(1)

            await self.test_text_summarization(framework)
            await asyncio.sleep(1)

            await self.test_pipeline_listing(framework)

            logger.info(f"✅ Completed testing {framework} framework\n")

    def print_test_summary(self):
        """Print a comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 MULTI-FRAMEWORK API TEST SUMMARY")
        logger.info("=" * 80)

        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results.values() if isinstance(r, dict) and r.get("success")])

        logger.info(f"📊 Total Tests: {total_tests}")
        logger.info(f"✅ Successful: {successful_tests}")
        logger.info(f"❌ Failed: {total_tests - successful_tests}")
        logger.info(f"📈 Success Rate: {(successful_tests / total_tests) * 100:.1f}%\n")

        # Detailed results by category
        categories = {
            "Framework Availability": ["framework_availability"],
            "Document Ingestion": [k for k in self.test_results.keys() if k.startswith("document_ingestion")],
            "RAG Queries": [k for k in self.test_results.keys() if k.startswith("rag_queries")],
            "Document Search": [k for k in self.test_results.keys() if k.startswith("document_search")],
            "Text Summarization": [k for k in self.test_results.keys() if k.startswith("summarization")],
            "Pipeline Listing": [k for k in self.test_results.keys() if k.startswith("pipeline_listing")],
        }

        for category, test_keys in categories.items():
            if test_keys:
                logger.info(f"📋 {category}:")
                for key in test_keys:
                    result = self.test_results.get(key, {})
                    status = "✅" if result.get("success") else "❌"
                    framework = key.split("_")[-1] if "_" in key else "general"
                    logger.info(
                        f"   {status} {framework}: {result.get('error', 'Success') if not result.get('success') else 'Success'}"
                    )
                logger.info("")

        # Save results to file
        with open("multi_framework_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        logger.info("💾 Detailed results saved to 'multi_framework_test_results.json'")


async def main():
    """Main test runner function."""

    # Check if API server is running
    api_url = os.environ.get("API_URL", "http://localhost:8000")

    logger.info(f"🌐 Testing API at: {api_url}")
    logger.info("🔑 Using mock API key for testing")
    logger.info("📝 Make sure the API server is running with both frameworks available")
    logger.info("🚨 Note: Tests will use mock data and may fail if OpenAI API key is not configured\n")

    try:
        async with MultiFrameworkAPITester(api_url) as tester:
            await tester.run_comprehensive_test()
            tester.print_test_summary()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test runner failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
