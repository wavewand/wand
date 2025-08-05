#!/usr/bin/env python3
"""
Framework Comparison Example

This example demonstrates how to use the same API endpoints with different
AI frameworks (Haystack and LlamaIndex) and compare their responses.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameworkComparison:
    """Compare different AI frameworks using the same API."""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.headers = {"x-api-key": "test-key", "Content-Type": "application/json"}

    async def ingest_sample_document(self, session: aiohttp.ClientSession, framework: str) -> str:
        """Ingest a sample document and return document ID."""

        sample_content = """
        Python is a high-level, interpreted programming language with dynamic semantics.
        Its high-level built-in data structures, combined with dynamic typing and dynamic
        binding, make it very attractive for Rapid Application Development, as well as for
        use as a scripting or glue language to connect existing components together.

        Python's simple, easy to learn syntax emphasizes readability and therefore reduces
        the cost of program maintenance. Python supports modules and packages, which
        encourages program modularity and code reuse.

        Machine Learning in Python is powered by libraries like scikit-learn, TensorFlow,
        PyTorch, and pandas. These libraries provide powerful tools for data analysis,
        statistical modeling, and artificial intelligence applications.
        """

        payload = {
            "filename": f"python_guide_{framework}.txt",
            "content": sample_content,
            "framework": framework,
            "content_type": "text/plain",
            "metadata": {"topic": "programming", "language": "python", "framework_test": framework},
        }

        async with session.post(
            f"{self.api_base_url}/api/v1/documents", headers=self.headers, json=payload
        ) as response:
            if response.status == 201:
                data = await response.json()
                return data.get("document_id", "")
            else:
                logger.error(f"Failed to ingest document for {framework}: {response.status}")
                return ""

    async def compare_rag_responses(
        self, session: aiohttp.ClientSession, query: str, frameworks: List[str]
    ) -> Dict[str, Any]:
        """Compare RAG responses from different frameworks."""

        results = {}

        for framework in frameworks:
            payload = {
                "query": query,
                "framework": framework,
                "temperature": 0.7,
                "max_tokens": 300,
                "framework_config": {"comparison_test": True},
            }

            try:
                async with session.post(
                    f"{self.api_base_url}/api/v1/rag", headers=self.headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[framework] = {
                            "success": data.get("success", False),
                            "answer": data.get("answer", ""),
                            "sources_count": len(data.get("sources", [])),
                            "execution_time_ms": data.get("execution_time_ms", 0),
                        }
                    else:
                        results[framework] = {"success": False, "error": f"HTTP {response.status}"}
            except Exception as e:
                results[framework] = {"success": False, "error": str(e)}

        return results

    async def compare_search_results(
        self, session: aiohttp.ClientSession, query: str, frameworks: List[str]
    ) -> Dict[str, Any]:
        """Compare search results from different frameworks."""

        results = {}

        for framework in frameworks:
            payload = {
                "query": query,
                "framework": framework,
                "search_type": "semantic",
                "max_results": 5,
                "framework_config": {"comparison_test": True},
            }

            try:
                async with session.post(
                    f"{self.api_base_url}/api/v1/search", headers=self.headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[framework] = {
                            "success": data.get("success", False),
                            "documents_found": len(data.get("documents", [])),
                            "search_type": data.get("search_type", ""),
                            "framework": data.get("framework", framework),
                        }
                    else:
                        results[framework] = {"success": False, "error": f"HTTP {response.status}"}
            except Exception as e:
                results[framework] = {"success": False, "error": str(e)}

        return results

    async def compare_summarization(
        self, session: aiohttp.ClientSession, text: str, frameworks: List[str]
    ) -> Dict[str, Any]:
        """Compare summarization results from different frameworks."""

        results = {}

        for framework in frameworks:
            payload = {
                "text": text,
                "framework": framework,
                "max_length": 150,
                "framework_config": {"comparison_test": True},
            }

            try:
                async with session.post(
                    f"{self.api_base_url}/api/v1/summarize", headers=self.headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results[framework] = {
                            "success": data.get("success", False),
                            "summary": data.get("summary", ""),
                            "original_length": data.get("original_length", 0),
                            "summary_length": data.get("summary_length", 0),
                            "compression_ratio": data.get("compression_ratio", 0),
                        }
                    else:
                        results[framework] = {"success": False, "error": f"HTTP {response.status}"}
            except Exception as e:
                results[framework] = {"success": False, "error": str(e)}

        return results

    def print_comparison_results(self, operation: str, query: str, results: Dict[str, Any]):
        """Print formatted comparison results."""

        print(f"\n{'=' * 80}")
        print(f"🔍 {operation.upper()} COMPARISON")
        print(f"{'=' * 80}")
        print(f"Query/Input: {query}")
        print("-" * 80)

        for framework, result in results.items():
            print(f"\n🤖 {framework.upper()} Framework:")
            print("-" * 40)

            if result.get("success"):
                if operation == "RAG":
                    print(f"✅ Success")
                    print(f"💬 Answer: {result.get('answer', '')[:200]}...")
                    print(f"📊 Sources: {result.get('sources_count', 0)}")
                    print(f"⏱️ Time: {result.get('execution_time_ms', 0)}ms")

                elif operation == "Search":
                    print(f"✅ Success")
                    print(f"🔍 Documents found: {result.get('documents_found', 0)}")
                    print(f"🎯 Search type: {result.get('search_type', 'N/A')}")

                elif operation == "Summarization":
                    print(f"✅ Success")
                    print(f"📄 Summary: {result.get('summary', '')[:150]}...")
                    print(f"📏 Original: {result.get('original_length', 0)} chars")
                    print(f"📝 Summary: {result.get('summary_length', 0)} chars")
                    print(f"📊 Compression: {result.get('compression_ratio', 0):.2%}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    async def run_comparison(self):
        """Run comprehensive framework comparison."""

        print("🚀 Starting AI Framework Comparison")
        print("🌐 Testing Haystack vs LlamaIndex using the same API")
        print("-" * 80)

        frameworks = ["haystack", "llamaindex"]

        async with aiohttp.ClientSession() as session:
            # 1. Check framework availability
            print("📦 Checking framework availability...")
            try:
                async with session.get(
                    f"{self.api_base_url}/api/v1/ai/frameworks", headers={"x-api-key": "test-key"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        available_frameworks = [f["name"] for f in data.get("frameworks", []) if f.get("available")]
                        print(f"✅ Available frameworks: {', '.join(available_frameworks)}")

                        # Filter to only test available frameworks
                        frameworks = [f for f in frameworks if f in available_frameworks]
                    else:
                        print(f"⚠️ Could not check framework availability: HTTP {response.status}")
            except Exception as e:
                print(f"⚠️ Framework availability check failed: {e}")

            if not frameworks:
                print("❌ No frameworks available for comparison")
                return

            # 2. Ingest sample documents
            print(f"\n📄 Ingesting sample documents for: {', '.join(frameworks)}")
            document_ids = {}
            for framework in frameworks:
                doc_id = await self.ingest_sample_document(session, framework)
                if doc_id:
                    document_ids[framework] = doc_id
                    print(f"✅ {framework}: Document ingested")
                else:
                    print(f"❌ {framework}: Document ingestion failed")

            # Wait a moment for indexing
            print("⏳ Waiting for document indexing...")
            await asyncio.sleep(3)

            # 3. Compare RAG responses
            test_questions = [
                "What is Python programming language?",
                "How is Python used in machine learning?",
                "What are the benefits of Python?",
            ]

            for question in test_questions:
                rag_results = await self.compare_rag_responses(session, question, frameworks)
                self.print_comparison_results("RAG", question, rag_results)

            # 4. Compare search results
            search_query = "Python machine learning libraries"
            search_results = await self.compare_search_results(session, search_query, frameworks)
            self.print_comparison_results("Search", search_query, search_results)

            # 5. Compare summarization
            long_text = """
            Python is a versatile programming language that has gained immense popularity
            in recent years, particularly in the fields of data science, artificial intelligence,
            and web development. Its syntax is clean and readable, making it an excellent choice
            for beginners while still being powerful enough for advanced applications.

            In the realm of machine learning, Python offers numerous libraries and frameworks
            such as scikit-learn for traditional machine learning algorithms, TensorFlow and
            PyTorch for deep learning, pandas for data manipulation, and NumPy for numerical
            computations. These tools have made Python the de facto standard for data science
            and AI research.

            The language's interpreted nature allows for rapid prototyping and interactive
            development, which is particularly valuable in research environments where
            experimentation and iteration are crucial. Additionally, Python's extensive
            standard library and vast ecosystem of third-party packages mean that developers
            can find solutions for almost any problem they encounter.
            """

            summarization_results = await self.compare_summarization(session, long_text, frameworks)
            self.print_comparison_results("Summarization", "Long text about Python", summarization_results)

            print(f"\n🎯 Framework comparison completed!")
            print("💾 Results show how the same API can work with different AI frameworks")


async def main():
    """Main function to run the framework comparison."""

    try:
        comparison = FrameworkComparison()
        await comparison.run_comparison()
    except KeyboardInterrupt:
        print("\n⏹️ Comparison interrupted by user")
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
