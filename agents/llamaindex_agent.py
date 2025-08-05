"""
LlamaIndex Agent for MCP Distributed System

This agent specializes in AI-powered document processing, RAG operations,
semantic search, and question answering using the LlamaIndex AI framework.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai_framework_registry import ai_framework_registry
from distributed.agent import BaseAgent
from distributed.types import AgentType

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LlamaIndexAgent(BaseAgent):
    """Agent specialized in LlamaIndex AI operations."""

    def __init__(self, agent_id: str = None, coordinator_address: str = "localhost:50051"):
        super().__init__(
            agent_id=agent_id or "llamaindex-agent",
            agent_type=AgentType.LLAMAINDEX,
            coordinator_address=coordinator_address,
            capabilities=[
                "rag",
                "document_search",
                "question_answering",
                "document_processing",
                "index_management",
                "semantic_search",
                "summarization",
                "query_engines",
            ],
        )

        # Initialize framework through registry
        self.framework = ai_framework_registry.get_framework("llamaindex")
        if not self.framework:
            self.logger.error("LlamaIndex framework not available in registry")
            raise RuntimeError("LlamaIndex framework not available")

        # Task type mappings
        self.task_handlers = {
            "rag_query": self._handle_rag_query,
            "document_search": self._handle_document_search,
            "document_ingestion": self._handle_document_ingestion,
            "question_answering": self._handle_question_answering,
            "text_summarization": self._handle_text_summarization,
            "semantic_search": self._handle_semantic_search,
            "index_management": self._handle_index_management,
            "document_analysis": self._handle_document_analysis,
            "llamaindex_operation": self._handle_llamaindex_operation,
        }

        self.logger.info("LlamaIndex agent initialized successfully")

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming tasks based on their type."""
        task_type = task.get("type", "").lower()

        # Handle various task types
        handler = self.task_handlers.get(task_type, self._handle_generic_task)

        try:
            result = await handler(task)

            # Log successful task completion
            self.logger.info(f"Successfully completed {task_type} task: {task.get('id', 'unknown')}")

            return {
                "success": True,
                "result": result,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to handle {task_type} task: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

    async def _handle_rag_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG (Retrieval-Augmented Generation) queries."""
        query = task.get("query", "")
        index_name = task.get("index_name", "default")
        temperature = task.get("temperature", 0.7)
        max_tokens = task.get("max_tokens", 500)
        context = task.get("context", {})

        if not query:
            raise ValueError("Query is required for RAG operations")

        result = await self.framework.execute_rag_query(
            query=query, index_name=index_name, temperature=temperature, max_tokens=max_tokens, context=context
        )

        return {
            "operation": "rag_query",
            "query": query,
            "index_name": index_name,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "success": result.get("success", False),
            "execution_time_ms": result.get("execution_time_ms", 0),
            "framework": "llamaindex",
        }

    async def _handle_document_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document search operations."""
        query = task.get("query", "")
        search_type = task.get("search_type", "semantic")
        max_results = task.get("max_results", 10)
        filters = task.get("filters", {})

        if not query:
            raise ValueError("Query is required for document search")

        result = await self.framework.search_documents(
            query=query, search_type=search_type, max_results=max_results, filters=filters
        )

        results = result.get("documents", [])

        return {
            "operation": "document_search",
            "query": query,
            "search_type": search_type,
            "documents": results,
            "total_results": len(results),
            "framework": "llamaindex",
        }

    async def _handle_document_ingestion(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document ingestion operations."""
        filename = task.get("filename", "")
        content = task.get("content", "")
        content_type = task.get("content_type", "text/plain")
        metadata = task.get("metadata", {})
        index_name = task.get("index_name", "default")

        if not filename or not content:
            raise ValueError("Filename and content are required for document ingestion")

        result = await self.framework.ingest_document(
            filename=filename, content=content, content_type=content_type, metadata=metadata, index_name=index_name
        )

        return {
            "operation": "document_ingestion",
            "filename": filename,
            "index_name": index_name,
            "document_id": result.get("document_id"),
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "framework": "llamaindex",
        }

    async def _handle_question_answering(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle question answering operations."""
        question = task.get("question", "")
        index_name = task.get("index_name", "default")
        max_results = task.get("max_results", 5)

        if not question:
            raise ValueError("Question is required for QA operations")

        # Use RAG query for question answering
        result = await self.framework.execute_rag_query(query=question, index_name=index_name, max_results=max_results)

        return {
            "operation": "question_answering",
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "framework": "llamaindex",
        }

    async def _handle_text_summarization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text summarization operations."""
        text = task.get("text", "")
        index_name = task.get("index_name", "default")
        max_length = task.get("max_length", 200)

        if not text:
            raise ValueError("Text is required for summarization")

        result = await self.framework.summarize_text(text=text, index_name=index_name, max_length=max_length)

        return {
            "operation": "text_summarization",
            "original_length": len(text),
            "summary": result.get("summary", ""),
            "summary_length": len(result.get("summary", "")),
            "compression_ratio": len(result.get("summary", "")) / len(text) if text else 0,
            "framework": "llamaindex",
        }

    async def _handle_semantic_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search operations."""
        query = task.get("query", "")
        top_k = task.get("top_k", 10)
        threshold = task.get("threshold", 0.0)
        filters = task.get("filters", {})
        index_name = task.get("index_name", "default")

        if not query:
            raise ValueError("Query is required for semantic search")

        result = await self.framework.search_documents(
            query=query, search_type="semantic", max_results=top_k, filters=filters, index_name=index_name
        )

        results = result.get("documents", [])

        # Filter by threshold if specified
        if threshold > 0:
            results = [r for r in results if r.get("similarity_score", 0) >= threshold]

        return {
            "operation": "semantic_search",
            "query": query,
            "results": results,
            "total_results": len(results),
            "framework": "llamaindex",
        }

    async def _handle_index_management(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle index management operations."""
        operation = task.get("operation", "list")
        index_name = task.get("index_name", "")

        if operation == "list":
            result = await self.framework.list_pipelines()
            return {
                "operation": "list_indexes",
                "indexes": result.get("pipelines", []),
                "total": len(result.get("pipelines", [])),
                "framework": "llamaindex",
            }
        elif operation == "info" and index_name:
            result = await self.framework.get_pipeline_info(index_name)
            return {"operation": "index_info", "index_name": index_name, "info": result, "framework": "llamaindex"}
        elif operation == "stats":
            result = await self.framework.get_status()
            return {"operation": "index_stats", "stats": result.get("stats", {}), "framework": "llamaindex"}
        elif operation == "create" and index_name:
            # Create new index
            result = await self.framework.create_index(index_name)
            return {
                "operation": "create_index",
                "index_name": index_name,
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "framework": "llamaindex",
            }
        else:
            raise ValueError(f"Unknown index operation: {operation}")

    async def _handle_document_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document analysis operations."""
        document_id = task.get("document_id", "")
        analysis_type = task.get("analysis_type", "summary")
        index_name = task.get("index_name", "default")

        if not document_id:
            raise ValueError("Document ID is required for analysis")

        # Get document info first
        document_result = await self.framework.get_document(document_id, index_name)
        if not document_result.get("success"):
            raise ValueError(f"Document {document_id} not found")

        document = document_result.get("document", {})
        content = document.get("content", "")

        if analysis_type == "summary":
            result = await self.framework.summarize_text(text=content, index_name=index_name)
            return {
                "operation": "document_analysis",
                "document_id": document_id,
                "analysis_type": "summary",
                "result": result.get("summary", ""),
                "original_length": len(content),
                "framework": "llamaindex",
            }
        elif analysis_type == "keywords":
            # Use search to find similar content for keyword extraction
            search_result = await self.framework.search_documents(
                query=content[:500],  # Use first 500 chars as query
                search_type="semantic",
                max_results=5,
                index_name=index_name,
            )
            return {
                "operation": "document_analysis",
                "document_id": document_id,
                "analysis_type": "keywords",
                "result": search_result.get("documents", []),
                "framework": "llamaindex",
            }
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    async def _handle_llamaindex_operation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic LlamaIndex operations."""
        operation = task.get("operation", "")
        parameters = task.get("parameters", {})

        if operation == "status":
            status_result = await self.framework.get_status()
            return {"operation": "status", "framework": "llamaindex", "status": status_result}
        elif operation == "health_check":
            status_result = await self.framework.get_status()
            return {
                "operation": "health_check",
                "status": "healthy" if status_result.get("available") else "unavailable",
                "framework": "llamaindex",
                "details": status_result,
            }
        else:
            raise ValueError(f"Unknown LlamaIndex operation: {operation}")

    async def _handle_generic_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic tasks that don't have specific handlers."""
        task_type = task.get("type", "unknown")

        # Try to determine the best handler based on task content
        if "query" in task and "rag" in task_type.lower():
            return await self._handle_rag_query(task)
        elif "search" in task_type.lower():
            return await self._handle_document_search(task)
        elif "question" in task or "answer" in task_type.lower():
            return await self._handle_question_answering(task)
        elif "summarize" in task_type.lower() or "summary" in task_type.lower():
            return await self._handle_text_summarization(task)
        else:
            return {
                "operation": "generic_task",
                "message": f"Processed generic task of type: {task_type}",
                "task_data": task,
                "framework": "llamaindex",
            }

    def get_agent_status(self) -> Dict[str, Any]:
        """Get the current status of the LlamaIndex agent."""
        base_status = super().get_agent_status()

        # Add LlamaIndex-specific status information
        llamaindex_status = {
            "framework": "llamaindex",
            "framework_available": self.framework is not None,
            "supported_operations": list(self.task_handlers.keys()),
        }

        base_status.update(llamaindex_status)
        return base_status


async def main():
    """Main entry point for the LlamaIndex agent."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Get configuration from environment
    agent_id = os.environ.get('AGENT_ID', f'llamaindex-agent-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    coordinator_address = os.environ.get('COORDINATOR_ADDRESS', 'localhost:50051')

    # Create and start the LlamaIndex agent
    agent = LlamaIndexAgent(agent_id=agent_id, coordinator_address=coordinator_address)

    try:
        await agent.start()

        # Keep the agent running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logging.info("Shutting down LlamaIndex agent...")
    except Exception as e:
        logging.error(f"Agent error: {e}")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
