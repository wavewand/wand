"""
LlamaIndex Query Engine

This module provides query execution capabilities for RAG, search, and QA operations.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.core.response.schema import Response
    from llama_index.core.retrievers import BaseRetriever

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logging.warning("LlamaIndex not available")
    LLAMAINDEX_AVAILABLE = False

    class BaseQueryEngine:
        def query(self, *args, **kwargs):
            pass

    class Response:
        def __init__(self, *args, **kwargs):
            pass

        response = ""
        source_nodes = []


class LlamaIndexQueryEngine:
    """Manages query operations for LlamaIndex."""

    def __init__(self, index_manager):
        self.logger = logging.getLogger(__name__)
        self.index_manager = index_manager
        self.query_history: List[Dict[str, Any]] = []

    def execute_rag_query(self, query: str, index_name: str = "default", **kwargs) -> Dict[str, Any]:
        """Execute a RAG query using LlamaIndex."""
        if not LLAMAINDEX_AVAILABLE:
            return {"success": False, "error": "LlamaIndex not available"}

        start_time = time.time()

        try:
            # Get query engine
            query_engine = self.index_manager.get_query_engine(
                index_name=index_name,
                similarity_top_k=kwargs.get("max_results", 5),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 500),
            )

            if not query_engine:
                return {"success": False, "error": f"Could not create query engine for index '{index_name}'"}

            # Execute query
            response = query_engine.query(query)

            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    source_info = {
                        "content": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                        "score": getattr(node, 'score', 0.0),
                        "metadata": getattr(node, 'metadata', {}),
                    }

                    # Add document reference if available
                    if hasattr(node, 'node_id'):
                        source_info["node_id"] = node.node_id

                    sources.append(source_info)

            execution_time = int((time.time() - start_time) * 1000)

            # Log query
            query_record = {
                "query": query,
                "index_name": index_name,
                "response": str(response),
                "sources_count": len(sources),
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat(),
                "parameters": kwargs,
            }
            self.query_history.append(query_record)

            # Keep only last 100 queries
            if len(self.query_history) > 100:
                self.query_history = self.query_history[-100:]

            self.logger.info(f"Executed RAG query on index '{index_name}' in {execution_time}ms")

            return {
                "success": True,
                "answer": str(response),
                "sources": sources,
                "index_name": index_name,
                "execution_time_ms": execution_time,
                "source_count": len(sources),
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"RAG query failed: {e}")

            return {"success": False, "error": str(e), "execution_time_ms": execution_time}

    def search_documents(self, query: str, index_name: str = "default", **kwargs) -> Dict[str, Any]:
        """Search documents using LlamaIndex retriever."""
        if not LLAMAINDEX_AVAILABLE:
            return {"success": False, "error": "LlamaIndex not available"}

        start_time = time.time()

        try:
            # Get index
            if index_name not in self.index_manager.indexes:
                return {"success": False, "error": f"Index '{index_name}' not found"}

            index = self.index_manager.indexes[index_name]

            # Create retriever
            retriever = index.as_retriever(similarity_top_k=kwargs.get("max_results", 10))

            # Retrieve documents
            nodes = retriever.retrieve(query)

            # Format results
            documents = []
            for node in nodes:
                doc_info = {
                    "id": getattr(node, 'node_id', str(len(documents))),
                    "content": node.text,
                    "score": getattr(node, 'score', 0.0),
                    "metadata": getattr(node, 'metadata', {}),
                }
                documents.append(doc_info)

            execution_time = int((time.time() - start_time) * 1000)

            self.logger.info(f"Document search on index '{index_name}' returned {len(documents)} results")

            return {
                "success": True,
                "documents": documents,
                "total_results": len(documents),
                "index_name": index_name,
                "execution_time_ms": execution_time,
                "search_type": "vector_similarity",
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Document search failed: {e}")

            return {"success": False, "error": str(e), "execution_time_ms": execution_time}

    def summarize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Summarize text using LlamaIndex."""
        if not LLAMAINDEX_AVAILABLE:
            return {"success": False, "error": "LlamaIndex not available"}

        start_time = time.time()

        try:
            from llama_index.core import Document, SummaryIndex

            # Create a temporary document
            document = Document(text=text)

            # Create summary index
            summary_index = SummaryIndex.from_documents([document])

            # Create query engine for summarization
            query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize", llm=self.index_manager._get_llm_with_params(**kwargs)
            )

            # Generate summary
            summary_prompt = kwargs.get("summary_prompt", "Please provide a concise summary of this text.")
            response = query_engine.query(summary_prompt)

            execution_time = int((time.time() - start_time) * 1000)

            summary_text = str(response)

            self.logger.info(f"Text summarization completed in {execution_time}ms")

            return {
                "success": True,
                "summary": summary_text,
                "original_length": len(text),
                "summary_length": len(summary_text),
                "compression_ratio": len(summary_text) / len(text) if text else 0,
                "execution_time_ms": execution_time,
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Text summarization failed: {e}")

            return {"success": False, "error": str(e), "execution_time_ms": execution_time}

    def question_answering(self, question: str, context: str = None, **kwargs) -> Dict[str, Any]:
        """Answer questions using LlamaIndex."""
        if context:
            # Use provided context
            return self._qa_with_context(question, context, **kwargs)
        else:
            # Use default index
            return self.execute_rag_query(question, **kwargs)

    def _qa_with_context(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """Answer question with provided context."""
        if not LLAMAINDEX_AVAILABLE:
            return {"success": False, "error": "LlamaIndex not available"}

        start_time = time.time()

        try:
            from llama_index.core import Document, VectorStoreIndex

            # Create temporary document from context
            document = Document(text=context)

            # Create temporary index
            temp_index = VectorStoreIndex.from_documents([document])

            # Create query engine
            query_engine = temp_index.as_query_engine(similarity_top_k=1, response_mode="compact")

            # Execute query
            response = query_engine.query(question)

            execution_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "answer": str(response),
                "question": question,
                "context_length": len(context),
                "execution_time_ms": execution_time,
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"QA with context failed: {e}")

            return {"success": False, "error": str(e), "execution_time_ms": execution_time}

    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent query history."""
        return self.query_history[-limit:]

    def clear_query_history(self) -> Dict[str, Any]:
        """Clear query history."""
        count = len(self.query_history)
        self.query_history.clear()

        return {"success": True, "message": f"Cleared {count} queries from history"}

    def get_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        if not self.query_history:
            return {"total_queries": 0, "average_execution_time_ms": 0, "llamaindex_available": LLAMAINDEX_AVAILABLE}

        # Calculate statistics
        total_queries = len(self.query_history)
        total_time = sum(q.get("execution_time_ms", 0) for q in self.query_history)
        avg_time = total_time / total_queries if total_queries > 0 else 0

        # Get recent query stats
        recent_queries = self.query_history[-10:] if len(self.query_history) >= 10 else self.query_history
        recent_avg_time = (
            sum(q.get("execution_time_ms", 0) for q in recent_queries) / len(recent_queries) if recent_queries else 0
        )

        return {
            "total_queries": total_queries,
            "average_execution_time_ms": avg_time,
            "recent_average_execution_time_ms": recent_avg_time,
            "total_execution_time_ms": total_time,
            "llamaindex_available": LLAMAINDEX_AVAILABLE,
        }
