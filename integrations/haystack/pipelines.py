"""
Haystack Pipeline Management for MCP System

This module manages Haystack AI pipelines for various AI operations including
RAG, question answering, document search, and summarization.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from haystack import Document, Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    HAYSTACK_AVAILABLE = True
except ImportError:
    logging.warning("Haystack not available. Install with: pip install haystack-ai")
    HAYSTACK_AVAILABLE = False
    # Create mock classes for development

    class Pipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return {"answer": "Haystack not available"}

    class Document:
        def __init__(self, *args, **kwargs):
            pass

    class InMemoryDocumentStore:
        def __init__(self, *args, **kwargs):
            pass


class HaystackPipelineManager:
    """Manages Haystack AI pipelines for the MCP system."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pipelines: Dict[str, Pipeline] = {}
        self.pipeline_configs: Dict[str, Dict[str, Any]] = {}
        self.document_store = None

        if HAYSTACK_AVAILABLE:
            self._initialize_document_store()
            self._create_default_pipelines()
        else:
            self.logger.warning("Haystack not available - running in mock mode")

    def _initialize_document_store(self):
        """Initialize the in-memory document store."""
        try:
            self.document_store = InMemoryDocumentStore()
            self.logger.info("Initialized Haystack document store")
        except Exception as e:
            self.logger.error(f"Failed to initialize document store: {e}")
            self.document_store = None

    def _create_default_pipelines(self):
        """Create default pipelines for common use cases."""
        try:
            # RAG Pipeline
            self._create_rag_pipeline()

            # QA Pipeline
            self._create_qa_pipeline()

            # Search Pipeline
            self._create_search_pipeline()

            # Summarization Pipeline
            self._create_summarization_pipeline()

            self.logger.info(f"Created {len(self.pipelines)} default pipelines")

        except Exception as e:
            self.logger.error(f"Failed to create default pipelines: {e}")

    def _create_rag_pipeline(self):
        """Create a Retrieval-Augmented Generation pipeline."""
        if not HAYSTACK_AVAILABLE or not self.document_store:
            return

        try:
            # Create RAG pipeline components
            retriever = InMemoryBM25Retriever(document_store=self.document_store)
            prompt_builder = PromptBuilder(
                template="""
                Context: {context}

                Question: {question}

                Please provide a comprehensive answer based on the context provided above.
                If the context doesn't contain relevant information, say so clearly.

                Answer:
                """
            )
            generator = OpenAIGenerator(model="gpt-3.5-turbo")

            # Build pipeline
            pipeline = Pipeline()
            pipeline.add_component("retriever", retriever)
            pipeline.add_component("prompt_builder", prompt_builder)
            pipeline.add_component("generator", generator)

            # Connect components
            pipeline.connect("retriever", "prompt_builder.context")
            pipeline.connect("prompt_builder", "generator")

            pipeline_id = "default_rag"
            self.pipelines[pipeline_id] = pipeline
            self.pipeline_configs[pipeline_id] = {
                "name": "Default RAG Pipeline",
                "type": "rag",
                "description": "Basic retrieval-augmented generation",
                "components": ["retriever", "prompt_builder", "generator"],
                "created_at": datetime.now().isoformat(),
            }

            self.logger.info("Created RAG pipeline")

        except Exception as e:
            self.logger.error(f"Failed to create RAG pipeline: {e}")

    def _create_qa_pipeline(self):
        """Create a Question Answering pipeline."""
        if not HAYSTACK_AVAILABLE or not self.document_store:
            return

        try:
            # Create QA pipeline (similar to RAG but optimized for Q&A)
            retriever = InMemoryBM25Retriever(document_store=self.document_store)
            prompt_builder = PromptBuilder(
                template="""
                Please answer the following question based on the provided context.
                If you cannot answer based on the context, say "I cannot answer this question based on the provided information."

                Context: {context}
                Question: {question}

                Answer:
                """
            )
            generator = OpenAIGenerator(model="gpt-3.5-turbo", generation_kwargs={"temperature": 0.1})

            pipeline = Pipeline()
            pipeline.add_component("retriever", retriever)
            pipeline.add_component("prompt_builder", prompt_builder)
            pipeline.add_component("generator", generator)

            pipeline.connect("retriever", "prompt_builder.context")
            pipeline.connect("prompt_builder", "generator")

            pipeline_id = "default_qa"
            self.pipelines[pipeline_id] = pipeline
            self.pipeline_configs[pipeline_id] = {
                "name": "Default QA Pipeline",
                "type": "qa",
                "description": "Question answering over documents",
                "components": ["retriever", "prompt_builder", "generator"],
                "created_at": datetime.now().isoformat(),
            }

            self.logger.info("Created QA pipeline")

        except Exception as e:
            self.logger.error(f"Failed to create QA pipeline: {e}")

    def _create_search_pipeline(self):
        """Create a document search pipeline."""
        if not HAYSTACK_AVAILABLE or not self.document_store:
            return

        try:
            retriever = InMemoryBM25Retriever(document_store=self.document_store)

            pipeline = Pipeline()
            pipeline.add_component("retriever", retriever)

            pipeline_id = "default_search"
            self.pipelines[pipeline_id] = pipeline
            self.pipeline_configs[pipeline_id] = {
                "name": "Default Search Pipeline",
                "type": "search",
                "description": "Document search and retrieval",
                "components": ["retriever"],
                "created_at": datetime.now().isoformat(),
            }

            self.logger.info("Created search pipeline")

        except Exception as e:
            self.logger.error(f"Failed to create search pipeline: {e}")

    def _create_summarization_pipeline(self):
        """Create a document summarization pipeline."""
        if not HAYSTACK_AVAILABLE:
            return

        try:
            prompt_builder = PromptBuilder(
                template="""
                Please provide a concise summary of the following text:

                Text: {text}

                Summary:
                """
            )
            generator = OpenAIGenerator(model="gpt-3.5-turbo", generation_kwargs={"temperature": 0.3})

            pipeline = Pipeline()
            pipeline.add_component("prompt_builder", prompt_builder)
            pipeline.add_component("generator", generator)

            pipeline.connect("prompt_builder", "generator")

            pipeline_id = "default_summarization"
            self.pipelines[pipeline_id] = pipeline
            self.pipeline_configs[pipeline_id] = {
                "name": "Default Summarization Pipeline",
                "type": "summarization",
                "description": "Document and text summarization",
                "components": ["prompt_builder", "generator"],
                "created_at": datetime.now().isoformat(),
            }

            self.logger.info("Created summarization pipeline")

        except Exception as e:
            self.logger.error(f"Failed to create summarization pipeline: {e}")

    def run_rag_query(
        self, query: str, pipeline_id: str = "default_rag", max_results: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """Execute a RAG query using the specified pipeline."""
        if not HAYSTACK_AVAILABLE:
            return {
                "success": False,
                "message": "Haystack not available",
                "answer": "Please install haystack-ai to use RAG functionality",
            }

        try:
            pipeline = self.pipelines.get(pipeline_id)
            if not pipeline:
                return {"success": False, "message": f"Pipeline {pipeline_id} not found"}

            # Run the pipeline
            result = pipeline.run(
                {"retriever": {"query": query, "top_k": max_results}, "prompt_builder": {"question": query}}
            )

            # Extract answer and sources
            answer = result.get("generator", {}).get("replies", ["No answer generated"])[0]
            sources = result.get("retriever", {}).get("documents", [])

            return {
                "success": True,
                "answer": answer,
                "sources": [self._document_to_dict(doc) for doc in sources],
                "pipeline_used": pipeline_id,
                "execution_time_ms": 0,  # Timing metrics can be added with performance monitoring
            }

        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return {"success": False, "message": f"RAG query failed: {str(e)}"}

    def run_search_query(
        self, query: str, pipeline_id: str = "default_search", max_results: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """Execute a search query using the specified pipeline."""
        if not HAYSTACK_AVAILABLE:
            return {"success": False, "message": "Haystack not available"}

        try:
            pipeline = self.pipelines.get(pipeline_id)
            if not pipeline:
                return {"success": False, "message": f"Pipeline {pipeline_id} not found"}

            result = pipeline.run({"retriever": {"query": query, "top_k": max_results}})

            documents = result.get("retriever", {}).get("documents", [])

            return {
                "success": True,
                "documents": [self._document_to_dict(doc) for doc in documents],
                "total_results": len(documents),
                "pipeline_used": pipeline_id,
            }

        except Exception as e:
            self.logger.error(f"Search query failed: {e}")
            return {"success": False, "message": f"Search query failed: {str(e)}"}

    def run_summarization(self, text: str, pipeline_id: str = "default_summarization", **kwargs) -> Dict[str, Any]:
        """Execute text summarization using the specified pipeline."""
        if not HAYSTACK_AVAILABLE:
            return {"success": False, "message": "Haystack not available"}

        try:
            pipeline = self.pipelines.get(pipeline_id)
            if not pipeline:
                return {"success": False, "message": f"Pipeline {pipeline_id} not found"}

            result = pipeline.run({"prompt_builder": {"text": text}})

            summary = result.get("generator", {}).get("replies", ["No summary generated"])[0]

            return {
                "success": True,
                "summary": summary,
                "pipeline_used": pipeline_id,
                "original_length": len(text),
                "summary_length": len(summary),
            }

        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return {"success": False, "message": f"Summarization failed: {str(e)}"}

    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to the document store."""
        if not HAYSTACK_AVAILABLE or not self.document_store:
            return {"success": False, "message": "Document store not available"}

        try:
            haystack_docs = []
            for doc in documents:
                haystack_doc = Document(
                    content=doc.get("content", ""), meta=doc.get("metadata", {}), id=doc.get("id", str(uuid.uuid4()))
                )
                haystack_docs.append(haystack_doc)

            self.document_store.write_documents(haystack_docs)

            return {
                "success": True,
                "message": f"Added {len(haystack_docs)} documents to store",
                "document_count": len(haystack_docs),
            }

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return {"success": False, "message": f"Failed to add documents: {str(e)}"}

    def get_pipeline_info(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific pipeline."""
        return self.pipeline_configs.get(pipeline_id)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all available pipelines."""
        return list(self.pipeline_configs.values())

    def _document_to_dict(self, document) -> Dict[str, Any]:
        """Convert a Haystack document to a dictionary."""
        if not hasattr(document, 'content'):
            return {"content": str(document)}

        return {
            "id": getattr(document, 'id', str(uuid.uuid4())),
            "content": document.content,
            "metadata": getattr(document, 'meta', {}),
            "score": getattr(document, 'score', 0.0),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline manager statistics."""
        return {
            "total_pipelines": len(self.pipelines),
            "pipeline_types": list(set(config["type"] for config in self.pipeline_configs.values())),
            "document_store_available": self.document_store is not None,
            "haystack_available": HAYSTACK_AVAILABLE,
        }
