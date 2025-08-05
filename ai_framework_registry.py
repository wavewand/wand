"""
AI Framework Registry

Provides a unified interface for different AI frameworks like Haystack, LlamaIndex, LangChain, etc.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type

# Import monitoring
try:
    from monitoring.framework_monitor import FrameworkMonitoringMiddleware, framework_monitor

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    framework_monitor = None


class FrameworkType(str, Enum):
    HAYSTACK = "haystack"
    LLAMAINDEX = "llamaindex"
    LANGCHAIN = "langchain"
    CUSTOM = "custom"


class AIFrameworkInterface(ABC):
    """Abstract interface for AI frameworks."""

    @abstractmethod
    async def execute_rag_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a RAG query."""
        pass

    @abstractmethod
    async def search_documents(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search documents."""
        pass

    @abstractmethod
    async def ingest_document(self, filename: str, content: str, **kwargs) -> Dict[str, Any]:
        """Ingest a document."""
        pass

    @abstractmethod
    async def summarize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Summarize text."""
        pass

    @abstractmethod
    async def list_pipelines(self, **kwargs) -> Dict[str, Any]:
        """List available pipelines."""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get framework status."""
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the framework name."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """Return list of supported capabilities."""
        pass


class HaystackFramework(AIFrameworkInterface):
    """Haystack AI framework implementation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._pipeline_manager = None
        self._document_store = None
        self._embedding_manager = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize Haystack components."""
        try:
            from integrations.haystack import HaystackDocumentStore, HaystackEmbeddingManager, HaystackPipelineManager

            self._pipeline_manager = HaystackPipelineManager()
            self._document_store = HaystackDocumentStore()
            self._embedding_manager = HaystackEmbeddingManager()
            self.logger.info("Haystack framework initialized successfully")
        except ImportError:
            self.logger.warning("Haystack not available")

    async def execute_rag_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a RAG query using Haystack."""
        if not self._pipeline_manager:
            return {"success": False, "error": "Haystack not available", "framework": self.framework_name}

        try:
            result = self._pipeline_manager.run_rag_query(
                query=query,
                pipeline_id=kwargs.get("pipeline_id", "default_rag"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 500),
                **kwargs.get("framework_config", {}),
            )
            result["framework"] = self.framework_name
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def search_documents(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search documents using Haystack."""
        if not self._pipeline_manager or not self._embedding_manager:
            return {"success": False, "error": "Haystack not available", "framework": self.framework_name}

        try:
            search_type = kwargs.get("search_type", "semantic")
            max_results = kwargs.get("max_results", 10)

            if search_type == "semantic" and self._embedding_manager:
                documents = self._document_store.list_documents(
                    filters=kwargs.get("filters", {}), limit=max_results * 2
                )

                if documents:
                    results = self._embedding_manager.semantic_search(
                        query=query, documents=documents, top_k=max_results
                    )
                else:
                    results = []

                return {
                    "success": True,
                    "documents": results,
                    "search_type": search_type,
                    "framework": self.framework_name,
                }
            else:
                result = self._pipeline_manager.run_search_query(query=query, max_results=max_results)
                result["framework"] = self.framework_name
                result["search_type"] = search_type
                return result

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def ingest_document(self, filename: str, content: str, **kwargs) -> Dict[str, Any]:
        """Ingest a document using Haystack."""
        if not self._document_store:
            return {"success": False, "error": "Haystack not available", "framework": self.framework_name}

        try:
            content_bytes = content.encode('utf-8')
            result = self._document_store.ingest_document(
                filename=filename,
                content=content_bytes,
                content_type=kwargs.get("content_type", "text/plain"),
                metadata=kwargs.get("metadata", {}),
            )

            if result.get("success") and self._pipeline_manager:
                doc_dict = {"id": result["document_id"], "content": content, "metadata": result.get("metadata", {})}
                self._pipeline_manager.add_documents([doc_dict])

            result["framework"] = self.framework_name
            return result

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def summarize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Summarize text using Haystack."""
        if not self._pipeline_manager:
            return {"success": False, "error": "Haystack not available", "framework": self.framework_name}

        try:
            result = self._pipeline_manager.run_summarization(
                text=text, pipeline_id=kwargs.get("pipeline_id", "default_summarization")
            )
            result["framework"] = self.framework_name
            return result

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def list_pipelines(self, **kwargs) -> Dict[str, Any]:
        """List Haystack pipelines."""
        if not self._pipeline_manager:
            return {"success": False, "error": "Haystack not available", "framework": self.framework_name}

        try:
            pipelines = self._pipeline_manager.list_pipelines()
            return {"success": True, "pipelines": pipelines, "total": len(pipelines), "framework": self.framework_name}
        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def get_status(self) -> Dict[str, Any]:
        """Get Haystack framework status."""
        try:
            return {
                "available": all([self._pipeline_manager, self._document_store, self._embedding_manager]),
                "framework": self.framework_name,
                "components": {
                    "pipeline_manager": self._pipeline_manager.get_stats() if self._pipeline_manager else None,
                    "document_store": self._document_store.get_stats() if self._document_store else None,
                    "embedding_manager": self._embedding_manager.get_embedding_stats()
                    if self._embedding_manager
                    else None,
                },
                "capabilities": self.capabilities,
            }
        except Exception as e:
            return {"available": False, "framework": self.framework_name, "error": str(e)}

    @property
    def framework_name(self) -> str:
        return "haystack"

    @property
    def capabilities(self) -> List[str]:
        return [
            "rag",
            "document_search",
            "question_answering",
            "document_processing",
            "pipeline_management",
            "semantic_search",
            "summarization",
        ]


class LlamaIndexFramework(AIFrameworkInterface):
    """LlamaIndex framework implementation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._index_manager = None
        self._document_processor = None
        self._query_engine = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize LlamaIndex components."""
        try:
            from integrations.llamaindex import LlamaIndexDocumentProcessor, LlamaIndexManager, LlamaIndexQueryEngine

            self._index_manager = LlamaIndexManager()
            self._document_processor = LlamaIndexDocumentProcessor()
            self._query_engine = LlamaIndexQueryEngine(self._index_manager)

            self.logger.info("LlamaIndex framework initialized successfully")
        except ImportError:
            self.logger.warning("LlamaIndex integration not available")

    async def execute_rag_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute a RAG query using LlamaIndex."""
        if not self._query_engine:
            return {"success": False, "error": "LlamaIndex not available", "framework": self.framework_name}

        try:
            result = self._query_engine.execute_rag_query(
                query=query,
                index_name=kwargs.get("pipeline_id", "default"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 500),
                max_results=kwargs.get("max_results", 5),
                **kwargs.get("framework_config", {}),
            )
            result["framework"] = self.framework_name
            return result

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def search_documents(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search documents using LlamaIndex."""
        if not self._query_engine:
            return {"success": False, "error": "LlamaIndex not available", "framework": self.framework_name}

        try:
            result = self._query_engine.search_documents(
                query=query,
                index_name=kwargs.get("index_name", "default"),
                max_results=kwargs.get("max_results", 10),
                **kwargs.get("framework_config", {}),
            )
            result["framework"] = self.framework_name
            return result

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def ingest_document(self, filename: str, content: str, **kwargs) -> Dict[str, Any]:
        """Ingest a document using LlamaIndex."""
        if not self._document_processor or not self._index_manager:
            return {"success": False, "error": "LlamaIndex not available", "framework": self.framework_name}

        try:
            # Process document
            content_bytes = content.encode('utf-8') if isinstance(content, str) else content
            process_result = self._document_processor.process_document(
                filename=filename,
                content=content_bytes,
                content_type=kwargs.get("content_type", "text/plain"),
                metadata=kwargs.get("metadata", {}),
            )

            if not process_result.get("success"):
                process_result["framework"] = self.framework_name
                return process_result

            # Add to index
            documents = process_result["documents"]
            index_name = kwargs.get("index_name", "default")

            add_result = self._index_manager.add_documents(documents, index_name)

            if add_result.get("success"):
                return {
                    "success": True,
                    "document_id": process_result["document_id"],
                    "message": f"Document {filename} ingested successfully",
                    "document_count": len(documents),
                    "index_name": index_name,
                    "framework": self.framework_name,
                }
            else:
                add_result["framework"] = self.framework_name
                return add_result

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def summarize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Summarize text using LlamaIndex."""
        if not self._query_engine:
            return {"success": False, "error": "LlamaIndex not available", "framework": self.framework_name}

        try:
            result = self._query_engine.summarize_text(
                text=text,
                max_length=kwargs.get("max_length"),
                min_length=kwargs.get("min_length"),
                **kwargs.get("framework_config", {}),
            )
            result["framework"] = self.framework_name
            return result

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def list_pipelines(self, **kwargs) -> Dict[str, Any]:
        """List LlamaIndex indexes (equivalent to pipelines)."""
        if not self._index_manager:
            return {"success": False, "error": "LlamaIndex not available", "framework": self.framework_name}

        try:
            result = self._index_manager.list_indexes()

            # Convert to pipeline format
            pipelines = []
            for index in result.get("indexes", []):
                pipelines.append(
                    {
                        "id": index["name"],
                        "name": index["name"],
                        "type": "vector_index",
                        "description": f"LlamaIndex vector index: {index['name']}",
                        "created_at": index["created_at"],
                        "storage_path": index["storage_path"],
                    }
                )

            return {"success": True, "pipelines": pipelines, "total": len(pipelines), "framework": self.framework_name}

        except Exception as e:
            return {"success": False, "error": str(e), "framework": self.framework_name}

    async def get_status(self) -> Dict[str, Any]:
        """Get LlamaIndex framework status."""
        try:
            components_available = all(
                [self._index_manager is not None, self._document_processor is not None, self._query_engine is not None]
            )

            status = {
                "available": components_available,
                "framework": self.framework_name,
                "components": {},
                "capabilities": self.capabilities,
            }

            if self._index_manager:
                status["components"]["index_manager"] = self._index_manager.get_stats()

            if self._document_processor:
                status["components"]["document_processor"] = self._document_processor.get_stats()

            if self._query_engine:
                status["components"]["query_engine"] = self._query_engine.get_stats()

            return status

        except Exception as e:
            return {"available": False, "framework": self.framework_name, "error": str(e)}

    @property
    def framework_name(self) -> str:
        return "llamaindex"

    @property
    def capabilities(self) -> List[str]:
        return [
            "rag",
            "document_search",
            "question_answering",
            "document_processing",
            "indexing",
            "query_engines",
            "summarization",
            "vector_search",
        ]


class AIFrameworkRegistry:
    """Registry for managing AI frameworks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._frameworks: Dict[str, AIFrameworkInterface] = {}
        self._initialize_frameworks()

    def _initialize_frameworks(self):
        """Initialize available frameworks."""
        # Register Haystack
        try:
            self._frameworks["haystack"] = HaystackFramework()
            self.logger.info("Registered Haystack framework")
        except Exception as e:
            self.logger.error(f"Failed to register Haystack framework: {e}")

        # Register LlamaIndex (placeholder)
        try:
            self._frameworks["llamaindex"] = LlamaIndexFramework()
            self.logger.info("Registered LlamaIndex framework (placeholder)")
        except Exception as e:
            self.logger.error(f"Failed to register LlamaIndex framework: {e}")

    def get_framework(self, framework_name: str) -> Optional[AIFrameworkInterface]:
        """Get a framework by name."""
        return self._frameworks.get(framework_name.lower())

    def list_frameworks(self) -> List[str]:
        """List available framework names."""
        return list(self._frameworks.keys())

    def get_framework_capabilities(self, framework_name: str) -> List[str]:
        """Get capabilities of a specific framework."""
        framework = self.get_framework(framework_name)
        return framework.capabilities if framework else []

    async def get_all_framework_status(self) -> Dict[str, Any]:
        """Get status of all registered frameworks."""
        status = {}
        for name, framework in self._frameworks.items():
            try:
                status[name] = await framework.get_status()
            except Exception as e:
                status[name] = {"available": False, "framework": name, "error": str(e)}
        return status

    def register_framework(self, framework: AIFrameworkInterface):
        """Register a new framework."""
        self._frameworks[framework.framework_name] = framework
        self.logger.info(f"Registered custom framework: {framework.framework_name}")


# Global registry instance
ai_framework_registry = AIFrameworkRegistry()
