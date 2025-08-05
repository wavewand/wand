"""
LlamaIndex Index Management

This module manages LlamaIndex indexes and provides vector storage capabilities.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from llama_index.core import (
        Document,
        Settings,
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.core.storage.index_store import SimpleIndexStore
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logging.warning("LlamaIndex not available. Install with: pip install llama-index")
    LLAMAINDEX_AVAILABLE = False
    # Mock classes for development

    class VectorStoreIndex:
        def __init__(self, *args, **kwargs):
            pass

        def as_query_engine(self, *args, **kwargs):
            pass

        def insert(self, *args, **kwargs):
            pass

    class Document:
        def __init__(self, *args, **kwargs):
            pass

    class Settings:
        llm = None
        embed_model = None

    class StorageContext:
        @classmethod
        def from_defaults(cls, *args, **kwargs):
            pass


class LlamaIndexManager:
    """Manages LlamaIndex indexes and vector stores."""

    def __init__(self, persist_dir: str = "./storage/llamaindex"):
        self.logger = logging.getLogger(__name__)
        self.persist_dir = persist_dir
        self.indexes: Dict[str, VectorStoreIndex] = {}
        self.documents: Dict[str, Document] = {}

        # Ensure storage directory exists
        os.makedirs(persist_dir, exist_ok=True)

        if LLAMAINDEX_AVAILABLE:
            self._initialize_settings()
            self._load_existing_indexes()
        else:
            self.logger.warning("LlamaIndex not available - running in mock mode")

    def _initialize_settings(self):
        """Initialize LlamaIndex global settings."""
        try:
            # Configure LLM
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1)

            # Configure embeddings
            Settings.embed_model = (
                OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-ada-002") if openai_api_key else None
            )

            self.logger.info("Initialized LlamaIndex settings")

        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex settings: {e}")

    def _load_existing_indexes(self):
        """Load existing indexes from storage."""
        try:
            # Check if storage exists
            storage_path = os.path.join(self.persist_dir, "default")
            if os.path.exists(storage_path):
                storage_context = StorageContext.from_defaults(persist_dir=storage_path)
                index = load_index_from_storage(storage_context)
                self.indexes["default"] = index
                self.logger.info("Loaded existing default index")
        except Exception as e:
            self.logger.info(f"No existing indexes found or failed to load: {e}")

    def create_index(self, index_name: str = "default", documents: List[Document] = None) -> Dict[str, Any]:
        """Create a new vector index."""
        if not LLAMAINDEX_AVAILABLE:
            return {"success": False, "error": "LlamaIndex not available"}

        try:
            if documents is None:
                documents = []

            # Create storage context
            storage_path = os.path.join(self.persist_dir, index_name)
            os.makedirs(storage_path, exist_ok=True)

            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
                vector_store=SimpleVectorStore(),
                persist_dir=storage_path,
            )

            # Create index
            if documents:
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            else:
                index = VectorStoreIndex([], storage_context=storage_context)

            # Store index
            self.indexes[index_name] = index

            # Persist to disk
            index.storage_context.persist(persist_dir=storage_path)

            self.logger.info(f"Created index '{index_name}' with {len(documents)} documents")

            return {
                "success": True,
                "index_name": index_name,
                "document_count": len(documents),
                "storage_path": storage_path,
            }

        except Exception as e:
            self.logger.error(f"Failed to create index '{index_name}': {e}")
            return {"success": False, "error": str(e)}

    def add_documents(self, documents: List[Document], index_name: str = "default") -> Dict[str, Any]:
        """Add documents to an existing index."""
        if not LLAMAINDEX_AVAILABLE:
            return {"success": False, "error": "LlamaIndex not available"}

        try:
            # Get or create index
            if index_name not in self.indexes:
                create_result = self.create_index(index_name)
                if not create_result["success"]:
                    return create_result

            index = self.indexes[index_name]

            # Add documents to index
            for doc in documents:
                # Store document reference
                doc_id = str(uuid.uuid4())
                self.documents[doc_id] = doc

                # Insert into index
                index.insert(doc)

            # Persist changes
            storage_path = os.path.join(self.persist_dir, index_name)
            index.storage_context.persist(persist_dir=storage_path)

            self.logger.info(f"Added {len(documents)} documents to index '{index_name}'")

            return {
                "success": True,
                "index_name": index_name,
                "documents_added": len(documents),
                "total_documents": len(self.documents),
            }

        except Exception as e:
            self.logger.error(f"Failed to add documents to index '{index_name}': {e}")
            return {"success": False, "error": str(e)}

    def get_query_engine(self, index_name: str = "default", **kwargs):
        """Get a query engine for the specified index."""
        if not LLAMAINDEX_AVAILABLE:
            return None

        try:
            if index_name not in self.indexes:
                self.logger.warning(f"Index '{index_name}' not found")
                return None

            index = self.indexes[index_name]

            # Configure query engine parameters
            query_params = {
                "similarity_top_k": kwargs.get("similarity_top_k", 5),
                "response_mode": kwargs.get("response_mode", "compact"),
            }

            # Add temperature/generation params if provided
            if "temperature" in kwargs:
                query_params["llm"] = OpenAI(
                    temperature=kwargs["temperature"], max_tokens=kwargs.get("max_tokens", 500)
                )

            query_engine = index.as_query_engine(**query_params)
            return query_engine

        except Exception as e:
            self.logger.error(f"Failed to get query engine for index '{index_name}': {e}")
            return None

    def list_indexes(self) -> Dict[str, Any]:
        """List all available indexes."""
        try:
            indexes_info = []

            for index_name, index in self.indexes.items():
                # Get index stats
                storage_path = os.path.join(self.persist_dir, index_name)
                storage_exists = os.path.exists(storage_path)

                indexes_info.append(
                    {
                        "name": index_name,
                        "storage_path": storage_path,
                        "storage_exists": storage_exists,
                        "created_at": datetime.now().isoformat(),  # Creation time tracked at runtime
                    }
                )

            return {"success": True, "indexes": indexes_info, "total_indexes": len(indexes_info)}

        except Exception as e:
            self.logger.error(f"Failed to list indexes: {e}")
            return {"success": False, "error": str(e)}

    def delete_index(self, index_name: str) -> Dict[str, Any]:
        """Delete an index and its storage."""
        try:
            if index_name not in self.indexes:
                return {"success": False, "error": f"Index '{index_name}' not found"}

            # Remove from memory
            del self.indexes[index_name]

            # Remove storage directory
            storage_path = os.path.join(self.persist_dir, index_name)
            if os.path.exists(storage_path):
                import shutil

                shutil.rmtree(storage_path)

            self.logger.info(f"Deleted index '{index_name}'")

            return {"success": True, "index_name": index_name, "message": f"Index '{index_name}' deleted successfully"}

        except Exception as e:
            self.logger.error(f"Failed to delete index '{index_name}': {e}")
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get index manager statistics."""
        return {
            "total_indexes": len(self.indexes),
            "total_documents": len(self.documents),
            "available_indexes": list(self.indexes.keys()),
            "llamaindex_available": LLAMAINDEX_AVAILABLE,
            "persist_directory": self.persist_dir,
            "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        }

    def _get_llm_with_params(self, **kwargs):
        """Get LLM with specific parameters."""
        try:
            from llama_index.llms.openai import OpenAI

            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                return None

            return OpenAI(
                api_key=openai_api_key,
                model=kwargs.get("model", "gpt-3.5-turbo"),
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 500),
            )
        except Exception:
            return None
