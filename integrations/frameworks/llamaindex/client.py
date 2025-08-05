"""
LlamaIndex Client

High-level client for interacting with LlamaIndex functionality,
including document indexing, querying, and index management.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from integrations.frameworks.base import BaseClient
from observability.logging import get_logger

from .models import (
    IndexType,
    LlamaIndexConfig,
    LlamaIndexDocument,
    LlamaIndexIndex,
    LlamaIndexMode,
    LlamaIndexQuery,
    LlamaIndexResponse,
)


class LlamaIndexClient(BaseClient):
    """High-level LlamaIndex client."""

    def __init__(self, config: LlamaIndexConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)

        # LlamaIndex components (initialized lazily)
        self._service_context = None
        self._storage_context = None
        self._indexes: Dict[str, Any] = {}

        # Initialize LlamaIndex
        self._initialize_llamaindex()

    def _initialize_llamaindex(self):
        """Initialize LlamaIndex components."""
        try:
            # Set OpenAI API key if available
            if hasattr(self.config, 'openai_api_key') and self.config.openai_api_key:
                os.environ['OPENAI_API_KEY'] = self.config.openai_api_key

            self.logger.info("LlamaIndex client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex: {e}")
            raise

    def _get_service_context(self):
        """Get or create service context."""
        if self._service_context is None:
            try:
                from llama_index.core import ServiceContext
                from llama_index.embeddings.openai import OpenAIEmbedding
                from llama_index.llms.openai import OpenAI

                # Configure LLM
                llm = OpenAI(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )

                # Configure embedding model
                embed_model = OpenAIEmbedding(
                    model=self.config.embed_model, embed_batch_size=self.config.embed_batch_size
                )

                # Create service context
                self._service_context = ServiceContext.from_defaults(
                    llm=llm,
                    embed_model=embed_model,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    **self.config.service_context_config,
                )

            except ImportError as e:
                self.logger.error(f"LlamaIndex import error: {e}")
                raise ImportError("LlamaIndex is not installed. Please install with: pip install llama-index")
            except Exception as e:
                self.logger.error(f"Failed to create service context: {e}")
                raise

        return self._service_context

    def _get_storage_context(self, persist_dir: Optional[str] = None):
        """Get or create storage context."""
        storage_dir = persist_dir or self.config.persist_dir

        try:
            from llama_index.core import StorageContext

            if storage_dir and Path(storage_dir).exists():
                # Load existing storage context
                return StorageContext.from_defaults(persist_dir=storage_dir)
            else:
                # Create new storage context
                storage_context = StorageContext.from_defaults(**self.config.storage_context_config)

                if storage_dir:
                    Path(storage_dir).mkdir(parents=True, exist_ok=True)

                return storage_context

        except Exception as e:
            self.logger.error(f"Failed to create storage context: {e}")
            raise

    async def create_index(
        self,
        documents: List[LlamaIndexDocument],
        index_id: str,
        index_type: IndexType = IndexType.VECTOR_STORE,
        persist_dir: Optional[str] = None,
    ) -> LlamaIndexIndex:
        """Create a new index from documents."""
        try:
            self.logger.info(f"Creating LlamaIndex index: {index_id}")

            # Convert documents to LlamaIndex format
            llamaindex_docs = [doc.to_llamaindex_document() for doc in documents]

            # Get contexts
            service_context = self._get_service_context()
            storage_context = self._get_storage_context(persist_dir)

            # Create index based on type
            if index_type == IndexType.VECTOR_STORE:
                from llama_index.core import VectorStoreIndex

                index = VectorStoreIndex.from_documents(
                    documents=llamaindex_docs, service_context=service_context, storage_context=storage_context
                )

            elif index_type == IndexType.LIST:
                from llama_index.core import ListIndex

                index = ListIndex.from_documents(
                    documents=llamaindex_docs, service_context=service_context, storage_context=storage_context
                )

            elif index_type == IndexType.TREE:
                from llama_index.core import TreeIndex

                index = TreeIndex.from_documents(
                    documents=llamaindex_docs, service_context=service_context, storage_context=storage_context
                )

            elif index_type == IndexType.KEYWORD_TABLE:
                from llama_index.core import KeywordTableIndex

                index = KeywordTableIndex.from_documents(
                    documents=llamaindex_docs, service_context=service_context, storage_context=storage_context
                )

            else:
                raise ValueError(f"Unsupported index type: {index_type}")

            # Persist index if directory specified
            if persist_dir:
                index.storage_context.persist(persist_dir=persist_dir)

            # Store index reference
            self._indexes[index_id] = index

            # Create index metadata
            index_metadata = LlamaIndexIndex(
                index_id=index_id,
                index_type=index_type,
                num_documents=len(documents),
                persist_dir=persist_dir,
                config=self.config.to_dict(),
            )

            self.logger.info(f"Created LlamaIndex index {index_id} with {len(documents)} documents")
            return index_metadata

        except Exception as e:
            self.logger.error(f"Failed to create index {index_id}: {e}")
            raise

    async def load_index(
        self, index_id: str, persist_dir: str, index_type: IndexType = IndexType.VECTOR_STORE
    ) -> LlamaIndexIndex:
        """Load an existing index from storage."""
        try:
            self.logger.info(f"Loading LlamaIndex index: {index_id}")

            if not Path(persist_dir).exists():
                raise FileNotFoundError(f"Index directory not found: {persist_dir}")

            # Get contexts
            service_context = self._get_service_context()
            storage_context = self._get_storage_context(persist_dir)

            # Load index based on type
            if index_type == IndexType.VECTOR_STORE:
                from llama_index.core import load_index_from_storage

                index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
            else:
                # For other types, use generic loader
                from llama_index.core import load_index_from_storage

                index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

            # Store index reference
            self._indexes[index_id] = index

            # Create index metadata
            index_metadata = LlamaIndexIndex(index_id=index_id, index_type=index_type, persist_dir=persist_dir)

            self.logger.info(f"Loaded LlamaIndex index: {index_id}")
            return index_metadata

        except Exception as e:
            self.logger.error(f"Failed to load index {index_id}: {e}")
            raise

    async def query(self, query: LlamaIndexQuery, index_id: str) -> LlamaIndexResponse:
        """Execute a query against an index."""
        try:
            self.logger.info(f"Executing LlamaIndex query on index {index_id}")

            # Get index
            if index_id not in self._indexes:
                raise ValueError(f"Index not found: {index_id}")

            index = self._indexes[index_id]

            # Create query engine
            query_engine = index.as_query_engine(
                similarity_top_k=query.similarity_top_k or self.config.similarity_top_k,
                response_mode=query.response_mode or "compact",
            )

            # Execute query
            if query.streaming:
                # Handle streaming query
                response = query_engine.query(query.query_str)
                # For now, convert streaming to regular response
                # In future, could implement proper streaming
                response_obj = LlamaIndexResponse.from_llamaindex_response(response, query)
            else:
                # Regular query
                response = query_engine.query(query.query_str)
                response_obj = LlamaIndexResponse.from_llamaindex_response(response, query)

            self.logger.info(f"Query executed successfully on index {index_id}")
            return response_obj

        except Exception as e:
            self.logger.error(f"Query failed on index {index_id}: {e}")
            return LlamaIndexResponse(success=False, error=str(e), framework="llamaindex")

    async def chat(
        self, query: LlamaIndexQuery, index_id: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> LlamaIndexResponse:
        """Execute a chat query against an index."""
        try:
            self.logger.info(f"Executing LlamaIndex chat query on index {index_id}")

            # Get index
            if index_id not in self._indexes:
                raise ValueError(f"Index not found: {index_id}")

            index = self._indexes[index_id]

            # Create chat engine
            chat_engine = index.as_chat_engine(
                similarity_top_k=query.similarity_top_k or self.config.similarity_top_k,
                chat_mode="context",  # or "condense_question", "react"
            )

            # Set chat history if provided
            if chat_history or query.chat_history:
                history = chat_history or query.chat_history
                for msg in history:
                    if msg.get('role') == 'user':
                        chat_engine.chat_history.append(('user', msg.get('content', '')))
                    elif msg.get('role') == 'assistant':
                        chat_engine.chat_history.append(('assistant', msg.get('content', '')))

            # Execute chat query
            response = chat_engine.chat(query.query_str)
            response_obj = LlamaIndexResponse.from_llamaindex_response(response, query)
            response_obj.is_chat_response = True

            self.logger.info(f"Chat query executed successfully on index {index_id}")
            return response_obj

        except Exception as e:
            self.logger.error(f"Chat query failed on index {index_id}: {e}")
            return LlamaIndexResponse(success=False, error=str(e), framework="llamaindex", is_chat_response=True)

    async def add_documents(self, documents: List[LlamaIndexDocument], index_id: str) -> bool:
        """Add documents to an existing index."""
        try:
            self.logger.info(f"Adding {len(documents)} documents to index {index_id}")

            # Get index
            if index_id not in self._indexes:
                raise ValueError(f"Index not found: {index_id}")

            index = self._indexes[index_id]

            # Convert documents to LlamaIndex format
            llamaindex_docs = [doc.to_llamaindex_document() for doc in documents]

            # Add documents to index
            for doc in llamaindex_docs:
                index.insert(doc)

            self.logger.info(f"Added {len(documents)} documents to index {index_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents to index {index_id}: {e}")
            return False

    async def delete_documents(self, doc_ids: List[str], index_id: str) -> bool:
        """Delete documents from an index."""
        try:
            self.logger.info(f"Deleting {len(doc_ids)} documents from index {index_id}")

            # Get index
            if index_id not in self._indexes:
                raise ValueError(f"Index not found: {index_id}")

            index = self._indexes[index_id]

            # Delete documents
            for doc_id in doc_ids:
                index.delete_ref_doc(doc_id, delete_from_docstore=True)

            self.logger.info(f"Deleted {len(doc_ids)} documents from index {index_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete documents from index {index_id}: {e}")
            return False

    async def get_index_stats(self, index_id: str) -> Dict[str, Any]:
        """Get statistics about an index."""
        try:
            if index_id not in self._indexes:
                raise ValueError(f"Index not found: {index_id}")

            index = self._indexes[index_id]

            # Get basic stats
            stats = {
                'index_id': index_id,
                'index_type': type(index).__name__,
                'num_nodes': len(index.docstore.docs) if hasattr(index, 'docstore') else 0,
                'storage_context_type': type(index.storage_context).__name__,
            }

            # Add vector store stats if applicable
            if hasattr(index, 'vector_store'):
                stats['vector_store_type'] = type(index.vector_store).__name__

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get stats for index {index_id}: {e}")
            return {}

    async def list_indexes(self) -> List[str]:
        """List all loaded indexes."""
        return list(self._indexes.keys())

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test LlamaIndex imports
            from llama_index.core import ServiceContext

            # Test service context creation
            service_context = self._get_service_context()

            return {
                'status': 'healthy',
                'framework': 'llamaindex',
                'service_context': str(type(service_context)),
                'loaded_indexes': len(self._indexes),
                'config': {
                    'llm_model': self.config.llm_model,
                    'embed_model': self.config.embed_model,
                    'index_type': self.config.index_type.value,
                },
            }

        except Exception as e:
            return {'status': 'unhealthy', 'framework': 'llamaindex', 'error': str(e)}
