"""
LlamaIndex Data Models

Defines data structures and models for LlamaIndex operations,
including queries, responses, documents, and configurations.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseDocument, BaseQuery, BaseResponse


class LlamaIndexMode(str, Enum):
    """LlamaIndex query modes."""

    DEFAULT = "default"
    EMBEDDING = "embedding"
    CHAT = "chat"
    SUMMARIZE = "summarize"
    TREE_SUMMARIZE = "tree_summarize"
    SIMPLE_SUMMARIZE = "simple_summarize"
    GENERATION = "generation"
    REFINE = "refine"
    COMPACT = "compact"


class IndexType(str, Enum):
    """Types of LlamaIndex indexes."""

    VECTOR_STORE = "vector_store"
    LIST = "list"
    TREE = "tree"
    KEYWORD_TABLE = "keyword_table"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SUMMARY = "summary"
    DOCUMENT_SUMMARY = "document_summary"
    PANDAS = "pandas"
    SQL = "sql"
    EMPTY = "empty"


@dataclass
class LlamaIndexConfig(BaseConfig):
    """LlamaIndex specific configuration."""

    # LLM Configuration
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.1
    llm_max_tokens: Optional[int] = None

    # Embedding Configuration
    embed_model: str = "text-embedding-ada-002"
    embed_batch_size: int = 10

    # Index Configuration
    index_type: IndexType = IndexType.VECTOR_STORE
    chunk_size: int = 1024
    chunk_overlap: int = 20

    # Vector Store Configuration
    vector_store_type: str = "simple"  # simple, chroma, pinecone, etc.
    vector_store_config: Dict[str, Any] = field(default_factory=dict)

    # Query Configuration
    similarity_top_k: int = 5
    max_iterations: int = 10

    # Storage Configuration
    persist_dir: Optional[str] = "./storage/llamaindex"

    # Advanced Configuration
    service_context_config: Dict[str, Any] = field(default_factory=dict)
    storage_context_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.vector_store_config:
            self.vector_store_config = {}
        if not self.service_context_config:
            self.service_context_config = {}
        if not self.storage_context_config:
            self.storage_context_config = {}


@dataclass
class LlamaIndexDocument(BaseDocument):
    """LlamaIndex document representation."""

    doc_id: Optional[str] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    excluded_embed_metadata_keys: List[str] = field(default_factory=list)
    excluded_llm_metadata_keys: List[str] = field(default_factory=list)
    relationships: Dict[str, Any] = field(default_factory=dict)
    hash: Optional[str] = None

    def to_llamaindex_document(self):
        """Convert to LlamaIndex Document format."""
        try:
            from llama_index.core import Document

            return Document(
                text=self.text or self.content,
                doc_id=self.doc_id,
                metadata=self.metadata,
                excluded_embed_metadata_keys=self.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=self.excluded_llm_metadata_keys,
                relationships=self.relationships,
                hash=self.hash,
            )
        except ImportError:
            raise ImportError("LlamaIndex is not installed. Please install with: pip install llama-index")

    @classmethod
    def from_llamaindex_document(cls, doc):
        """Create from LlamaIndex Document."""
        return cls(
            doc_id=doc.doc_id,
            text=doc.text,
            content=doc.text,
            metadata=doc.metadata or {},
            excluded_embed_metadata_keys=doc.excluded_embed_metadata_keys or [],
            excluded_llm_metadata_keys=doc.excluded_llm_metadata_keys or [],
            relationships=doc.relationships or {},
            hash=doc.hash,
        )


@dataclass
class LlamaIndexQuery(BaseQuery):
    """LlamaIndex query representation."""

    query_str: str
    mode: LlamaIndexMode = LlamaIndexMode.DEFAULT
    similarity_top_k: Optional[int] = None
    streaming: bool = False

    # Chat-specific parameters
    chat_history: Optional[List[Dict[str, str]]] = None

    # Advanced query parameters
    required_keywords: Optional[List[str]] = None
    exclude_keywords: Optional[List[str]] = None

    # Retrieval parameters
    node_ids: Optional[List[str]] = None
    doc_ids: Optional[List[str]] = None

    # Response synthesis parameters
    response_mode: Optional[str] = None
    text_qa_template: Optional[str] = None
    refine_template: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.chat_history is None:
            self.chat_history = []

        # Set query_text for base class compatibility
        if not hasattr(self, 'query_text') or not self.query_text:
            self.query_text = self.query_str


@dataclass
class LlamaIndexResponse(BaseResponse):
    """LlamaIndex response representation."""

    response: Optional[str] = None
    source_nodes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Chat-specific fields
    is_chat_response: bool = False

    # Additional response information
    formatted_sources: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        # Set content for base class compatibility
        if not self.content and self.response:
            self.content = self.response

    def get_formatted_sources(self) -> str:
        """Get formatted source information."""
        if self.formatted_sources:
            return self.formatted_sources

        if not self.source_nodes:
            return "No sources available."

        formatted = []
        for i, node in enumerate(self.source_nodes, 1):
            score = node.get('score', 'N/A')
            text = node.get('text', '')[:200] + "..." if len(node.get('text', '')) > 200 else node.get('text', '')
            metadata = node.get('metadata', {})

            source_info = f"Source {i} (Score: {score}):\n"
            source_info += f"Text: {text}\n"

            if metadata:
                source_info += f"Metadata: {json.dumps(metadata, indent=2)}\n"

            formatted.append(source_info)

        self.formatted_sources = "\n".join(formatted)
        return self.formatted_sources

    @classmethod
    def from_llamaindex_response(cls, response, query: LlamaIndexQuery):
        """Create from LlamaIndex response object."""
        source_nodes = []

        # Extract source nodes if available
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                node_data = {
                    'text': getattr(node, 'text', ''),
                    'score': getattr(node, 'score', None),
                    'metadata': getattr(node, 'metadata', {}),
                    'node_id': getattr(node, 'node_id', None),
                    'doc_id': getattr(node, 'ref_doc_id', None),
                }
                source_nodes.append(node_data)

        # Extract metadata
        metadata = {}
        if hasattr(response, 'metadata') and response.metadata:
            metadata = response.metadata

        return cls(
            response=str(response) if response else None,
            content=str(response) if response else None,
            source_nodes=source_nodes,
            metadata=metadata,
            is_chat_response=query.mode == LlamaIndexMode.CHAT,
            success=True,
            framework="llamaindex",
        )


@dataclass
class LlamaIndexIndex:
    """LlamaIndex index representation."""

    index_id: str
    index_type: IndexType
    name: Optional[str] = None
    description: Optional[str] = None

    # Index statistics
    num_documents: int = 0
    num_nodes: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Storage information
    persist_dir: Optional[str] = None
    storage_size_bytes: Optional[int] = None

    # Configuration used to create the index
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.config is None:
            self.config = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'index_id': self.index_id,
            'index_type': self.index_type.value,
            'name': self.name,
            'description': self.description,
            'num_documents': self.num_documents,
            'num_nodes': self.num_nodes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'persist_dir': self.persist_dir,
            'storage_size_bytes': self.storage_size_bytes,
            'config': self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LlamaIndexIndex':
        """Create from dictionary representation."""
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])

        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])

        return cls(
            index_id=data['index_id'],
            index_type=IndexType(data['index_type']),
            name=data.get('name'),
            description=data.get('description'),
            num_documents=data.get('num_documents', 0),
            num_nodes=data.get('num_nodes', 0),
            created_at=created_at,
            updated_at=updated_at,
            persist_dir=data.get('persist_dir'),
            storage_size_bytes=data.get('storage_size_bytes'),
            config=data.get('config', {}),
        )
