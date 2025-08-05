"""
LangChain Data Models

Defines data structures and models for LangChain operations,
including queries, responses, documents, chains, and agents.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseDocument, BaseQuery, BaseResponse


class LangChainChainType(str, Enum):
    """Types of LangChain chains."""

    LLM = "llm"
    SEQUENTIAL = "sequential"
    ROUTER = "router"
    TRANSFORM = "transform"
    CONVERSATION = "conversation"
    QA = "qa"
    RETRIEVAL_QA = "retrieval_qa"
    SUMMARIZATION = "summarization"
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"
    STUFF = "stuff"
    SQL_DATABASE = "sql_database"
    API = "api"
    CONSTITUTIONAL_AI = "constitutional_ai"


class LangChainAgentType(str, Enum):
    """Types of LangChain agents."""

    ZERO_SHOT_REACT = "zero-shot-react-description"
    REACT_DOCSTORE = "react-docstore"
    SELF_ASK_WITH_SEARCH = "self-ask-with-search"
    CONVERSATIONAL_REACT = "conversational-react-description"
    CHAT_ZERO_SHOT_REACT = "chat-zero-shot-react-description"
    CHAT_CONVERSATIONAL_REACT = "chat-conversational-react-description"
    STRUCTURED_CHAT_ZERO_SHOT_REACT = "structured-chat-zero-shot-react-description"
    OPENAI_FUNCTIONS = "openai-functions"
    OPENAI_MULTI_FUNCTIONS = "openai-multi-functions"


class LangChainVectorStoreType(str, Enum):
    """Types of vector stores."""

    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    FAISS = "faiss"
    ANNOY = "annoy"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"
    PGVECTOR = "pgvector"


@dataclass
class LangChainConfig(BaseConfig):
    """LangChain specific configuration."""

    # LLM Configuration
    llm_type: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: Optional[int] = None
    llm_streaming: bool = False

    # Embedding Configuration
    embedding_type: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    embedding_chunk_size: int = 1000

    # Vector Store Configuration
    vector_store_type: LangChainVectorStoreType = LangChainVectorStoreType.CHROMA
    vector_store_config: Dict[str, Any] = field(default_factory=dict)

    # Memory Configuration
    memory_type: str = "buffer"  # buffer, summary, entity, kg, etc.
    memory_config: Dict[str, Any] = field(default_factory=dict)

    # Chain Configuration
    default_chain_type: LangChainChainType = LangChainChainType.LLM
    chain_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Agent Configuration
    default_agent_type: LangChainAgentType = LangChainAgentType.ZERO_SHOT_REACT
    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Retrieval Configuration
    retrieval_k: int = 4
    retrieval_search_type: str = "similarity"
    retrieval_score_threshold: float = 0.0

    # Text Splitter Configuration
    text_splitter_type: str = "recursive"
    text_splitter_chunk_size: int = 1000
    text_splitter_chunk_overlap: int = 200

    # Tool Configuration
    available_tools: List[str] = field(default_factory=list)
    tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.vector_store_config:
            self.vector_store_config = {}
        if not self.memory_config:
            self.memory_config = {}
        if not self.chain_configs:
            self.chain_configs = {}
        if not self.agent_configs:
            self.agent_configs = {}
        if not self.available_tools:
            self.available_tools = ["python_repl", "wikipedia", "ddg-search"]
        if not self.tool_configs:
            self.tool_configs = {}


@dataclass
class LangChainDocument(BaseDocument):
    """LangChain document representation."""

    page_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Ensure content compatibility
        if not self.content and self.page_content:
            self.content = self.page_content
        elif not self.page_content and self.content:
            self.page_content = self.content

    def to_langchain_document(self):
        """Convert to LangChain Document format."""
        try:
            from langchain.schema import Document

            return Document(page_content=self.page_content, metadata=self.metadata)
        except ImportError:
            raise ImportError("LangChain is not installed. Please install with: pip install langchain")

    @classmethod
    def from_langchain_document(cls, doc) -> 'LangChainDocument':
        """Create from LangChain Document."""
        return cls(page_content=doc.page_content, content=doc.page_content, metadata=doc.metadata or {})


@dataclass
class LangChainQuery(BaseQuery):
    """LangChain query representation."""

    query: str
    chain_type: Optional[LangChainChainType] = None
    agent_type: Optional[LangChainAgentType] = None

    # Chain-specific parameters
    chain_config: Dict[str, Any] = field(default_factory=dict)

    # Agent-specific parameters
    agent_config: Dict[str, Any] = field(default_factory=dict)
    tools: Optional[List[str]] = None

    # Memory/conversation parameters
    chat_history: Optional[List[Dict[str, str]]] = None
    memory_key: Optional[str] = None

    # Retrieval parameters
    retrieval_k: Optional[int] = None
    retrieval_filter: Optional[Dict[str, Any]] = None

    # Advanced parameters
    callbacks: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Set query_text for base class compatibility
        if not hasattr(self, 'query_text') or not self.query_text:
            self.query_text = self.query

        if self.chat_history is None:
            self.chat_history = []
        if self.tools is None:
            self.tools = []
        if self.callbacks is None:
            self.callbacks = []
        if self.tags is None:
            self.tags = []


@dataclass
class LangChainResponse(BaseResponse):
    """LangChain response representation."""

    result: Optional[str] = None
    output: Optional[str] = None

    # Chain-specific outputs
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    chain_outputs: Dict[str, Any] = field(default_factory=dict)

    # Agent-specific outputs
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Source documents (for retrieval chains)
    source_documents: List[Dict[str, Any]] = field(default_factory=list)

    # Memory/conversation state
    memory_state: Optional[Dict[str, Any]] = None

    # Execution metadata
    token_usage: Optional[Dict[str, int]] = None

    def __post_init__(self):
        super().__post_init__()
        # Set content for base class compatibility
        if not self.content:
            self.content = self.result or self.output or ""

    def get_formatted_sources(self) -> str:
        """Get formatted source information."""
        if not self.source_documents:
            return "No sources available."

        formatted = []
        for i, doc in enumerate(self.source_documents, 1):
            content = (
                doc.get('page_content', '')[:200] + "..."
                if len(doc.get('page_content', '')) > 200
                else doc.get('page_content', '')
            )
            metadata = doc.get('metadata', {})

            source_info = f"Source {i}:\n"
            source_info += f"Content: {content}\n"

            if metadata:
                source_info += f"Metadata: {json.dumps(metadata, indent=2)}\n"

            formatted.append(source_info)

        return "\n".join(formatted)

    @classmethod
    def from_langchain_output(
        cls, output: Any, query: LangChainQuery, chain_type: Optional[LangChainChainType] = None
    ) -> 'LangChainResponse':
        """Create from LangChain output."""

        # Handle different output types
        if isinstance(output, dict):
            result = output.get('result') or output.get('output') or output.get('answer') or str(output)

            response = cls(result=result, content=result, chain_outputs=output, success=True, framework="langchain")

            # Extract source documents if available
            if 'source_documents' in output:
                source_docs = []
                for doc in output['source_documents']:
                    if hasattr(doc, 'page_content'):
                        source_docs.append({'page_content': doc.page_content, 'metadata': getattr(doc, 'metadata', {})})
                response.source_documents = source_docs

            # Extract intermediate steps for agents
            if 'intermediate_steps' in output:
                response.intermediate_steps = output['intermediate_steps']

            return response

        elif isinstance(output, str):
            return cls(result=output, content=output, success=True, framework="langchain")

        else:
            return cls(result=str(output), content=str(output), success=True, framework="langchain")


@dataclass
class LangChainChain:
    """LangChain chain representation."""

    chain_id: str
    chain_type: LangChainChainType
    name: Optional[str] = None
    description: Optional[str] = None

    # Chain configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Chain metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    usage_count: int = 0

    # Performance metrics
    avg_execution_time: Optional[float] = None
    success_rate: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'chain_id': self.chain_id,
            'chain_type': self.chain_type.value,
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'usage_count': self.usage_count,
            'avg_execution_time': self.avg_execution_time,
            'success_rate': self.success_rate,
        }


@dataclass
class LangChainAgent:
    """LangChain agent representation."""

    agent_id: str
    agent_type: LangChainAgentType
    name: Optional[str] = None
    description: Optional[str] = None

    # Agent configuration
    tools: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    # Agent metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    usage_count: int = 0

    # Performance metrics
    avg_execution_time: Optional[float] = None
    success_rate: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'name': self.name,
            'description': self.description,
            'tools': self.tools,
            'config': self.config,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'usage_count': self.usage_count,
            'avg_execution_time': self.avg_execution_time,
            'success_rate': self.success_rate,
        }
