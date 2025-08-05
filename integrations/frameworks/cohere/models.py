"""
Cohere Data Models

Defines data structures for Cohere API operations including
text generation, embeddings, classification, and reranking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseQuery, BaseResponse


class CohereModel(str, Enum):
    """Cohere model names."""

    # Generation Models
    COMMAND = "command"
    COMMAND_NIGHTLY = "command-nightly"
    COMMAND_LIGHT = "command-light"
    COMMAND_LIGHT_NIGHTLY = "command-light-nightly"

    # Embedding Models
    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    # Classification Models
    CLASSIFY_ENGLISH_V3 = "classify-english-v3.0"
    CLASSIFY_MULTILINGUAL_V3 = "classify-multilingual-v3.0"

    # Rerank Models
    RERANK_ENGLISH_V3 = "rerank-english-v3.0"
    RERANK_MULTILINGUAL_V3 = "rerank-multilingual-v3.0"


class CohereTask(str, Enum):
    """Cohere task types."""

    GENERATE = "generate"
    EMBED = "embed"
    CLASSIFY = "classify"
    RERANK = "rerank"
    TOKENIZE = "tokenize"
    DETOKENIZE = "detokenize"
    CHAT = "chat"


class CohereInputType(str, Enum):
    """Input types for embeddings."""

    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


@dataclass
class CohereConfig(BaseConfig):
    """Cohere specific configuration."""

    # API Configuration
    api_key: Optional[str] = None
    base_url: str = "https://api.cohere.ai/v1"

    # Default Model Settings
    default_model: CohereModel = CohereModel.COMMAND
    default_max_tokens: int = 256
    default_temperature: float = 0.7

    # Request Configuration
    timeout: float = 60.0
    max_retries: int = 3

    # Usage Tracking
    track_usage: bool = True
    usage_limits: Dict[str, int] = field(default_factory=dict)

    # Streaming
    enable_streaming: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.usage_limits:
            self.usage_limits = {"requests_per_minute": 1000, "tokens_per_minute": 100000}


@dataclass
class CohereQuery(BaseQuery):
    """Base Cohere query."""

    task: CohereTask
    model: Optional[CohereModel] = None

    def __post_init__(self):
        super().__post_init__()


@dataclass
class CohereGenerateQuery(CohereQuery):
    """Cohere text generation query."""

    prompt: str

    # Generation parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)

    # Advanced parameters
    return_likelihoods: str = "NONE"  # NONE, GENERATION, ALL
    truncate: str = "END"  # START, END

    # Streaming
    stream: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.task = CohereTask.GENERATE
        if not self.query_text:
            self.query_text = self.prompt


@dataclass
class CohereChatMessage:
    """Cohere chat message."""

    role: str  # USER, CHATBOT, SYSTEM
    message: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "message": self.message}


@dataclass
class CohereChatQuery(CohereQuery):
    """Cohere chat query."""

    message: str
    chat_history: List[CohereChatMessage] = field(default_factory=list)

    # Chat parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False

    # Conversation parameters
    conversation_id: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.task = CohereTask.CHAT
        if not self.query_text:
            self.query_text = self.message

    def add_message(self, role: str, message: str):
        """Add message to chat history."""
        self.chat_history.append(CohereChatMessage(role=role, message=message))


@dataclass
class CohereEmbedQuery(CohereQuery):
    """Cohere embedding query."""

    texts: List[str]
    input_type: Optional[CohereInputType] = None
    truncate: str = "END"

    def __post_init__(self):
        super().__post_init__()
        self.task = CohereTask.EMBED
        if not self.query_text:
            self.query_text = "; ".join(self.texts[:3])


@dataclass
class CohereClassifyExample:
    """Example for classification."""

    text: str
    label: str


@dataclass
class CohereClassifyQuery(CohereQuery):
    """Cohere classification query."""

    inputs: List[str]
    examples: List[CohereClassifyExample] = field(default_factory=list)

    # Classification parameters
    truncate: str = "END"

    def __post_init__(self):
        super().__post_init__()
        self.task = CohereTask.CLASSIFY
        if not self.query_text:
            self.query_text = "; ".join(self.inputs[:3])


@dataclass
class CohereRerankDocument:
    """Document for reranking."""

    text: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"text": self.text}


@dataclass
class CohereRerankQuery(CohereQuery):
    """Cohere rerank query."""

    query: str
    documents: List[CohereRerankDocument]

    # Rerank parameters
    top_n: Optional[int] = None
    max_chunks_per_doc: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.task = CohereTask.RERANK
        if not self.query_text:
            self.query_text = self.query


@dataclass
class CohereTokenizeQuery(CohereQuery):
    """Cohere tokenization query."""

    text: str

    def __post_init__(self):
        super().__post_init__()
        self.task = CohereTask.TOKENIZE
        if not self.query_text:
            self.query_text = self.text


@dataclass
class CohereDetokenizeQuery(CohereQuery):
    """Cohere detokenization query."""

    tokens: List[int]

    def __post_init__(self):
        super().__post_init__()
        self.task = CohereTask.DETOKENIZE
        if not self.query_text:
            self.query_text = f"Tokens: {self.tokens[:10]}..."


@dataclass
class CohereUsage:
    """Cohere API usage information."""

    billed_units: Dict[str, int] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return sum(self.tokens.values())


@dataclass
class CohereGeneration:
    """Cohere generation result."""

    id: str
    text: str
    likelihood: Optional[float] = None
    token_likelihoods: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None


@dataclass
class CohereEmbedding:
    """Cohere embedding result."""

    embedding: List[float]
    index: int


@dataclass
class CohereClassification:
    """Cohere classification result."""

    input: str
    prediction: str
    confidence: float
    labels: Dict[str, float] = field(default_factory=dict)


@dataclass
class CohereRerankResult:
    """Cohere rerank result."""

    index: int
    relevance_score: float
    document: Optional[Dict[str, Any]] = None


@dataclass
class CohereResponse(BaseResponse):
    """Cohere API response."""

    task: Optional[CohereTask] = None
    model: Optional[str] = None

    # Generation results
    generations: List[CohereGeneration] = field(default_factory=list)

    # Embedding results
    embeddings: List[CohereEmbedding] = field(default_factory=list)

    # Classification results
    classifications: List[CohereClassification] = field(default_factory=list)

    # Rerank results
    rerank_results: List[CohereRerankResult] = field(default_factory=list)

    # Tokenization results
    tokens: List[int] = field(default_factory=list)
    token_strings: List[str] = field(default_factory=list)

    # Chat results
    text: Optional[str] = None
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    # Usage information
    usage: Optional[CohereUsage] = None

    # Streaming support
    is_streaming: bool = False
    stream_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # Set content from results
        if not self.content:
            if self.text:
                self.content = self.text
            elif self.generations:
                self.content = self.generations[0].text
            elif self.classifications:
                self.content = f"{self.classifications[0].prediction} ({self.classifications[0].confidence:.3f})"

    def get_text(self) -> str:
        """Get the main text response."""
        if self.text:
            return self.text
        elif self.generations:
            return self.generations[0].text
        return self.content or ""

    def get_generations(self) -> List[str]:
        """Get all generation texts."""
        return [gen.text for gen in self.generations]

    def get_embeddings_list(self) -> List[List[float]]:
        """Get embeddings as list of lists."""
        return [emb.embedding for emb in self.embeddings]

    def get_classifications(self) -> List[Dict[str, Any]]:
        """Get classification results."""
        return [
            {"input": cls.input, "prediction": cls.prediction, "confidence": cls.confidence, "labels": cls.labels}
            for cls in self.classifications
        ]

    def get_rerank_results(self) -> List[Dict[str, Any]]:
        """Get rerank results."""
        return [
            {"index": result.index, "relevance_score": result.relevance_score, "document": result.document}
            for result in self.rerank_results
        ]

    def add_stream_event(self, event_type: str, data: Any):
        """Add streaming event."""
        event = {"event_type": event_type, "data": data, "timestamp": datetime.now().isoformat()}
        self.stream_events.append(event)

    @classmethod
    def from_cohere_response(cls, response: Any, query: CohereQuery) -> 'CohereResponse':
        """Create response from Cohere API response."""
        cohere_response = cls(
            success=True, framework="cohere", task=query.task, model=query.model.value if query.model else None
        )

        # Handle different response types based on task
        if query.task == CohereTask.GENERATE:
            if hasattr(response, 'generations'):
                cohere_response.generations = [
                    CohereGeneration(
                        id=getattr(gen, 'id', ''),
                        text=getattr(gen, 'text', ''),
                        likelihood=getattr(gen, 'likelihood', None),
                        finish_reason=getattr(gen, 'finish_reason', None),
                    )
                    for gen in response.generations
                ]

        elif query.task == CohereTask.CHAT:
            if hasattr(response, 'text'):
                cohere_response.text = response.text
            if hasattr(response, 'chat_history'):
                cohere_response.chat_history = [
                    {"role": msg.role, "message": msg.message} for msg in response.chat_history
                ]

        elif query.task == CohereTask.EMBED:
            if hasattr(response, 'embeddings'):
                cohere_response.embeddings = [
                    CohereEmbedding(embedding=emb, index=i) for i, emb in enumerate(response.embeddings)
                ]

        elif query.task == CohereTask.CLASSIFY:
            if hasattr(response, 'classifications'):
                cohere_response.classifications = [
                    CohereClassification(
                        input=getattr(cls, 'input', ''),
                        prediction=getattr(cls, 'prediction', ''),
                        confidence=getattr(cls, 'confidence', 0.0),
                        labels=getattr(cls, 'labels', {}),
                    )
                    for cls in response.classifications
                ]

        elif query.task == CohereTask.RERANK:
            if hasattr(response, 'results'):
                cohere_response.rerank_results = [
                    CohereRerankResult(
                        index=getattr(result, 'index', 0),
                        relevance_score=getattr(result, 'relevance_score', 0.0),
                        document=getattr(result, 'document', None),
                    )
                    for result in response.results
                ]

        elif query.task in [CohereTask.TOKENIZE, CohereTask.DETOKENIZE]:
            if hasattr(response, 'tokens'):
                cohere_response.tokens = response.tokens
            if hasattr(response, 'token_strings'):
                cohere_response.token_strings = response.token_strings

        # Handle usage information
        if hasattr(response, 'meta') and hasattr(response.meta, 'billed_units'):
            cohere_response.usage = CohereUsage(
                billed_units=response.meta.billed_units, tokens=getattr(response.meta, 'tokens', {})
            )

        return cohere_response
