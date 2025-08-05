"""
OpenAI Data Models

Defines data structures for OpenAI API operations including
completions, chat, embeddings, and other OpenAI services.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseQuery, BaseResponse


class OpenAIModel(str, Enum):
    """OpenAI model names."""

    # GPT Models
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_TURBO = "gpt-4-1106-preview"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"

    # Embedding Models
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    # Legacy Models
    TEXT_DAVINCI_003 = "text-davinci-003"
    TEXT_CURIE_001 = "text-curie-001"


class OpenAIQueryType(str, Enum):
    """Types of OpenAI queries."""

    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    MODERATION = "moderation"
    FINE_TUNING = "fine_tuning"


@dataclass
class OpenAIConfig(BaseConfig):
    """OpenAI specific configuration."""

    # API Configuration
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"

    # Default Model Settings
    default_model: OpenAIModel = OpenAIModel.GPT_3_5_TURBO
    default_temperature: float = 0.7
    default_max_tokens: Optional[int] = None

    # Request Configuration
    timeout: float = 43200.0  # 12 hours
    max_retries: int = 3
    request_timeout: float = 43200.0  # 12 hours

    # Usage Tracking
    track_usage: bool = True
    usage_limits: Dict[str, int] = field(default_factory=dict)

    # Streaming
    enable_streaming: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.usage_limits:
            self.usage_limits = {"requests_per_minute": 3500, "tokens_per_minute": 90000}


@dataclass
class OpenAIQuery(BaseQuery):
    """Base OpenAI query."""

    query_type: OpenAIQueryType
    model: Optional[OpenAIModel] = None

    # Common parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    # Advanced parameters
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Streaming
    stream: bool = False

    def __post_init__(self):
        super().__post_init__()


@dataclass
class OpenAICompletionQuery(OpenAIQuery):
    """OpenAI completion query."""

    prompt: str
    suffix: Optional[str] = None
    echo: bool = False
    n: int = 1
    best_of: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.query_type = OpenAIQueryType.COMPLETION
        if not self.query_text:
            self.query_text = self.prompt


@dataclass
class OpenAIChatMessage:
    """OpenAI chat message."""

    role: str  # system, user, assistant, function
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


@dataclass
class OpenAIChatQuery(OpenAIQuery):
    """OpenAI chat completion query."""

    messages: List[OpenAIChatMessage]

    # Function calling
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None

    # Chat-specific parameters
    n: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.query_type = OpenAIQueryType.CHAT
        if not self.query_text and self.messages:
            # Use last user message as query_text
            for msg in reversed(self.messages):
                if msg.role == "user":
                    self.query_text = msg.content
                    break


@dataclass
class OpenAIEmbeddingQuery(OpenAIQuery):
    """OpenAI embedding query."""

    input: Union[str, List[str]]
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.query_type = OpenAIQueryType.EMBEDDING
        if not self.query_text:
            if isinstance(self.input, str):
                self.query_text = self.input
            else:
                self.query_text = "; ".join(self.input[:3])  # First 3 items


@dataclass
class OpenAIUsage:
    """OpenAI API usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class OpenAIChoice:
    """OpenAI response choice."""

    index: int
    text: Optional[str] = None
    message: Optional[OpenAIChatMessage] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class OpenAIResponse(BaseResponse):
    """OpenAI API response."""

    # OpenAI specific fields
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None

    # Response data
    choices: List[OpenAIChoice] = field(default_factory=list)
    usage: Optional[OpenAIUsage] = None

    # Embedding specific
    data: Optional[List[Dict[str, Any]]] = None

    # Function calling
    function_call: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        # Set content from choices
        if not self.content and self.choices:
            if self.choices[0].text:
                self.content = self.choices[0].text
            elif self.choices[0].message:
                self.content = self.choices[0].message.content

    def get_text(self) -> str:
        """Get the main text response."""
        if self.choices:
            choice = self.choices[0]
            if choice.text:
                return choice.text
            elif choice.message:
                return choice.message.content
        return self.content or ""

    def get_function_call(self) -> Optional[Dict[str, Any]]:
        """Get function call if present."""
        if self.choices and self.choices[0].message:
            return self.choices[0].message.function_call
        return self.function_call

    def get_embeddings(self) -> List[List[float]]:
        """Get embeddings data."""
        if self.data:
            return [item.get("embedding", []) for item in self.data]
        return []

    @classmethod
    def from_openai_response(cls, response: Any, query: OpenAIQuery) -> 'OpenAIResponse':
        """Create from OpenAI API response."""
        openai_response = cls(
            id=getattr(response, 'id', None),
            object=getattr(response, 'object', None),
            created=getattr(response, 'created', None),
            model=getattr(response, 'model', None),
            success=True,
            framework="openai",
        )

        # Handle different response types
        if hasattr(response, 'choices') and response.choices:
            choices = []
            for i, choice in enumerate(response.choices):
                openai_choice = OpenAIChoice(
                    index=getattr(choice, 'index', i),
                    text=getattr(choice, 'text', None),
                    finish_reason=getattr(choice, 'finish_reason', None),
                )

                # Handle chat messages
                if hasattr(choice, 'message'):
                    message = choice.message
                    openai_choice.message = OpenAIChatMessage(
                        role=getattr(message, 'role', 'assistant'),
                        content=getattr(message, 'content', ''),
                        function_call=getattr(message, 'function_call', None),
                    )

                choices.append(openai_choice)

            openai_response.choices = choices

        # Handle usage information
        if hasattr(response, 'usage'):
            usage = response.usage
            openai_response.usage = OpenAIUsage(
                prompt_tokens=getattr(usage, 'prompt_tokens', 0),
                completion_tokens=getattr(usage, 'completion_tokens', 0),
                total_tokens=getattr(usage, 'total_tokens', 0),
            )

        # Handle embedding data
        if hasattr(response, 'data'):
            openai_response.data = [
                {
                    "object": getattr(item, 'object', 'embedding'),
                    "embedding": getattr(item, 'embedding', []),
                    "index": getattr(item, 'index', i),
                }
                for i, item in enumerate(response.data)
            ]

        # Set content
        if openai_response.choices:
            choice = openai_response.choices[0]
            if choice.text:
                openai_response.content = choice.text
            elif choice.message:
                openai_response.content = choice.message.content

        return openai_response
