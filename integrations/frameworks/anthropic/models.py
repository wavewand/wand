"""
Anthropic Data Models

Defines data structures for Anthropic Claude API operations
including messages, completions, and advanced reasoning tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseQuery, BaseResponse


class AnthropicModel(str, Enum):
    """Anthropic Claude models."""

    # Claude 3 Models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # Claude 2 Models
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2_0 = "claude-2.0"

    # Claude Instant
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"
    CLAUDE_INSTANT_1_1 = "claude-instant-1.1"


class AnthropicMessageRole(str, Enum):
    """Message roles in Anthropic conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class AnthropicConfig(BaseConfig):
    """Anthropic specific configuration."""

    # API Configuration
    api_key: Optional[str] = None
    base_url: str = "https://api.anthropic.com"

    # Default Model Settings
    default_model: AnthropicModel = AnthropicModel.CLAUDE_3_SONNET
    default_max_tokens: int = 4096
    default_temperature: float = 0.7

    # Request Configuration
    timeout: float = 60.0
    max_retries: int = 3

    # Usage Tracking
    track_usage: bool = True
    usage_limits: Dict[str, int] = field(default_factory=dict)

    # Streaming
    enable_streaming: bool = True

    # Safety
    safety_settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.usage_limits:
            self.usage_limits = {"requests_per_minute": 1000, "tokens_per_minute": 100000}
        if not self.safety_settings:
            self.safety_settings = {"filter_harmful_content": True, "block_unsafe_requests": True}


@dataclass
class AnthropicMessage:
    """Anthropic message structure."""

    role: AnthropicMessageRole
    content: str

    # Optional fields
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnthropicMessage':
        """Create from dictionary."""
        return cls(role=AnthropicMessageRole(data["role"]), content=data["content"], metadata=data.get("metadata", {}))


@dataclass
class AnthropicQuery(BaseQuery):
    """Anthropic query structure."""

    model: Optional[AnthropicModel] = None
    messages: List[AnthropicMessage] = field(default_factory=list)

    # Generation parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    # Advanced parameters
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None

    # Streaming
    stream: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        # Set query_text from messages if not provided
        if not self.query_text and self.messages:
            user_messages = [msg.content for msg in self.messages if msg.role == AnthropicMessageRole.USER]
            if user_messages:
                self.query_text = user_messages[-1]  # Use last user message

    def add_message(self, role: AnthropicMessageRole, content: str):
        """Add a message to the conversation."""
        self.messages.append(AnthropicMessage(role=role, content=content))

        # Update query_text if this is a user message
        if role == AnthropicMessageRole.USER:
            self.query_text = content

    def add_user_message(self, content: str):
        """Add a user message."""
        self.add_message(AnthropicMessageRole.USER, content)

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.add_message(AnthropicMessageRole.ASSISTANT, content)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dictionaries."""
        return [msg.to_dict() for msg in self.messages]


@dataclass
class AnthropicUsage:
    """Anthropic API usage information."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class AnthropicResponse(BaseResponse):
    """Anthropic API response."""

    # Anthropic specific fields
    model: Optional[str] = None
    stop_reason: Optional[str] = None

    # Usage information
    usage: Optional[AnthropicUsage] = None

    # Response content
    message: Optional[AnthropicMessage] = None

    # Streaming support
    is_streaming: bool = False
    stream_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # Set content from message if not provided
        if not self.content and self.message:
            self.content = self.message.content

    def get_text(self) -> str:
        """Get the response text."""
        if self.message:
            return self.message.content
        return self.content or ""

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message."""
        if self.message and self.message.role == AnthropicMessageRole.ASSISTANT:
            return self.message.content
        return None

    def add_stream_event(self, event_type: str, data: Any):
        """Add streaming event."""
        event = {"event_type": event_type, "data": data, "timestamp": datetime.now().isoformat()}
        self.stream_events.append(event)

    @classmethod
    def from_anthropic_response(cls, response: Any, query: AnthropicQuery) -> 'AnthropicResponse':
        """Create response from Anthropic API response."""
        anthropic_response = cls(success=True, framework="anthropic", model=getattr(response, 'model', None))

        # Handle message content
        if hasattr(response, 'content') and response.content:
            # Anthropic returns content as a list of content blocks
            content_text = ""
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    content_text += content_block.text

            anthropic_response.message = AnthropicMessage(role=AnthropicMessageRole.ASSISTANT, content=content_text)
            anthropic_response.content = content_text

        # Handle stop reason
        if hasattr(response, 'stop_reason'):
            anthropic_response.stop_reason = response.stop_reason

        # Handle usage information
        if hasattr(response, 'usage'):
            usage = response.usage
            anthropic_response.usage = AnthropicUsage(
                input_tokens=getattr(usage, 'input_tokens', 0), output_tokens=getattr(usage, 'output_tokens', 0)
            )

        return anthropic_response

    @classmethod
    def from_stream_event(cls, event: Dict[str, Any], query: AnthropicQuery) -> 'AnthropicResponse':
        """Create response from streaming event."""
        response = cls(success=True, framework="anthropic", is_streaming=True)

        response.add_stream_event(event.get("type", "unknown"), event.get("data", {}))

        # Extract content from stream event
        if event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if "text" in delta:
                response.content = delta["text"]
                response.message = AnthropicMessage(role=AnthropicMessageRole.ASSISTANT, content=delta["text"])

        return response


@dataclass
class AnthropicConversation:
    """Anthropic conversation management."""

    conversation_id: str
    messages: List[AnthropicMessage] = field(default_factory=list)
    system_prompt: Optional[str] = None

    # Conversation metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_message(self, role: AnthropicMessageRole, content: str):
        """Add message to conversation."""
        message = AnthropicMessage(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()

    def add_user_message(self, content: str):
        """Add user message."""
        self.add_message(AnthropicMessageRole.USER, content)

    def add_assistant_message(self, content: str):
        """Add assistant message."""
        self.add_message(AnthropicMessageRole.ASSISTANT, content)

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for Anthropic API."""
        return [msg.to_dict() for msg in self.messages]

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message."""
        for message in reversed(self.messages):
            if message.role == AnthropicMessageRole.USER:
                return message.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the last assistant message."""
        for message in reversed(self.messages):
            if message.role == AnthropicMessageRole.ASSISTANT:
                return message.content
        return None

    def clear_messages(self):
        """Clear all messages."""
        self.messages.clear()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "system_prompt": self.system_prompt,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
        }
