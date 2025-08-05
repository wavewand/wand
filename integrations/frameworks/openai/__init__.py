"""
OpenAI Framework Integration

Direct integration with OpenAI APIs for completions, chat, embeddings,
and other OpenAI services.
"""

from .client import OpenAIClient
from .models import (
    OpenAIChatQuery,
    OpenAICompletionQuery,
    OpenAIConfig,
    OpenAIEmbeddingQuery,
    OpenAIQuery,
    OpenAIResponse,
)
from .service import OpenAIService

__all__ = [
    'OpenAIClient',
    'OpenAIService',
    'OpenAIQuery',
    'OpenAIResponse',
    'OpenAIConfig',
    'OpenAICompletionQuery',
    'OpenAIChatQuery',
    'OpenAIEmbeddingQuery',
]
