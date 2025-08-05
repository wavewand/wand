"""
Cohere Framework Integration

Direct integration with Cohere's API for text generation,
embeddings, classification, and reranking capabilities.
"""

from .client import CohereClient
from .models import CohereConfig, CohereModel, CohereQuery, CohereResponse, CohereTask
from .service import CohereService

__all__ = [
    'CohereClient',
    'CohereService',
    'CohereQuery',
    'CohereResponse',
    'CohereConfig',
    'CohereModel',
    'CohereTask',
]
