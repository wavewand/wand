"""
Haystack AI Integration Module for MCP Distributed System

This module provides integration with the Haystack AI framework for:
- Retrieval-Augmented Generation (RAG)
- Document search and processing
- Question answering systems
- Semantic search capabilities
- Pipeline management
"""

from .document_store import HaystackDocumentStore
from .embeddings import HaystackEmbeddingManager
from .pipelines import HaystackPipelineManager

__version__ = "1.0.0"
__all__ = ["HaystackPipelineManager", "HaystackDocumentStore", "HaystackEmbeddingManager"]
