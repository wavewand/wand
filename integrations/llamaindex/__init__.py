"""
LlamaIndex AI Integration Module for MCP Distributed System

This module provides integration with the LlamaIndex framework for:
- Retrieval-Augmented Generation (RAG)
- Document indexing and querying
- Question answering systems
- Text summarization
- Vector-based similarity search
"""

from .document_processor import LlamaIndexDocumentProcessor
from .index_manager import LlamaIndexManager
from .query_engine import LlamaIndexQueryEngine

__version__ = "1.0.0"
__all__ = ["LlamaIndexManager", "LlamaIndexDocumentProcessor", "LlamaIndexQueryEngine"]
