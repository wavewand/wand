"""
Haystack Embedding Management for MCP System

This module manages text embeddings for semantic search and retrieval
operations within the Haystack integration.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from sentence_transformers import SentenceTransformer

    EMBEDDING_AVAILABLE = True
except ImportError:
    logging.warning("Embedding libraries not available. Install with: pip install sentence-transformers")
    EMBEDDING_AVAILABLE = False

    class SentenceTransformersTextEmbedder:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return {"embedding": []}


class HaystackEmbeddingManager:
    """Manages text embeddings for semantic search and similarity operations."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.cache_size = cache_size
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.embedder = None
        self.model = None

        self._initialize_embedder()

    def _initialize_embedder(self):
        """Initialize the embedding model."""
        if not EMBEDDING_AVAILABLE:
            self.logger.warning("Embedding libraries not available - semantic search disabled")
            return

        try:
            # Initialize Haystack embedder
            self.embedder = SentenceTransformersTextEmbedder(
                model=self.model_name, device="cpu"  # Use CPU for compatibility
            )

            # Also initialize the model directly for some operations
            self.model = SentenceTransformer(self.model_name)

            self.logger.info(f"Initialized embedding model: {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.embedder = None
            self.model = None

    def get_text_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """Get embedding for a single text."""
        if not self.model:
            return None

        # Create cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Check cache first
        if use_cache and text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        try:
            # Generate embedding
            embedding = self.model.encode([text])[0]

            # Cache the result
            if use_cache:
                self._cache_embedding(text_hash, embedding)

            return embedding

        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None

    def get_batch_embeddings(self, texts: List[str], use_cache: bool = True) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts efficiently."""
        if not self.model:
            return [None] * len(texts)

        results = []
        texts_to_embed = []
        indices_to_embed = []

        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            if use_cache and text_hash in self.embedding_cache:
                results.append(self.embedding_cache[text_hash])
            else:
                results.append(None)  # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Generate embeddings for uncached texts
        if texts_to_embed:
            try:
                embeddings = self.model.encode(texts_to_embed)

                for idx, embedding in zip(indices_to_embed, embeddings):
                    results[idx] = embedding

                    # Cache the result
                    if use_cache:
                        text_hash = hashlib.sha256(texts[idx].encode()).hexdigest()
                        self._cache_embedding(text_hash, embedding)

            except Exception as e:
                self.logger.error(f"Failed to generate batch embeddings: {e}")
                # Return None for failed embeddings
                for idx in indices_to_embed:
                    results[idx] = None

        return results

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embedding1 = self.get_text_embedding(text1)
        embedding2 = self.get_text_embedding(text2)

        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    def find_similar_texts(
        self, query_text: str, candidate_texts: List[str], top_k: int = 5, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Find the most similar texts to a query."""
        if not self.model or not candidate_texts:
            return []

        try:
            # Get query embedding
            query_embedding = self.get_text_embedding(query_text)
            if query_embedding is None:
                return []

            # Get candidate embeddings
            candidate_embeddings = self.get_batch_embeddings(candidate_texts)

            # Calculate similarities
            similarities = []
            for i, (text, embedding) in enumerate(zip(candidate_texts, candidate_embeddings)):
                if embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    if similarity >= threshold:
                        similarities.append({"text": text, "similarity": similarity, "index": i})

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            self.logger.error(f"Failed to find similar texts: {e}")
            return []

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

        except Exception:
            return 0.0

    def _cache_embedding(self, text_hash: str, embedding: np.ndarray):
        """Cache an embedding with size management."""
        # Remove oldest entries if cache is full
        if len(self.embedding_cache) >= self.cache_size:
            # Remove 20% of cache entries (simple FIFO)
            keys_to_remove = list(self.embedding_cache.keys())[: self.cache_size // 5]
            for key in keys_to_remove:
                del self.embedding_cache[key]

        self.embedding_cache[text_hash] = embedding

    def semantic_search(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = 10, content_field: str = "content"
    ) -> List[Dict[str, Any]]:
        """Perform semantic search over a collection of documents."""
        if not documents:
            return []

        try:
            # Extract text content from documents
            texts = []
            for doc in documents:
                if content_field in doc:
                    texts.append(str(doc[content_field]))
                else:
                    texts.append(str(doc))

            # Find similar texts
            similar_results = self.find_similar_texts(query, texts, top_k=top_k)

            # Map results back to original documents
            results = []
            for result in similar_results:
                doc_index = result["index"]
                doc_result = {**documents[doc_index], "similarity_score": result["similarity"], "query": query}
                results.append(doc_result)

            return results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics."""
        return {
            "model_name": self.model_name,
            "embedding_available": EMBEDDING_AVAILABLE,
            "model_loaded": self.model is not None,
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.cache_size,
            "embedding_dimension": self._get_embedding_dimension(),
        }

    def _get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings from the current model."""
        if not self.model:
            return None

        try:
            # Generate a test embedding to get dimension
            test_embedding = self.model.encode(["test"])
            return len(test_embedding[0])
        except Exception:
            return None

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")

    def change_model(self, model_name: str) -> Dict[str, Any]:
        """Change the embedding model."""
        if not EMBEDDING_AVAILABLE:
            return {"success": False, "message": "Embedding libraries not available"}

        try:
            old_model = self.model_name
            self.model_name = model_name

            # Reinitialize embedder
            self._initialize_embedder()

            if self.model is not None:
                # Clear cache since model changed
                self.clear_cache()

                return {
                    "success": True,
                    "message": f"Changed model from {old_model} to {model_name}",
                    "old_model": old_model,
                    "new_model": model_name,
                    "embedding_dimension": self._get_embedding_dimension(),
                }
            else:
                # Revert to old model if new one failed
                self.model_name = old_model
                self._initialize_embedder()

                return {"success": False, "message": f"Failed to load model {model_name}, reverted to {old_model}"}

        except Exception as e:
            self.logger.error(f"Failed to change model: {e}")
            return {"success": False, "message": f"Failed to change model: {str(e)}"}
