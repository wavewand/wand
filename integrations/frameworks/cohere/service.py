"""
Cohere Service

Service layer for Cohere operations with intelligent task routing,
caching, and advanced functionality.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from integrations.frameworks.base import BaseService
from observability.logging import get_logger

from .client import CohereClient
from .models import (
    CohereChatQuery,
    CohereClassifyExample,
    CohereClassifyQuery,
    CohereConfig,
    CohereEmbedQuery,
    CohereGenerateQuery,
    CohereInputType,
    CohereModel,
    CohereQuery,
    CohereRerankDocument,
    CohereRerankQuery,
    CohereResponse,
    CohereTask,
)


class CohereService(BaseService):
    """Cohere service with advanced functionality."""

    def __init__(self, config: CohereConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)
        self.client = CohereClient(config)

        # Response cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=30)

        # Task-specific model recommendations
        self._task_models = {
            CohereTask.GENERATE: CohereModel.COMMAND,
            CohereTask.CHAT: CohereModel.COMMAND,
            CohereTask.EMBED: CohereModel.EMBED_ENGLISH_V3,
            CohereTask.CLASSIFY: CohereModel.CLASSIFY_ENGLISH_V3,
            CohereTask.RERANK: CohereModel.RERANK_ENGLISH_V3,
        }

    def _get_cache_key(self, query: CohereQuery) -> str:
        """Generate cache key for query."""
        cache_data = {
            "task": query.task.value,
            "model": query.model.value if query.model else None,
            "query_text": query.query_text[:100] if query.query_text else "",
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"cohere:{hashlib.md5(cache_string.encode()).hexdigest()[:16]}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False

        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.now() - cached_time < self._cache_ttl

    def _get_from_cache(self, query: CohereQuery) -> Optional[CohereResponse]:
        """Get response from cache if available."""
        cache_key = self._get_cache_key(query)

        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                self.logger.debug(f"Cache hit for query: {cache_key}")
                response_data = cache_entry["response"]
                return CohereResponse(**response_data)

        return None

    def _store_in_cache(self, query: CohereQuery, response: CohereResponse):
        """Store response in cache."""
        if not response.success:
            return

        cache_key = self._get_cache_key(query)
        cache_entry = {"timestamp": datetime.now().isoformat(), "response": response.__dict__}

        self._cache[cache_key] = cache_entry
        self.logger.debug(f"Cached response for query: {cache_key}")

    async def generate_text(
        self,
        prompt: str,
        model: Optional[CohereModel] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> CohereResponse:
        """Generate text with intelligent model selection."""
        try:
            query = CohereGenerateQuery(
                prompt=prompt,
                model=model or self._task_models[CohereTask.GENERATE],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Check cache
            cached_response = self._get_from_cache(query)
            if cached_response:
                return cached_response

            # Execute query
            response = await self.client.generate(query)

            # Cache response
            if response.success:
                self._store_in_cache(query, response)

            return response

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def chat_completion(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[CohereModel] = None,
        **kwargs,
    ) -> CohereResponse:
        """Chat completion with conversation management."""
        try:
            query = CohereChatQuery(message=message, model=model or self._task_models[CohereTask.CHAT], **kwargs)

            # Add chat history
            if chat_history:
                for msg in chat_history:
                    query.add_message(msg["role"], msg["message"])

            return await self.client.chat(query)

        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def generate_embeddings(
        self, texts: List[str], input_type: Optional[CohereInputType] = None, model: Optional[CohereModel] = None
    ) -> CohereResponse:
        """Generate embeddings for texts."""
        try:
            query = CohereEmbedQuery(
                texts=texts, input_type=input_type, model=model or self._task_models[CohereTask.EMBED]
            )

            # Check cache
            cached_response = self._get_from_cache(query)
            if cached_response:
                return cached_response

            # Execute query
            response = await self.client.embed(query)

            # Cache response
            if response.success:
                self._store_in_cache(query, response)

            return response

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def classify_texts(
        self, texts: List[str], examples: Optional[List[Dict[str, str]]] = None, model: Optional[CohereModel] = None
    ) -> CohereResponse:
        """Classify texts with examples."""
        try:
            query = CohereClassifyQuery(inputs=texts, model=model or self._task_models[CohereTask.CLASSIFY])

            # Add examples
            if examples:
                query.examples = [CohereClassifyExample(text=ex["text"], label=ex["label"]) for ex in examples]

            return await self.client.classify(query)

        except Exception as e:
            self.logger.error(f"Text classification failed: {e}")
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def rerank_documents(
        self, query: str, documents: List[str], top_n: Optional[int] = None, model: Optional[CohereModel] = None
    ) -> CohereResponse:
        """Rerank documents by relevance."""
        try:
            rerank_query = CohereRerankQuery(
                query=query,
                documents=[CohereRerankDocument(text=doc) for doc in documents],
                top_n=top_n,
                model=model or self._task_models[CohereTask.RERANK],
            )

            return await self.client.rerank(rerank_query)

        except Exception as e:
            self.logger.error(f"Document reranking failed: {e}")
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings and reranking."""
        try:
            # First, rerank documents
            rerank_response = await self.rerank_documents(query=query, documents=documents, top_n=top_k)

            if not rerank_response.success:
                return []

            # Return ranked results
            results = []
            for result in rerank_response.get_rerank_results():
                results.append(
                    {
                        "text": documents[result["index"]],
                        "relevance_score": result["relevance_score"],
                        "index": result["index"],
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        valid_entries = sum(1 for entry in self._cache.values() if self._is_cache_valid(entry))

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "cache_hit_rate": valid_entries / max(total_entries, 1),
            "cache_ttl_minutes": self._cache_ttl.total_seconds() / 60,
        }

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        self.logger.info("Cohere response cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        client_health = await self.client.health_check()
        cache_stats = await self.get_cache_stats()

        return {
            **client_health,
            "service_type": "cohere_service",
            "cache_stats": cache_stats,
            "supported_tasks": [task.value for task in CohereTask],
            "available_models": [model.value for model in CohereModel],
        }
