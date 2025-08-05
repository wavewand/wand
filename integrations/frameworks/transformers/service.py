"""
Transformers Service

Service layer for Transformers operations with model management,
optimization, and advanced functionality.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseService
from observability.logging import get_logger

from .client import TransformersClient
from .models import (
    TransformersClassificationQuery,
    TransformersConfig,
    TransformersEmbeddingQuery,
    TransformersModelType,
    TransformersQAQuery,
    TransformersQuery,
    TransformersResponse,
    TransformersSummarizationQuery,
    TransformersTask,
    TransformersTextGenerationQuery,
)


class TransformersService(BaseService):
    """Transformers service with advanced functionality."""

    def __init__(self, config: TransformersConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)
        self.client = TransformersClient(config)

        # Response cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=30)

        # Model warmup cache
        self._warmed_models: Dict[str, datetime] = {}

        # Popular models for different tasks
        self._task_models = {
            TransformersTask.TEXT_GENERATION: [
                TransformersModelType.GPT2,
                TransformersModelType.BLOOM,
                TransformersModelType.T5,
            ],
            TransformersTask.TEXT_CLASSIFICATION: [
                TransformersModelType.BERT,
                TransformersModelType.ROBERTA,
                TransformersModelType.DISTILBERT,
            ],
            TransformersTask.QUESTION_ANSWERING: [
                TransformersModelType.BERT,
                TransformersModelType.ROBERTA,
                TransformersModelType.ELECTRA,
            ],
            TransformersTask.SUMMARIZATION: [
                TransformersModelType.BART,
                TransformersModelType.T5,
                TransformersModelType.PEGASUS,
            ],
            TransformersTask.FEATURE_EXTRACTION: [
                TransformersModelType.SENTENCE_TRANSFORMERS,
                TransformersModelType.E5_BASE,
                TransformersModelType.BGE_BASE,
            ],
        }

    def _get_cache_key(self, query: TransformersQuery) -> str:
        """Generate cache key for query."""
        cache_data = {
            "task": query.task.value,
            "model_name": query.model_name or self.config.model_name,
            "inputs": str(query.inputs)[:200],  # Truncate long inputs
            "max_length": query.max_length,
            "temperature": query.temperature,
            "top_k": query.top_k,
            "top_p": query.top_p,
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"transformers:{hashlib.md5(cache_string.encode()).hexdigest()[:16]}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False

        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.now() - cached_time < self._cache_ttl

    def _get_from_cache(self, query: TransformersQuery) -> Optional[TransformersResponse]:
        """Get response from cache if available."""
        if not self.config.use_cache:
            return None

        cache_key = self._get_cache_key(query)

        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                self.logger.debug(f"Cache hit for query: {cache_key}")
                response_data = cache_entry["response"]
                return TransformersResponse(**response_data)

        return None

    def _store_in_cache(self, query: TransformersQuery, response: TransformersResponse):
        """Store response in cache."""
        if not self.config.use_cache or not response.success:
            return

        cache_key = self._get_cache_key(query)
        cache_entry = {"timestamp": datetime.now().isoformat(), "response": response.__dict__}

        self._cache[cache_key] = cache_entry
        self.logger.debug(f"Cached response for query: {cache_key}")

    async def query(self, query: TransformersQuery) -> TransformersResponse:
        """Execute Transformers query with caching."""
        try:
            self.logger.info(f"Processing Transformers query: {query.task}")

            # Check cache first
            cached_response = self._get_from_cache(query)
            if cached_response:
                return cached_response

            # Execute query
            response = await self.client.query(query)

            # Cache successful responses
            if response.success:
                self._store_in_cache(query, response)

            return response

        except Exception as e:
            self.logger.error(f"Transformers query failed: {e}")
            return TransformersResponse(success=False, error=str(e), framework="transformers")

    async def generate_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> TransformersResponse:
        """Generate text with intelligent model selection."""
        try:
            # Auto-select model if not provided
            if not model_name:
                model_name = self._select_best_model(TransformersTask.TEXT_GENERATION)

            query = TransformersTextGenerationQuery(
                prompt=prompt, model_name=model_name, max_length=max_length, temperature=temperature, **kwargs
            )

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return TransformersResponse(success=False, error=str(e), framework="transformers")

    async def classify_text(
        self, text: str, labels: Optional[List[str]] = None, model_name: Optional[str] = None
    ) -> TransformersResponse:
        """Classify text with zero-shot capability."""
        try:
            # Auto-select model if not provided
            if not model_name:
                model_name = self._select_best_model(TransformersTask.TEXT_CLASSIFICATION)

            query = TransformersClassificationQuery(text=text, labels=labels, model_name=model_name)

            # Use zero-shot classification if labels provided
            if labels:
                query.task = TransformersTask.ZERO_SHOT_CLASSIFICATION

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Text classification failed: {e}")
            return TransformersResponse(success=False, error=str(e), framework="transformers")

    async def answer_question(
        self, question: str, context: str, model_name: Optional[str] = None
    ) -> TransformersResponse:
        """Answer question based on context."""
        try:
            if not model_name:
                model_name = self._select_best_model(TransformersTask.QUESTION_ANSWERING)

            query = TransformersQAQuery(question=question, context=context, model_name=model_name)

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Question answering failed: {e}")
            return TransformersResponse(success=False, error=str(e), framework="transformers")

    async def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> TransformersResponse:
        """Summarize text."""
        try:
            if not model_name:
                model_name = self._select_best_model(TransformersTask.SUMMARIZATION)

            query = TransformersSummarizationQuery(
                text=text, model_name=model_name, max_length=max_length, min_length=min_length
            )

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Text summarization failed: {e}")
            return TransformersResponse(success=False, error=str(e), framework="transformers")

    async def extract_embeddings(
        self, text: Union[str, List[str]], model_name: Optional[str] = None
    ) -> TransformersResponse:
        """Extract embeddings from text."""
        try:
            if not model_name:
                model_name = self._select_best_model(TransformersTask.FEATURE_EXTRACTION)

            query = TransformersEmbeddingQuery(text=text, model_name=model_name)

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {e}")
            return TransformersResponse(success=False, error=str(e), framework="transformers")

    async def semantic_similarity(self, text1: str, text2: str, model_name: Optional[str] = None) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Get embeddings for both texts
            embeddings_response = await self.extract_embeddings([text1, text2], model_name)

            if not embeddings_response.success:
                return 0.0

            embeddings = embeddings_response.get_embeddings()
            if len(embeddings) < 2:
                return 0.0

            # Calculate cosine similarity
            import numpy as np

            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])

            cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(cosine_sim)

        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0

    async def batch_classify(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[TransformersResponse]:
        """Classify multiple texts efficiently."""
        try:
            if not model_name:
                model_name = self._select_best_model(TransformersTask.TEXT_CLASSIFICATION)

            queries = [
                TransformersClassificationQuery(text=text, labels=labels, model_name=model_name) for text in texts
            ]

            return await self.client.batch_inference(queries, batch_size)

        except Exception as e:
            self.logger.error(f"Batch classification failed: {e}")
            return [TransformersResponse(success=False, error=str(e), framework="transformers") for _ in texts]

    def _select_best_model(self, task: TransformersTask) -> str:
        """Select the best model for a given task."""
        if task in self._task_models:
            # Return the first (preferred) model for the task
            return self._task_models[task][0].value

        # Fallback to configured model
        return self.config.model_name

    async def warm_up_models(self, tasks: Optional[List[TransformersTask]] = None):
        """Pre-load models for faster inference."""
        try:
            tasks_to_warm = tasks or [
                TransformersTask.TEXT_GENERATION,
                TransformersTask.TEXT_CLASSIFICATION,
                TransformersTask.FEATURE_EXTRACTION,
            ]

            for task in tasks_to_warm:
                model_name = self._select_best_model(task)

                # Create a simple test query
                if task == TransformersTask.TEXT_GENERATION:
                    test_query = TransformersTextGenerationQuery(prompt="Hello", model_name=model_name, max_length=10)
                elif task == TransformersTask.TEXT_CLASSIFICATION:
                    test_query = TransformersClassificationQuery(text="This is a test", model_name=model_name)
                else:
                    test_query = TransformersEmbeddingQuery(text="Test text", model_name=model_name)

                # Execute warm-up query
                await self.query(test_query)
                self._warmed_models[f"{model_name}:{task.value}"] = datetime.now()

                self.logger.info(f"Warmed up model: {model_name} for {task.value}")

            self.logger.info(f"Completed warm-up for {len(tasks_to_warm)} models")

        except Exception as e:
            self.logger.error(f"Model warm-up failed: {e}")

    def get_model_recommendations(self, task: TransformersTask) -> List[Dict[str, Any]]:
        """Get model recommendations for a task."""
        if task not in self._task_models:
            return []

        recommendations = []
        for model in self._task_models[task]:
            recommendations.append(
                {
                    "model_name": model.value,
                    "task": task.value,
                    "description": f"Recommended model for {task.value}",
                    "is_warmed": f"{model.value}:{task.value}" in self._warmed_models,
                }
            )

        return recommendations

    async def optimize_memory(self):
        """Optimize memory usage by clearing unused models."""
        try:
            # Clear client cache
            self.client.clear_cache()

            # Clear service cache
            self._cache.clear()

            # Reset warm-up tracking
            self._warmed_models.clear()

            self.logger.info("Memory optimization completed")

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        valid_entries = sum(1 for entry in self._cache.values() if self._is_cache_valid(entry))

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "cache_hit_rate": valid_entries / max(total_entries, 1),
            "cache_ttl_minutes": self._cache_ttl.total_seconds() / 60,
            "warmed_models": len(self._warmed_models),
        }

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        self.logger.info("Transformers response cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        client_health = await self.client.health_check()
        cache_stats = await self.get_cache_stats()

        return {
            **client_health,
            "service_type": "transformers_service",
            "cache_stats": cache_stats,
            "available_tasks": [task.value for task in TransformersTask],
            "recommended_models": {
                task.value: [model.value for model in models] for task, models in self._task_models.items()
            },
        }
