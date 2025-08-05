"""
Transformers Client

Direct client for interacting with Hugging Face Transformers library
for local and cloud-based model inference.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from integrations.frameworks.base import BaseClient
from observability.logging import get_logger

from .models import (
    TransformersClassificationQuery,
    TransformersConfig,
    TransformersDevice,
    TransformersEmbeddingQuery,
    TransformersModelType,
    TransformersQAQuery,
    TransformersQuery,
    TransformersResponse,
    TransformersSummarizationQuery,
    TransformersTask,
    TransformersTextGenerationQuery,
    TransformersTranslationQuery,
)


class TransformersClient(BaseClient):
    """Hugging Face Transformers client."""

    def __init__(self, config: TransformersConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)

        # Model cache
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}

        # Device management
        self._device = self._get_device()

        # Performance tracking
        self._stats = {"models_loaded": 0, "inferences": 0, "total_time_ms": 0, "errors": 0}

        self.logger.info(f"Transformers client initialized on device: {self._device}")

    def _get_device(self) -> str:
        """Determine the best device for model execution."""
        if self.config.device == TransformersDevice.AUTO:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return self.config.device.value

    def _get_model_key(self, model_name: str, task: TransformersTask) -> str:
        """Generate cache key for model."""
        return f"{model_name}:{task.value}"

    def _load_pipeline(self, model_name: str, task: TransformersTask) -> Any:
        """Load or get cached pipeline."""
        try:
            from transformers import pipeline

            model_key = self._get_model_key(model_name, task)

            if model_key not in self._pipelines:
                self.logger.info(f"Loading Transformers pipeline: {model_name} for {task.value}")

                # Prepare pipeline arguments
                pipeline_kwargs = {
                    "model": model_name,
                    "task": task.value,
                    "device": self._device if self._device != "auto" else None,
                    **self.config.pipeline_kwargs,
                }

                # Model-specific kwargs
                if self.config.model_kwargs:
                    pipeline_kwargs["model_kwargs"] = self.config.model_kwargs

                # Tokenizer kwargs
                if self.config.tokenizer_kwargs:
                    pipeline_kwargs["tokenizer_kwargs"] = self.config.tokenizer_kwargs

                # Memory optimizations
                if self.config.torch_dtype != "auto":
                    pipeline_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

                if self.config.load_in_8bit:
                    pipeline_kwargs["load_in_8bit"] = True

                if self.config.load_in_4bit:
                    pipeline_kwargs["load_in_4bit"] = True

                # Cache directory
                if self.config.cache_dir:
                    os.makedirs(self.config.cache_dir, exist_ok=True)
                    pipeline_kwargs["cache_dir"] = self.config.cache_dir

                # Authentication
                if self.config.use_auth_token:
                    pipeline_kwargs["use_auth_token"] = self.config.use_auth_token

                # Safety
                pipeline_kwargs["trust_remote_code"] = self.config.trust_remote_code

                # Create pipeline
                pipe = pipeline(**pipeline_kwargs)
                self._pipelines[model_key] = pipe
                self._stats["models_loaded"] += 1

                self.logger.info(f"Loaded pipeline: {model_name}")

            return self._pipelines[model_key]

        except ImportError as e:
            self.logger.error(f"Transformers import error: {e}")
            raise ImportError("Transformers is not installed. Please install with: pip install transformers torch")
        except Exception as e:
            self.logger.error(f"Failed to load pipeline {model_name}: {e}")
            raise

    def _prepare_generation_kwargs(self, query: TransformersQuery) -> Dict[str, Any]:
        """Prepare generation kwargs from query."""
        kwargs = {}

        # Add non-None parameters
        if query.max_length is not None:
            kwargs["max_length"] = query.max_length
        elif self.config.max_length:
            kwargs["max_length"] = self.config.max_length

        if query.max_new_tokens is not None:
            kwargs["max_new_tokens"] = query.max_new_tokens

        if query.min_length is not None:
            kwargs["min_length"] = query.min_length

        if query.temperature is not None:
            kwargs["temperature"] = query.temperature
        elif self.config.temperature != 1.0:
            kwargs["temperature"] = self.config.temperature

        if query.top_k is not None:
            kwargs["top_k"] = query.top_k
        elif self.config.top_k != 50:
            kwargs["top_k"] = self.config.top_k

        if query.top_p is not None:
            kwargs["top_p"] = query.top_p
        elif self.config.top_p != 1.0:
            kwargs["top_p"] = self.config.top_p

        if query.do_sample is not None:
            kwargs["do_sample"] = query.do_sample
        elif self.config.do_sample:
            kwargs["do_sample"] = self.config.do_sample

        if query.num_return_sequences is not None:
            kwargs["num_return_sequences"] = query.num_return_sequences
        elif self.config.num_return_sequences != 1:
            kwargs["num_return_sequences"] = self.config.num_return_sequences

        if query.repetition_penalty is not None:
            kwargs["repetition_penalty"] = query.repetition_penalty
        elif self.config.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.config.repetition_penalty

        # Add task-specific kwargs
        kwargs.update(query.task_kwargs)

        return kwargs

    async def query(self, query: TransformersQuery) -> TransformersResponse:
        """Execute Transformers query."""
        try:
            start_time = time.time()

            self.logger.info(f"Executing Transformers query: {query.task}")

            # Get model name
            model_name = query.model_name or self.config.model_name

            # Load pipeline
            pipeline = self._load_pipeline(model_name, query.task)

            # Prepare generation kwargs
            generation_kwargs = self._prepare_generation_kwargs(query)

            # Execute inference
            output = pipeline(query.inputs, **generation_kwargs)

            # Calculate timing
            generation_time_ms = (time.time() - start_time) * 1000

            # Create response
            response = TransformersResponse.from_transformers_output(output, query, generation_time_ms)
            response.model_name = model_name

            # Update stats
            self._stats["inferences"] += 1
            self._stats["total_time_ms"] += generation_time_ms

            self.logger.info(f"Transformers query completed in {generation_time_ms:.2f}ms")
            return response

        except Exception as e:
            self.logger.error(f"Transformers query failed: {e}")
            self._stats["errors"] += 1
            return TransformersResponse(
                success=False,
                error=str(e),
                framework="transformers",
                task=query.task,
                model_name=query.model_name or self.config.model_name,
            )

    async def text_generation(self, query: TransformersTextGenerationQuery) -> TransformersResponse:
        """Generate text."""
        return await self.query(query)

    async def text_classification(self, query: TransformersClassificationQuery) -> TransformersResponse:
        """Classify text."""
        return await self.query(query)

    async def question_answering(self, query: TransformersQAQuery) -> TransformersResponse:
        """Answer questions."""
        return await self.query(query)

    async def summarization(self, query: TransformersSummarizationQuery) -> TransformersResponse:
        """Summarize text."""
        return await self.query(query)

    async def translation(self, query: TransformersTranslationQuery) -> TransformersResponse:
        """Translate text."""
        return await self.query(query)

    async def feature_extraction(self, query: TransformersEmbeddingQuery) -> TransformersResponse:
        """Extract features/embeddings."""
        return await self.query(query)

    async def batch_inference(
        self, queries: List[TransformersQuery], batch_size: Optional[int] = None
    ) -> List[TransformersResponse]:
        """Execute batch inference."""
        try:
            batch_size = batch_size or self.config.batch_size
            responses = []

            # Group queries by model and task
            grouped_queries = {}
            for i, query in enumerate(queries):
                model_name = query.model_name or self.config.model_name
                key = self._get_model_key(model_name, query.task)

                if key not in grouped_queries:
                    grouped_queries[key] = []
                grouped_queries[key].append((i, query))

            # Process each group
            for group_queries in grouped_queries.values():
                # Sort back to original order
                group_queries.sort(key=lambda x: x[0])

                # Process in batches
                for i in range(0, len(group_queries), batch_size):
                    batch = group_queries[i : i + batch_size]
                    batch_responses = []

                    for _, query in batch:
                        response = await self.query(query)
                        batch_responses.append(response)

                    responses.extend(batch_responses)

            return responses

        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            return [TransformersResponse(success=False, error=str(e), framework="transformers") for _ in queries]

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models."""
        return list(self._pipelines.keys())

    def unload_model(self, model_name: str, task: TransformersTask):
        """Unload a specific model."""
        model_key = self._get_model_key(model_name, task)
        if model_key in self._pipelines:
            del self._pipelines[model_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"Unloaded model: {model_key}")

    def clear_cache(self):
        """Clear all loaded models."""
        self._pipelines.clear()
        self._models.clear()
        self._tokenizers.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Cleared all model caches")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        memory_info = {"loaded_models": len(self._pipelines), "device": self._device}

        if torch.cuda.is_available():
            memory_info.update(
                {
                    "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "cuda_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                }
            )

        return memory_info

    async def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._stats.copy()
        stats.update(
            {
                "average_inference_time_ms": (stats["total_time_ms"] / max(stats["inferences"], 1)),
                "memory_usage": self.get_memory_usage(),
            }
        )
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic functionality
            from transformers import pipeline

            # Test simple inference
            test_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            test_result = test_pipeline("This is a test.")

            return {
                'status': 'healthy',
                'framework': 'transformers',
                'device': self._device,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'loaded_models': len(self._pipelines),
                'stats': await self.get_stats(),
                'test_result': test_result[0] if test_result else None,
            }

        except Exception as e:
            return {'status': 'unhealthy', 'framework': 'transformers', 'error': str(e)}
