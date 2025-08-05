"""
Cohere Client

Direct client for interacting with Cohere's API including
text generation, embeddings, classification, and reranking.
"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from integrations.frameworks.base import BaseClient
from observability.logging import get_logger

from .models import (
    CohereChatQuery,
    CohereClassifyQuery,
    CohereConfig,
    CohereDetokenizeQuery,
    CohereEmbedQuery,
    CohereGenerateQuery,
    CohereInputType,
    CohereModel,
    CohereQuery,
    CohereRerankQuery,
    CohereResponse,
    CohereTask,
    CohereTokenizeQuery,
)


class CohereClient(BaseClient):
    """Cohere API client."""

    def __init__(self, config: CohereConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)

        # Cohere client (initialized lazily)
        self._client = None
        self._async_client = None

        # Usage tracking
        self._usage_stats = {"requests": 0, "tokens": 0, "errors": 0, "last_request": None}

        # Initialize Cohere
        self._initialize_cohere()

    def _initialize_cohere(self):
        """Initialize Cohere client."""
        try:
            # Set API key
            if self.config.api_key:
                os.environ['COHERE_API_KEY'] = self.config.api_key

            self.logger.info("Cohere client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Cohere: {e}")
            raise

    def _get_client(self):
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere

                self._client = cohere.Client(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )

            except ImportError as e:
                self.logger.error(f"Cohere import error: {e}")
                raise ImportError("Cohere is not installed. Please install with: pip install cohere")
            except Exception as e:
                self.logger.error(f"Failed to create Cohere client: {e}")
                raise

        return self._client

    def _get_async_client(self):
        """Get or create async Cohere client."""
        if self._async_client is None:
            try:
                import cohere

                self._async_client = cohere.AsyncClient(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )

            except ImportError as e:
                self.logger.error(f"Cohere import error: {e}")
                raise ImportError("Cohere is not installed. Please install with: pip install cohere")
            except Exception as e:
                self.logger.error(f"Failed to create async Cohere client: {e}")
                raise

        return self._async_client

    def _update_usage_stats(self, usage_data: Optional[Dict[str, Any]] = None):
        """Update usage statistics."""
        self._usage_stats["requests"] += 1
        self._usage_stats["last_request"] = time.time()

        if usage_data and "tokens" in usage_data:
            total_tokens = sum(usage_data["tokens"].values())
            self._usage_stats["tokens"] += total_tokens

        if self.config.track_usage:
            self.logger.debug(f"Usage stats: {self._usage_stats}")

    def _check_rate_limits(self):
        """Check if we're within rate limits."""
        if not self.config.usage_limits:
            return True

        current_time = time.time()

        # Simple rate limiting check
        if self._usage_stats["last_request"]:
            time_since_last = current_time - self._usage_stats["last_request"]
            if time_since_last < 1.0:  # Basic throttling
                time.sleep(0.1)

        return True

    async def generate(self, query: CohereGenerateQuery) -> CohereResponse:
        """Generate text using Cohere."""
        try:
            self.logger.info("Executing Cohere generation")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "prompt": query.prompt,
                "max_tokens": query.max_tokens or self.config.default_max_tokens,
                "stream": query.stream and self.config.enable_streaming,
            }

            # Add optional parameters
            if query.temperature is not None:
                params["temperature"] = query.temperature
            elif self.config.default_temperature != 0.7:
                params["temperature"] = self.config.default_temperature

            if query.top_k is not None:
                params["k"] = query.top_k

            if query.top_p is not None:
                params["p"] = query.top_p

            if query.frequency_penalty is not None:
                params["frequency_penalty"] = query.frequency_penalty

            if query.presence_penalty is not None:
                params["presence_penalty"] = query.presence_penalty

            if query.stop_sequences:
                params["stop_sequences"] = query.stop_sequences

            if query.return_likelihoods != "NONE":
                params["return_likelihoods"] = query.return_likelihoods

            if query.truncate != "END":
                params["truncate"] = query.truncate

            # Execute request
            response = await client.generate(**params)

            # Convert response
            cohere_response = CohereResponse.from_cohere_response(response, query)

            # Update usage stats
            if cohere_response.usage:
                self._update_usage_stats({"tokens": cohere_response.usage.tokens})
            else:
                self._update_usage_stats()

            self.logger.info("Cohere generation completed successfully")
            return cohere_response

        except Exception as e:
            self.logger.error(f"Cohere generation failed: {e}")
            self._usage_stats["errors"] += 1
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def chat(self, query: CohereChatQuery) -> CohereResponse:
        """Chat using Cohere."""
        try:
            self.logger.info("Executing Cohere chat")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "message": query.message,
                "stream": query.stream and self.config.enable_streaming,
            }

            # Add chat history
            if query.chat_history:
                params["chat_history"] = [msg.to_dict() for msg in query.chat_history]

            # Add optional parameters
            if query.max_tokens is not None:
                params["max_tokens"] = query.max_tokens

            if query.temperature is not None:
                params["temperature"] = query.temperature

            if query.top_k is not None:
                params["k"] = query.top_k

            if query.top_p is not None:
                params["p"] = query.top_p

            if query.conversation_id:
                params["conversation_id"] = query.conversation_id

            # Execute request
            response = await client.chat(**params)

            # Convert response
            cohere_response = CohereResponse.from_cohere_response(response, query)

            # Update usage stats
            if cohere_response.usage:
                self._update_usage_stats({"tokens": cohere_response.usage.tokens})
            else:
                self._update_usage_stats()

            self.logger.info("Cohere chat completed successfully")
            return cohere_response

        except Exception as e:
            self.logger.error(f"Cohere chat failed: {e}")
            self._usage_stats["errors"] += 1
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def embed(self, query: CohereEmbedQuery) -> CohereResponse:
        """Generate embeddings using Cohere."""
        try:
            self.logger.info("Executing Cohere embedding")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else CohereModel.EMBED_ENGLISH_V3.value,
                "texts": query.texts,
            }

            # Add optional parameters
            if query.input_type:
                params["input_type"] = query.input_type.value

            if query.truncate != "END":
                params["truncate"] = query.truncate

            # Execute request
            response = await client.embed(**params)

            # Convert response
            cohere_response = CohereResponse.from_cohere_response(response, query)

            # Update usage stats
            if cohere_response.usage:
                self._update_usage_stats({"tokens": cohere_response.usage.tokens})
            else:
                self._update_usage_stats()

            self.logger.info("Cohere embedding completed successfully")
            return cohere_response

        except Exception as e:
            self.logger.error(f"Cohere embedding failed: {e}")
            self._usage_stats["errors"] += 1
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def classify(self, query: CohereClassifyQuery) -> CohereResponse:
        """Classify text using Cohere."""
        try:
            self.logger.info("Executing Cohere classification")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else CohereModel.CLASSIFY_ENGLISH_V3.value,
                "inputs": query.inputs,
            }

            # Add examples
            if query.examples:
                params["examples"] = [{"text": ex.text, "label": ex.label} for ex in query.examples]

            if query.truncate != "END":
                params["truncate"] = query.truncate

            # Execute request
            response = await client.classify(**params)

            # Convert response
            cohere_response = CohereResponse.from_cohere_response(response, query)

            # Update usage stats
            if cohere_response.usage:
                self._update_usage_stats({"tokens": cohere_response.usage.tokens})
            else:
                self._update_usage_stats()

            self.logger.info("Cohere classification completed successfully")
            return cohere_response

        except Exception as e:
            self.logger.error(f"Cohere classification failed: {e}")
            self._usage_stats["errors"] += 1
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def rerank(self, query: CohereRerankQuery) -> CohereResponse:
        """Rerank documents using Cohere."""
        try:
            self.logger.info("Executing Cohere reranking")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else CohereModel.RERANK_ENGLISH_V3.value,
                "query": query.query,
                "documents": [doc.to_dict() for doc in query.documents],
            }

            # Add optional parameters
            if query.top_n is not None:
                params["top_n"] = query.top_n

            if query.max_chunks_per_doc is not None:
                params["max_chunks_per_doc"] = query.max_chunks_per_doc

            # Execute request
            response = await client.rerank(**params)

            # Convert response
            cohere_response = CohereResponse.from_cohere_response(response, query)

            # Update usage stats
            if cohere_response.usage:
                self._update_usage_stats({"tokens": cohere_response.usage.tokens})
            else:
                self._update_usage_stats()

            self.logger.info("Cohere reranking completed successfully")
            return cohere_response

        except Exception as e:
            self.logger.error(f"Cohere reranking failed: {e}")
            self._usage_stats["errors"] += 1
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def tokenize(self, query: CohereTokenizeQuery) -> CohereResponse:
        """Tokenize text using Cohere."""
        try:
            client = self._get_async_client()

            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "text": query.text,
            }

            response = await client.tokenize(**params)
            cohere_response = CohereResponse.from_cohere_response(response, query)

            self._update_usage_stats()
            return cohere_response

        except Exception as e:
            self.logger.error(f"Cohere tokenization failed: {e}")
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def detokenize(self, query: CohereDetokenizeQuery) -> CohereResponse:
        """Detokenize tokens using Cohere."""
        try:
            client = self._get_async_client()

            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "tokens": query.tokens,
            }

            response = await client.detokenize(**params)
            cohere_response = CohereResponse.from_cohere_response(response, query)

            self._update_usage_stats()
            return cohere_response

        except Exception as e:
            self.logger.error(f"Cohere detokenization failed: {e}")
            return CohereResponse(success=False, error=str(e), framework="cohere")

    async def stream_generate(self, query: CohereGenerateQuery) -> AsyncIterator[CohereResponse]:
        """Stream text generation."""
        try:
            query.stream = True
            client = self._get_async_client()

            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "prompt": query.prompt,
                "max_tokens": query.max_tokens or self.config.default_max_tokens,
                "stream": True,
            }

            if query.temperature is not None:
                params["temperature"] = query.temperature

            stream = client.generate(**params)

            async for event in stream:
                response = CohereResponse(success=True, framework="cohere", task=query.task, is_streaming=True)

                if hasattr(event, 'text'):
                    response.content = event.text

                response.add_stream_event("text_delta", {"text": getattr(event, 'text', '')})
                yield response

            self._update_usage_stats()

        except Exception as e:
            self.logger.error(f"Cohere streaming failed: {e}")
            yield CohereResponse(success=False, error=str(e), framework="cohere")

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self._usage_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic API access
            client = self._get_async_client()

            # Simple tokenization test
            response = await client.tokenize(model=self.config.default_model.value, text="Hello world")

            return {
                'status': 'healthy',
                'framework': 'cohere',
                'base_url': self.config.base_url,
                'default_model': self.config.default_model.value,
                'usage_stats': self._usage_stats,
                'test_successful': True,
            }

        except Exception as e:
            return {'status': 'unhealthy', 'framework': 'cohere', 'error': str(e)}
