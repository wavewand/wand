"""
OpenAI Client

Direct client for interacting with OpenAI APIs including completions,
chat completions, embeddings, and other OpenAI services.
"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from integrations.frameworks.base import BaseClient
from observability.logging import get_logger

from .models import (
    OpenAIChatMessage,
    OpenAIChatQuery,
    OpenAICompletionQuery,
    OpenAIConfig,
    OpenAIEmbeddingQuery,
    OpenAIModel,
    OpenAIQuery,
    OpenAIQueryType,
    OpenAIResponse,
)


class OpenAIClient(BaseClient):
    """Direct OpenAI API client."""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)

        # OpenAI client (initialized lazily)
        self._client = None
        self._async_client = None

        # Usage tracking
        self._usage_stats = {"requests": 0, "tokens": 0, "errors": 0, "last_request": None}

        # Initialize OpenAI
        self._initialize_openai()

    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            # Set API key
            if self.config.api_key:
                os.environ['OPENAI_API_KEY'] = self.config.api_key

            # Set organization if provided
            if self.config.organization:
                os.environ['OPENAI_ORG_ID'] = self.config.organization

            self.logger.info("OpenAI client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")
            raise

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    organization=self.config.organization,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )

            except ImportError as e:
                self.logger.error(f"OpenAI import error: {e}")
                raise ImportError("OpenAI is not installed. Please install with: pip install openai")
            except Exception as e:
                self.logger.error(f"Failed to create OpenAI client: {e}")
                raise

        return self._client

    def _get_async_client(self):
        """Get or create async OpenAI client."""
        if self._async_client is None:
            try:
                import openai

                self._async_client = openai.AsyncOpenAI(
                    api_key=self.config.api_key,
                    organization=self.config.organization,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )

            except ImportError as e:
                self.logger.error(f"OpenAI import error: {e}")
                raise ImportError("OpenAI is not installed. Please install with: pip install openai")
            except Exception as e:
                self.logger.error(f"Failed to create async OpenAI client: {e}")
                raise

        return self._async_client

    def _update_usage_stats(self, usage_data: Optional[Dict[str, Any]] = None):
        """Update usage statistics."""
        self._usage_stats["requests"] += 1
        self._usage_stats["last_request"] = time.time()

        if usage_data:
            tokens = usage_data.get("total_tokens", 0)
            self._usage_stats["tokens"] += tokens

        if self.config.track_usage:
            self.logger.debug(f"Usage stats: {self._usage_stats}")

    def _check_rate_limits(self):
        """Check if we're within rate limits."""
        if not self.config.usage_limits:
            return True

        current_time = time.time()

        # Simple rate limiting check (more sophisticated logic can be added)
        if self._usage_stats["last_request"]:
            time_since_last = current_time - self._usage_stats["last_request"]
            if time_since_last < 1.0:  # Basic throttling
                time.sleep(0.1)

        return True

    async def completion(self, query: OpenAICompletionQuery) -> OpenAIResponse:
        """Execute completion request."""
        try:
            self.logger.info(f"Executing OpenAI completion")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "prompt": query.prompt,
                "temperature": query.temperature or self.config.default_temperature,
                "max_tokens": query.max_tokens or self.config.default_max_tokens,
                "stream": query.stream and self.config.enable_streaming,
            }

            # Add optional parameters
            if query.suffix:
                params["suffix"] = query.suffix
            if query.top_p:
                params["top_p"] = query.top_p
            if query.frequency_penalty:
                params["frequency_penalty"] = query.frequency_penalty
            if query.presence_penalty:
                params["presence_penalty"] = query.presence_penalty
            if query.stop:
                params["stop"] = query.stop
            if query.logit_bias:
                params["logit_bias"] = query.logit_bias
            if query.user:
                params["user"] = query.user
            if query.echo:
                params["echo"] = query.echo
            if query.n:
                params["n"] = query.n
            if query.best_of:
                params["best_of"] = query.best_of

            # Execute request
            response = await client.completions.create(**params)

            # Convert response
            openai_response = OpenAIResponse.from_openai_response(response, query)

            # Update usage stats
            if openai_response.usage:
                self._update_usage_stats({"total_tokens": openai_response.usage.total_tokens})
            else:
                self._update_usage_stats()

            self.logger.info("OpenAI completion executed successfully")
            return openai_response

        except Exception as e:
            self.logger.error(f"OpenAI completion failed: {e}")
            self._usage_stats["errors"] += 1
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def chat_completion(self, query: OpenAIChatQuery) -> OpenAIResponse:
        """Execute chat completion request."""
        try:
            self.logger.info(f"Executing OpenAI chat completion")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare messages
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **({"function_call": msg.function_call} if msg.function_call else {}),
                }
                for msg in query.messages
            ]

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "messages": messages,
                "temperature": query.temperature or self.config.default_temperature,
                "max_tokens": query.max_tokens or self.config.default_max_tokens,
                "stream": query.stream and self.config.enable_streaming,
            }

            # Add optional parameters
            if query.top_p:
                params["top_p"] = query.top_p
            if query.frequency_penalty:
                params["frequency_penalty"] = query.frequency_penalty
            if query.presence_penalty:
                params["presence_penalty"] = query.presence_penalty
            if query.stop:
                params["stop"] = query.stop
            if query.logit_bias:
                params["logit_bias"] = query.logit_bias
            if query.user:
                params["user"] = query.user
            if query.n:
                params["n"] = query.n
            if query.functions:
                params["functions"] = query.functions
            if query.function_call:
                params["function_call"] = query.function_call

            # Execute request
            response = await client.chat.completions.create(**params)

            # Convert response
            openai_response = OpenAIResponse.from_openai_response(response, query)

            # Update usage stats
            if openai_response.usage:
                self._update_usage_stats({"total_tokens": openai_response.usage.total_tokens})
            else:
                self._update_usage_stats()

            self.logger.info("OpenAI chat completion executed successfully")
            return openai_response

        except Exception as e:
            self.logger.error(f"OpenAI chat completion failed: {e}")
            self._usage_stats["errors"] += 1
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def embedding(self, query: OpenAIEmbeddingQuery) -> OpenAIResponse:
        """Execute embedding request."""
        try:
            self.logger.info(f"Executing OpenAI embedding")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else OpenAIModel.TEXT_EMBEDDING_ADA_002.value,
                "input": query.input,
                "encoding_format": query.encoding_format,
            }

            # Add optional parameters
            if query.dimensions:
                params["dimensions"] = query.dimensions
            if query.user:
                params["user"] = query.user

            # Execute request
            response = await client.embeddings.create(**params)

            # Convert response
            openai_response = OpenAIResponse.from_openai_response(response, query)

            # Update usage stats
            if openai_response.usage:
                self._update_usage_stats({"total_tokens": openai_response.usage.total_tokens})
            else:
                self._update_usage_stats()

            self.logger.info("OpenAI embedding executed successfully")
            return openai_response

        except Exception as e:
            self.logger.error(f"OpenAI embedding failed: {e}")
            self._usage_stats["errors"] += 1
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def stream_chat_completion(self, query: OpenAIChatQuery) -> AsyncIterator[OpenAIResponse]:
        """Stream chat completion responses."""
        try:
            self.logger.info("Starting OpenAI streaming chat completion")

            # Force streaming
            query.stream = True

            client = self._get_async_client()

            # Prepare messages
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **({"function_call": msg.function_call} if msg.function_call else {}),
                }
                for msg in query.messages
            ]

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "messages": messages,
                "temperature": query.temperature or self.config.default_temperature,
                "max_tokens": query.max_tokens or self.config.default_max_tokens,
                "stream": True,
            }

            # Add optional parameters
            if query.top_p:
                params["top_p"] = query.top_p
            if query.frequency_penalty:
                params["frequency_penalty"] = query.frequency_penalty
            if query.presence_penalty:
                params["presence_penalty"] = query.presence_penalty
            if query.stop:
                params["stop"] = query.stop
            if query.functions:
                params["functions"] = query.functions
            if query.function_call:
                params["function_call"] = query.function_call

            # Execute streaming request
            stream = await client.chat.completions.create(**params)

            async for chunk in stream:
                # Convert chunk to response
                response = OpenAIResponse.from_openai_response(chunk, query)
                yield response

            self._update_usage_stats()
            self.logger.info("OpenAI streaming completed")

        except Exception as e:
            self.logger.error(f"OpenAI streaming failed: {e}")
            self._usage_stats["errors"] += 1
            yield OpenAIResponse(success=False, error=str(e), framework="openai")

    async def moderate_content(self, content: str) -> OpenAIResponse:
        """Moderate content using OpenAI moderation API."""
        try:
            client = self._get_async_client()

            response = await client.moderations.create(input=content)

            # Convert response
            openai_response = OpenAIResponse(
                success=True,
                framework="openai",
                content=str(response.results[0].flagged),
                metadata={
                    "flagged": response.results[0].flagged,
                    "categories": response.results[0].categories.model_dump(),
                    "category_scores": response.results[0].category_scores.model_dump(),
                },
            )

            self._update_usage_stats()
            return openai_response

        except Exception as e:
            self.logger.error(f"OpenAI moderation failed: {e}")
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self._usage_stats.copy()

    async def list_models(self) -> List[str]:
        """List available OpenAI models."""
        try:
            client = self._get_async_client()
            models = await client.models.list()
            return [model.id for model in models.data]

        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic API access
            client = self._get_async_client()
            models = await client.models.list()

            return {
                'status': 'healthy',
                'framework': 'openai',
                'base_url': self.config.base_url,
                'available_models': len(models.data),
                'usage_stats': self._usage_stats,
                'rate_limits': self.config.usage_limits,
            }

        except Exception as e:
            return {'status': 'unhealthy', 'framework': 'openai', 'error': str(e)}
