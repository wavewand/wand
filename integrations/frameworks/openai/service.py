"""
OpenAI Service

Service layer for OpenAI API operations with caching, batching,
and advanced functionality.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseService
from observability.logging import get_logger

from .client import OpenAIClient
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


class OpenAIService(BaseService):
    """OpenAI service with advanced functionality."""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)
        self.client = OpenAIClient(config)

        # Response cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(hours=1)

        # Batch processing
        self._batch_queue: List[OpenAIQuery] = []
        self._batch_size = 10
        self._batch_timeout = 5.0

        # Conversation management
        self._conversations: Dict[str, List[OpenAIChatMessage]] = {}

    def _get_cache_key(self, query: OpenAIQuery) -> str:
        """Generate cache key for query."""
        cache_data = {
            "query_type": query.query_type.value,
            "query_text": query.query_text,
            "model": query.model.value if query.model else None,
            "temperature": query.temperature,
            "max_tokens": query.max_tokens,
        }

        # Add query-specific data
        if isinstance(query, OpenAICompletionQuery):
            cache_data["prompt"] = query.prompt
        elif isinstance(query, OpenAIChatQuery):
            cache_data["messages"] = [{"role": msg.role, "content": msg.content} for msg in query.messages]
        elif isinstance(query, OpenAIEmbeddingQuery):
            cache_data["input"] = query.input

        return f"openai:{hash(json.dumps(cache_data, sort_keys=True))}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False

        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.now() - cached_time < self._cache_ttl

    def _get_from_cache(self, query: OpenAIQuery) -> Optional[OpenAIResponse]:
        """Get response from cache if available."""
        cache_key = self._get_cache_key(query)

        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                self.logger.debug(f"Cache hit for query: {cache_key}")
                response_data = cache_entry["response"]
                return OpenAIResponse(**response_data)

        return None

    def _store_in_cache(self, query: OpenAIQuery, response: OpenAIResponse):
        """Store response in cache."""
        if not response.success:
            return  # Don't cache failed responses

        cache_key = self._get_cache_key(query)
        cache_entry = {"timestamp": datetime.now().isoformat(), "response": response.__dict__}

        self._cache[cache_key] = cache_entry
        self.logger.debug(f"Cached response for query: {cache_key}")

    async def query(self, query: OpenAIQuery) -> OpenAIResponse:
        """Execute OpenAI query with caching and routing."""
        try:
            self.logger.info(f"Processing OpenAI query: {query.query_type}")

            # Check cache first
            cached_response = self._get_from_cache(query)
            if cached_response:
                return cached_response

            # Route to appropriate method
            if query.query_type == OpenAIQueryType.COMPLETION:
                response = await self.client.completion(query)
            elif query.query_type == OpenAIQueryType.CHAT:
                response = await self.client.chat_completion(query)
            elif query.query_type == OpenAIQueryType.EMBEDDING:
                response = await self.client.embedding(query)
            else:
                raise ValueError(f"Unsupported query type: {query.query_type}")

            # Cache successful responses
            if response.success:
                self._store_in_cache(query, response)

            return response

        except Exception as e:
            self.logger.error(f"OpenAI query failed: {e}")
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def chat_with_history(
        self,
        user_message: str,
        conversation_id: str,
        system_message: Optional[str] = None,
        model: Optional[OpenAIModel] = None,
        **kwargs,
    ) -> OpenAIResponse:
        """Chat with conversation history management."""
        try:
            # Initialize conversation if needed
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []

                # Add system message if provided
                if system_message:
                    self._conversations[conversation_id].append(
                        OpenAIChatMessage(role="system", content=system_message)
                    )

            # Add user message to conversation
            self._conversations[conversation_id].append(OpenAIChatMessage(role="user", content=user_message))

            # Create chat query
            query = OpenAIChatQuery(
                messages=self._conversations[conversation_id].copy(), model=model or self.config.default_model, **kwargs
            )

            # Execute query
            response = await self.client.chat_completion(query)

            # Add assistant response to conversation
            if response.success and response.content:
                self._conversations[conversation_id].append(
                    OpenAIChatMessage(role="assistant", content=response.content)
                )

            return response

        except Exception as e:
            self.logger.error(f"Chat conversation failed: {e}")
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def batch_embeddings(
        self, texts: List[str], model: Optional[OpenAIModel] = None, batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        try:
            embeddings = []
            batch_size = batch_size or self._batch_size

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                query = OpenAIEmbeddingQuery(input=batch_texts, model=model or OpenAIModel.TEXT_EMBEDDING_ADA_002)

                response = await self.client.embedding(query)

                if response.success:
                    batch_embeddings = response.get_embeddings()
                    embeddings.extend(batch_embeddings)
                else:
                    # Add empty embeddings for failed batch
                    embeddings.extend([[] for _ in batch_texts])

                # Small delay between batches to respect rate limits
                await asyncio.sleep(0.1)

            return embeddings

        except Exception as e:
            self.logger.error(f"Batch embeddings failed: {e}")
            return [[] for _ in texts]

    async def summarize_text(
        self, text: str, max_length: Optional[int] = None, style: str = "concise"
    ) -> OpenAIResponse:
        """Summarize text using chat completion."""
        try:
            # Prepare summarization prompt
            prompts = {
                "concise": "Summarize the following text concisely:",
                "detailed": "Provide a detailed summary of the following text:",
                "bullet": "Summarize the following text in bullet points:",
                "key_points": "Extract the key points from the following text:",
            }

            prompt = prompts.get(style, prompts["concise"])

            if max_length:
                prompt += f" Keep the summary under {max_length} words."

            messages = [
                OpenAIChatMessage(
                    role="system", content="You are a helpful assistant that creates clear, accurate summaries."
                ),
                OpenAIChatMessage(role="user", content=f"{prompt}\n\n{text}"),
            ]

            query = OpenAIChatQuery(messages=messages)
            return await self.client.chat_completion(query)

        except Exception as e:
            self.logger.error(f"Text summarization failed: {e}")
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def analyze_sentiment(self, text: str) -> OpenAIResponse:
        """Analyze sentiment of text."""
        try:
            prompt = """Analyze the sentiment of the following text. Respond with:
1. Overall sentiment (positive/negative/neutral)
2. Confidence score (0-1)
3. Key emotional indicators
4. Brief explanation

Text: """

            messages = [
                OpenAIChatMessage(
                    role="system",
                    content="You are an expert in sentiment analysis. Provide clear, structured responses.",
                ),
                OpenAIChatMessage(role="user", content=f"{prompt}{text}"),
            ]

            query = OpenAIChatQuery(messages=messages)
            return await self.client.chat_completion(query)

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    async def translate_text(
        self, text: str, target_language: str, source_language: Optional[str] = None
    ) -> OpenAIResponse:
        """Translate text to target language."""
        try:
            if source_language:
                prompt = f"Translate the following text from {source_language} to {target_language}:"
            else:
                prompt = f"Translate the following text to {target_language}:"

            messages = [
                OpenAIChatMessage(
                    role="system", content="You are a professional translator. Provide accurate, natural translations."
                ),
                OpenAIChatMessage(role="user", content=f"{prompt}\n\n{text}"),
            ]

            query = OpenAIChatQuery(messages=messages)
            return await self.client.chat_completion(query)

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return OpenAIResponse(success=False, error=str(e), framework="openai")

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        if conversation_id not in self._conversations:
            return []

        return [{"role": msg.role, "content": msg.content} for msg in self._conversations[conversation_id]]

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self._conversations.keys())

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        valid_entries = sum(1 for entry in self._cache.values() if self._is_cache_valid(entry))

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "cache_hit_rate": valid_entries / max(total_entries, 1),
            "cache_ttl_hours": self._cache_ttl.total_seconds() / 3600,
        }

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        self.logger.info("OpenAI response cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        client_health = await self.client.health_check()
        cache_stats = await self.get_cache_stats()

        return {
            **client_health,
            "service_type": "openai_service",
            "active_conversations": len(self._conversations),
            "cache_stats": cache_stats,
        }
