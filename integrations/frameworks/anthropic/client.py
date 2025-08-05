"""
Anthropic Client

Direct client for interacting with Anthropic's Claude API
for advanced conversational AI capabilities.
"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from integrations.frameworks.base import BaseClient
from observability.logging import get_logger

from .models import (
    AnthropicConfig,
    AnthropicConversation,
    AnthropicMessage,
    AnthropicMessageRole,
    AnthropicModel,
    AnthropicQuery,
    AnthropicResponse,
    AnthropicUsage,
)


class AnthropicClient(BaseClient):
    """Anthropic Claude API client."""

    def __init__(self, config: AnthropicConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)

        # Anthropic client (initialized lazily)
        self._client = None
        self._async_client = None

        # Usage tracking
        self._usage_stats = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0, "last_request": None}

        # Initialize Anthropic
        self._initialize_anthropic()

    def _initialize_anthropic(self):
        """Initialize Anthropic client."""
        try:
            # Set API key
            if self.config.api_key:
                os.environ['ANTHROPIC_API_KEY'] = self.config.api_key

            self.logger.info("Anthropic client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic: {e}")
            raise

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )

            except ImportError as e:
                self.logger.error(f"Anthropic import error: {e}")
                raise ImportError("Anthropic is not installed. Please install with: pip install anthropic")
            except Exception as e:
                self.logger.error(f"Failed to create Anthropic client: {e}")
                raise

        return self._client

    def _get_async_client(self):
        """Get or create async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic

                self._async_client = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )

            except ImportError as e:
                self.logger.error(f"Anthropic import error: {e}")
                raise ImportError("Anthropic is not installed. Please install with: pip install anthropic")
            except Exception as e:
                self.logger.error(f"Failed to create async Anthropic client: {e}")
                raise

        return self._async_client

    def _update_usage_stats(self, usage: Optional[AnthropicUsage] = None):
        """Update usage statistics."""
        self._usage_stats["requests"] += 1
        self._usage_stats["last_request"] = time.time()

        if usage:
            self._usage_stats["input_tokens"] += usage.input_tokens
            self._usage_stats["output_tokens"] += usage.output_tokens

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

    async def message(self, query: AnthropicQuery) -> AnthropicResponse:
        """Send message to Claude."""
        try:
            self.logger.info("Executing Anthropic message request")

            self._check_rate_limits()

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "max_tokens": query.max_tokens or self.config.default_max_tokens,
                "messages": [msg.to_dict() for msg in query.messages],
                "stream": query.stream and self.config.enable_streaming,
            }

            # Add optional parameters
            if query.temperature is not None:
                params["temperature"] = query.temperature
            elif self.config.default_temperature != 0.7:
                params["temperature"] = self.config.default_temperature

            if query.top_p is not None:
                params["top_p"] = query.top_p

            if query.top_k is not None:
                params["top_k"] = query.top_k

            if query.stop_sequences:
                params["stop_sequences"] = query.stop_sequences

            if query.system_prompt:
                params["system"] = query.system_prompt

            # Execute request
            response = await client.messages.create(**params)

            # Convert response
            anthropic_response = AnthropicResponse.from_anthropic_response(response, query)

            # Update usage stats
            if anthropic_response.usage:
                self._update_usage_stats(anthropic_response.usage)
            else:
                self._update_usage_stats()

            self.logger.info("Anthropic message request completed successfully")
            return anthropic_response

        except Exception as e:
            self.logger.error(f"Anthropic message request failed: {e}")
            self._usage_stats["errors"] += 1
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def stream_message(self, query: AnthropicQuery) -> AsyncIterator[AnthropicResponse]:
        """Stream message responses from Claude."""
        try:
            self.logger.info("Starting Anthropic streaming message")

            # Force streaming
            query.stream = True

            client = self._get_async_client()

            # Prepare parameters
            params = {
                "model": query.model.value if query.model else self.config.default_model.value,
                "max_tokens": query.max_tokens or self.config.default_max_tokens,
                "messages": [msg.to_dict() for msg in query.messages],
                "stream": True,
            }

            # Add optional parameters
            if query.temperature is not None:
                params["temperature"] = query.temperature

            if query.top_p is not None:
                params["top_p"] = query.top_p

            if query.top_k is not None:
                params["top_k"] = query.top_k

            if query.stop_sequences:
                params["stop_sequences"] = query.stop_sequences

            if query.system_prompt:
                params["system"] = query.system_prompt

            # Execute streaming request
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    # Convert event to response
                    response = AnthropicResponse.from_stream_event(
                        {"type": event.type, "data": event.data if hasattr(event, 'data') else {}}, query
                    )
                    yield response

            self._update_usage_stats()
            self.logger.info("Anthropic streaming completed")

        except Exception as e:
            self.logger.error(f"Anthropic streaming failed: {e}")
            self._usage_stats["errors"] += 1
            yield AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def simple_completion(
        self,
        prompt: str,
        model: Optional[AnthropicModel] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> AnthropicResponse:
        """Simple completion interface."""
        try:
            query = AnthropicQuery(
                model=model, max_tokens=max_tokens, temperature=temperature, system_prompt=system_prompt
            )
            query.add_user_message(prompt)

            return await self.message(query)

        except Exception as e:
            self.logger.error(f"Simple completion failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def analyze_text(
        self, text: str, analysis_type: str = "general", instructions: Optional[str] = None
    ) -> AnthropicResponse:
        """Analyze text with Claude."""
        try:
            analysis_prompts = {
                "sentiment": "Analyze the sentiment of this text. Provide a detailed assessment including emotional tone, confidence level, and key indicators.",
                "summary": "Provide a concise but comprehensive summary of this text, highlighting the main points and key takeaways.",
                "keywords": "Extract the most important keywords and key phrases from this text, organizing them by relevance and topic.",
                "tone": "Analyze the tone and writing style of this text, including formality level, target audience, and rhetorical techniques used.",
                "general": "Provide a comprehensive analysis of this text, including its main themes, structure, key insights, and overall assessment.",
            }

            system_prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
            if instructions:
                system_prompt += f"\n\nAdditional instructions: {instructions}"

            prompt = f"Please analyze the following text:\n\n{text}"

            return await self.simple_completion(prompt=prompt, system_prompt=system_prompt)

        except Exception as e:
            self.logger.error(f"Text analysis failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def generate_creative_content(
        self,
        content_type: str,
        topic: str,
        style: Optional[str] = None,
        length: Optional[str] = None,
        additional_requirements: Optional[str] = None,
    ) -> AnthropicResponse:
        """Generate creative content with Claude."""
        try:
            content_prompts = {
                "story": "Write a creative story",
                "poem": "Write a poem",
                "essay": "Write an essay",
                "article": "Write an article",
                "dialogue": "Write a dialogue",
                "script": "Write a script",
                "blog_post": "Write a blog post",
            }

            base_prompt = content_prompts.get(content_type, "Write creative content")
            prompt = f"{base_prompt} about: {topic}"

            if style:
                prompt += f"\nStyle: {style}"

            if length:
                prompt += f"\nLength: {length}"

            if additional_requirements:
                prompt += f"\nAdditional requirements: {additional_requirements}"

            system_prompt = f"You are a creative writer specializing in {content_type}. Create engaging, original, and high-quality content that meets the specified requirements."

            return await self.simple_completion(
                prompt=prompt, system_prompt=system_prompt, temperature=0.8  # Higher temperature for creativity
            )

        except Exception as e:
            self.logger.error(f"Creative content generation failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def reasoning_task(
        self, problem: str, task_type: str = "general", step_by_step: bool = True
    ) -> AnthropicResponse:
        """Perform reasoning tasks with Claude."""
        try:
            reasoning_prompts = {
                "math": "Solve this mathematical problem step by step, showing all work and explaining your reasoning.",
                "logic": "Analyze this logical problem systematically, identifying premises, conclusions, and any logical fallacies.",
                "analysis": "Break down this complex problem into components and analyze each part systematically.",
                "general": "Think through this problem carefully and provide a reasoned solution.",
            }

            base_prompt = reasoning_prompts.get(task_type, reasoning_prompts["general"])

            if step_by_step:
                base_prompt += " Use clear step-by-step reasoning and explain your thought process."

            prompt = f"{base_prompt}\n\nProblem: {problem}"

            system_prompt = "You are an expert problem solver. Approach each problem methodically, use clear reasoning, and provide detailed explanations for your conclusions."

            return await self.simple_completion(
                prompt=prompt, system_prompt=system_prompt, temperature=0.3  # Lower temperature for reasoning
            )

        except Exception as e:
            self.logger.error(f"Reasoning task failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self._usage_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic API access with a minimal request
            test_response = await self.simple_completion(prompt="Hello", max_tokens=10)

            return {
                'status': 'healthy',
                'framework': 'anthropic',
                'base_url': self.config.base_url,
                'default_model': self.config.default_model.value,
                'usage_stats': self._usage_stats,
                'test_successful': test_response.success,
            }

        except Exception as e:
            return {'status': 'unhealthy', 'framework': 'anthropic', 'error': str(e)}
