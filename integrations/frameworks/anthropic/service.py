"""
Anthropic Service

Service layer for Anthropic Claude operations with conversation management,
advanced prompting, and intelligent task routing.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from integrations.frameworks.base import BaseService
from observability.logging import get_logger

from .client import AnthropicClient
from .models import (
    AnthropicConfig,
    AnthropicConversation,
    AnthropicMessage,
    AnthropicMessageRole,
    AnthropicModel,
    AnthropicQuery,
    AnthropicResponse,
)


class AnthropicService(BaseService):
    """Anthropic service with advanced conversation and task management."""

    def __init__(self, config: AnthropicConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)
        self.client = AnthropicClient(config)

        # Response cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(hours=1)

        # Conversation management
        self._conversations: Dict[str, AnthropicConversation] = {}

        # Task-specific prompts
        self._system_prompts = {
            "assistant": "You are Claude, a helpful AI assistant created by Anthropic. You are thoughtful, nuanced, and provide detailed responses.",
            "analyst": "You are an expert analyst. You break down complex problems systematically and provide data-driven insights.",
            "writer": "You are a professional writer. You create clear, engaging, and well-structured content tailored to your audience.",
            "researcher": "You are a research assistant. You gather, analyze, and synthesize information to provide comprehensive answers.",
            "tutor": "You are an educational tutor. You explain concepts clearly, ask probing questions, and adapt to the learner's level.",
            "creative": "You are a creative AI. You think outside the box and generate original, innovative ideas and content.",
            "technical": "You are a technical expert. You provide precise, accurate information and explain complex technical concepts clearly.",
        }

        # Model selection for different tasks
        self._task_models = {
            "complex_reasoning": AnthropicModel.CLAUDE_3_OPUS,
            "creative_writing": AnthropicModel.CLAUDE_3_SONNET,
            "code_analysis": AnthropicModel.CLAUDE_3_SONNET,
            "general_chat": AnthropicModel.CLAUDE_3_SONNET,
            "quick_tasks": AnthropicModel.CLAUDE_3_HAIKU,
            "analysis": AnthropicModel.CLAUDE_3_OPUS,
            "summarization": AnthropicModel.CLAUDE_3_HAIKU,
        }

    def _get_cache_key(self, query: AnthropicQuery) -> str:
        """Generate cache key for query."""
        cache_data = {
            "model": query.model.value if query.model else self.config.default_model.value,
            "messages": [{"role": msg.role.value, "content": msg.content[:100]} for msg in query.messages],
            "system_prompt": query.system_prompt,
            "temperature": query.temperature,
            "max_tokens": query.max_tokens,
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"anthropic:{hashlib.md5(cache_string.encode()).hexdigest()[:16]}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False

        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.now() - cached_time < self._cache_ttl

    def _get_from_cache(self, query: AnthropicQuery) -> Optional[AnthropicResponse]:
        """Get response from cache if available."""
        cache_key = self._get_cache_key(query)

        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                self.logger.debug(f"Cache hit for query: {cache_key}")
                response_data = cache_entry["response"]
                return AnthropicResponse(**response_data)

        return None

    def _store_in_cache(self, query: AnthropicQuery, response: AnthropicResponse):
        """Store response in cache."""
        if not response.success:
            return  # Don't cache failed responses

        cache_key = self._get_cache_key(query)
        cache_entry = {"timestamp": datetime.now().isoformat(), "response": response.__dict__}

        self._cache[cache_key] = cache_entry
        self.logger.debug(f"Cached response for query: {cache_key}")

    def _select_model_for_task(self, task_type: str) -> AnthropicModel:
        """Select appropriate model for task type."""
        return self._task_models.get(task_type, self.config.default_model)

    async def query(self, query: AnthropicQuery) -> AnthropicResponse:
        """Execute Anthropic query with caching."""
        try:
            self.logger.info("Processing Anthropic query")

            # Check cache first
            cached_response = self._get_from_cache(query)
            if cached_response:
                return cached_response

            # Execute query
            response = await self.client.message(query)

            # Cache successful responses
            if response.success:
                self._store_in_cache(query, response)

            return response

        except Exception as e:
            self.logger.error(f"Anthropic query failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def chat(
        self,
        message: str,
        conversation_id: str,
        system_prompt: Optional[str] = None,
        assistant_type: str = "assistant",
        model: Optional[AnthropicModel] = None,
    ) -> AnthropicResponse:
        """Chat with conversation history management."""
        try:
            # Get or create conversation
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = AnthropicConversation(
                    conversation_id=conversation_id,
                    system_prompt=system_prompt
                    or self._system_prompts.get(assistant_type, self._system_prompts["assistant"]),
                )

            conversation = self._conversations[conversation_id]

            # Add user message
            conversation.add_user_message(message)

            # Create query
            query = AnthropicQuery(
                model=model or self._select_model_for_task("general_chat"),
                messages=conversation.messages,
                system_prompt=conversation.system_prompt,
            )

            # Execute query
            response = await self.client.message(query)

            # Add assistant response to conversation
            if response.success and response.message:
                conversation.add_assistant_message(response.message.content)

            return response

        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def analyze_document(
        self, document_text: str, analysis_type: str = "comprehensive", focus_areas: Optional[List[str]] = None
    ) -> AnthropicResponse:
        """Analyze documents with specialized prompting."""
        try:
            analysis_prompts = {
                "comprehensive": "Provide a comprehensive analysis of this document, including key themes, main arguments, structure, and conclusions.",
                "summary": "Provide a concise summary highlighting the most important points and key takeaways.",
                "critique": "Provide a critical analysis, evaluating strengths, weaknesses, and areas for improvement.",
                "fact_check": "Analyze this document for factual accuracy, identifying claims that need verification.",
                "sentiment": "Analyze the emotional tone and sentiment throughout this document.",
                "bias": "Identify potential biases, assumptions, and perspectives in this document.",
            }

            prompt = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])

            if focus_areas:
                prompt += f"\n\nPay special attention to these areas: {', '.join(focus_areas)}"

            prompt += f"\n\nDocument to analyze:\n\n{document_text}"

            query = AnthropicQuery(
                model=self._select_model_for_task("analysis"), system_prompt=self._system_prompts["analyst"]
            )
            query.add_user_message(prompt)

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def creative_writing(
        self,
        prompt: str,
        genre: str = "general",
        style: Optional[str] = None,
        length: str = "medium",
        additional_constraints: Optional[List[str]] = None,
    ) -> AnthropicResponse:
        """Generate creative content with style control."""
        try:
            genre_prompts = {
                "fiction": "Write a piece of fiction",
                "poetry": "Write a poem",
                "screenplay": "Write a screenplay scene",
                "dialogue": "Write a dialogue",
                "monologue": "Write a dramatic monologue",
                "essay": "Write a creative essay",
                "story": "Write a short story",
            }

            base_prompt = genre_prompts.get(genre, "Write creative content")

            full_prompt = f"{base_prompt} based on this prompt: {prompt}"

            if style:
                full_prompt += f"\n\nStyle: {style}"

            length_guidance = {
                "short": "Keep it concise (1-2 paragraphs)",
                "medium": "Write a medium-length piece (3-5 paragraphs)",
                "long": "Write an extended piece (6+ paragraphs)",
            }

            if length in length_guidance:
                full_prompt += f"\n\nLength: {length_guidance[length]}"

            if additional_constraints:
                full_prompt += f"\n\nAdditional constraints: {', '.join(additional_constraints)}"

            query = AnthropicQuery(
                model=self._select_model_for_task("creative_writing"),
                system_prompt=self._system_prompts["creative"],
                temperature=0.8,  # Higher creativity
            )
            query.add_user_message(full_prompt)

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Creative writing failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def solve_problem(
        self, problem: str, domain: str = "general", approach: str = "systematic", show_reasoning: bool = True
    ) -> AnthropicResponse:
        """Solve complex problems with structured reasoning."""
        try:
            domain_contexts = {
                "math": "You are solving a mathematical problem. Use precise mathematical reasoning and show all steps.",
                "science": "You are solving a scientific problem. Apply scientific principles and methodology.",
                "business": "You are solving a business problem. Consider practical constraints, costs, and stakeholder impacts.",
                "technical": "You are solving a technical problem. Focus on implementation details and technical feasibility.",
                "ethical": "You are addressing an ethical dilemma. Consider multiple perspectives and moral frameworks.",
                "general": "You are solving a complex problem. Use logical reasoning and systematic analysis.",
            }

            approach_instructions = {
                "systematic": "Break down the problem into steps and solve systematically.",
                "creative": "Think creatively and consider unconventional solutions.",
                "analytical": "Use data-driven analysis and evidence-based reasoning.",
                "collaborative": "Consider multiple perspectives and stakeholder viewpoints.",
            }

            system_context = domain_contexts.get(domain, domain_contexts["general"])
            approach_instruction = approach_instructions.get(approach, approach_instructions["systematic"])

            prompt = f"{approach_instruction}\n\nProblem to solve: {problem}"

            if show_reasoning:
                prompt += "\n\nPlease show your reasoning process and explain how you arrived at your solution."

            query = AnthropicQuery(
                model=self._select_model_for_task("complex_reasoning"),
                system_prompt=system_context,
                temperature=0.4,  # Balanced creativity and precision
            )
            query.add_user_message(prompt)

            return await self.query(query)

        except Exception as e:
            self.logger.error(f"Problem solving failed: {e}")
            return AnthropicResponse(success=False, error=str(e), framework="anthropic")

    async def batch_process(self, tasks: List[Dict[str, Any]], batch_size: int = 5) -> List[AnthropicResponse]:
        """Process multiple tasks in batches."""
        try:
            responses = []

            # Process in batches
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                batch_responses = []

                # Create queries for batch
                batch_queries = []
                for task in batch:
                    query = AnthropicQuery(
                        model=task.get("model"),
                        system_prompt=task.get("system_prompt"),
                        temperature=task.get("temperature"),
                    )
                    query.add_user_message(task["message"])
                    batch_queries.append(query)

                # Execute batch concurrently
                batch_tasks = [self.query(query) for query in batch_queries]
                batch_responses = await asyncio.gather(*batch_tasks)

                responses.extend(batch_responses)

                # Small delay between batches
                await asyncio.sleep(0.5)

            return responses

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [AnthropicResponse(success=False, error=str(e), framework="anthropic") for _ in tasks]

    def get_conversation(self, conversation_id: str) -> Optional[AnthropicConversation]:
        """Get conversation by ID."""
        return self._conversations.get(conversation_id)

    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self._conversations.keys())

    def clear_conversation(self, conversation_id: str):
        """Clear a specific conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]

    def clear_all_conversations(self):
        """Clear all conversations."""
        self._conversations.clear()

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
        self.logger.info("Anthropic response cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        client_health = await self.client.health_check()
        cache_stats = await self.get_cache_stats()

        return {
            **client_health,
            "service_type": "anthropic_service",
            "active_conversations": len(self._conversations),
            "cache_stats": cache_stats,
            "available_models": [model.value for model in AnthropicModel],
            "supported_tasks": list(self._task_models.keys()),
            "assistant_types": list(self._system_prompts.keys()),
        }
