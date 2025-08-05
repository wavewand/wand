"""
Core AI/ML platform integrations for Wand
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class HuggingFaceIntegration(BaseIntegration):
    """HuggingFace Hub and Inference API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_token": os.getenv("HUGGINGFACE_TOKEN", ""),
            "base_url": "https://api-inference.huggingface.co",
            "hub_url": "https://huggingface.co/api",
            "default_model": "gpt2",
            "max_tokens": 100,
            "temperature": 0.7,
        }
        super().__init__("huggingface", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize HuggingFace integration"""
        if not self.config["api_token"]:
            logger.warning("⚠️  HuggingFace token not configured - limited functionality")
        logger.info("✅ HuggingFace integration initialized")

    async def cleanup(self):
        """Cleanup HuggingFace resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check HuggingFace API health"""
        try:
            headers = {}
            if self.config["api_token"]:
                headers["Authorization"] = f"Bearer {self.config['api_token']}"

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.hub_url}/whoami", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy", "authenticated": bool(self.config["api_token"])}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute HuggingFace operations"""

        if operation == "generate_text":
            return await self._generate_text(**kwargs)
        elif operation == "classify_text":
            return await self._classify_text(**kwargs)
        elif operation == "embed_text":
            return await self._embed_text(**kwargs)
        elif operation == "search_models":
            return await self._search_models(**kwargs)
        elif operation == "get_model_info":
            return await self._get_model_info(**kwargs)
        elif operation == "generate_image":
            return await self._generate_image(**kwargs)
        elif operation == "transcribe_audio":
            return await self._transcribe_audio(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate text using HuggingFace model"""
        model = model or self.config["default_model"]
        max_tokens = max_tokens or self.config["max_tokens"]
        temperature = temperature or self.config["temperature"]

        headers = {"Content-Type": "application/json"}
        if self.config["api_token"]:
            headers["Authorization"] = f"Bearer {self.config['api_token']}"

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature, "return_full_text": False},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/models/{model}", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Handle different response formats
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                        else:
                            generated_text = str(result)

                        return {
                            "success": True,
                            "generated_text": generated_text,
                            "model": model,
                            "prompt": prompt,
                            "parameters": {"max_tokens": max_tokens, "temperature": temperature},
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _classify_text(
        self, text: str, model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ) -> Dict[str, Any]:
        """Classify text using HuggingFace model"""
        headers = {"Content-Type": "application/json"}
        if self.config["api_token"]:
            headers["Authorization"] = f"Bearer {self.config['api_token']}"

        payload = {"inputs": text}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/models/{model}", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        return {"success": True, "classifications": result, "model": model, "text": text}
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _embed_text(self, text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
        """Generate embeddings for text"""
        headers = {"Content-Type": "application/json"}
        if self.config["api_token"]:
            headers["Authorization"] = f"Bearer {self.config['api_token']}"

        payload = {"inputs": text}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/models/{model}", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        return {
                            "success": True,
                            "embeddings": result,
                            "model": model,
                            "text": text,
                            "dimensions": len(result) if isinstance(result, list) else None,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _search_models(self, query: str, task: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """Search HuggingFace model hub"""
        params = {"search": query, "limit": limit, "full": "true"}
        if task:
            params["pipeline_tag"] = task

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.hub_url}/models", params=params) as response:
                    if response.status == 200:
                        models = await response.json()

                        model_list = []
                        for model in models:
                            model_list.append(
                                {
                                    "id": model.get("id", ""),
                                    "author": model.get("author", ""),
                                    "pipeline_tag": model.get("pipeline_tag", ""),
                                    "downloads": model.get("downloads", 0),
                                    "likes": model.get("likes", 0),
                                    "created_at": model.get("createdAt", ""),
                                    "description": model.get("description", ""),
                                }
                            )

                        return {
                            "success": True,
                            "models": model_list,
                            "query": query,
                            "task": task,
                            "total": len(model_list),
                        }
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.hub_url}/models/{model_id}") as response:
                    if response.status == 200:
                        model_info = await response.json()

                        return {
                            "success": True,
                            "model": {
                                "id": model_info.get("id", ""),
                                "author": model_info.get("author", ""),
                                "pipeline_tag": model_info.get("pipeline_tag", ""),
                                "downloads": model_info.get("downloads", 0),
                                "likes": model_info.get("likes", 0),
                                "library_name": model_info.get("library_name", ""),
                                "tags": model_info.get("tags", []),
                                "created_at": model_info.get("createdAt", ""),
                                "last_modified": model_info.get("lastModified", ""),
                                "description": model_info.get("description", ""),
                            },
                        }
                    else:
                        return {"success": False, "error": f"Model not found or API error: {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


class OpenAIIntegration(BaseIntegration):
    """OpenAI API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-3.5-turbo",
            "max_tokens": 100,
            "temperature": 0.7,
        }
        super().__init__("openai", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize OpenAI integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  OpenAI API key not configured")
            raise Exception("OpenAI API key required")
        logger.info("✅ OpenAI integration initialized")

    async def cleanup(self):
        """Cleanup OpenAI resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health"""
        try:
            headers = {"Authorization": f"Bearer {self.config['api_key']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/models", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute OpenAI operations"""

        if operation == "chat":
            return await self._chat_completion(**kwargs)
        elif operation == "complete":
            return await self._text_completion(**kwargs)
        elif operation == "generate_image":
            return await self._generate_image(**kwargs)
        elif operation == "transcribe":
            return await self._transcribe_audio(**kwargs)
        elif operation == "embed":
            return await self._create_embeddings(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create chat completion"""
        headers = {"Authorization": f"Bearer {self.config['api_key']}", "Content-Type": "application/json"}

        payload = {
            "model": model or self.config["default_model"],
            "messages": messages,
            "max_tokens": max_tokens or self.config["max_tokens"],
            "temperature": temperature or self.config["temperature"],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/chat/completions", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        return {
                            "success": True,
                            "response": result["choices"][0]["message"]["content"],
                            "model": result["model"],
                            "usage": result["usage"],
                            "finish_reason": result["choices"][0]["finish_reason"],
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", {}).get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_image(
        self, prompt: str, size: str = "1024x1024", quality: str = "standard", n: int = 1
    ) -> Dict[str, Any]:
        """Generate image using DALL-E"""
        headers = {"Authorization": f"Bearer {self.config['api_key']}", "Content-Type": "application/json"}

        payload = {"model": "dall-e-3", "prompt": prompt, "size": size, "quality": quality, "n": n}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/images/generations", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        return {
                            "success": True,
                            "images": result["data"],
                            "prompt": prompt,
                            "size": size,
                            "quality": quality,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", {}).get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class AnthropicIntegration(BaseIntegration):
    """Anthropic Claude API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "base_url": "https://api.anthropic.com/v1",
            "default_model": "claude-3-sonnet-20240229",
            "max_tokens": 100,
        }
        super().__init__("anthropic", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Anthropic integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  Anthropic API key not configured")
            raise Exception("Anthropic API key required")
        logger.info("✅ Anthropic integration initialized")

    async def cleanup(self):
        """Cleanup Anthropic resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health"""
        return {"status": "healthy"}  # Simplified for demo

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Anthropic operations"""

        if operation == "chat":
            return await self._create_message(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_message(
        self, messages: List[Dict[str, str]], model: Optional[str] = None, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create message with Claude"""
        headers = {
            "x-api-key": self.config["api_key"],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": model or self.config["default_model"],
            "messages": messages,
            "max_tokens": max_tokens or self.config["max_tokens"],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/messages", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        return {
                            "success": True,
                            "response": result["content"][0]["text"],
                            "model": result["model"],
                            "usage": result["usage"],
                            "stop_reason": result["stop_reason"],
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", {}).get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class CohereIntegration(BaseIntegration):
    """Cohere API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("COHERE_API_KEY", ""),
            "base_url": "https://api.cohere.ai/v1",
            "default_model": "command",
            "max_tokens": 100,
        }
        super().__init__("cohere", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Cohere integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  Cohere API key not configured")
        logger.info("✅ Cohere integration initialized")

    async def cleanup(self):
        """Cleanup Cohere resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Cohere API health"""
        if not self.config["api_key"]:
            return {"status": "unhealthy", "error": "API key not configured"}
        return {"status": "healthy"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Cohere operations"""

        if operation == "generate":
            return await self._generate_text(**kwargs)
        elif operation == "embed":
            return await self._embed_text(**kwargs)
        elif operation == "classify":
            return await self._classify_text(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _generate_text(
        self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate text with Cohere"""
        headers = {"Authorization": f"Bearer {self.config['api_key']}", "Content-Type": "application/json"}

        payload = {
            "model": model or self.config["default_model"],
            "prompt": prompt,
            "max_tokens": max_tokens or self.config["max_tokens"],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/generate", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        return {
                            "success": True,
                            "generated_text": result["generations"][0]["text"],
                            "model": model or self.config["default_model"],
                            "prompt": prompt,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class ReplicateIntegration(BaseIntegration):
    """Replicate API integration for running ML models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"api_token": os.getenv("REPLICATE_API_TOKEN", ""), "base_url": "https://api.replicate.com/v1"}
        super().__init__("replicate", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Replicate integration"""
        if not self.config["api_token"]:
            logger.warning("⚠️  Replicate API token not configured")
        logger.info("✅ Replicate integration initialized")

    async def cleanup(self):
        """Cleanup Replicate resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Replicate API health"""
        if not self.config["api_token"]:
            return {"status": "unhealthy", "error": "API token not configured"}

        try:
            headers = {"Authorization": f"Token {self.config['api_token']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/models", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Replicate operations"""

        if operation == "run_model":
            return await self._run_model(**kwargs)
        elif operation == "get_prediction":
            return await self._get_prediction(**kwargs)
        elif operation == "list_models":
            return await self._list_models(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _run_model(self, model: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a model on Replicate"""
        headers = {"Authorization": f"Token {self.config['api_token']}", "Content-Type": "application/json"}

        payload = {"version": model, "input": input_data}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/predictions", headers=headers, json=payload
                ) as response:
                    if response.status == 201:
                        result = await response.json()

                        return {
                            "success": True,
                            "prediction_id": result["id"],
                            "status": result["status"],
                            "model": model,
                            "input": input_data,
                            "urls": result["urls"],
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("detail", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}
