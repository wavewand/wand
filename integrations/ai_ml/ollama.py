"""
Ollama local AI model integration for Wand
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class OllamaIntegration(BaseIntegration):
    """Ollama local AI model server integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "api_url": os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1"),
            "default_model": os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M"),
            "timeout": int(os.getenv("OLLAMA_TIMEOUT", "43200")),  # 12 hours for generation
            "client_timeout": int(os.getenv("OLLAMA_CLIENT_TIMEOUT", "43200")),  # 12 hours client timeout
            "health_timeout": int(os.getenv("OLLAMA_HEALTH_TIMEOUT", "30")),  # 30s for health checks
            "pull_timeout": int(os.getenv("OLLAMA_PULL_TIMEOUT", "3600")),  # 1 hour for model pulls
            "temperature": 0.7,
            "max_tokens": 8192,
        }
        super().__init__("ollama", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Ollama integration"""
        logger.info("✅ Ollama integration initialized")
        await self._check_server_health()

    async def cleanup(self):
        """Cleanup Ollama resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama server health"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if server is running
                async with session.get(
                    f"{self.config['base_url']}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=self.config["health_timeout"]),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]

                        return {
                            "status": "healthy",
                            "server": self.config["base_url"],
                            "available_models": models,
                            "default_model": self.config["default_model"],
                            "model_count": len(models),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"Server returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _check_server_health(self):
        """Internal health check during initialization"""
        health = await self.health_check()
        if health["status"] == "healthy":
            logger.info(f"🔗 Ollama server connected: {len(health['available_models'])} models available")
        else:
            logger.warning(f"⚠️  Ollama server issue: {health.get('error', 'Unknown error')}")

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Ollama operations"""

        if operation == "generate":
            return await self._generate(**kwargs)
        elif operation == "chat":
            return await self._chat(**kwargs)
        elif operation == "embed":
            return await self._embed(**kwargs)
        elif operation == "list_models":
            return await self._list_models(**kwargs)
        elif operation == "pull_model":
            return await self._pull_model(**kwargs)
        elif operation == "show_model":
            return await self._show_model(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _generate(
        self, prompt: Optional[str] = None, model: Optional[str] = None, stream: bool = False, **options
    ) -> Dict[str, Any]:
        """Generate text completion using Ollama"""
        if not prompt:
            return {"success": False, "error": "prompt parameter is required for generate operation"}

        model = model or self.config["default_model"]

        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": options.get("temperature", self.config["temperature"]),
                "num_ctx": options.get("num_ctx", 32768),
                "num_predict": options.get("max_tokens", self.config["max_tokens"]),
            },
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.config["client_timeout"])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.config['base_url']}/api/generate", json=data, headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "model": model,
                            "prompt": prompt,
                            "response": result.get("response", ""),
                            "done": result.get("done", True),
                            "total_duration": result.get("total_duration"),
                            "load_duration": result.get("load_duration"),
                            "prompt_eval_count": result.get("prompt_eval_count"),
                            "eval_count": result.get("eval_count"),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Ollama API error {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}

    async def _chat(
        self, messages: Optional[List[Dict[str, str]]] = None, model: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        """Chat completion using Ollama (OpenAI-compatible endpoint)"""
        if not messages:
            return {"success": False, "error": "messages parameter is required for chat operation"}

        model = model or self.config["default_model"]

        data = {
            "model": model,
            "messages": messages,
            "temperature": options.get("temperature", self.config["temperature"]),
            "max_tokens": options.get("max_tokens", self.config["max_tokens"]),
            "stream": options.get("stream", False),
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.config["client_timeout"])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.config['api_url']}/chat/completions",
                    json=data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        choice = result["choices"][0] if result.get("choices") else {}

                        return {
                            "success": True,
                            "model": model,
                            "messages": messages,
                            "response": choice.get("message", {}).get("content", ""),
                            "finish_reason": choice.get("finish_reason"),
                            "usage": result.get("usage", {}),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Ollama chat API error {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Chat request failed: {str(e)}"}

    async def _embed(self, input_text: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate embeddings using Ollama"""
        if not input_text:
            return {"success": False, "error": "input_text parameter is required for embed operation"}

        model = model or "nomic-embed-text"  # Common embedding model

        data = {"model": model, "prompt": input_text}

        try:
            timeout = aiohttp.ClientTimeout(total=self.config["client_timeout"])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.config['base_url']}/api/embeddings", json=data, headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "model": model,
                            "input": input_text,
                            "embedding": result.get("embedding", []),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Ollama embeddings error {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Embeddings request failed: {str(e)}"}

    async def _list_models(self, **kwargs) -> Dict[str, Any]:
        """List available models on Ollama server"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.config["health_timeout"])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.config['base_url']}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []

                        for model in data.get("models", []):
                            models.append(
                                {
                                    "name": model.get("name"),
                                    "modified_at": model.get("modified_at"),
                                    "size": model.get("size"),
                                    "digest": model.get("digest"),
                                    "details": model.get("details", {}),
                                }
                            )

                        return {
                            "success": True,
                            "models": models,
                            "count": len(models),
                            "server": self.config["base_url"],
                        }
                    else:
                        return {"success": False, "error": f"Failed to list models: {response.status}"}
        except Exception as e:
            return {"success": False, "error": f"List models failed: {str(e)}"}

    async def _pull_model(self, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Pull/download a model to Ollama server"""
        if not model_name:
            return {"success": False, "error": "model_name parameter is required for pull_model operation"}

        data = {"name": model_name}

        try:
            timeout = aiohttp.ClientTimeout(total=self.config["pull_timeout"])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.config['base_url']}/api/pull", json=data) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "model": model_name,
                            "status": "pulled",
                            "message": f"Model {model_name} pulled successfully",
                        }
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Pull failed {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": f"Pull model failed: {str(e)}"}

    async def _show_model(self, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Show detailed information about a model"""
        if not model_name:
            return {"success": False, "error": "model_name parameter is required for show_model operation"}

        data = {"name": model_name}

        try:
            timeout = aiohttp.ClientTimeout(total=self.config["health_timeout"])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.config['base_url']}/api/show", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "model": model_name,
                            "license": result.get("license"),
                            "modelfile": result.get("modelfile"),
                            "parameters": result.get("parameters"),
                            "template": result.get("template"),
                            "details": result.get("details", {}),
                            "modified_at": result.get("modified_at"),
                        }
                    else:
                        return {"success": False, "error": f"Show model failed: {response.status}"}
        except Exception as e:
            return {"success": False, "error": f"Show model failed: {str(e)}"}
