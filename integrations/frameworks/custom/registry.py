"""
Framework Registry

Registry system for managing custom framework adapter definitions
and providing a central repository of available adapters.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from observability.logging import get_logger

from .models import (
    AdapterDefinition,
    AuthenticationType,
    CustomFrameworkConfig,
    EndpointDefinition,
    ParameterDefinition,
    ParameterType,
    RequestMethod,
)


class FrameworkRegistry:
    """Registry for custom framework adapters."""

    def __init__(self, registry_path: Optional[str] = None):
        self.logger = get_logger(__name__)

        # Registry storage
        self.registry_path = registry_path or "./config/framework_registry.json"
        self.adapters: Dict[str, AdapterDefinition] = {}

        # Built-in adapters
        self._builtin_adapters: Dict[str, AdapterDefinition] = {}

        # Load registry
        self._load_registry()
        self._load_builtin_adapters()

        self.logger.info(f"Framework registry initialized with {len(self.adapters)} adapters")

    def _load_registry(self):
        """Load adapter definitions from registry file."""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)

                for adapter_data in registry_data.get("adapters", []):
                    adapter = self._deserialize_adapter(adapter_data)
                    self.adapters[adapter.name] = adapter

                self.logger.info(f"Loaded {len(self.adapters)} adapters from registry")
            else:
                self.logger.info("Registry file not found, starting with empty registry")

        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save adapter definitions to registry file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

            registry_data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "adapters": [self._serialize_adapter(adapter) for adapter in self.adapters.values()],
            }

            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)

            self.logger.info(f"Saved {len(self.adapters)} adapters to registry")

        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")

    def _serialize_adapter(self, adapter: AdapterDefinition) -> Dict[str, Any]:
        """Serialize adapter definition to dictionary."""
        return {
            "name": adapter.name,
            "version": adapter.version,
            "description": adapter.description,
            "base_url": adapter.base_url,
            "authentication": adapter.authentication.value,
            "auth_config": adapter.auth_config,
            "default_headers": adapter.default_headers,
            "rate_limit": adapter.rate_limit,
            "tags": adapter.tags,
            "created_at": adapter.created_at.isoformat() if adapter.created_at else None,
            "endpoints": {name: self._serialize_endpoint(endpoint) for name, endpoint in adapter.endpoints.items()},
        }

    def _serialize_endpoint(self, endpoint: EndpointDefinition) -> Dict[str, Any]:
        """Serialize endpoint definition to dictionary."""
        return {
            "name": endpoint.name,
            "path": endpoint.path,
            "method": endpoint.method.value,
            "description": endpoint.description,
            "headers": endpoint.headers,
            "timeout": endpoint.timeout,
            "response_key": endpoint.response_key,
            "success_codes": endpoint.success_codes,
            "parameters": [self._serialize_parameter(param) for param in endpoint.parameters],
        }

    def _serialize_parameter(self, parameter: ParameterDefinition) -> Dict[str, Any]:
        """Serialize parameter definition to dictionary."""
        return {
            "name": parameter.name,
            "type": parameter.type.value,
            "required": parameter.required,
            "default_value": parameter.default_value,
            "description": parameter.description,
            "min_value": parameter.min_value,
            "max_value": parameter.max_value,
            "allowed_values": parameter.allowed_values,
            "pattern": parameter.pattern,
            "location": parameter.location,
        }

    def _deserialize_adapter(self, data: Dict[str, Any]) -> AdapterDefinition:
        """Deserialize adapter definition from dictionary."""
        adapter = AdapterDefinition(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            base_url=data["base_url"],
            authentication=AuthenticationType(data.get("authentication", "none")),
            auth_config=data.get("auth_config", {}),
            default_headers=data.get("default_headers", {}),
            rate_limit=data.get("rate_limit"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )

        # Deserialize endpoints
        for endpoint_data in data.get("endpoints", {}).values():
            endpoint = self._deserialize_endpoint(endpoint_data)
            adapter.add_endpoint(endpoint)

        return adapter

    def _deserialize_endpoint(self, data: Dict[str, Any]) -> EndpointDefinition:
        """Deserialize endpoint definition from dictionary."""
        endpoint = EndpointDefinition(
            name=data["name"],
            path=data["path"],
            method=RequestMethod(data.get("method", "POST")),
            description=data.get("description"),
            headers=data.get("headers", {}),
            timeout=data.get("timeout", 30.0),
            response_key=data.get("response_key"),
            success_codes=data.get("success_codes", [200, 201]),
        )

        # Deserialize parameters
        for param_data in data.get("parameters", []):
            parameter = self._deserialize_parameter(param_data)
            endpoint.parameters.append(parameter)

        return endpoint

    def _deserialize_parameter(self, data: Dict[str, Any]) -> ParameterDefinition:
        """Deserialize parameter definition from dictionary."""
        return ParameterDefinition(
            name=data["name"],
            type=ParameterType(data["type"]),
            required=data.get("required", False),
            default_value=data.get("default_value"),
            description=data.get("description"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            allowed_values=data.get("allowed_values"),
            pattern=data.get("pattern"),
            location=data.get("location", "body"),
        )

    def _load_builtin_adapters(self):
        """Load built-in adapter definitions."""
        # Example: OpenAI-compatible API adapter
        openai_adapter = self._create_openai_adapter()
        self._builtin_adapters["openai_compatible"] = openai_adapter

        # Example: Generic REST API adapter
        rest_adapter = self._create_generic_rest_adapter()
        self._builtin_adapters["generic_rest"] = rest_adapter

        self.logger.info(f"Loaded {len(self._builtin_adapters)} built-in adapters")

    def _create_openai_adapter(self) -> AdapterDefinition:
        """Create OpenAI-compatible API adapter definition."""
        adapter = AdapterDefinition(
            name="openai_compatible",
            version="1.0",
            description="Generic OpenAI-compatible API adapter",
            base_url="https://api.openai.com/v1",
            authentication=AuthenticationType.BEARER_TOKEN,
            auth_config={"token_header": "Authorization"},
            default_headers={"Content-Type": "application/json"},
            tags=["llm", "text-generation", "openai"],
        )

        # Chat completions endpoint
        chat_endpoint = EndpointDefinition(
            name="chat_completions",
            path="/chat/completions",
            method=RequestMethod.POST,
            description="Create a chat completion",
            response_key="choices",
        )

        chat_endpoint.parameters = [
            ParameterDefinition("model", ParameterType.STRING, required=True, description="Model to use"),
            ParameterDefinition("messages", ParameterType.LIST, required=True, description="List of messages"),
            ParameterDefinition("temperature", ParameterType.FLOAT, default_value=1.0, min_value=0.0, max_value=2.0),
            ParameterDefinition("max_tokens", ParameterType.INTEGER, min_value=1),
            ParameterDefinition("stream", ParameterType.BOOLEAN, default_value=False),
        ]

        adapter.add_endpoint(chat_endpoint)

        # Embeddings endpoint
        embeddings_endpoint = EndpointDefinition(
            name="embeddings",
            path="/embeddings",
            method=RequestMethod.POST,
            description="Create embeddings",
            response_key="data",
        )

        embeddings_endpoint.parameters = [
            ParameterDefinition("model", ParameterType.STRING, required=True),
            ParameterDefinition("input", ParameterType.STRING, required=True),
            ParameterDefinition("encoding_format", ParameterType.STRING, default_value="float"),
        ]

        adapter.add_endpoint(embeddings_endpoint)

        return adapter

    def _create_generic_rest_adapter(self) -> AdapterDefinition:
        """Create generic REST API adapter definition."""
        adapter = AdapterDefinition(
            name="generic_rest",
            version="1.0",
            description="Generic REST API adapter template",
            base_url="https://api.example.com",
            authentication=AuthenticationType.API_KEY,
            auth_config={"api_key_header": "X-API-Key"},
            default_headers={"Content-Type": "application/json"},
            tags=["generic", "rest", "template"],
        )

        # Generic POST endpoint
        post_endpoint = EndpointDefinition(
            name="post_data", path="/api/data", method=RequestMethod.POST, description="Submit data to API"
        )

        post_endpoint.parameters = [
            ParameterDefinition("data", ParameterType.DICT, required=True, description="Data to submit"),
            ParameterDefinition("format", ParameterType.STRING, default_value="json", allowed_values=["json", "xml"]),
        ]

        adapter.add_endpoint(post_endpoint)

        return adapter

    def register_adapter(self, adapter: AdapterDefinition, save_to_file: bool = True):
        """Register a new adapter definition."""
        if not adapter.validate_auth_config():
            raise ValueError(f"Invalid authentication configuration for adapter: {adapter.name}")

        self.adapters[adapter.name] = adapter

        if save_to_file:
            self._save_registry()

        self.logger.info(f"Registered adapter: {adapter.name}")

    def get_adapter(self, name: str) -> Optional[AdapterDefinition]:
        """Get an adapter definition by name."""
        # Check user-defined adapters first
        if name in self.adapters:
            return self.adapters[name]

        # Check built-in adapters
        if name in self._builtin_adapters:
            return self._builtin_adapters[name]

        return None

    def list_adapters(self, include_builtin: bool = True) -> List[Dict[str, Any]]:
        """List all available adapters."""
        adapters = []

        # User-defined adapters
        for adapter in self.adapters.values():
            adapters.append(
                {
                    "name": adapter.name,
                    "version": adapter.version,
                    "description": adapter.description,
                    "base_url": adapter.base_url,
                    "authentication": adapter.authentication.value,
                    "endpoints": len(adapter.endpoints),
                    "tags": adapter.tags,
                    "builtin": False,
                }
            )

        # Built-in adapters
        if include_builtin:
            for adapter in self._builtin_adapters.values():
                adapters.append(
                    {
                        "name": adapter.name,
                        "version": adapter.version,
                        "description": adapter.description,
                        "base_url": adapter.base_url,
                        "authentication": adapter.authentication.value,
                        "endpoints": len(adapter.endpoints),
                        "tags": adapter.tags,
                        "builtin": True,
                    }
                )

        return adapters

    def remove_adapter(self, name: str, save_to_file: bool = True):
        """Remove an adapter from the registry."""
        if name in self.adapters:
            del self.adapters[name]

            if save_to_file:
                self._save_registry()

            self.logger.info(f"Removed adapter: {name}")
        else:
            raise ValueError(f"Adapter not found: {name}")

    def search_adapters(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        authentication: Optional[AuthenticationType] = None,
    ) -> List[Dict[str, Any]]:
        """Search for adapters based on criteria."""
        all_adapters = {**self.adapters, **self._builtin_adapters}
        results = []

        for adapter in all_adapters.values():
            # Text search
            if query:
                query_lower = query.lower()
                if query_lower not in adapter.name.lower() and query_lower not in adapter.description.lower():
                    continue

            # Tag filter
            if tags:
                if not any(tag in adapter.tags for tag in tags):
                    continue

            # Authentication filter
            if authentication and adapter.authentication != authentication:
                continue

            results.append(
                {
                    "name": adapter.name,
                    "version": adapter.version,
                    "description": adapter.description,
                    "base_url": adapter.base_url,
                    "authentication": adapter.authentication.value,
                    "endpoints": len(adapter.endpoints),
                    "tags": adapter.tags,
                    "builtin": adapter.name in self._builtin_adapters,
                }
            )

        return results

    def create_config(self, adapter_name: str, **config_overrides) -> CustomFrameworkConfig:
        """Create a configuration for an adapter."""
        adapter = self.get_adapter(adapter_name)
        if not adapter:
            raise ValueError(f"Adapter not found: {adapter_name}")

        config = CustomFrameworkConfig(adapter_definition=adapter, adapter_name=adapter_name, **config_overrides)

        return config

    def export_adapter(self, name: str, file_path: str):
        """Export an adapter definition to a file."""
        adapter = self.get_adapter(name)
        if not adapter:
            raise ValueError(f"Adapter not found: {name}")

        adapter_data = self._serialize_adapter(adapter)

        with open(file_path, 'w') as f:
            json.dump(adapter_data, f, indent=2, default=str)

        self.logger.info(f"Exported adapter '{name}' to {file_path}")

    def import_adapter(self, file_path: str, save_to_registry: bool = True):
        """Import an adapter definition from a file."""
        with open(file_path, 'r') as f:
            adapter_data = json.load(f)

        adapter = self._deserialize_adapter(adapter_data)

        if save_to_registry:
            self.register_adapter(adapter)

        self.logger.info(f"Imported adapter '{adapter.name}' from {file_path}")
        return adapter
