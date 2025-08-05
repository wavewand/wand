"""
Custom Framework Adapter

Implementation of the custom framework adapter that can integrate
with any AI framework or API through configuration.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

from integrations.frameworks.base import BaseClient
from observability.logging import get_logger

from .models import (
    AdapterDefinition,
    AuthenticationType,
    CustomFrameworkConfig,
    CustomFrameworkQuery,
    CustomFrameworkResponse,
    EndpointDefinition,
    ParameterDefinition,
    ParameterType,
    RequestMethod,
)


class CustomFrameworkAdapter(BaseClient):
    """Flexible adapter for custom AI frameworks and APIs."""

    def __init__(self, config: CustomFrameworkConfig):
        super().__init__(config)
        self.config = config
        self.logger = get_logger(__name__)

        # Get adapter definition
        self.adapter_definition = config.adapter_definition
        if not self.adapter_definition:
            raise ValueError("Adapter definition is required")

        # HTTP session (initialized lazily)
        self._session: Optional[aiohttp.ClientSession] = None

        # Response cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(seconds=config.cache_ttl_seconds)

        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._rate_limit_window_start = time.time()

        # Statistics
        self._stats = {
            "requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_response_time_ms": 0,
        }

        self.logger.info(f"Custom adapter initialized: {self.adapter_definition.name}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            # Create session with base configuration
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            headers = {}
            headers.update(self.adapter_definition.default_headers)
            headers.update(self.config.custom_headers)

            # Add authentication headers
            auth_headers = self._get_auth_headers()
            headers.update(auth_headers)

            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)

        return self._session

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on adapter configuration."""
        headers = {}
        auth_config = {**self.adapter_definition.auth_config, **self.config.auth_config}

        if self.adapter_definition.authentication == AuthenticationType.API_KEY:
            api_key = auth_config.get("api_key")
            key_header = auth_config.get("api_key_header", "X-API-Key")
            if api_key:
                headers[key_header] = api_key

        elif self.adapter_definition.authentication == AuthenticationType.BEARER_TOKEN:
            token = auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif self.adapter_definition.authentication == AuthenticationType.BASIC_AUTH:
            username = auth_config.get("username")
            password = auth_config.get("password")
            if username and password:
                import base64

                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"

        return headers

    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""
        if not self.config.enable_rate_limiting or not self.adapter_definition.rate_limit:
            return True

        current_time = time.time()
        rate_limit = self.adapter_definition.rate_limit

        # Reset window if needed
        window_duration = rate_limit.get("window_seconds", 60)
        if current_time - self._rate_limit_window_start > window_duration:
            self._rate_limit_window_start = current_time
            self._request_count = 0

        # Check limits
        max_requests = rate_limit.get("max_requests", 100)
        if self._request_count >= max_requests:
            self.logger.warning("Rate limit exceeded")
            return False

        return True

    def _get_cache_key(self, query: CustomFrameworkQuery) -> str:
        """Generate cache key for query."""
        cache_data = {"endpoint": query.endpoint_name, "parameters": query.parameters}
        import hashlib

        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"custom:{hashlib.md5(cache_string.encode()).hexdigest()[:16]}"

    def _get_from_cache(self, query: CustomFrameworkQuery) -> Optional[CustomFrameworkResponse]:
        """Get response from cache if available."""
        if not self.config.enable_caching:
            return None

        cache_key = self._get_cache_key(query)

        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            cached_time = datetime.fromisoformat(cache_entry["timestamp"])

            if datetime.now() - cached_time < self._cache_ttl:
                self.logger.debug(f"Cache hit for query: {cache_key}")
                self._stats["cache_hits"] += 1
                response_data = cache_entry["response"]
                return CustomFrameworkResponse(**response_data)

        return None

    def _store_in_cache(self, query: CustomFrameworkQuery, response: CustomFrameworkResponse):
        """Store response in cache."""
        if not self.config.enable_caching or not response.success:
            return

        cache_key = self._get_cache_key(query)
        cache_entry = {"timestamp": datetime.now().isoformat(), "response": response.__dict__}

        self._cache[cache_key] = cache_entry
        self.logger.debug(f"Cached response for query: {cache_key}")

    def _validate_parameters(self, endpoint: EndpointDefinition, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process parameters."""
        processed_params = {}

        # Check required parameters
        for param in endpoint.get_required_parameters():
            if param.name not in parameters:
                raise ValueError(f"Required parameter '{param.name}' is missing")

        # Validate and process all parameters
        for param in endpoint.parameters:
            value = parameters.get(param.name, param.default_value)

            if value is not None:
                if not param.validate(value):
                    raise ValueError(f"Invalid value for parameter '{param.name}': {value}")
                processed_params[param.name] = value

        return processed_params

    def _build_request_data(self, endpoint: EndpointDefinition, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Build request data from parameters."""
        request_data = {}
        query_params = {}
        headers = {}
        path_params = {}

        for param in endpoint.parameters:
            if param.name in parameters:
                value = parameters[param.name]

                if param.location == "body":
                    request_data[param.name] = value
                elif param.location == "query":
                    query_params[param.name] = value
                elif param.location == "header":
                    headers[param.name] = str(value)
                elif param.location == "path":
                    path_params[param.name] = str(value)

        return {"body": request_data, "query": query_params, "headers": headers, "path": path_params}

    async def execute_query(self, query: CustomFrameworkQuery) -> CustomFrameworkResponse:
        """Execute a query against the custom framework."""
        try:
            start_time = time.time()

            self.logger.info(f"Executing custom query: {query.endpoint_name}")

            # Check cache first
            cached_response = self._get_from_cache(query)
            if cached_response:
                return cached_response

            # Check rate limits
            if not self._check_rate_limits():
                return CustomFrameworkResponse(
                    success=False, error="Rate limit exceeded", framework="custom", endpoint_name=query.endpoint_name
                )

            # Get endpoint definition
            endpoint = self.adapter_definition.get_endpoint(query.endpoint_name)
            if not endpoint:
                return CustomFrameworkResponse(
                    success=False,
                    error=f"Endpoint '{query.endpoint_name}' not found",
                    framework="custom",
                    endpoint_name=query.endpoint_name,
                )

            # Validate parameters
            try:
                validated_params = self._validate_parameters(endpoint, query.parameters)
            except ValueError as e:
                return CustomFrameworkResponse(
                    success=False, error=str(e), framework="custom", endpoint_name=query.endpoint_name
                )

            # Build request data
            request_components = self._build_request_data(endpoint, validated_params)

            # Apply request transformer if available
            if endpoint.request_transformer:
                request_components = endpoint.request_transformer(request_components)

            # Build URL
            base_url = self.config.base_url or self.adapter_definition.base_url
            url_path = endpoint.path

            # Replace path parameters
            for key, value in request_components["path"].items():
                url_path = url_path.replace(f"{{{key}}}", value)

            url = f"{base_url.rstrip('/')}/{url_path.lstrip('/')}"

            # Prepare request
            session = await self._get_session()

            request_kwargs = {
                "url": url,
                "params": request_components["query"],
                "headers": {**endpoint.headers, **request_components["headers"], **query.headers},
                "timeout": aiohttp.ClientTimeout(total=query.timeout or endpoint.timeout),
            }

            # Add body for POST/PUT/PATCH requests
            if endpoint.method in [RequestMethod.POST, RequestMethod.PUT, RequestMethod.PATCH]:
                if request_components["body"]:
                    request_kwargs["json"] = request_components["body"]

            # Execute request
            async with session.request(endpoint.method.value, **request_kwargs) as response:
                duration_ms = (time.time() - start_time) * 1000

                # Create response
                custom_response = CustomFrameworkResponse.from_http_response(response, query, endpoint, duration_ms)

                # Update statistics
                self._stats["requests"] += 1
                self._stats["total_response_time_ms"] += duration_ms
                self._request_count += 1

                if custom_response.success:
                    self._stats["successful_requests"] += 1
                else:
                    self._stats["failed_requests"] += 1

                # Cache successful responses
                if custom_response.success:
                    self._store_in_cache(query, custom_response)

                self.logger.info(f"Custom query completed in {duration_ms:.2f}ms")
                return custom_response

        except Exception as e:
            self.logger.error(f"Custom query failed: {e}")
            self._stats["requests"] += 1
            self._stats["failed_requests"] += 1

            return CustomFrameworkResponse(
                success=False, error=str(e), framework="custom", endpoint_name=query.endpoint_name
            )

    async def execute_simple_query(self, endpoint_name: str, **parameters) -> CustomFrameworkResponse:
        """Execute a simple query with parameters as kwargs."""
        query = CustomFrameworkQuery(endpoint_name=endpoint_name, parameters=parameters)
        return await self.execute_query(query)

    def get_available_endpoints(self) -> List[str]:
        """Get list of available endpoints."""
        return list(self.adapter_definition.endpoints.keys())

    def get_endpoint_info(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an endpoint."""
        endpoint = self.adapter_definition.get_endpoint(endpoint_name)
        if not endpoint:
            return None

        return {
            "name": endpoint.name,
            "path": endpoint.path,
            "method": endpoint.method.value,
            "description": endpoint.description,
            "parameters": [
                {
                    "name": param.name,
                    "type": param.type.value,
                    "required": param.required,
                    "description": param.description,
                    "default_value": param.default_value,
                    "location": param.location,
                }
                for param in endpoint.parameters
            ],
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        stats = self._stats.copy()

        if stats["requests"] > 0:
            stats["average_response_time_ms"] = stats["total_response_time_ms"] / stats["requests"]
            stats["success_rate"] = stats["successful_requests"] / stats["requests"]
        else:
            stats["average_response_time_ms"] = 0
            stats["success_rate"] = 0

        stats["cache_entries"] = len(self._cache)
        return stats

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        self.logger.info("Custom adapter cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Basic connectivity test (if adapter has a health endpoint)
            health_endpoint = self.adapter_definition.get_endpoint("health")

            if health_endpoint:
                query = CustomFrameworkQuery(endpoint_name="health")
                response = await self.execute_query(query)

                return {
                    'status': 'healthy' if response.success else 'unhealthy',
                    'framework': 'custom',
                    'adapter_name': self.adapter_definition.name,
                    'adapter_version': self.adapter_definition.version,
                    'base_url': self.config.base_url or self.adapter_definition.base_url,
                    'available_endpoints': len(self.adapter_definition.endpoints),
                    'stats': await self.get_stats(),
                    'health_check_response': response.success,
                }
            else:
                # No health endpoint, just return configuration info
                return {
                    'status': 'healthy',
                    'framework': 'custom',
                    'adapter_name': self.adapter_definition.name,
                    'adapter_version': self.adapter_definition.version,
                    'base_url': self.config.base_url or self.adapter_definition.base_url,
                    'available_endpoints': len(self.adapter_definition.endpoints),
                    'stats': await self.get_stats(),
                    'note': 'No health endpoint defined',
                }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'framework': 'custom',
                'adapter_name': self.adapter_definition.name if self.adapter_definition else 'unknown',
                'error': str(e),
            }

    async def close(self):
        """Close the adapter and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

        if self.adapter_definition and self.adapter_definition.cleanup_function:
            try:
                self.adapter_definition.cleanup_function()
            except Exception as e:
                self.logger.error(f"Cleanup function failed: {e}")
