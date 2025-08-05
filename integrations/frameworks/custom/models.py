"""
Custom Framework Adapter Models

Defines data structures for the custom framework adapter system
including adapter definitions, configurations, and responses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseQuery, BaseResponse


class ParameterType(str, Enum):
    """Parameter types for adapter definitions."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    FILE = "file"
    JSON = "json"


class AuthenticationType(str, Enum):
    """Authentication types for frameworks."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


class RequestMethod(str, Enum):
    """HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class ParameterDefinition:
    """Definition of a parameter for framework endpoints."""

    name: str
    type: ParameterType
    required: bool = False
    default_value: Any = None
    description: Optional[str] = None

    # Validation rules
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings

    # Location in request
    location: str = "body"  # body, query, header, path

    def validate(self, value: Any) -> bool:
        """Validate parameter value."""
        try:
            # Type validation
            if self.type == ParameterType.STRING and not isinstance(value, str):
                return False
            elif self.type == ParameterType.INTEGER and not isinstance(value, int):
                return False
            elif self.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
                return False
            elif self.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                return False
            elif self.type == ParameterType.LIST and not isinstance(value, list):
                return False
            elif self.type == ParameterType.DICT and not isinstance(value, dict):
                return False

            # Range validation
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

            # Allowed values validation
            if self.allowed_values is not None and value not in self.allowed_values:
                return False

            # Pattern validation for strings
            if self.pattern and isinstance(value, str):
                import re

                if not re.match(self.pattern, value):
                    return False

            return True

        except Exception:
            return False


@dataclass
class EndpointDefinition:
    """Definition of a framework endpoint."""

    name: str
    path: str
    method: RequestMethod = RequestMethod.POST
    description: Optional[str] = None

    # Parameters
    parameters: List[ParameterDefinition] = field(default_factory=list)

    # Request configuration
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0

    # Response handling
    response_key: Optional[str] = None  # Key to extract from response
    success_codes: List[int] = field(default_factory=lambda: [200, 201])

    # Custom processing functions
    request_transformer: Optional[Callable] = None
    response_transformer: Optional[Callable] = None

    def get_required_parameters(self) -> List[ParameterDefinition]:
        """Get list of required parameters."""
        return [param for param in self.parameters if param.required]

    def get_parameter(self, name: str) -> Optional[ParameterDefinition]:
        """Get parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None


@dataclass
class AdapterDefinition:
    """Complete definition of a custom framework adapter."""

    name: str
    version: str
    description: str

    # Base configuration
    base_url: str
    authentication: AuthenticationType = AuthenticationType.NONE

    # Endpoints
    endpoints: Dict[str, EndpointDefinition] = field(default_factory=dict)

    # Authentication configuration
    auth_config: Dict[str, Any] = field(default_factory=dict)

    # Default headers
    default_headers: Dict[str, str] = field(default_factory=dict)

    # Rate limiting
    rate_limit: Optional[Dict[str, int]] = None

    # Custom initialization
    init_function: Optional[Callable] = None
    cleanup_function: Optional[Callable] = None

    # Metadata
    created_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def add_endpoint(self, endpoint: EndpointDefinition):
        """Add endpoint to adapter."""
        self.endpoints[endpoint.name] = endpoint

    def get_endpoint(self, name: str) -> Optional[EndpointDefinition]:
        """Get endpoint by name."""
        return self.endpoints.get(name)

    def validate_auth_config(self) -> bool:
        """Validate authentication configuration."""
        if self.authentication == AuthenticationType.API_KEY:
            return "api_key" in self.auth_config
        elif self.authentication == AuthenticationType.BEARER_TOKEN:
            return "token" in self.auth_config
        elif self.authentication == AuthenticationType.BASIC_AUTH:
            return "username" in self.auth_config and "password" in self.auth_config
        elif self.authentication == AuthenticationType.OAUTH2:
            return "client_id" in self.auth_config and "client_secret" in self.auth_config

        return True  # No validation needed for NONE or CUSTOM


@dataclass
class CustomFrameworkConfig(BaseConfig):
    """Configuration for custom framework adapter."""

    # Adapter definition
    adapter_definition: Optional[AdapterDefinition] = None
    adapter_name: Optional[str] = None  # Load from registry

    # Runtime configuration
    base_url: Optional[str] = None  # Override adapter base_url
    timeout: float = 30.0
    max_retries: int = 3

    # Authentication overrides
    auth_config: Dict[str, Any] = field(default_factory=dict)

    # Custom headers
    custom_headers: Dict[str, str] = field(default_factory=dict)

    # Rate limiting
    enable_rate_limiting: bool = True

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

    # Debug mode
    debug_mode: bool = False

    def __post_init__(self):
        super().__post_init__()


@dataclass
class CustomFrameworkQuery(BaseQuery):
    """Query for custom framework adapter."""

    endpoint_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Override settings
    timeout: Optional[float] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.query_text:
            self.query_text = f"Custom query to {self.endpoint_name}"


@dataclass
class CustomFrameworkResponse(BaseResponse):
    """Response from custom framework adapter."""

    endpoint_name: Optional[str] = None

    # Raw response data
    raw_response: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None

    # Processed response
    data: Optional[Any] = None

    # Request metadata
    request_duration_ms: float = 0

    def __post_init__(self):
        super().__post_init__()
        # Set content from data if not provided
        if not self.content and self.data:
            if isinstance(self.data, str):
                self.content = self.data
            else:
                self.content = str(self.data)

    def get_data(self) -> Any:
        """Get processed response data."""
        return self.data

    def get_raw_response(self) -> Optional[Dict[str, Any]]:
        """Get raw response data."""
        return self.raw_response

    @classmethod
    def from_http_response(
        cls, response: Any, query: CustomFrameworkQuery, endpoint: EndpointDefinition, duration_ms: float = 0
    ) -> 'CustomFrameworkResponse':
        """Create response from HTTP response."""
        custom_response = cls(
            success=response.status_code in endpoint.success_codes,
            framework="custom",
            endpoint_name=query.endpoint_name,
            status_code=response.status_code,
            request_duration_ms=duration_ms,
        )

        try:
            # Parse JSON response
            raw_data = response.json() if hasattr(response, 'json') else {}
            custom_response.raw_response = raw_data

            # Extract data using response key
            if endpoint.response_key and endpoint.response_key in raw_data:
                custom_response.data = raw_data[endpoint.response_key]
            else:
                custom_response.data = raw_data

            # Apply response transformer if available
            if endpoint.response_transformer:
                custom_response.data = endpoint.response_transformer(custom_response.data)

        except Exception as e:
            custom_response.success = False
            custom_response.error = f"Response parsing failed: {str(e)}"

        return custom_response


class BaseFrameworkAdapter(ABC):
    """Abstract base class for framework adapters."""

    def __init__(self, config: CustomFrameworkConfig):
        self.config = config
        self.adapter_definition = config.adapter_definition

    @abstractmethod
    async def execute_query(self, query: CustomFrameworkQuery) -> CustomFrameworkResponse:
        """Execute a query against the framework."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass

    def get_available_endpoints(self) -> List[str]:
        """Get list of available endpoints."""
        if self.adapter_definition:
            return list(self.adapter_definition.endpoints.keys())
        return []

    def get_endpoint_info(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an endpoint."""
        if not self.adapter_definition:
            return None

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
                }
                for param in endpoint.parameters
            ],
        }
