"""
Base classes for AI framework integrations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BaseConfig:
    """Base configuration for framework clients"""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    extra_params: Dict[str, Any] = None


@dataclass
class BaseQuery:
    """Base query structure for framework requests"""

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class BaseResponse:
    """Base response structure for framework responses"""

    content: str
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = None
    timestamp: Optional[datetime] = None


@dataclass
class BaseDocument:
    """Base document structure for RAG systems"""

    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None
    id: Optional[str] = None


class BaseClient(ABC):
    """Abstract base class for framework clients"""

    def __init__(self, config: BaseConfig):
        self.config = config

    @abstractmethod
    async def query(self, query: BaseQuery) -> BaseResponse:
        """Execute a query against the framework"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health/status of the framework connection"""
        pass


class BaseService(ABC):
    """Abstract base class for framework services"""

    def __init__(self, client: BaseClient):
        self.client = client

    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a service request"""
        pass
