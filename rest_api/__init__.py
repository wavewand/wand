"""
REST API Gateway for the distributed MCP system.

Provides external HTTP/REST interface to the internal gRPC services.
"""

from .gateway import RestAPIGateway
from .models import *

__all__ = [
    "RestAPIGateway",
]
