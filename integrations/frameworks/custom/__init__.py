"""
Custom Framework Adapter System

Flexible adapter system for integrating any AI framework or API
with the MCP platform through a standardized interface.
"""

from .adapter import CustomFrameworkAdapter
from .models import (
    AdapterDefinition,
    CustomFrameworkConfig,
    CustomFrameworkQuery,
    CustomFrameworkResponse,
    EndpointDefinition,
    ParameterDefinition,
)
from .registry import FrameworkRegistry

__all__ = [
    'CustomFrameworkAdapter',
    'FrameworkRegistry',
    'CustomFrameworkConfig',
    'CustomFrameworkQuery',
    'CustomFrameworkResponse',
    'AdapterDefinition',
    'EndpointDefinition',
    'ParameterDefinition',
]
