"""
Test Suite for Multi-Framework AI Platform

Provides comprehensive testing utilities and fixtures for all platform components.
"""

import asyncio
import logging
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock

import pytest

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
pytest_plugins = ["tests.fixtures"]
