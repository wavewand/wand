"""
Test Fixtures

Provides common test fixtures and utilities for the test suite.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest

from api.batch_processing import BatchProcessor
from caching.response_cache import ResponseCache
from config.manager import ConfigManager, Environment
from config.settings import Settings, initialize_settings
from monitoring.framework_monitor import FrameworkPerformanceMonitor
from security.auth import AuthManager, initialize_auth
from security.rate_limiter import RateLimiter
from utils.error_handling import ErrorHandler


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def config_manager(temp_dir):
    """Create test configuration manager."""
    config_manager = ConfigManager(base_path=str(temp_dir), environment=Environment.TESTING)

    # Add test configuration file
    test_config = {
        "api_host": "localhost",
        "api_port": 8001,
        "database_url": "sqlite:///:memory:",
        "cache_enabled": True,
        "security_jwt_secret": "test-secret-key",
    }

    config_file = temp_dir / "test_config.json"
    import json

    with open(config_file, 'w') as f:
        json.dump(test_config, f)

    config_manager.add_config_file("test_config.json")

    return config_manager


@pytest.fixture
def settings(config_manager):
    """Create test settings."""
    return initialize_settings(config_manager)


@pytest.fixture
def auth_manager():
    """Create test authentication manager."""
    initialize_auth("test-jwt-secret")
    from security.auth import auth_manager

    return auth_manager


@pytest.fixture
def rate_limiter():
    """Create test rate limiter."""
    return RateLimiter()


@pytest.fixture
def error_handler():
    """Create test error handler."""
    return ErrorHandler()


@pytest.fixture
def response_cache():
    """Create test response cache."""
    return ResponseCache(max_size=100, default_ttl=60)


@pytest.fixture
def framework_monitor():
    """Create test framework monitor."""
    return FrameworkPerformanceMonitor()


@pytest.fixture
def batch_processor():
    """Create test batch processor."""
    return BatchProcessor()


@pytest.fixture
def mock_framework():
    """Create mock AI framework."""
    framework = Mock()
    framework.name = "test_framework"
    framework.capabilities = ["rag_query", "document_search", "text_summarization"]

    # Mock async methods
    framework.execute_rag_query = AsyncMock(
        return_value={
            "success": True,
            "answer": "Test answer",
            "sources": ["doc1", "doc2"],
            "framework": "test_framework",
        }
    )

    framework.search_documents = AsyncMock(
        return_value={
            "success": True,
            "documents": [
                {"id": "doc1", "title": "Test Doc 1", "content": "Test content 1"},
                {"id": "doc2", "title": "Test Doc 2", "content": "Test content 2"},
            ],
            "framework": "test_framework",
        }
    )

    framework.summarize_text = AsyncMock(
        return_value={"success": True, "summary": "Test summary", "framework": "test_framework"}
    )

    framework.ingest_document = AsyncMock(
        return_value={"success": True, "document_id": "test_doc_123", "framework": "test_framework"}
    )

    framework.get_status = AsyncMock(
        return_value={"available": True, "framework": "test_framework", "version": "1.0.0"}
    )

    return framework


@pytest.fixture
def sample_user():
    """Create sample user for testing."""
    from security.auth import User, UserRole

    return User(user_id="test_user_123", username="testuser", email="test@example.com", role=UserRole.USER)


@pytest.fixture
def sample_admin_user():
    """Create sample admin user for testing."""
    from security.auth import User, UserRole

    return User(user_id="admin_user_123", username="admin", email="admin@example.com", role=UserRole.ADMIN)


@pytest.fixture
def test_documents():
    """Create sample test documents."""
    return [
        {
            "filename": "doc1.txt",
            "content": "This is a test document about artificial intelligence and machine learning.",
            "metadata": {"category": "AI", "author": "Test Author"},
        },
        {
            "filename": "doc2.txt",
            "content": "This document discusses natural language processing and neural networks.",
            "metadata": {"category": "NLP", "author": "Another Author"},
        },
        {
            "filename": "doc3.txt",
            "content": "A comprehensive guide to deep learning and computer vision applications.",
            "metadata": {"category": "Computer Vision", "author": "Expert Author"},
        },
    ]


@pytest.fixture
def test_queries():
    """Create sample test queries."""
    return [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain machine learning algorithms",
        "What are the applications of NLP?",
        "How does computer vision work?",
    ]


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    websocket = Mock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.close = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.receive_json = AsyncMock()

    return websocket


@pytest.fixture
async def test_client():
    """Create test HTTP client."""
    from fastapi.testclient import TestClient

    from api.server import app

    return TestClient(app)


@pytest.fixture
def performance_data():
    """Create sample performance data for testing."""
    return {
        "framework": "test_framework",
        "operation": "rag_query",
        "response_time_ms": 250.5,
        "success": True,
        "query": "test query",
        "timestamp": "2024-01-01T12:00:00Z",
    }


@pytest.fixture
def validation_schemas():
    """Create sample validation schemas for testing."""
    from security.validator import ValidationRule

    return {
        "username": [ValidationRule.REQUIRED, ValidationRule.ALPHANUMERIC, {"rule": "min_length", "min": 3}],
        "email": [ValidationRule.REQUIRED, ValidationRule.EMAIL],
        "password": [ValidationRule.REQUIRED, {"rule": "min_length", "min": 8}],
        "query": [ValidationRule.REQUIRED, ValidationRule.NO_XSS, {"rule": "max_length", "max": 1000}],
        "filename": [ValidationRule.REQUIRED, ValidationRule.SAFE_FILENAME],
    }


class AsyncContextManager:
    """Helper for testing async context managers."""

    def __init__(self, obj):
        self.obj = obj

    async def __aenter__(self):
        return self.obj

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Create async context manager helper."""
    return AsyncContextManager
