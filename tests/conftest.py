"""
Pytest Configuration

Global pytest configuration and markers.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")
    config.addinivalue_line("markers", "redis: mark test as requiring Redis")
    config.addinivalue_line("markers", "database: mark test as requiring database")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on path."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add slow marker for certain tests
        if "benchmark" in str(item.fspath) or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# Mock context fixture used by many tests
@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    return Mock()


# Process backend fixture
@pytest.fixture
def mock_process_backend():
    """Mock process execution backend."""
    from tools.execution.base import ExecutionResult, ExecutionStatus

    backend = Mock()
    backend.execute = AsyncMock(
        return_value=ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="test output", stderr="", exit_code=0)
    )
    backend.execute_with_timeout = AsyncMock(
        return_value=ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="test output", stderr="", exit_code=0)
    )
    return backend
