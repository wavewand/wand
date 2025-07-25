import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import json

from server import (
    get_time, get_uptime, calculate, echo, 
    get_system_info, get_recent_logs, get_config,
    greeting, summary, code_review, process_data
)
from mcp.server.fastmcp import Context
from mcp.types import PromptMessage, TextContent


@pytest.fixture
def mock_context():
    """Create a mock context for testing"""
    return Mock(spec=Context)


@pytest.mark.asyncio
async def test_get_time(mock_context):
    """Test get_time returns current time in ISO format"""
    result = await get_time(mock_context)
    assert "Current time:" in result
    # Verify it's a valid ISO timestamp
    timestamp = result.replace("Current time: ", "")
    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))


@pytest.mark.asyncio
async def test_get_uptime(mock_context):
    """Test get_uptime returns server uptime"""
    result = await get_uptime(mock_context)
    assert "Server uptime:" in result
    assert "h" in result and "m" in result and "s" in result


@pytest.mark.asyncio
async def test_calculate_valid(mock_context):
    """Test calculate with valid expressions"""
    test_cases = [
        ("2 + 2", "4"),
        ("10 * 5", "50"),
        ("(10 + 5) * 2", "30"),
        ("100 / 4", "25"),
    ]
    
    for expression, expected in test_cases:
        result = await calculate(mock_context, expression)
        assert f"Result: {expression} = {expected}" in result


@pytest.mark.asyncio
async def test_calculate_invalid(mock_context):
    """Test calculate with invalid expressions"""
    invalid_expressions = [
        "import os",
        "2 + 2; print('hack')",
        "lambda x: x",
    ]
    
    for expression in invalid_expressions:
        result = await calculate(mock_context, expression)
        assert "Error" in result


@pytest.mark.asyncio
async def test_echo(mock_context):
    """Test echo function"""
    # Test normal echo
    result = await echo(mock_context, "hello world")
    assert result == "Echo: hello world"
    
    # Test uppercase echo
    result = await echo(mock_context, "hello world", uppercase=True)
    assert result == "Echo: HELLO WORLD"


@pytest.mark.asyncio
async def test_get_system_info():
    """Test system info resource"""
    result = await get_system_info()
    assert "System Information:" in result
    assert "OS:" in result
    assert "Python Version:" in result
    assert "Process ID:" in result


@pytest.mark.asyncio
async def test_get_recent_logs():
    """Test recent logs resource"""
    result = await get_recent_logs()
    assert "Recent logs" in result
    assert "Server started" in result
    assert "All systems operational" in result


@pytest.mark.asyncio
async def test_get_config():
    """Test config resource returns valid JSON"""
    result = await get_config()
    config = json.loads(result)
    
    assert config["server_name"] == "example-mcp-server"
    assert config["version"] == "1.0.0"
    assert "tools" in config["features"]
    assert "resources" in config["features"]
    assert "prompts" in config["features"]


@pytest.mark.asyncio
async def test_greeting_prompt():
    """Test greeting prompt generation"""
    # Test informal greeting
    messages = await greeting("Alice")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "Hey Alice!" in messages[0].content.text
    
    # Test formal greeting
    messages = await greeting("Dr. Smith", formal=True)
    assert len(messages) == 1
    assert "Good day, Dr. Smith" in messages[0].content.text


@pytest.mark.asyncio
async def test_summary_prompt():
    """Test summary prompt generation"""
    text = "This is a long text that needs summarization."
    messages = await summary(text, max_length=50)
    
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "summarize" in messages[0].content.text.lower()
    assert "50 characters" in messages[0].content.text
    assert text in messages[0].content.text


@pytest.mark.asyncio
async def test_code_review_prompt():
    """Test code review prompt generation"""
    code = "def hello():\n    print('Hello')"
    
    # Test general review
    messages = await code_review(code)
    assert len(messages) == 1
    assert "correctness" in messages[0].content.text
    assert code in messages[0].content.text
    
    # Test security review
    messages = await code_review(code, language="python", focus="security")
    assert "security" in messages[0].content.text.lower()
    
    # Test performance review
    messages = await code_review(code, focus="performance")
    assert "performance" in messages[0].content.text.lower()


@pytest.mark.asyncio
async def test_process_data_filter(mock_context):
    """Test process_data filter operation"""
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 30}
    ]
    
    result = await process_data(
        mock_context,
        data,
        "filter",
        {"key": "age", "value": 30}
    )
    
    assert "Filtered 2 items out of 3" in result


@pytest.mark.asyncio
async def test_process_data_transform(mock_context):
    """Test process_data transform operation"""
    data = [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": 25}
    ]
    
    result = await process_data(
        mock_context,
        data,
        "transform",
        {"key": "name", "type": "uppercase"}
    )
    
    assert "Transformed 2 items" in result
    assert data[0]["name"] == "ALICE"
    assert data[1]["name"] == "BOB"


@pytest.mark.asyncio
async def test_process_data_aggregate(mock_context):
    """Test process_data aggregate operation"""
    data = [
        {"name": "Alice", "score": 90},
        {"name": "Bob", "score": 85},
        {"name": "Charlie", "score": 95}
    ]
    
    # Test count
    result = await process_data(
        mock_context,
        data,
        "aggregate",
        {"type": "count"}
    )
    assert "Aggregation result: 3" in result
    
    # Test sum
    result = await process_data(
        mock_context,
        data,
        "aggregate",
        {"key": "score", "type": "sum"}
    )
    assert "Aggregation result: 270" in result


@pytest.mark.asyncio
async def test_process_data_error_handling(mock_context):
    """Test process_data error handling"""
    # Test unknown operation
    result = await process_data(
        mock_context,
        [{"test": 1}],
        "unknown_op"
    )
    assert "Unknown operation" in result
    
    # Test with invalid data
    result = await process_data(
        mock_context,
        "not a list",  # Invalid data type
        "filter"
    )
    assert "Error processing data" in result