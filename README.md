# MCP Python Implementation

A robust Model Context Protocol (MCP) server implementation in Python using the official MCP SDK.

## Features

- **Tools**: Time utilities, calculator, echo, and data processing
- **Resources**: System info, logs, and configuration
- **Prompts**: Greeting, summary, and code review templates
- **Transport**: Stdio-based communication for Claude Desktop integration
- **Type Safety**: Full type hints and validation

## Installation

```bash
# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Usage

### Running the Server

```bash
python server.py
```

### Integration with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "example-python": {
      "command": "python",
      "args": ["/path/to/mcp-python/server.py"],
      "env": {}
    }
  }
}
```

## API Reference

### Tools

- `get_time()`: Returns current server time in ISO format
- `get_uptime()`: Returns server uptime duration
- `calculate(expression: str)`: Evaluates mathematical expressions
- `echo(message: str, uppercase: bool = False)`: Echoes messages
- `process_data(data: List[Dict], operation: str, options: Dict)`: Advanced data processing

### Resources

- `system://info`: System information (OS, Python version, uptime)
- `logs://recent`: Recent server logs with timestamps
- `config://current`: Current server configuration in JSON

### Prompts

- `greeting(name: str, formal: bool = False)`: Generates greetings
- `summary(text: str, max_length: int = 100)`: Creates text summaries
- `code_review(code: str, language: str = "python", focus: str = "general")`: Code review requests

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=server --cov-report=html

# Run specific test
pytest test_server.py::test_get_time
```

### Project Structure

```
mcp-python/
├── server.py          # Main MCP server implementation
├── test_server.py     # Comprehensive test suite
├── pyproject.toml     # Project configuration
└── README.md          # This file
```

## Examples

### Using Tools

```python
# Get current time
result = await get_time(ctx)
# Returns: "Current time: 2024-01-20T10:30:00Z"

# Calculate expression
result = await calculate(ctx, "2 + 2 * 3")
# Returns: "Result: 2 + 2 * 3 = 8"

# Process data
data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
result = await process_data(ctx, data, "filter", {"key": "age", "value": 30})
# Returns: "Filtered 1 items out of 2"
```

### Using Resources

Resources are accessed via their URIs:
- `system://info` - System information
- `logs://recent` - Recent logs
- `config://current` - Server configuration

### Using Prompts

Prompts generate structured messages for LLM interactions:

```python
# Generate greeting
messages = await greeting(ctx, "Alice", formal=True)
# Returns prompt for formal greeting

# Request code review
messages = await code_review(ctx, code_snippet, focus="security")
# Returns prompt for security-focused review
```

## License

MIT