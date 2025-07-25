.PHONY: install test run clean dev lint format coverage

# Install dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest -v

# Run the server
run:
	python server.py

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run linter
lint:
	ruff check .
	mypy server.py

# Format code
format:
	black .
	isort .

# Run tests with coverage
coverage:
	pytest --cov=server --cov-report=html --cov-report=term

# Run server with debug logging
debug:
	MCP_LOG_LEVEL=debug python server.py

# Type check
typecheck:
	mypy server.py test_server.py