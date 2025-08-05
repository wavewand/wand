.PHONY: install test run clean dev lint format coverage setup-venv generate-grpc start debug status demo docker-build docker-start docker-dev docker-stop docker-logs docker-clean

# Default Python interpreter
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin

# Setup virtual environment and install dependencies
setup-venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt

# Install dependencies (assumes venv is activated)
install:
	pip install -r requirements.txt

# Install development dependencies
dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio black isort mypy ruff

# Generate gRPC code from protobuf definitions
generate-grpc:
	$(VENV_BIN)/python scripts/generate_grpc.py

# Start the complete distributed system
start:
	$(VENV_BIN)/python main.py

# Start with debug logging
debug:
	$(VENV_BIN)/python main.py --log-level DEBUG

# Check system status
status:
	$(VENV_BIN)/python main.py --status

# Run the demo client
demo:
	$(VENV_BIN)/python examples/demo_client.py

# Run simple test
simple-test:
	$(VENV_BIN)/python examples/simple_test.py

# Run tests
test:
	$(VENV_BIN)/pytest -v

# Run tests with coverage
coverage:
	$(VENV_BIN)/pytest --cov=grpc_services --cov=rest_api --cov=distributed --cov-report=html --cov-report=term

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info
	rm -rf logs/
	rm -rf generated/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Format code
format:
	$(VENV_BIN)/black .
	$(VENV_BIN)/isort .

# Run linter
lint:
	$(VENV_BIN)/ruff check .

# Type check
typecheck:
	$(VENV_BIN)/mypy grpc_services/ rest_api/ distributed/ orchestrator/

# Build and start system (complete setup)
build: clean setup-venv generate-grpc
	@echo "✅ MCP Distributed System ready!"
	@echo "Run 'make start' to launch the system"

# Docker commands
docker-build:
	./scripts/docker-build.sh

docker-start:
	docker-compose up -d

docker-dev:
	docker-compose -f docker-compose.dev.yml up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# Help
help:
	@echo "MCP Distributed System - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup-venv     - Create virtual environment and install dependencies"
	@echo "  make install        - Install dependencies (assumes venv activated)"
	@echo "  make generate-grpc  - Generate gRPC code from protobuf definitions"
	@echo "  make build          - Complete setup (clean, venv, generate gRPC)"
	@echo ""
	@echo "Running (Local):"
	@echo "  make start          - Start the distributed system"
	@echo "  make debug          - Start with debug logging"
	@echo "  make status         - Check system status"
	@echo ""
	@echo "Running (Docker):"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-start   - Start with Docker Compose"
	@echo "  make docker-dev     - Start development environment"
	@echo "  make docker-stop    - Stop Docker containers"
	@echo "  make docker-logs    - View Docker logs"
	@echo "  make docker-clean   - Clean Docker resources"
	@echo ""
	@echo "Testing:"
	@echo "  make demo           - Run full demonstration"
	@echo "  make simple-test    - Run simple connectivity test"
	@echo "  make test           - Run unit tests"
	@echo "  make coverage       - Run tests with coverage report"
	@echo ""
	@echo "Development:"
	@echo "  make format         - Format code with black and isort"
	@echo "  make lint           - Run linting checks"
	@echo "  make typecheck      - Run type checking"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Local: http://localhost:8000/docs | Docker: http://localhost:3000"
