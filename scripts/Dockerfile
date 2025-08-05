# Multi-stage Docker build for MCP Python Platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mcp && useradd -r -g mcp mcp

# Set working directory
WORKDIR /app

# Layer 1: Core dependencies (most stable)
COPY requirements-core.txt .
RUN pip install -r requirements-core.txt

# Layer 2: AI/ML dependencies
COPY requirements-ai.txt .
RUN pip install -r requirements-ai.txt

# Layer 3: Multimedia dependencies
COPY requirements-multimedia.txt .
RUN pip install -r requirements-multimedia.txt

# Layer 4: Integration dependencies
COPY requirements-integrations.txt .
RUN pip install -r requirements-integrations.txt

# Layer 5: Specialized/problematic dependencies
COPY requirements-specialized.txt .
RUN pip install -r requirements-specialized.txt

# Development stage
FROM base as development

# Copy development requirements if exists
COPY requirements-dev.txt* ./
RUN if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi

# Copy source code (this layer rebuilds when code changes)
COPY . .

# Change ownership to mcp user
RUN chown -R mcp:mcp /app

USER mcp

# Expose port
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=mcp:mcp config ./config
COPY --chown=mcp:mcp database ./database
COPY --chown=mcp:mcp security ./security
COPY --chown=mcp:mcp utils ./utils
COPY --chown=mcp:mcp observability ./observability
COPY --chown=mcp:mcp frameworks ./frameworks
COPY --chown=mcp:mcp api ./api
COPY --chown=mcp:mcp main.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R mcp:mcp /app

USER mcp

# Expose ports
EXPOSE 8000 8001

# Health check (port 8001 is the HTTP API port)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Production command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
