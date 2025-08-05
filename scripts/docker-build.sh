#!/bin/bash

# Docker build script for MCP Distributed System
set -e

echo "🐳 Building MCP Distributed System Docker Images"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="mcp-distributed-system"
TAG="${1:-latest}"
BUILD_CONTEXT="."

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the main MCP system image
log_info "Building MCP Distributed System image..."
docker build \
    --tag "${IMAGE_NAME}:${TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    --file Dockerfile \
    --progress=plain \
    "${BUILD_CONTEXT}"

if [ $? -eq 0 ]; then
    log_success "MCP Distributed System image built successfully"
else
    log_error "Failed to build MCP Distributed System image"
    exit 1
fi

# Check if ui directory exists within mcp-python and build it
if [ -d "ui" ]; then
    log_info "Building MCP UI image..."
    cd ui
    if [ -f "package.json" ]; then
        docker build \
            --tag "mcp-ui:${TAG}" \
            --tag "mcp-ui:latest" \
            --progress=plain \
            .

        if [ $? -eq 0 ]; then
            log_success "MCP UI image built successfully"
        else
            log_error "Failed to build MCP UI image"
            cd - > /dev/null
            exit 1
        fi
    else
        log_warning "No package.json found in ui directory, skipping UI build"
    fi
    cd - > /dev/null
else
    log_info "No ui directory found in mcp-python, skipping UI build"
    log_info "To add UI support, place the React UI files in ./ui/ directory"
fi

# Show built images
log_info "Built images:"
docker images | grep -E "(mcp-distributed-system|mcp-ui)" | head -10

# Show image sizes
log_info "Image sizes:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "(mcp-distributed-system|mcp-ui)"

log_success "Build completed successfully!"
echo ""
echo "🚀 To start the system:"
echo "   docker-compose up -d"
echo ""
echo "📊 To start with monitoring:"
echo "   docker-compose -f docker-compose.yml up -d"
echo ""
echo "🔧 For development:"
echo "   docker-compose -f docker-compose.dev.yml up -d"
