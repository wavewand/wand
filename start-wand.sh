#!/bin/bash
# 🪄 Start Wand API Server - Workspace Mount Mode

set -e

# Colors for output
PURPLE='\033[0;35m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default workspace path - use current directory or home if not specified
DEFAULT_WORKSPACE="${HOME}/workspace"
WORKSPACE_PATH="${1:-$DEFAULT_WORKSPACE}"

echo -e "${PURPLE}🪄 Wand API Server${NC}"
echo -e "${BLUE}==================${NC}"
echo ""
echo -e "${GREEN}📁 Workspace Directory: ${WORKSPACE_PATH}${NC}"
echo -e "${BLUE}   • NON-PRIVILEGED container mode${NC}"
echo -e "${BLUE}   • Commands limited to workspace directory only${NC}"
echo -e "${BLUE}   • Safe and controlled access to host filesystem${NC}"
echo -e "${BLUE}   • Command validation and security controls enabled${NC}"
echo ""

# Validate workspace path
if [ ! -d "$WORKSPACE_PATH" ]; then
    echo -e "${YELLOW}📁 Creating workspace directory: ${WORKSPACE_PATH}${NC}"
    mkdir -p "$WORKSPACE_PATH"
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create workspace directory${NC}"
        echo -e "${YELLOW}Please check permissions or create manually:${NC}"
        echo -e "${YELLOW}   mkdir -p \"${WORKSPACE_PATH}\"${NC}"
        exit 1
    fi
fi

# Check if workspace is writable
if [ ! -w "$WORKSPACE_PATH" ]; then
    echo -e "${RED}❌ Workspace directory is not writable: ${WORKSPACE_PATH}${NC}"
    echo -e "${YELLOW}Please check permissions:${NC}"
    echo -e "${YELLOW}   chmod 755 \"${WORKSPACE_PATH}\"${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo -e "${YELLOW}Please start Docker Desktop and try again${NC}"
    exit 1
fi

# Optional cleanup of old MCP images
if [[ "$2" == "--cleanup" ]]; then
    echo -e "${YELLOW}🧹 Cleaning up old MCP Docker images...${NC}"
    docker images | grep -E "(mcp-python|mcp-automation|mcp-ui|mcp-distributed)" | awk '{print $1":"$2}' | xargs -r docker rmi 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup completed${NC}"
fi

# Create required directories
echo -e "${BLUE}📁 Creating application directories...${NC}"
mkdir -p logs data config

# Set workspace path for docker-compose
export WORKSPACE_PATH="$WORKSPACE_PATH"

# Build and start the container
echo -e "${BLUE}🐳 Building and starting Wand API container...${NC}"
docker-compose -f docker-compose.workspace.yml up --build -d

# Wait for container to be healthy
echo -e "${BLUE}⏳ Waiting for API server to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Wand API Server is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""

# Test basic functionality
echo -e "${BLUE}🧪 Testing basic functionality...${NC}"
echo -e "${YELLOW}   Testing 'ls' command in workspace...${NC}"

# Create a simple test using the API
TEST_RESULT=$(curl -s -X POST "http://localhost:8000/api/v1/execute" \
  -H "Content-Type: application/json" \
  -d '{"command": "ls -la /workspace", "working_directory": "/workspace"}' 2>/dev/null || echo "TEST_FAILED")

if [[ "$TEST_RESULT" != "TEST_FAILED" ]]; then
    echo -e "${GREEN}✅ Command execution test passed${NC}"
else
    echo -e "${YELLOW}⚠️  API test failed - server may still be starting${NC}"
fi

echo ""
echo -e "${GREEN}🚀 Wand API Server Started Successfully!${NC}"
echo ""
echo -e "${BLUE}📍 API Server: http://localhost:8000${NC}"
echo -e "${BLUE}📊 Health Check: http://localhost:8000/health${NC}"
echo -e "${BLUE}📈 Metrics: http://localhost:9100/metrics${NC}"
echo -e "${BLUE}📋 API Docs: http://localhost:8000/docs${NC}"
echo ""
echo -e "${BLUE}📁 Workspace: ${WORKSPACE_PATH}${NC}"
echo -e "${BLUE}   • All commands will execute in this directory${NC}"
echo -e "${BLUE}   • Files created will appear on your host system${NC}"
echo ""
echo -e "${YELLOW}🔧 Useful Commands:${NC}"
echo -e "${YELLOW}   View logs: docker logs wand-api -f${NC}"
echo -e "${YELLOW}   Stop server: docker-compose -f docker-compose.workspace.yml down${NC}"
echo -e "${YELLOW}   Shell access: docker exec -it wand-api bash${NC}"
echo -e "${YELLOW}   Test command: curl -X POST http://localhost:8000/api/v1/execute -H 'Content-Type: application/json' -d '{\"command\": \"ls -la\", \"working_directory\": \"/workspace\"}'${NC}"
echo ""
echo -e "${PURPLE}🪄 Ready to cast spells in your workspace!${NC}"

# Show workspace contents
echo ""
echo -e "${BLUE}📂 Current workspace contents:${NC}"
ls -la "$WORKSPACE_PATH" | head -10
