#!/bin/bash
# 🪄 Start Wand API Server - Privileged Mode for Development

set -e

# Colors for output
PURPLE='\033[0;35m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🪄 Wand API Server - Privileged Mode${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: This runs in privileged mode!${NC}"
echo -e "${RED}   - Full host system access${NC}"
echo -e "${RED}   - Only for development/testing${NC}"
echo -e "${RED}   - NOT for production use${NC}"
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo -e "${YELLOW}Please start Docker Desktop and try again${NC}"
    exit 1
fi

# Create required directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p logs data config

# Create workspace directory (ask for sudo if needed)
if [ ! -d "/workspace" ]; then
    echo -e "${YELLOW}📁 Creating /workspace directory (requires sudo)...${NC}"
    sudo mkdir -p /workspace
    sudo chmod 755 /workspace
    echo -e "${GREEN}✅ /workspace directory created${NC}"
else
    echo -e "${GREEN}✅ /workspace directory already exists${NC}"
fi

# Build and start the container
echo -e "${BLUE}🐳 Building and starting Wand API container...${NC}"
docker-compose -f docker-compose.privileged.yml up --build -d

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
echo -e "${GREEN}🚀 Wand API Server Started Successfully!${NC}"
echo ""
echo -e "${BLUE}📍 API Server: http://localhost:8000${NC}"
echo -e "${BLUE}📊 Health Check: http://localhost:8000/health${NC}"
echo -e "${BLUE}📈 Metrics: http://localhost:9100/metrics${NC}"
echo -e "${BLUE}📋 API Docs: http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}🔧 Useful Commands:${NC}"
echo -e "${YELLOW}   View logs: docker logs wand-api -f${NC}"
echo -e "${YELLOW}   Stop server: docker-compose -f docker-compose.privileged.yml down${NC}"
echo -e "${YELLOW}   Shell access: docker exec -it wand-api bash${NC}"
echo ""
echo -e "${PURPLE}🪄 Ready to cast API spells!${NC}"
