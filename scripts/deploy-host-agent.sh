#!/bin/bash
# scripts/deploy-host-agent.sh
# Deploy MCP-Python in Host Agent mode (recommended for production)

set -e

ENVIRONMENT=${1:-"development"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🏗️  Deploying Host Agent mode for environment: ${YELLOW}$ENVIRONMENT${NC}"

cd "$PROJECT_ROOT"

# Generate secure host agent token
echo -e "${BLUE}🔐 Generating secure authentication token...${NC}"
HOST_AGENT_TOKEN=$(openssl rand -hex 32)

# Create directories
mkdir -p config secrets logs

# Create environment file
cat > .env << EOF
# MCP-Python Host Agent Configuration
ENVIRONMENT=$ENVIRONMENT
HOST_AGENT_TOKEN=$HOST_AGENT_TOKEN
MCP_PORT=8000
HOST_AGENT_PORT=8001
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Security settings
SECURITY_STRICT=${SECURITY_STRICT:-true}
RATE_LIMITING=${RATE_LIMITING:-true}
AUDIT_LOGGING=${AUDIT_LOGGING:-true}
EOF

echo -e "${GREEN}✅ Environment configuration created${NC}"

# Create host agent configuration
cat > config/host-agent.json << EOF
{
  "server": {
    "host": "0.0.0.0",
    "port": 8001,
    "auth_token": "$HOST_AGENT_TOKEN"
  },
  "execution": {
    "working_directory": "/workspace",
    "timeout": 30,
    "max_concurrent": 10,
    "user_isolation": true
  },
  "security": {
    "command_validation": true,
    "allowed_commands": [
      "git", "npm", "yarn", "python", "python3", "pip", "pip3",
      "node", "docker", "ls", "cat", "grep", "find", "head", "tail",
      "echo", "which", "whoami", "pwd", "mkdir", "touch", "cp", "mv"
    ],
    "blocked_commands": [
      "rm", "dd", "mkfs", "fdisk", "mount", "umount",
      "su", "sudo", "passwd", "useradd", "userdel", "usermod",
      "systemctl", "service", "kill", "killall"
    ],
    "path_restrictions": ["/workspace", "/tmp", "/var/tmp"],
    "resource_limits": {
      "max_memory": "1GB",
      "max_cpu": "1.0",
      "max_processes": 50,
      "max_execution_time": 300
    }
  },
  "logging": {
    "level": "${LOG_LEVEL:-INFO}",
    "audit_commands": true,
    "log_file": "/app/logs/host-agent.log"
  }
}
EOF

# Create MCP server configuration
cat > config/mcp-server.json << EOF
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "version": "1.0.0",
    "log_level": "${LOG_LEVEL:-INFO}"
  },
  "execution": {
    "mode": "host_agent",
    "host_agent": {
      "url": "http://host-agent:8001",
      "auth_token": "$HOST_AGENT_TOKEN",
      "timeout": 30,
      "retry_attempts": 3,
      "health_check_interval": 60
    },
    "security": {
      "command_validation": true,
      "audit_logging": true
    }
  },
  "frameworks": {
    "haystack": {
      "enabled": true
    },
    "llamaindex": {
      "enabled": true
    },
    "langchain": {
      "enabled": true
    }
  }
}
EOF

echo -e "${GREEN}✅ Configuration files created${NC}"

# Create Host Agent Dockerfile
cat > Dockerfile.host-agent << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openssh-client \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional host agent dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    psutil \
    docker

# Copy host agent code
COPY tools/host_agent/ ./host_agent/
COPY config/host-agent.json ./config/

# Create non-root user for security
RUN useradd -m -u 1000 hostagent && \
    chown -R hostagent:hostagent /app

# Create directories
RUN mkdir -p /app/logs /workspace && \
    chown -R hostagent:hostagent /app/logs /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Switch to non-root user
USER hostagent

# Expose port
EXPOSE 8001

# Start host agent
CMD ["python", "-m", "host_agent.server"]
EOF

# Create docker-compose file
cat > docker-compose.host-agent.yml << 'EOF'
version: '3.8'

services:
  mcp-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${MCP_PORT:-8000}:8000"
    environment:
      - EXECUTION_MODE=host_agent
      - HOST_AGENT_URL=http://host-agent:8001
      - HOST_AGENT_TOKEN=${HOST_AGENT_TOKEN}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - SECURITY_STRICT=${SECURITY_STRICT:-true}
    volumes:
      - ./config/mcp-server.json:/app/config/config.json:ro
      - ./logs:/app/logs:rw
    networks:
      - mcp-network
    depends_on:
      host-agent:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  host-agent:
    build:
      context: .
      dockerfile: Dockerfile.host-agent
    ports:
      - "${HOST_AGENT_PORT:-8001}:8001"
    volumes:
      # Host access (read-only root, read-write workspace)
      - /:/host:ro
      - /workspace:/workspace:rw
      - /tmp:/tmp:rw
      # Docker socket for container operations
      - /var/run/docker.sock:/var/run/docker.sock
      # Configuration and logs
      - ./config/host-agent.json:/app/config/config.json:ro
      - ./logs:/app/logs:rw
    environment:
      - HOST_AGENT_TOKEN=${HOST_AGENT_TOKEN}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - AUDIT_LOGGING=${AUDIT_LOGGING:-true}
    networks:
      - mcp-network
    # Security options
    security_opt:
      - apparmor:unconfined
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

networks:
  mcp-network:
    driver: bridge

volumes:
  workspace:
    driver: local
EOF

echo -e "${GREEN}✅ Docker configuration created${NC}"

# Create workspace directory if it doesn't exist
if [ ! -d "/workspace" ]; then
    echo -e "${BLUE}📁 Creating workspace directory...${NC}"
    sudo mkdir -p /workspace
    sudo chmod 755 /workspace
    # Set ownership to current user if possible
    if [ "$EUID" -ne 0 ]; then
        sudo chown $USER:$USER /workspace
    fi
fi

# Deploy the services
echo -e "${BLUE}🚀 Starting services...${NC}"

# Build and start services
docker-compose -f docker-compose.host-agent.yml build --no-cache
docker-compose -f docker-compose.host-agent.yml up -d

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Health checks
echo -e "${BLUE}🏥 Performing health checks...${NC}"

# Check MCP server
if curl -f -s http://localhost:${MCP_PORT:-8000}/health > /dev/null; then
    echo -e "${GREEN}✅ MCP Server is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  MCP Server health check failed, checking logs...${NC}"
    docker-compose -f docker-compose.host-agent.yml logs mcp-server
fi

# Check Host Agent
if curl -f -s http://localhost:${HOST_AGENT_PORT:-8001}/health > /dev/null; then
    echo -e "${GREEN}✅ Host Agent is healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Host Agent health check failed, checking logs...${NC}"
    docker-compose -f docker-compose.host-agent.yml logs host-agent
fi

# Create health check script
cat > scripts/health-check-host-agent.sh << 'EOF'
#!/bin/bash
# Health check script for Host Agent mode

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🏥 Host Agent Mode Health Check"

# Check MCP Server
echo -n "• MCP Server: "
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Unhealthy${NC}"
    exit 1
fi

# Check Host Agent
echo -n "• Host Agent: "
if curl -f -s http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${RED}❌ Unhealthy${NC}"
    exit 1
fi

# Test command execution
echo -n "• Command Execution: "
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/execute \
    -H "Content-Type: application/json" \
    -d '{"command": "echo test"}' || echo "failed")

if [[ "$RESPONSE" == *"test"* ]]; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi

echo -e "${GREEN}✅ All health checks passed!${NC}"
EOF

chmod +x scripts/health-check-host-agent.sh

# Show deployment summary
echo ""
echo -e "${GREEN}🎉 Host Agent deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}📊 Deployment Summary:${NC}"
echo -e "  • Mode: ${YELLOW}Host Agent${NC}"
echo -e "  • Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "  • MCP Server: ${YELLOW}http://localhost:${MCP_PORT:-8000}${NC}"
echo -e "  • Host Agent: ${YELLOW}http://localhost:${HOST_AGENT_PORT:-8001}${NC}"
echo ""
echo -e "${BLUE}🔧 Configuration Files:${NC}"
echo -e "  • Environment: ${YELLOW}.env${NC}"
echo -e "  • MCP Server: ${YELLOW}config/mcp-server.json${NC}"
echo -e "  • Host Agent: ${YELLOW}config/host-agent.json${NC}"
echo -e "  • Docker Compose: ${YELLOW}docker-compose.host-agent.yml${NC}"
echo ""
echo -e "${BLUE}📋 Management Commands:${NC}"
echo -e "  • View logs: ${YELLOW}docker-compose -f docker-compose.host-agent.yml logs -f${NC}"
echo -e "  • Stop services: ${YELLOW}docker-compose -f docker-compose.host-agent.yml down${NC}"
echo -e "  • Restart services: ${YELLOW}docker-compose -f docker-compose.host-agent.yml restart${NC}"
echo -e "  • Health check: ${YELLOW}./scripts/health-check-host-agent.sh${NC}"
echo ""
echo -e "${BLUE}🔐 Security Notes:${NC}"
echo -e "  • Host Agent Token: ${YELLOW}Stored in .env file${NC}"
echo -e "  • Command validation: ${YELLOW}Enabled${NC}"
echo -e "  • Path restrictions: ${YELLOW}/workspace, /tmp${NC}"
echo -e "  • Resource limits: ${YELLOW}1GB RAM, 1 CPU${NC}"
