# MCP Distributed System - Docker Deployment Guide

## 🐳 Container Architecture

The MCP Distributed System is fully containerized with the following services:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Nginx    │  │   MCP UI    │  │   MCP API   │             │
│  │  (Port 80)  │  │ (Port 3000) │  │ (Port 8000) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ PostgreSQL  │  │    Redis    │  │   Agents    │             │
│  │ (Port 5432) │  │ (Port 6379) │  │ (50100-106) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 10GB+ disk space

### 2. Environment Setup

```bash
# Clone the repository (if not already done)
cd mcp-python

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. Build and Start

```bash
# Build all images
./scripts/docker-build.sh

# Start the complete system
docker-compose up -d

# Or start with logs
docker-compose up
```

### 4. Access Services

- **MCP UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Direct API**: http://localhost:8000/api/v1/
- **Health Check**: http://localhost:8000/health
- **Nginx Proxy**: http://localhost:80

## 📋 Available Configurations

### Development Setup
```bash
# Minimal setup for development
docker-compose -f docker-compose.dev.yml up -d
```

### Production Setup
```bash
# Full production setup with all services
docker-compose -f docker-compose.yml up -d
```

### Custom Configuration
```bash
# Use custom environment file
docker-compose --env-file .env.prod up -d
```

## 🔧 Service Configuration

### MCP Distributed System
- **Image**: `mcp-distributed-system:latest`
- **Ports**: 8000 (API), 50051 (Coordinator), 50200 (Integration), 50100-50106 (Agents)
- **Health Check**: `/health` endpoint
- **Logs**: `./logs` directory mounted

### MCP UI
- **Image**: `mcp-ui:latest`
- **Port**: 3000 (nginx)
- **API Backend**: Connects to MCP system on port 8000
- **Built**: React SPA with nginx serving

### PostgreSQL (Optional)
- **Image**: `postgres:15-alpine`
- **Port**: 5432
- **Database**: `mcp`
- **User**: `mcp_user`
- **Schema**: Auto-initialized from `scripts/init-db.sql`

### Redis (Optional)
- **Image**: `redis:7-alpine`
- **Port**: 6379
- **Persistence**: Enabled with volume

### Nginx (Optional)
- **Image**: `nginx:alpine`
- **Ports**: 80, 443
- **Configuration**: Reverse proxy for UI and API
- **SSL**: Ready for certificates in `nginx/ssl/`

## 🛠️ Management Commands

### System Operations
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart mcp-system

# View logs
docker-compose logs -f mcp-system
docker-compose logs -f mcp-ui

# Scale services (if needed)
docker-compose up -d --scale mcp-system=2
```

### Health Monitoring
```bash
# Check service status
docker-compose ps

# Health check all services
docker-compose exec mcp-system curl -f http://localhost:8000/health

# System status
curl http://localhost:8000/api/v1/system/status
```

### Database Operations
```bash
# Connect to database
docker-compose exec postgres psql -U mcp_user -d mcp

# Backup database
docker-compose exec postgres pg_dump -U mcp_user mcp > backup.sql

# Restore database
docker-compose exec -T postgres psql -U mcp_user mcp < backup.sql
```

## 📊 Monitoring and Debugging

### Container Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mcp-system

# Last 100 lines
docker-compose logs --tail=100 mcp-system
```

### Resource Usage
```bash
# Container stats
docker-compose exec mcp-system top
docker stats

# Disk usage
docker system df
```

### Debug Mode
```bash
# Start with debug logging
LOG_LEVEL=DEBUG docker-compose up

# Or edit docker-compose.yml:
environment:
  - LOG_LEVEL=DEBUG
```

## 🔒 Security Configuration

### Environment Variables
Configure these in your `.env` file:

```env
# API Security
API_KEY=your-secure-api-key
JWT_SECRET=your-jwt-secret

# Database Security
POSTGRES_PASSWORD=secure-database-password

# Integration Credentials
SLACK_BOT_TOKEN=xoxb-your-token
GITHUB_TOKEN=ghp_your-token
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### SSL/TLS Setup
```bash
# Generate self-signed certificates
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# Update nginx config to enable SSL block
# Uncomment SSL server block in nginx/nginx.conf
```

### Network Security
```bash
# Use custom network
docker network create mcp-secure-network

# Update docker-compose.yml to use custom network
```

## 🚨 Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Check what's using ports
sudo lsof -i :8000
sudo lsof -i :3000

# Change ports in docker-compose.yml
```

**Memory Issues**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Monitor container memory
docker stats
```

**Build Failures**
```bash
# Clean build cache
docker builder prune

# Rebuild from scratch
docker-compose build --no-cache

# Check Docker space
docker system df
docker system prune
```

**Service Health Issues**
```bash
# Check service logs
docker-compose logs mcp-system

# Test health endpoint directly
docker-compose exec mcp-system curl http://localhost:8000/health

# Restart unhealthy services
docker-compose restart mcp-system
```

### Reset Everything
```bash
# Stop and remove all containers, networks, volumes
docker-compose down -v --remove-orphans

# Remove all images
docker rmi $(docker images -q "mcp-*")

# Clean system
docker system prune -a

# Rebuild and restart
./scripts/docker-build.sh
docker-compose up -d
```

## 📈 Performance Tuning

### Resource Limits
```yaml
# Add to docker-compose.yml services
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      memory: 2G
```

### Volume Optimization
```bash
# Use named volumes for better performance
volumes:
  mcp_data:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
```

### Network Optimization
```bash
# Use overlay network for multi-host
docker network create -d overlay mcp-cluster
```

## 🔄 Updating

### Update Images
```bash
# Pull latest base images
docker-compose pull

# Rebuild custom images
./scripts/docker-build.sh

# Rolling update
docker-compose up -d --no-deps mcp-system
```

### Backup Before Update
```bash
# Backup volumes
docker run --rm -v mcp_data:/data -v $(pwd):/backup alpine tar czf /backup/mcp_data.tar.gz -C /data .

# Backup database
docker-compose exec postgres pg_dump -U mcp_user mcp > backup.sql
```

## 🎯 Production Deployment

### Production Checklist
- [ ] Update all default passwords
- [ ] Configure SSL certificates
- [ ] Set up log rotation
- [ ] Configure monitoring
- [ ] Set resource limits
- [ ] Enable backup automation
- [ ] Configure firewall rules
- [ ] Test disaster recovery

### Docker Swarm (Optional)
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml mcp
```

### Kubernetes (Advanced)
See `k8s/` directory for Kubernetes manifests.

---

## 🎉 Success!

Your MCP Distributed System is now fully containerized and ready for deployment!

**Available Services:**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **System Status**: http://localhost:8000/api/v1/system/status
- **Direct API**: http://localhost:8000/api/v1/

**To add UI support:**
1. Place React UI files in `./ui/` directory within mcp-python
2. Uncomment the mcp-ui service in docker-compose.yml
3. Rebuild with `./scripts/docker-build.sh`

The system is now ready for production deployment with full containerization!
