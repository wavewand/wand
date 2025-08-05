# MCP Platform Deployment Guide

This directory contains deployment configurations and scripts for the MCP Platform across different environments and orchestration platforms.

## Quick Start

### Docker Compose (Recommended for Development/Testing)

1. **Setup Environment**:
   ```bash
   # Create environment file from template
   cp .env.example .env

   # Edit .env with your configuration
   nano .env
   ```

2. **Deploy with Script**:
   ```bash
   # Make deploy script executable
   chmod +x deployment/scripts/deploy.sh

   # Full deployment (build + start + migrate)
   ./deployment/scripts/deploy.sh deploy

   # Or step by step:
   ./deployment/scripts/deploy.sh build
   ./deployment/scripts/deploy.sh start
   ./deployment/scripts/deploy.sh migrate
   ```

3. **Verify Deployment**:
   ```bash
   # Check service status
   ./deployment/scripts/deploy.sh status

   # Check health
   ./deployment/scripts/deploy.sh health

   # View logs
   ./deployment/scripts/deploy.sh logs
   ```

### Kubernetes (Production)

1. **Prepare Cluster**:
   ```bash
   # Create namespace and resources
   kubectl apply -f deployment/kubernetes/namespace.yaml
   kubectl apply -f deployment/kubernetes/configmap.yaml
   ```

2. **Deploy Services**:
   ```bash
   # Deploy PostgreSQL
   kubectl apply -f deployment/kubernetes/postgres.yaml

   # Deploy Redis
   kubectl apply -f deployment/kubernetes/redis.yaml

   # Deploy MCP API
   kubectl apply -f deployment/kubernetes/mcp-api.yaml
   ```

3. **Verify Deployment**:
   ```bash
   # Check pods
   kubectl get pods -n mcp-platform

   # Check services
   kubectl get services -n mcp-platform

   # Check ingress
   kubectl get ingress -n mcp-platform
   ```

## Architecture

### Services

- **mcp-api**: Main application API server
- **postgres**: PostgreSQL database for persistent storage
- **redis**: Redis for caching and session storage
- **nginx**: Reverse proxy and load balancer
- **prometheus**: Metrics collection
- **grafana**: Metrics visualization

### Ports

- **80/443**: HTTP/HTTPS (Nginx)
- **3000**: Grafana dashboard
- **5432**: PostgreSQL (internal)
- **6379**: Redis (internal)
- **8000**: MCP API (internal)
- **9090**: Application metrics (internal)
- **9091**: Prometheus (external)

## Configuration

### Environment Variables

See `.env.example` for all available configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `REDIS_URL` | Redis connection URL | - |
| `JWT_SECRET_KEY` | JWT signing secret | **Must change** |
| `API_KEY_SECRET` | API key signing secret | **Must change** |

### Security Configuration

**Critical**: Always change default secrets in production:

```bash
# Generate secure secrets
openssl rand -hex 32  # For JWT_SECRET_KEY
openssl rand -hex 32  # For API_KEY_SECRET
openssl rand -hex 16  # For passwords
```

## Monitoring

### Grafana Dashboard

Access Grafana at `http://localhost:3000`:
- Username: `admin`
- Password: Set via `GRAFANA_PASSWORD` environment variable

### Prometheus Metrics

Access Prometheus at `http://localhost:9091` for:
- Application metrics
- System metrics
- Custom business metrics

### Application Health

Health check endpoint: `http://localhost:8000/health`

Returns:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "frameworks": "healthy"
  }
}
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
./deployment/scripts/deploy.sh backup

# Manual backup
docker-compose exec postgres pg_dump -U mcp_user mcp_platform > backup.sql
```

### Data Recovery

```bash
# Restore from backup
docker-compose exec -T postgres psql -U mcp_user mcp_platform < backup.sql
```

## Scaling

### Docker Compose Scaling

```bash
# Scale API service
docker-compose up -d --scale mcp-api=3
```

### Kubernetes Scaling

```bash
# Scale deployment
kubectl scale deployment mcp-api --replicas=5 -n mcp-platform

# Horizontal Pod Autoscaler
kubectl autoscale deployment mcp-api --cpu-percent=70 --min=2 --max=10 -n mcp-platform
```

## SSL/TLS Configuration

### Development (Self-Signed)

```bash
# Generate self-signed certificate
mkdir -p deployment/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/nginx/ssl/key.pem \
  -out deployment/nginx/ssl/cert.pem
```

### Production (Let's Encrypt)

```bash
# Use Certbot for Let's Encrypt
certbot certonly --webroot -w /var/www/html -d your-domain.com
```

## Troubleshooting

### Common Issues

1. **Services won't start**:
   ```bash
   # Check logs
   ./deployment/scripts/deploy.sh logs

   # Check Docker resources
   docker system df
   docker system prune
   ```

2. **Database connection issues**:
   ```bash
   # Test database connectivity
   docker-compose exec postgres pg_isready -U mcp_user -d mcp_platform

   # Check database logs
   ./deployment/scripts/deploy.sh logs postgres
   ```

3. **Memory issues**:
   ```bash
   # Check resource usage
   docker stats

   # Adjust memory limits in docker-compose.yml
   ```

### Log Analysis

```bash
# Application logs
./deployment/scripts/deploy.sh logs mcp-api

# Database logs
./deployment/scripts/deploy.sh logs postgres

# All services
./deployment/scripts/deploy.sh logs
```

### Performance Tuning

1. **Database optimization**:
   - Adjust PostgreSQL configuration
   - Monitor query performance
   - Use connection pooling

2. **Application optimization**:
   - Adjust worker count
   - Configure caching
   - Monitor memory usage

3. **Network optimization**:
   - Use nginx caching
   - Enable compression
   - Configure keep-alive

## Development vs Production

### Development Setup

```bash
# Use development target
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Or use development script flags
./deployment/scripts/deploy.sh start --dev
```

### Production Checklist

- [ ] Change all default passwords and secrets
- [ ] Configure SSL/TLS certificates
- [ ] Set up proper backup strategy
- [ ] Configure monitoring and alerting
- [ ] Review resource limits and requests
- [ ] Set up log rotation
- [ ] Configure firewall rules
- [ ] Test disaster recovery procedures

## Contributing

When adding new services or configurations:

1. Update docker-compose.yml
2. Add Kubernetes manifests
3. Update this documentation
4. Test in both development and production modes
5. Update backup/restore procedures if needed

## Support

For issues and questions:
1. Check the logs first
2. Review this documentation
3. Check the main project README
4. Create an issue with detailed information
