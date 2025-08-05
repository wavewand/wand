#!/bin/bash

# 🪄 Wand Integration System - Production Deployment Script
# Automated deployment with health checks and rollback capabilities

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
WAND_VERSION=${WAND_VERSION:-"latest"}
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"production"}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-7}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-300}
COMPOSE_FILE="docker-compose.production.yml"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_magic() {
    echo -e "${PURPLE}🪄 [WAND]${NC} $1"
}

check_prerequisites() {
    log_info "Checking deployment prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check if running as root for production
    if [[ "$DEPLOYMENT_ENV" == "production" && $EUID -ne 0 ]]; then
        log_warning "Running production deployment as non-root user"
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Check disk space (minimum 5GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then
        log_warning "Low disk space detected (less than 5GB available)"
    fi

    log_success "Prerequisites check passed"
}

create_directories() {
    log_info "Creating necessary directories..."

    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "${PROJECT_ROOT}/config"
    mkdir -p "${PROJECT_ROOT}/data"

    log_success "Directories created"
}

backup_current_deployment() {
    log_info "Creating backup of current deployment..."

    local backup_timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_path="${BACKUP_DIR}/wand_backup_${backup_timestamp}"

    mkdir -p "$backup_path"

    # Backup configuration
    if [[ -d "${PROJECT_ROOT}/config" ]]; then
        cp -r "${PROJECT_ROOT}/config" "$backup_path/"
        log_info "Configuration backed up"
    fi

    # Backup data
    if [[ -d "${PROJECT_ROOT}/data" ]]; then
        cp -r "${PROJECT_ROOT}/data" "$backup_path/"
        log_info "Data backed up"
    fi

    # Export current containers if running
    if docker-compose -f "$COMPOSE_FILE" ps -q > /dev/null 2>&1; then
        docker-compose -f "$COMPOSE_FILE" config > "${backup_path}/docker-compose.yml"
        log_info "Docker configuration backed up"
    fi

    # Cleanup old backups
    find "$BACKUP_DIR" -name "wand_backup_*" -type d -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} +

    log_success "Backup created at $backup_path"
    echo "$backup_path" > "${PROJECT_ROOT}/.last_backup"
}

create_production_compose() {
    log_info "Creating production Docker Compose configuration..."

    cat > "$COMPOSE_FILE" << EOF
version: '3.8'

services:
  wand-api:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        WAND_VERSION: ${WAND_VERSION}
    image: wand-integration-system:${WAND_VERSION}
    container_name: wand-api-prod
    restart: unless-stopped
    ports:
      - "8001:8001"
      - "9100:9100"  # Metrics port
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - \${WORKSPACE_DIR:-./workspace}:/workspace
    environment:
      - WAND_ENV=production
      - WAND_LOG_LEVEL=INFO
      - WAND_METRICS_ENABLED=true
      - WAND_HEALTH_CHECK_ENABLED=true
    networks:
      - wand-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'

  wand-ui:
    build:
      context: ../mcp-ui
      dockerfile: Dockerfile
    image: wand-ui:${WAND_VERSION}
    container_name: wand-ui-prod
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - REACT_APP_API_URL=http://localhost:8001
      - REACT_APP_DISABLE_AUTH=false
    networks:
      - wand-network
    depends_on:
      wand-api:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: wand-redis-prod
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - wand-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: wand-nginx-prod
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    networks:
      - wand-network
    depends_on:
      - wand-api
      - wand-ui

volumes:
  redis-data:
    driver: local

networks:
  wand-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

    log_success "Production Docker Compose file created"
}

create_nginx_config() {
    log_info "Creating Nginx reverse proxy configuration..."

    cat > nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream wand-api {
        server wand-api:8001;
    }

    upstream wand-ui {
        server wand-ui:3000;
    }

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=ui:10m rate=50r/s;

    server {
        listen 80;
        server_name _;

        # Redirect HTTP to HTTPS in production
        # return 301 https://\$server_name\$request_uri;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://wand-api/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;

            # CORS headers
            add_header Access-Control-Allow-Origin "*" always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization" always;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://wand-api/health;
            access_log off;
        }

        # Metrics endpoint (restricted)
        location /metrics {
            proxy_pass http://wand-api:9100/metrics;
            allow 127.0.0.1;
            allow 172.20.0.0/16;
            deny all;
        }

        # UI application
        location / {
            limit_req zone=ui burst=100 nodelay;
            proxy_pass http://wand-ui/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }

    # HTTPS configuration (uncomment for production with SSL)
    # server {
    #     listen 443 ssl http2;
    #     server_name your-domain.com;
    #
    #     ssl_certificate /etc/nginx/ssl/cert.pem;
    #     ssl_certificate_key /etc/nginx/ssl/key.pem;
    #
    #     # SSL configuration
    #     ssl_protocols TLSv1.2 TLSv1.3;
    #     ssl_ciphers HIGH:!aNULL:!MD5;
    #     ssl_prefer_server_ciphers on;
    #
    #     # Same location blocks as HTTP server
    # }
}
EOF

    log_success "Nginx configuration created"
}

deploy_application() {
    log_info "Deploying Wand Integration System..."

    # Build and start services
    log_info "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache

    log_info "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d

    log_success "Application deployed"
}

wait_for_health_checks() {
    log_info "Waiting for services to become healthy..."

    local timeout=$HEALTH_CHECK_TIMEOUT
    local interval=10
    local elapsed=0

    while [[ $elapsed -lt $timeout ]]; do
        local healthy_services=0
        local total_services=0

        # Check each service health
        for service in wand-api wand-ui redis; do
            total_services=$((total_services + 1))
            if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "healthy\|Up"; then
                healthy_services=$((healthy_services + 1))
            fi
        done

        log_info "Health check: $healthy_services/$total_services services healthy"

        if [[ $healthy_services -eq $total_services ]]; then
            log_success "All services are healthy!"
            return 0
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log_error "Health check timeout after ${timeout}s"
    return 1
}

run_smoke_tests() {
    log_info "Running smoke tests..."

    # Test API health endpoint
    if curl -f -s http://localhost:8001/health > /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi

    # Test UI availability
    if curl -f -s http://localhost:3000 > /dev/null; then
        log_success "UI availability check passed"
    else
        log_error "UI availability check failed"
        return 1
    fi

    # Test integration endpoint
    if curl -f -s http://localhost:8001/status > /dev/null; then
        log_success "Integration status check passed"
    else
        log_warning "Integration status check failed (non-critical)"
    fi

    log_success "Smoke tests completed"
}

rollback_deployment() {
    log_error "Deployment failed, initiating rollback..."

    # Stop current deployment
    docker-compose -f "$COMPOSE_FILE" down

    # Restore from last backup if available
    if [[ -f "${PROJECT_ROOT}/.last_backup" ]]; then
        local backup_path=$(cat "${PROJECT_ROOT}/.last_backup")
        if [[ -d "$backup_path" ]]; then
            log_info "Restoring from backup: $backup_path"

            # Restore configuration and data
            if [[ -d "${backup_path}/config" ]]; then
                rm -rf "${PROJECT_ROOT}/config"
                cp -r "${backup_path}/config" "${PROJECT_ROOT}/"
            fi

            if [[ -d "${backup_path}/data" ]]; then
                rm -rf "${PROJECT_ROOT}/data"
                cp -r "${backup_path}/data" "${PROJECT_ROOT}/"
            fi

            log_success "Backup restored"
        fi
    fi

    log_info "Please check logs and fix issues before redeploying"
    exit 1
}

cleanup_resources() {
    log_info "Cleaning up unused Docker resources..."

    # Remove unused images
    docker image prune -f

    # Remove unused volumes (be careful with this)
    # docker volume prune -f

    log_success "Cleanup completed"
}

show_deployment_info() {
    log_magic "🌟 Wand Integration System Deployment Complete! 🌟"
    echo
    log_info "Access URLs:"
    echo "  • Web UI: http://localhost:3000"
    echo "  • API: http://localhost:8001"
    echo "  • Health Check: http://localhost:8001/health"
    echo "  • Metrics: http://localhost:9100/metrics (local access only)"
    echo
    log_info "Management Commands:"
    echo "  • View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  • Restart services: docker-compose -f $COMPOSE_FILE restart"
    echo "  • Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  • View status: docker-compose -f $COMPOSE_FILE ps"
    echo
    log_info "Configuration:"
    echo "  • Config directory: ${PROJECT_ROOT}/config"
    echo "  • Data directory: ${PROJECT_ROOT}/data"
    echo "  • Logs directory: ${PROJECT_ROOT}/logs"
    echo "  • Backups directory: ${BACKUP_DIR}"
    echo
    log_magic "Happy magical automation! ✨"
}

# Main deployment function
main() {
    log_magic "🪄 Starting Wand Integration System Production Deployment"
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Version: $WAND_VERSION"
    echo "Timestamp: $(date)"
    echo

    # Deployment steps
    check_prerequisites
    create_directories
    backup_current_deployment
    create_production_compose
    create_nginx_config
    deploy_application

    # Health checks and validation
    if wait_for_health_checks; then
        if run_smoke_tests; then
            cleanup_resources
            show_deployment_info
            log_success "🎉 Deployment completed successfully!"
        else
            rollback_deployment
        fi
    else
        rollback_deployment
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "cleanup")
        cleanup_resources
        ;;
    "status")
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    "stop")
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    "restart")
        docker-compose -f "$COMPOSE_FILE" restart "${2:-}"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|cleanup|status|logs|stop|restart}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy the Wand system (default)"
        echo "  rollback - Rollback to previous deployment"
        echo "  cleanup  - Clean up unused Docker resources"
        echo "  status   - Show service status"
        echo "  logs     - Show service logs"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart services"
        exit 1
        ;;
esac
