#!/bin/bash

# MCP Platform Deployment Script
# This script provides various deployment utilities for the MCP platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=${ENVIRONMENT:-production}
DOCKER_COMPOSE_FILE=${DOCKER_COMPOSE_FILE:-docker-compose.yml}
PROJECT_NAME=${PROJECT_NAME:-mcp-platform}

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

check_requirements() {
    log_info "Checking deployment requirements..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if .env file exists
    if [ ! -f .env ]; then
        log_warning ".env file not found. Creating from template..."
        create_env_file
    fi

    log_success "Requirements check passed"
}

create_env_file() {
    cat > .env << EOF
# MCP Platform Environment Configuration

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security (CHANGE THESE IN PRODUCTION!)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production-$(openssl rand -hex 32)
API_KEY_SECRET=your-api-key-secret-change-in-production-$(openssl rand -hex 32)

# Database
POSTGRES_PASSWORD=mcp_password_$(openssl rand -hex 16)

# Monitoring
GRAFANA_PASSWORD=admin_$(openssl rand -hex 8)
ENABLE_METRICS=true

# Framework Settings
HAYSTACK_ENABLED=true
LLAMAINDEX_ENABLED=true

# Optional: External API Keys
# OPENAI_API_KEY=your-openai-api-key
# HUGGING_FACE_API_TOKEN=your-hugging-face-token
EOF

    log_success "Created .env file with random secrets"
    log_warning "Please review and update the .env file with your specific configuration"
}

build_images() {
    log_info "Building Docker images..."
    docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME build --no-cache
    log_success "Docker images built successfully"
}

start_services() {
    log_info "Starting MCP Platform services..."
    docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME up -d
    log_success "Services started successfully"

    log_info "Waiting for services to be healthy..."
    sleep 30

    # Check service health
    check_health
}

stop_services() {
    log_info "Stopping MCP Platform services..."
    docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME down
    log_success "Services stopped successfully"
}

restart_services() {
    log_info "Restarting MCP Platform services..."
    stop_services
    start_services
}

check_health() {
    log_info "Checking service health..."

    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_success "MCP API is healthy"
            break
        fi

        if [ $i -eq 30 ]; then
            log_error "MCP API health check failed after 30 attempts"
            show_logs mcp-api
            exit 1
        fi

        log_info "Waiting for API to be ready... (attempt $i/30)"
        sleep 10
    done

    # Check database health
    if docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME exec -T postgres pg_isready -U mcp_user -d mcp_platform > /dev/null 2>&1; then
        log_success "PostgreSQL is healthy"
    else
        log_error "PostgreSQL health check failed"
    fi

    # Check Redis health
    if docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME exec -T redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis is healthy"
    else
        log_error "Redis health check failed"
    fi
}

show_logs() {
    local service=${1:-}
    if [ -z "$service" ]; then
        log_info "Showing logs for all services..."
        docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME logs -f
    else
        log_info "Showing logs for service: $service"
        docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME logs -f $service
    fi
}

show_status() {
    log_info "Service status:"
    docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME ps

    echo
    log_info "Service endpoints:"
    echo "  - API: http://localhost:8000"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo "  - Health Check: http://localhost:8000/health"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9091"
}

backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $backup_dir

    log_info "Creating backup in $backup_dir..."

    # Backup database
    docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME exec -T postgres pg_dump -U mcp_user mcp_platform > $backup_dir/database.sql

    # Backup application data
    docker cp ${PROJECT_NAME}_mcp-api_1:/app/data $backup_dir/app_data 2>/dev/null || true

    # Backup logs
    docker cp ${PROJECT_NAME}_mcp-api_1:/app/logs $backup_dir/app_logs 2>/dev/null || true

    log_success "Backup created in $backup_dir"
}

run_migrations() {
    log_info "Running database migrations..."
    docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME exec mcp-api python -c "
from database.init import quick_database_setup
from database.migrations import get_migration_manager
import os

# Initialize database
db_url = os.getenv('DATABASE_URL', 'postgresql://mcp_user:mcp_password@postgres:5432/mcp_platform')
initializer = quick_database_setup(db_url)

# Run migrations
manager = get_migration_manager()
applied = manager.migrate()
print(f'Applied migrations: {applied}')
"
    log_success "Database migrations completed"
}

cleanup() {
    log_info "Cleaning up Docker resources..."

    # Remove stopped containers
    docker-compose -f $DOCKER_COMPOSE_FILE -p $PROJECT_NAME down --remove-orphans

    # Remove unused images
    docker image prune -f

    # Remove unused volumes (be careful!)
    read -p "Do you want to remove unused volumes? This will delete data! (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
        log_success "Volumes cleaned up"
    fi

    log_success "Cleanup completed"
}

# Main script logic
case "${1:-}" in
    "start")
        check_requirements
        start_services
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        show_status
        ;;
    "build")
        check_requirements
        build_images
        ;;
    "deploy")
        check_requirements
        build_images
        start_services
        run_migrations
        show_status
        ;;
    "logs")
        show_logs ${2:-}
        ;;
    "status")
        show_status
        ;;
    "health")
        check_health
        ;;
    "backup")
        backup_data
        ;;
    "migrate")
        run_migrations
        ;;
    "cleanup")
        cleanup
        ;;
    "env")
        create_env_file
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|build|deploy|logs|status|health|backup|migrate|cleanup|env}"
        echo
        echo "Commands:"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  build    - Build Docker images"
        echo "  deploy   - Full deployment (build + start + migrate)"
        echo "  logs     - Show logs (optionally for specific service)"
        echo "  status   - Show service status and endpoints"
        echo "  health   - Check service health"
        echo "  backup   - Create backup of data and database"
        echo "  migrate  - Run database migrations"
        echo "  cleanup  - Clean up Docker resources"
        echo "  env      - Create .env file from template"
        echo
        echo "Examples:"
        echo "  $0 deploy                 # Full deployment"
        echo "  $0 logs mcp-api          # Show API logs"
        echo "  $0 start                 # Start services"
        exit 1
        ;;
esac
