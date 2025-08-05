#!/bin/bash
# scripts/deploy-quick.sh
# Quick deployment script for MCP-Python with configurable execution modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
EXECUTION_MODE=${1:-"host_agent"}
ENVIRONMENT=${2:-"development"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Banner
echo -e "${BLUE}"
cat << 'EOF'
  __  __  ______ _____     _____       _   _
 |  \/  |/ _____||  __ \   |  __ \    | | | |
 | |\/| | |      | |__) |  | |__) |   | |_| |__   ___  _ __
 | |  | | |      |  ___/   |  ___/    | __| '_ \ / _ \| '_ \
 | |  | | |_____ | |       | |        | |_| | | | (_) | | | |
 |_|  |_|\______||_|       |_|         \__|_| |_|\___/|_| |_|

           Host Command Execution Deployment
EOF
echo -e "${NC}"

echo -e "${GREEN}🚀 Deploying MCP-Python with execution mode: ${YELLOW}$EXECUTION_MODE${NC}"
echo -e "${GREEN}🌍 Environment: ${YELLOW}$ENVIRONMENT${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}📋 Checking prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is required but not installed${NC}"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is required but not installed${NC}"
        exit 1
    fi

    # Check Python for native mode
    if [[ "$EXECUTION_MODE" == "native" ]] && ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is required for native mode${NC}"
        exit 1
    fi

    # Check SSH for ssh_remote mode
    if [[ "$EXECUTION_MODE" == "ssh_remote" ]] && ! command -v ssh &> /dev/null; then
        echo -e "${RED}❌ SSH is required for ssh_remote mode${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Show deployment information
show_deployment_info() {
    echo -e "${BLUE}📊 Deployment Information:${NC}"
    echo -e "  Mode: ${YELLOW}$EXECUTION_MODE${NC}"
    echo -e "  Environment: ${YELLOW}$ENVIRONMENT${NC}"
    echo -e "  Project Root: ${YELLOW}$PROJECT_ROOT${NC}"
    echo ""

    case $EXECUTION_MODE in
        "native")
            echo -e "${YELLOW}ℹ️  Native Mode:${NC} Direct execution on host system"
            echo -e "   • Best for: Development, single-user environments"
            echo -e "   • Security: Medium (user-limited)"
            ;;
        "host_agent")
            echo -e "${YELLOW}ℹ️  Host Agent Mode:${NC} Separate privileged agent service"
            echo -e "   • Best for: Production, multi-tenant environments"
            echo -e "   • Security: High (configurable isolation)"
            ;;
        "docker_socket")
            echo -e "${YELLOW}ℹ️  Docker Socket Mode:${NC} Execute via Docker containers"
            echo -e "   • Best for: CI/CD, container workflows"
            echo -e "   • Security: Medium (container isolation)"
            ;;
        "ssh_remote")
            echo -e "${YELLOW}ℹ️  SSH Remote Mode:${NC} SSH back to host from container"
            echo -e "   • Best for: Secure isolated environments"
            echo -e "   • Security: High (SSH user limits)"
            ;;
        "volume_mount")
            echo -e "${YELLOW}ℹ️  Volume Mount Mode:${NC} Mount host binaries and directories"
            echo -e "   • Best for: Simple containerization"
            echo -e "   • Security: Low-Medium (mount-limited)"
            ;;
        "privileged")
            echo -e "${RED}⚠️  Privileged Mode:${NC} Full host access (DANGEROUS)"
            echo -e "   • Best for: Testing only"
            echo -e "   • Security: Critical Risk (breaks container security)"
            ;;
    esac
    echo ""
}

# Confirmation prompt
confirm_deployment() {
    if [[ "$EXECUTION_MODE" == "privileged" ]]; then
        echo -e "${RED}⚠️  WARNING: Privileged mode is insecure and breaks container isolation!${NC}"
        echo -e "${RED}    This mode should ONLY be used for development/testing.${NC}"
        echo ""
        read -p "$(echo -e ${RED}Are you absolutely sure you want to continue? ${NC}(y/N): )" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}🚫 Deployment cancelled${NC}"
            exit 0
        fi
    else
        read -p "$(echo -e ${GREEN}Proceed with deployment? ${NC}(Y/n): )" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo -e "${YELLOW}🚫 Deployment cancelled${NC}"
            exit 0
        fi
    fi
}

# Deploy based on execution mode
deploy_mode() {
    case $EXECUTION_MODE in
        "native")
            echo -e "${GREEN}📦 Deploying Native Mode...${NC}"
            "$SCRIPT_DIR/deploy-native.sh" "$ENVIRONMENT"
            ;;
        "host_agent")
            echo -e "${GREEN}🏗️  Deploying Host Agent Mode...${NC}"
            "$SCRIPT_DIR/deploy-host-agent.sh" "$ENVIRONMENT"
            ;;
        "docker_socket")
            echo -e "${GREEN}🐳 Deploying Docker Socket Mode...${NC}"
            "$SCRIPT_DIR/deploy-docker-socket.sh" "$ENVIRONMENT"
            ;;
        "ssh_remote")
            echo -e "${GREEN}🔐 Deploying SSH Remote Mode...${NC}"
            "$SCRIPT_DIR/deploy-ssh-remote.sh" "$ENVIRONMENT"
            ;;
        "volume_mount")
            echo -e "${GREEN}📁 Deploying Volume Mount Mode...${NC}"
            "$SCRIPT_DIR/deploy-volume-mount.sh" "$ENVIRONMENT"
            ;;
        "privileged")
            echo -e "${RED}⚠️  Deploying Privileged Mode...${NC}"
            "$SCRIPT_DIR/deploy-privileged.sh" "$ENVIRONMENT"
            ;;
        *)
            echo -e "${RED}❌ Unknown execution mode: $EXECUTION_MODE${NC}"
            echo -e "${YELLOW}Available modes:${NC}"
            echo "  • native       - Direct host execution"
            echo "  • host_agent   - Separate privileged agent (recommended)"
            echo "  • docker_socket - Docker container execution"
            echo "  • ssh_remote   - SSH back to host"
            echo "  • volume_mount - Mount host binaries"
            echo "  • privileged   - Privileged container (dangerous)"
            exit 1
            ;;
    esac
}

# Show post-deployment information
show_post_deployment() {
    echo ""
    echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}🔗 Service URLs:${NC}"

    case $EXECUTION_MODE in
        "native")
            echo -e "  • MCP Server: ${YELLOW}http://localhost:8000${NC}"
            ;;
        "host_agent")
            echo -e "  • MCP Server: ${YELLOW}http://localhost:8000${NC}"
            echo -e "  • Host Agent: ${YELLOW}http://localhost:8001${NC}"
            ;;
        *)
            echo -e "  • MCP Server: ${YELLOW}http://localhost:8000${NC}"
            ;;
    esac

    echo ""
    echo -e "${BLUE}📖 Useful commands:${NC}"
    echo -e "  • Check status: ${YELLOW}docker-compose ps${NC}"
    echo -e "  • View logs: ${YELLOW}docker-compose logs -f${NC}"
    echo -e "  • Stop services: ${YELLOW}docker-compose down${NC}"
    echo -e "  • Health check: ${YELLOW}$SCRIPT_DIR/health-check.sh $EXECUTION_MODE${NC}"
    echo ""
    echo -e "${BLUE}📚 For more information, see:${NC}"
    echo -e "  • ${YELLOW}docs/HOST_COMMAND_EXECUTION_GUIDE.md${NC}"
    echo -e "  • ${YELLOW}docs/OPENCODE_AGENT_REQUIREMENTS.md${NC}"
}

# Cleanup on failure
cleanup_on_failure() {
    echo -e "${RED}❌ Deployment failed. Cleaning up...${NC}"

    # Stop any running containers
    if [ -f "docker-compose.yml" ] || [ -f "docker-compose.${EXECUTION_MODE}.yml" ]; then
        docker-compose -f "docker-compose.${EXECUTION_MODE}.yml" down 2>/dev/null || true
    fi

    # Remove temporary files
    rm -f .env.tmp config/temp.json 2>/dev/null || true

    echo -e "${YELLOW}🧹 Cleanup completed${NC}"
    exit 1
}

# Set up error handling
trap cleanup_on_failure ERR

# Main execution
main() {
    cd "$PROJECT_ROOT"

    check_prerequisites
    show_deployment_info
    confirm_deployment

    echo -e "${GREEN}🔄 Starting deployment...${NC}"
    deploy_mode

    show_post_deployment
}

# Run main function
main "$@"
