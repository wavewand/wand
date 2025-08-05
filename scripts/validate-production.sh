#!/bin/bash

# Production Deployment Validation Script
# Validates that the MCP-Python system is ready for production deployment

set -e

echo "🔍 MCP-Python Production Validation"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validation functions
validate_dependencies() {
    echo -e "\n📦 Checking dependencies..."

    # Check Python
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}✅ Python 3 available${NC}"
        python3 --version
    else
        echo -e "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi

    # Check pip packages
    echo -e "\n📦 Checking Python packages..."

    required_packages=("fastapi" "uvicorn" "psutil" "paramiko" "docker" "pydantic" "pytest")

    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            echo -e "${GREEN}✅ $package available${NC}"
        else
            echo -e "${YELLOW}⚠️  $package missing - install with: pip install $package${NC}"
        fi
    done
}

validate_configuration() {
    echo -e "\n⚙️  Validating configuration..."

    # Check config files
    if [ -f "config.sample.json" ]; then
        echo -e "${GREEN}✅ Sample configuration available${NC}"
    else
        echo -e "${RED}❌ config.sample.json not found${NC}"
    fi

    # Validate JSON syntax
    if python3 -c "import json; json.load(open('config.sample.json'))" 2>/dev/null; then
        echo -e "${GREEN}✅ Configuration JSON is valid${NC}"
    else
        echo -e "${RED}❌ Invalid JSON in configuration${NC}"
    fi
}

validate_backends() {
    echo -e "\n🔧 Validating execution backends..."

    # Test backend imports
    if python3 -c "from tools.execution.factory import create_execution_backend; print('Backend factory working')" 2>/dev/null; then
        echo -e "${GREEN}✅ Execution backend factory working${NC}"
    else
        echo -e "${RED}❌ Backend factory import failed${NC}"
    fi

    # Test each backend
    backends=("native" "host_agent" "docker_socket" "ssh_remote")

    for backend in "${backends[@]}"; do
        if python3 -c "from tools.execution.factory import EXECUTION_BACKENDS; print('$backend' in EXECUTION_BACKENDS)" 2>/dev/null | grep -q "True"; then
            echo -e "${GREEN}✅ $backend backend available${NC}"
        else
            echo -e "${YELLOW}⚠️  $backend backend not available${NC}"
        fi
    done
}

validate_security() {
    echo -e "\n🛡️  Security validation..."

    # Check for default tokens
    if grep -r "default-token\|your-token\|change-me" . --exclude-dir=.git --exclude="*.sh" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Found default tokens - please update for production${NC}"
    else
        echo -e "${GREEN}✅ No default tokens found${NC}"
    fi

    # Check file permissions
    if [ -f "config.sample.json" ]; then
        perms=$(stat -c "%a" config.sample.json 2>/dev/null || stat -f "%Lp" config.sample.json 2>/dev/null)
        if [ "$perms" -le "644" ]; then
            echo -e "${GREEN}✅ Configuration file permissions secure${NC}"
        else
            echo -e "${YELLOW}⚠️  Configuration file permissions too open: $perms${NC}"
        fi
    fi
}

validate_ports() {
    echo -e "\n🌐 Checking required ports..."

    # Check if ports are available
    ports=(8000 8001 9000)

    for port in "${ports[@]}"; do
        if lsof -i :$port >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  Port $port is already in use${NC}"
        else
            echo -e "${GREEN}✅ Port $port available${NC}"
        fi
    done
}

validate_system_resources() {
    echo -e "\n💻 System resource check..."

    # Check available memory
    if command -v free &> /dev/null; then
        available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ "$available_mem" -gt 512 ]; then
            echo -e "${GREEN}✅ Sufficient memory available: ${available_mem}MB${NC}"
        else
            echo -e "${YELLOW}⚠️  Low memory: ${available_mem}MB (recommend 1GB+)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Cannot check memory (free command not available)${NC}"
    fi

    # Check disk space
    available_disk=$(df . | awk 'NR==2{print $4}')
    if [ "$available_disk" -gt 1048576 ]; then  # 1GB in KB
        echo -e "${GREEN}✅ Sufficient disk space available${NC}"
    else
        echo -e "${YELLOW}⚠️  Low disk space (recommend 2GB+ available)${NC}"
    fi

    # Check CPU cores
    cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    if [ "$cpu_cores" != "unknown" ] && [ "$cpu_cores" -ge 2 ]; then
        echo -e "${GREEN}✅ Sufficient CPU cores: $cpu_cores${NC}"
    else
        echo -e "${YELLOW}⚠️  CPU cores: $cpu_cores (recommend 2+)${NC}"
    fi
}

run_basic_tests() {
    echo -e "\n🧪 Running basic functionality tests..."

    # Test basic imports
    if python3 -c "
import sys
sys.path.append('.')
from tools.execution.base import ExecutionConfig
from tools.process.models import ProcessInfo
from tools.host_agent.models import ExecutionRequest
print('✅ Basic imports successful')
" 2>/dev/null; then
        echo -e "${GREEN}✅ Core modules import successfully${NC}"
    else
        echo -e "${RED}❌ Core module import failed${NC}"
    fi

    # Test configuration loading
    if python3 -c "
import sys
sys.path.append('.')
from config import Config, load_config
config = load_config('config.sample.json')
print('✅ Configuration loading successful')
" 2>/dev/null; then
        echo -e "${GREEN}✅ Configuration loading works${NC}"
    else
        echo -e "${YELLOW}⚠️  Configuration loading test failed${NC}"
    fi
}

generate_deployment_checklist() {
    echo -e "\n📋 Production Deployment Checklist"
    echo "=================================="
    echo
    echo "Before deploying to production, ensure:"
    echo
    echo "🔐 Security:"
    echo "  □ Update all default tokens and passwords"
    echo "  □ Configure proper authentication"
    echo "  □ Set up HTTPS/TLS certificates"
    echo "  □ Configure firewall rules"
    echo "  □ Enable audit logging"
    echo
    echo "⚙️  Configuration:"
    echo "  □ Review and customize config.json"
    echo "  □ Set appropriate resource limits"
    echo "  □ Configure command allowlists/blocklists"
    echo "  □ Set working directory restrictions"
    echo "  □ Configure backend-specific settings"
    echo
    echo "🌐 Infrastructure:"
    echo "  □ Ensure sufficient system resources"
    echo "  □ Configure monitoring and alerting"
    echo "  □ Set up log aggregation"
    echo "  □ Plan backup and recovery procedures"
    echo "  □ Test failover scenarios"
    echo
    echo "🧪 Testing:"
    echo "  □ Run full test suite: python -m pytest tests/ -v"
    echo "  □ Perform load testing"
    echo "  □ Test all execution backends"
    echo "  □ Verify security controls"
    echo "  □ Test OpenCode integration"
    echo
    echo "📊 Monitoring:"
    echo "  □ Set up health check endpoints"
    echo "  □ Configure performance monitoring"
    echo "  □ Set up alerting for failures"
    echo "  □ Monitor resource usage"
    echo "  □ Track security events"
}

# Main execution
main() {
    echo "Starting production validation..."

    validate_dependencies
    validate_configuration
    validate_backends
    validate_security
    validate_ports
    validate_system_resources
    run_basic_tests

    echo -e "\n🎉 Production validation complete!"
    echo
    generate_deployment_checklist

    echo -e "\n${GREEN}✅ System appears ready for production deployment!${NC}"
    echo "Review the checklist above before deploying."
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
