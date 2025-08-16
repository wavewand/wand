#!/bin/bash
# Single Local CI Script for Wand MCP Server
# Handles setup, dependency management, and execution of CI pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Function to print colored output
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%H:%M:%S')

    case $level in
        "info")
            echo -e "${CYAN}[$timestamp] INFO:${NC} $message"
            ;;
        "success")
            echo -e "${GREEN}[$timestamp] SUCCESS:${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}[$timestamp] WARNING:${NC} $message"
            ;;
        "error")
            echo -e "${RED}[$timestamp] ERROR:${NC} $message"
            ;;
        "header")
            echo
            echo -e "${PURPLE}===============================================================${NC}"
            echo -e "${PURPLE}$message${NC}"
            echo -e "${PURPLE}===============================================================${NC}"
            ;;
    esac
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run command with logging
run_cmd() {
    local description="$1"
    shift
    local cmd=("$@")

    log "info" "Running: $description"

    if [ "$VERBOSE" = "true" ]; then
        log "info" "Command: ${cmd[*]}"
        "${cmd[@]}"
        local exit_code=$?
    else
        local output
        output=$("${cmd[@]}" 2>&1)
        local exit_code=$?

        if [ $exit_code -ne 0 ]; then
            echo "$output"
        fi
    fi

    if [ $exit_code -eq 0 ]; then
        log "success" "✅ $description"
        return 0
    else
        log "error" "❌ $description (Exit code: $exit_code)"
        return $exit_code
    fi
}

# Function to setup environment
setup_environment() {
    log "header" "🪄 Setting Up Wand CI Environment"

    # Check Python installation
    if ! command_exists python3; then
        log "error" "Python 3 is not installed. Please install Python $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR+ first."
        exit 1
    fi

    local python_version
    python_version=$(python3 --version | cut -d' ' -f2)
    log "info" "Found Python $python_version"

    # Check Python version
    local python_major python_minor
    python_major=$(echo "$python_version" | cut -d'.' -f1)
    python_minor=$(echo "$python_version" | cut -d'.' -f2)

    if [ "$python_major" -lt $PYTHON_MIN_MAJOR ] || ([ "$python_major" -eq $PYTHON_MIN_MAJOR ] && [ "$python_minor" -lt $PYTHON_MIN_MINOR ]); then
        log "error" "Python $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR+ is required. Found $python_version"
        exit 1
    fi

    # Setup virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        log "info" "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    if [ -z "$VIRTUAL_ENV" ]; then
        log "error" "Failed to activate virtual environment"
        exit 1
    fi

    log "success" "Virtual environment ready"

    # Install/upgrade dependencies
    log "info" "Installing dependencies..."

    run_cmd "Upgrade pip" python -m pip install --upgrade pip

    if [ -f "requirements-base.txt" ]; then
        run_cmd "Install base requirements" pip install -r requirements-base.txt
    else
        log "warning" "requirements-base.txt not found, installing minimal requirements"
        run_cmd "Install minimal requirements" pip install fastapi uvicorn aiohttp
    fi

    # Install test dependencies
    run_cmd "Install test dependencies" pip install pytest pytest-cov pytest-asyncio

    # Install development tools
    run_cmd "Install dev tools" pip install ruff mypy bandit safety

    # Install optional enterprise dependencies (don't fail if unavailable)
    log "info" "Installing optional enterprise dependencies..."
    pip install pysnc 2>/dev/null || log "warning" "pysnc (ServiceNow) not available - tests will be skipped"
    pip install azure-identity msgraph-core 2>/dev/null || log "warning" "Azure dependencies not available - tests will be skipped"
    pip install britive 2>/dev/null || log "warning" "Britive SDK not available - tests will be skipped"

    log "success" "Environment setup complete"
}

# Function to run linting
run_linting() {
    log "header" "Running Code Linting"

    if run_cmd "Run ruff linting" ruff check .; then
        return 0
    else
        log "warning" "Linting issues found (non-critical)"
        return 0  # Non-critical
    fi
}

# Function to run type checking
run_type_checking() {
    log "header" "Running Type Checking"

    if run_cmd "Run mypy type checking" mypy . --ignore-missing-imports; then
        return 0
    else
        log "warning" "Type checking issues found (non-critical)"
        return 0  # Non-critical
    fi
}

# Function to run tests
run_tests() {
    log "header" "Running Test Suite"

    local test_args=("pytest" "-v")

    case "$TEST_MODE" in
        "enterprise")
            test_args+=("tests/test_enterprise_integrations.py")
            ;;
        "basic")
            test_args+=("tests/test_enterprise_integrations_basic.py")
            ;;
        "all"|*)
            test_args+=("tests/" "--cov=." "--cov-report=xml" "--cov-report=term")
            ;;
    esac

    if run_cmd "Run tests" "${test_args[@]}"; then
        log "success" "All tests passed! 🎉"
        return 0
    else
        log "error" "Some tests failed"
        return 1
    fi
}

# Function to run security checks
run_security_checks() {
    log "header" "Running Security Checks"

    local security_passed=true

    if ! run_cmd "Run Bandit security scan" bandit -r . -ll; then
        log "warning" "Bandit security issues found (non-critical)"
    fi

    if ! run_cmd "Run Safety dependency check" safety check --json; then
        log "warning" "Safety dependency issues found (non-critical)"
    fi

    return 0  # Security checks are non-critical
}

# Function to run full CI pipeline
run_full_pipeline() {
    log "header" "🪄 Starting Wand Local CI Pipeline"

    local failed_steps=()
    local successful_steps=()

    # Critical steps
    local critical_steps=(
        "Environment Setup:setup_environment"
        "Test Suite:run_tests"
    )

    # Non-critical steps
    local optional_steps=(
        "Code Linting:run_linting"
        "Type Checking:run_type_checking"
        "Security Checks:run_security_checks"
    )

    # Run critical steps
    local critical_failed=false
    for step in "${critical_steps[@]}"; do
        local name="${step%:*}"
        local func="${step#*:}"

        if $func; then
            successful_steps+=("$name")
        else
            failed_steps+=("$name")
            critical_failed=true
        fi
    done

    # Run optional steps only if critical steps passed
    if [ "$critical_failed" = false ]; then
        for step in "${optional_steps[@]}"; do
            local name="${step%:*}"
            local func="${step#*:}"

            if $func; then
                successful_steps+=("$name")
            else
                failed_steps+=("$name")
            fi
        done
    fi

    # Print summary
    log "header" "CI Pipeline Summary"

    if [ ${#successful_steps[@]} -gt 0 ]; then
        log "success" "✅ Successful steps:"
        for step in "${successful_steps[@]}"; do
            echo "  - $step"
        done
    fi

    if [ ${#failed_steps[@]} -gt 0 ]; then
        log "error" "❌ Failed steps:"
        for step in "${failed_steps[@]}"; do
            echo "  - $step"
        done
    else
        log "success" "🎉 All CI checks passed!"
    fi

    local total_steps=$((${#successful_steps[@]} + ${#failed_steps[@]}))
    local success_rate=0
    if [ $total_steps -gt 0 ]; then
        success_rate=$(( ${#successful_steps[@]} * 100 / total_steps ))
    fi

    log "info" "Success rate: $success_rate% (${#successful_steps[@]}/$total_steps)"

    return $critical_failed
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Local CI pipeline for Wand MCP Server"
    echo
    echo "Options:"
    echo "  --setup              Setup environment only"
    echo "  --enterprise         Run enterprise integration tests only"
    echo "  --basic              Run basic enterprise tests only"
    echo "  --tests-only         Run tests without setup"
    echo "  --lint-only          Run linting only"
    echo "  --type-check-only    Run type checking only"
    echo "  --security-only      Run security checks only"
    echo "  -v, --verbose        Verbose output"
    echo "  -h, --help           Show this help"
    echo
    echo "Examples:"
    echo "  $0                   # Run full CI pipeline"
    echo "  $0 --enterprise -v   # Run enterprise tests with verbose output"
    echo "  $0 --setup           # Setup environment only"
}

# Main function
main() {
    local mode="full"
    VERBOSE="false"
    TEST_MODE="all"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --setup)
                mode="setup"
                shift
                ;;
            --enterprise)
                mode="tests"
                TEST_MODE="enterprise"
                shift
                ;;
            --basic)
                mode="tests"
                TEST_MODE="basic"
                shift
                ;;
            --tests-only)
                mode="tests"
                shift
                ;;
            --lint-only)
                mode="lint"
                shift
                ;;
            --type-check-only)
                mode="typecheck"
                shift
                ;;
            --security-only)
                mode="security"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log "error" "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Change to script directory
    cd "$SCRIPT_DIR"

    # Ensure we have a virtual environment for all operations except setup
    if [ "$mode" != "setup" ] && [ ! -d "$VENV_DIR" ]; then
        log "warning" "Virtual environment not found, setting up first..."
        setup_environment
    elif [ "$mode" != "setup" ]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Execute based on mode
    case $mode in
        "setup")
            setup_environment
            ;;
        "tests")
            run_tests
            ;;
        "lint")
            run_linting
            ;;
        "typecheck")
            run_type_checking
            ;;
        "security")
            run_security_checks
            ;;
        "full"|*)
            run_full_pipeline
            ;;
    esac
}

# Run main function with all arguments
main "$@"
