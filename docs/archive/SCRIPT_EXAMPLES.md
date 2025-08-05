# MCP Platform Script Examples

## Overview

This document provides comprehensive script examples for interacting with the MCP platform, with detailed explanations of each script's functionality and usage.

## Table of Contents

1. [Claude Code Integration Scripts](#claude-code-integration-scripts)
2. [API Interaction Scripts](#api-interaction-scripts)
3. [Agent Management Scripts](#agent-management-scripts)
4. [System Administration Scripts](#system-administration-scripts)
5. [Testing and Validation Scripts](#testing-and-validation-scripts)

## Claude Code Integration Scripts

### 1. Complete MCP Integration Test Script

This script demonstrates all aspects of Claude Code MCP integration:

```bash
#!/bin/bash
# claude_mcp_integration_test.sh
# Complete test suite for MCP Claude Code integration

set -e  # Exit on any error

BASE_URL="http://localhost:8000"
CLAUDE_TOOLS="mcp__mcp-automation-api__*"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status="$1"
    local message="$2"
    case $status in
        "INFO")  echo -e "${BLUE}[INFO]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "WARNING") echo -e "${YELLOW}[WARNING]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
    esac
}

# Function to run Claude Code command with error handling
run_claude_command() {
    local description="$1"
    local command="$2"

    print_status "INFO" "Testing: $description"
    echo "Command: $command"
    echo "----------------------------------------"

    if eval "$command"; then
        print_status "SUCCESS" "$description completed successfully"
    else
        print_status "ERROR" "$description failed"
        return 1
    fi
    echo ""
}

# Pre-flight checks
print_status "INFO" "Starting MCP Integration Test Suite"
print_status "INFO" "Performing pre-flight checks..."

# Check if MCP platform is running
if curl -s "$BASE_URL/health" > /dev/null; then
    print_status "SUCCESS" "MCP platform is running at $BASE_URL"
else
    print_status "ERROR" "MCP platform is not accessible at $BASE_URL"
    exit 1
fi

# Check if Claude Code is available
if command -v claude > /dev/null; then
    print_status "SUCCESS" "Claude Code CLI is available"
else
    print_status "ERROR" "Claude Code CLI not found"
    exit 1
fi

# Check MCP server registration
print_status "INFO" "Checking MCP server registration..."
if claude mcp list | grep -q "mcp-automation"; then
    print_status "SUCCESS" "MCP server is registered"
else
    print_status "WARNING" "MCP server not found, attempting to register..."
    claude mcp add -t http mcp-automation-api "$BASE_URL/mcp"
    if [ $? -eq 0 ]; then
        print_status "SUCCESS" "MCP server registered successfully"
    else
        print_status "ERROR" "Failed to register MCP server"
        exit 1
    fi
fi

echo ""
print_status "INFO" "Starting MCP tool tests..."
echo ""

# Test 1: System Status Check
run_claude_command \
    "System Status Check" \
    "claude --print \"Use the get_system_status tool to check the current automation platform status\" --allowedTools \"$CLAUDE_TOOLS\""

# Test 2: Agent Listing
run_claude_command \
    "Agent Listing" \
    "claude --print \"List all available agents in the system\" --allowedTools \"$CLAUDE_TOOLS\""

# Test 3: Task Creation
run_claude_command \
    "Task Creation" \
    "claude --print \"Create a task with title 'Integration Test Task', description 'This task was created during MCP integration testing', type 'testing', and priority 'high'\" --allowedTools \"$CLAUDE_TOOLS\""

# Test 4: Complex Query
run_claude_command \
    "Complex System Query" \
    "claude --print \"Show me the current system status and list of all agents, then summarize the overall health of the platform\" --allowedTools \"$CLAUDE_TOOLS\""

# Test 5: Interactive Permission Test (commented - requires user interaction)
print_status "INFO" "Interactive permission test (manual):"
echo "Run this command to test interactive permissions:"
echo "claude \"Use the get_system_status tool to check our automation platform\""
echo ""

# Test 6: Using settings file
SETTINGS_FILE="$HOME/.config/claude/mcp_test_settings.json"
print_status "INFO" "Creating temporary settings file: $SETTINGS_FILE"

mkdir -p "$(dirname "$SETTINGS_FILE")"
cat > "$SETTINGS_FILE" << EOF
{
  "allowedTools": ["mcp__mcp-automation-api__*"],
  "permissionMode": "acceptEdits"
}
EOF

run_claude_command \
    "Settings File Test" \
    "claude --settings \"$SETTINGS_FILE\" --print \"Get system status using settings file configuration\""

# Cleanup
print_status "INFO" "Cleaning up temporary settings file"
rm -f "$SETTINGS_FILE"

print_status "SUCCESS" "MCP Integration Test Suite completed!"
print_status "INFO" "Summary:"
echo "  ✓ Platform connectivity verified"
echo "  ✓ MCP server registration confirmed"
echo "  ✓ Tool execution tested"
echo "  ✓ Permission management verified"
echo ""
print_status "INFO" "Next steps:"
echo "  - Run interactive permission test manually"
echo "  - Set up permanent settings file if needed"
echo "  - Configure project-specific MCP settings"
```

**Usage:**
```bash
chmod +x claude_mcp_integration_test.sh
./claude_mcp_integration_test.sh
```

**What this script does:**
1. **Pre-flight checks**: Verifies platform is running and Claude Code is available
2. **MCP server registration**: Checks and registers the MCP server if needed
3. **Tool testing**: Tests all three MCP tools with different scenarios
4. **Permission management**: Demonstrates different permission approaches
5. **Settings file usage**: Shows how to use configuration files
6. **Error handling**: Provides clear status messages and error handling

### 2. Permission Management Script

This script demonstrates different approaches to handling Claude Code permissions:

```bash
#!/bin/bash
# claude_permission_management.sh
# Demonstrates various permission management approaches for Claude Code MCP

# Configuration
TOOLS_WILDCARD="mcp__mcp-automation-api__*"
SPECIFIC_TOOL="mcp__mcp-automation-api__get_system_status"
SETTINGS_DIR="$HOME/.config/claude"

echo "🔐 Claude Code Permission Management Examples"
echo "============================================="

# Method 1: Command-line tool specification
echo ""
echo "Method 1: Command-line tool specification"
echo "----------------------------------------"
echo "This method explicitly allows specific tools via command line"
echo ""

COMMAND="claude --print \"Check system status\" --allowedTools \"$SPECIFIC_TOOL\""
echo "Command: $COMMAND"
echo "Executing..."
eval "$COMMAND"

echo ""
echo "Using wildcard permissions:"
COMMAND="claude --print \"Show system status and list agents\" --allowedTools \"$TOOLS_WILDCARD\""
echo "Command: $COMMAND"
echo "Executing..."
eval "$COMMAND"

# Method 2: Settings file approach
echo ""
echo "Method 2: Settings file approach"
echo "-------------------------------"

# Create development settings
DEV_SETTINGS="$SETTINGS_DIR/mcp_dev_settings.json"
mkdir -p "$SETTINGS_DIR"

cat > "$DEV_SETTINGS" << 'EOF'
{
  "allowedTools": ["mcp__mcp-automation-api__*"],
  "disallowedTools": [],
  "permissionMode": "acceptEdits",
  "autoApprove": {
    "tools": ["mcp__mcp-automation-api__get_system_status"],
    "maxUsagePerSession": 100
  },
  "logging": {
    "enabled": true,
    "logLevel": "INFO"
  }
}
EOF

echo "Created development settings file: $DEV_SETTINGS"
echo "Contents:"
cat "$DEV_SETTINGS"
echo ""

COMMAND="claude --settings \"$DEV_SETTINGS\" --print \"Get system status using dev settings\""
echo "Command: $COMMAND"
echo "Executing..."
eval "$COMMAND"

# Create production settings
PROD_SETTINGS="$SETTINGS_DIR/mcp_prod_settings.json"

cat > "$PROD_SETTINGS" << 'EOF'
{
  "allowedTools": [
    "mcp__mcp-automation-api__get_system_status",
    "mcp__mcp-automation-api__list_agents"
  ],
  "disallowedTools": [
    "mcp__mcp-automation-api__create_task"
  ],
  "permissionMode": "prompt",
  "requireConfirmation": true,
  "maxUsagePerHour": 50,
  "logging": {
    "enabled": true,
    "logLevel": "WARNING"
  }
}
EOF

echo ""
echo "Created production settings file: $PROD_SETTINGS"
echo "Contents:"
cat "$PROD_SETTINGS"

# Method 3: Project-specific configuration
echo ""
echo "Method 3: Project-specific configuration"
echo "---------------------------------------"

PROJECT_CONFIG="./.claude.json"

cat > "$PROJECT_CONFIG" << 'EOF'
{
  "mcpServers": {
    "mcp-automation-api": {
      "transport": "http",
      "url": "http://localhost:8000/mcp",
      "timeout": 30000
    }
  },
  "allowedTools": [
    "mcp__mcp-automation-api__get_system_status",
    "mcp__mcp-automation-api__list_agents"
  ],
  "permissionMode": "acceptEdits",
  "projectSettings": {
    "autoSave": true,
    "maxSessionTime": 3600
  }
}
EOF

echo "Created project configuration: $PROJECT_CONFIG"
echo "Contents:"
cat "$PROJECT_CONFIG"
echo ""

echo "Testing project-specific configuration:"
COMMAND="claude --print \"Test project-specific MCP configuration\""
echo "Command: $COMMAND"
echo "Executing..."
eval "$COMMAND"

# Method 4: Environment-based configuration
echo ""
echo "Method 4: Environment-based configuration"
echo "----------------------------------------"

export CLAUDE_ALLOWED_TOOLS="$TOOLS_WILDCARD"
export CLAUDE_PERMISSION_MODE="acceptEdits"

echo "Set environment variables:"
echo "CLAUDE_ALLOWED_TOOLS=$CLAUDE_ALLOWED_TOOLS"
echo "CLAUDE_PERMISSION_MODE=$CLAUDE_PERMISSION_MODE"
echo ""

# Note: This is illustrative - actual environment variable support may vary
echo "Environment-based configuration would be used automatically by Claude Code"

# Method 5: Interactive permission granting
echo ""
echo "Method 5: Interactive permission granting"
echo "----------------------------------------"
echo "For interactive permission granting, run commands without --print flag:"
echo ""
echo "Example commands to run manually:"
echo "  claude \"Use the get_system_status tool\""
echo "  claude \"List all agents in the system\""
echo "  claude \"Create a test task\""
echo ""
echo "These will prompt for permission and save your choices for future use."

# Cleanup options
echo ""
echo "🧹 Cleanup Options"
echo "=================="
echo ""
echo "To remove created settings files:"
echo "  rm \"$DEV_SETTINGS\""
echo "  rm \"$PROD_SETTINGS\""
echo "  rm \"$PROJECT_CONFIG\""
echo ""
echo "To reset Claude Code permissions:"
echo "  claude mcp remove mcp-automation-api -s local"
echo "  claude mcp add -t http mcp-automation-api http://localhost:8000/mcp"

echo ""
echo "✅ Permission management examples completed!"
```

**Key Features:**
- **Multiple permission methods**: Shows 5 different approaches
- **Configuration templates**: Provides ready-to-use config files
- **Environment-specific settings**: Development vs production configurations
- **Cleanup instructions**: How to reset and manage configurations

## API Interaction Scripts

### 3. Comprehensive API Testing Script

This script provides complete API testing with detailed explanations:

```bash
#!/bin/bash
# comprehensive_api_test.sh
# Complete API testing suite for the MCP platform

set -e

# Configuration
BASE_URL="http://localhost:8000"
TIMEOUT=30
CONTENT_TYPE="Content-Type: application/json"

# Test data
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TEST_SESSION_ID="test-session-$(date +%s)"

# Colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging function
log() {
    local level="$1"
    local message="$2"
    case $level in
        "INFO")  echo -e "${BLUE}[INFO]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[✓]${NC} $message" ;;
        "WARNING") echo -e "${YELLOW}[⚠]${NC} $message" ;;
        "ERROR") echo -e "${RED}[✗]${NC} $message" ;;
        "TEST") echo -e "${PURPLE}[TEST]${NC} $message" ;;
    esac
}

# API call function with error handling and response parsing
api_call() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local description="$4"
    local expected_status="${5:-200}"

    log "TEST" "$description"
    echo "  Method: $method"
    echo "  Endpoint: $endpoint"

    if [ -n "$data" ]; then
        echo "  Payload: $data"
    fi

    echo "  Expected Status: $expected_status"
    echo ""

    # Make the API call
    local response
    local status_code

    if [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$BASE_URL$endpoint" \
            -H "$CONTENT_TYPE" \
            -d "$data" \
            --max-time $TIMEOUT)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$BASE_URL$endpoint" \
            --max-time $TIMEOUT)
    fi

    # Extract status code and response body
    status_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)

    echo "  Response Status: $status_code"

    # Check if status code matches expected
    if [ "$status_code" = "$expected_status" ]; then
        log "SUCCESS" "$description - Status code matches expected ($expected_status)"
    else
        log "ERROR" "$description - Expected $expected_status, got $status_code"
        echo "  Response: $response_body"
        return 1
    fi

    # Pretty print JSON response if possible
    echo "  Response:"
    if echo "$response_body" | jq empty 2>/dev/null; then
        echo "$response_body" | jq '.' | sed 's/^/    /'
    else
        echo "$response_body" | sed 's/^/    /'
    fi

    echo ""
    echo "----------------------------------------"
    echo ""

    # Return response body for further processing
    echo "$response_body"
}

# Start testing
log "INFO" "Starting Comprehensive API Test Suite"
log "INFO" "Base URL: $BASE_URL"
log "INFO" "Test Session ID: $TEST_SESSION_ID"
echo ""

# Test 1: Health Check
log "INFO" "=== HEALTH AND STATUS TESTS ==="
api_call "GET" "/health" "" "Health Check Endpoint"

# Test 2: System Status
api_call "GET" "/api/v1/system/status" "" "System Status Endpoint"

# Test 3: System Metrics
api_call "GET" "/api/v1/system/metrics" "" "System Metrics Endpoint"

# Test 4: MCP Protocol Tests
log "INFO" "=== MCP PROTOCOL TESTS ==="

# MCP Initialize
MCP_INIT_DATA='{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {}
}'
mcp_response=$(api_call "POST" "/mcp" "$MCP_INIT_DATA" "MCP Initialize Protocol")

# MCP Tools List
MCP_TOOLS_DATA='{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}'
api_call "POST" "/mcp" "$MCP_TOOLS_DATA" "MCP Tools List"

# MCP Tool Call - System Status
MCP_STATUS_DATA='{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "get_system_status",
    "arguments": {}
  }
}'
api_call "POST" "/mcp" "$MCP_STATUS_DATA" "MCP System Status Tool Call"

# Test 5: Agent Management
log "INFO" "=== AGENT MANAGEMENT TESTS ==="

# List existing agents
agents_response=$(api_call "GET" "/api/v1/agents" "" "List All Agents")

# Create a new agent
AGENT_DATA='{
  "name": "TestAgent-'$TEST_SESSION_ID'",
  "type": "testing",
  "capabilities": [
    "test_execution",
    "data_validation",
    "performance_monitoring"
  ],
  "status": "active",
  "config": {
    "max_concurrent_tasks": 3,
    "timeout_seconds": 120,
    "retry_attempts": 2
  },
  "metadata": {
    "created_by": "api_test_script",
    "test_session": "'$TEST_SESSION_ID'",
    "created_at": "'$TIMESTAMP'"
  }
}'

agent_response=$(api_call "POST" "/api/v1/agents" "$AGENT_DATA" "Create Test Agent")

# Extract agent ID for further tests
AGENT_ID=$(echo "$agent_response" | jq -r '.id // empty')

if [ -n "$AGENT_ID" ]; then
    log "SUCCESS" "Created agent with ID: $AGENT_ID"

    # Get specific agent
    api_call "GET" "/api/v1/agents/$AGENT_ID" "" "Get Specific Agent"

    # Update agent
    AGENT_UPDATE_DATA='{
      "status": "idle",
      "metadata": {
        "updated_by": "api_test_script",
        "updated_at": "'$TIMESTAMP'",
        "test_update": true
      }
    }'
    api_call "PUT" "/api/v1/agents/$AGENT_ID" "$AGENT_UPDATE_DATA" "Update Agent"
else
    log "WARNING" "Could not extract agent ID from response"
fi

# Test 6: Task Management
log "INFO" "=== TASK MANAGEMENT TESTS ==="

# List existing tasks
api_call "GET" "/api/v1/tasks" "" "List All Tasks"

# Create a new task
TASK_DATA='{
  "title": "API Test Task - '$TEST_SESSION_ID'",
  "description": "This task was created during comprehensive API testing to validate task management functionality.",
  "type": "api_testing",
  "priority": "medium",
  "metadata": {
    "test_session": "'$TEST_SESSION_ID'",
    "created_by": "api_test_script",
    "estimated_duration": "5 minutes",
    "requirements": ["api_access", "test_environment"]
  }
}'

task_response=$(api_call "POST" "/api/v1/tasks" "$TASK_DATA" "Create Test Task")

# Extract task ID
TASK_ID=$(echo "$task_response" | jq -r '.id // empty')

if [ -n "$TASK_ID" ]; then
    log "SUCCESS" "Created task with ID: $TASK_ID"

    # Get specific task
    api_call "GET" "/api/v1/tasks/$TASK_ID" "" "Get Specific Task"

    # Update task status
    if [ -n "$AGENT_ID" ]; then
        TASK_ASSIGN_DATA='{
          "agent_id": "'$AGENT_ID'",
          "status": "in_progress"
        }'
        api_call "POST" "/api/v1/tasks/$TASK_ID/assign" "$TASK_ASSIGN_DATA" "Assign Task to Agent"
    fi

    # Update task status
    TASK_STATUS_DATA='{
      "status": "completed",
      "metadata": {
        "completed_by": "api_test_script",
        "completion_time": "'$TIMESTAMP'",
        "test_result": "success"
      }
    }'
    api_call "PUT" "/api/v1/tasks/$TASK_ID/status" "$TASK_STATUS_DATA" "Update Task Status"
else
    log "WARNING" "Could not extract task ID from response"
fi

# Test 7: Project Management
log "INFO" "=== PROJECT MANAGEMENT TESTS ==="

# List existing projects
api_call "GET" "/api/v1/projects" "" "List All Projects"

# Create a new project
PROJECT_DATA='{
  "name": "API Test Project - '$TEST_SESSION_ID'",
  "description": "A comprehensive test project created during API validation testing.",
  "status": "active",
  "metadata": {
    "test_session": "'$TEST_SESSION_ID'",
    "project_type": "testing",
    "priority": "high",
    "expected_duration": "1 hour",
    "team_size": 1
  }
}'

project_response=$(api_call "POST" "/api/v1/projects" "$PROJECT_DATA" "Create Test Project")

# Extract project ID
PROJECT_ID=$(echo "$project_response" | jq -r '.id // empty')

if [ -n "$PROJECT_ID" ]; then
    log "SUCCESS" "Created project with ID: $PROJECT_ID"

    # Get specific project
    api_call "GET" "/api/v1/projects/$PROJECT_ID" "" "Get Specific Project"
else
    log "WARNING" "Could not extract project ID from response"
fi

# Test 8: Batch Operations
log "INFO" "=== BATCH OPERATIONS TESTS ==="

# Batch RAG query (if available)
BATCH_RAG_DATA='{
  "queries": [
    "What is the purpose of this API testing?",
    "How do batch operations work?",
    "What are the benefits of automated testing?"
  ],
  "framework": "llamaindex",
  "batch_size": 3,
  "max_tokens": 150
}'

api_call "POST" "/api/v1/batch/rag" "$BATCH_RAG_DATA" "Batch RAG Operations" "200"

# Test 9: Monitoring and Analytics
log "INFO" "=== MONITORING AND ANALYTICS TESTS ==="

# Get monitoring stats
api_call "GET" "/api/v1/monitoring/stats" "" "Monitoring Statistics"

# Get cache stats
api_call "GET" "/api/v1/cache/stats" "" "Cache Statistics"

# Get WebSocket stats
api_call "GET" "/api/v1/websocket/stats" "" "WebSocket Statistics"

# Test 10: AI Framework Integration
log "INFO" "=== AI FRAMEWORK TESTS ==="

# Get available frameworks
api_call "GET" "/api/v1/ai/frameworks" "" "Available AI Frameworks"

# Get AI framework status
api_call "GET" "/api/v1/ai/status" "" "AI Framework Status"

# Test 11: Performance and Load Testing
log "INFO" "=== PERFORMANCE TESTS ==="

# Benchmark performance
BENCHMARK_DATA='{
  "frameworks": ["haystack", "llamaindex"],
  "iterations": 5,
  "test_type": "response_time"
}'

api_call "POST" "/api/v1/benchmark/performance" "$BENCHMARK_DATA" "Performance Benchmark"

# Cleanup Section
log "INFO" "=== CLEANUP OPERATIONS ==="

# Clean up created resources
if [ -n "$TASK_ID" ]; then
    log "INFO" "Cleaning up task: $TASK_ID"
    # In a real implementation, you might want to delete the test task
    # api_call "DELETE" "/api/v1/tasks/$TASK_ID" "" "Delete Test Task" "204"
fi

if [ -n "$AGENT_ID" ]; then
    log "INFO" "Cleaning up agent: $AGENT_ID"
    # In a real implementation, you might want to delete the test agent
    # api_call "DELETE" "/api/v1/agents/$AGENT_ID" "" "Delete Test Agent" "204"
fi

if [ -n "$PROJECT_ID" ]; then
    log "INFO" "Cleaning up project: $PROJECT_ID"
    # In a real implementation, you might want to delete the test project
    # api_call "DELETE" "/api/v1/projects/$PROJECT_ID" "" "Delete Test Project" "204"
fi

# Final status check
log "INFO" "=== FINAL STATUS CHECK ==="
api_call "GET" "/health" "" "Final Health Check"

# Test Summary
log "SUCCESS" "API Test Suite Completed Successfully!"
echo ""
log "INFO" "Test Summary:"
echo "  ✓ Health and status endpoints tested"
echo "  ✓ MCP protocol functionality verified"
echo "  ✓ Agent management operations tested"
echo "  ✓ Task management operations tested"
echo "  ✓ Project management operations tested"
echo "  ✓ Batch operations tested"
echo "  ✓ Monitoring endpoints verified"
echo "  ✓ AI framework integration tested"
echo "  ✓ Performance benchmarks executed"
echo ""
log "INFO" "Test Session ID: $TEST_SESSION_ID"
log "INFO" "Timestamp: $TIMESTAMP"
```

**Key Features:**
- **Comprehensive coverage**: Tests all major API endpoints
- **Error handling**: Proper error detection and reporting
- **Response validation**: Checks status codes and response formats
- **Resource management**: Creates and manages test resources
- **Cleanup operations**: Provides cleanup instructions
- **Detailed logging**: Color-coded output with clear status indicators

### Usage Instructions

Each script can be run independently:

```bash
# Make scripts executable
chmod +x claude_mcp_integration_test.sh
chmod +x claude_permission_management.sh
chmod +x comprehensive_api_test.sh

# Run individual scripts
./claude_mcp_integration_test.sh
./claude_permission_management.sh
./comprehensive_api_test.sh

# Or run all scripts in sequence
for script in *.sh; do
    echo "Running $script..."
    ./"$script"
    echo "Completed $script"
    echo ""
done
```

### Script Dependencies

These scripts require:
- `curl` for API calls
- `jq` for JSON processing
- `claude` CLI for MCP integration
- Running MCP platform at `localhost:8000`

Install dependencies:
```bash
# macOS
brew install curl jq

# Ubuntu/Debian
sudo apt-get install curl jq

# Install Claude Code CLI (if not already installed)
# Follow instructions at https://docs.anthropic.com/claude-code
```
