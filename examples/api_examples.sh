#!/bin/bash

# MCP Platform REST API Examples
# This script demonstrates how to interact directly with the MCP platform API

BASE_URL="http://localhost:8000"

echo "🔧 MCP Platform API Examples"
echo "============================"

# Function to make API calls with proper formatting
api_call() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local description="$4"

    echo ""
    echo "📋 $description"
    echo "Method: $method"
    echo "Endpoint: $endpoint"

    if [ -n "$data" ]; then
        echo "Data: $data"
        echo "Response:"
        curl -s -X "$method" "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" | jq '.' 2>/dev/null || echo "Response received (not JSON)"
    else
        echo "Response:"
        curl -s -X "$method" "$BASE_URL$endpoint" | jq '.' 2>/dev/null || echo "Response received (not JSON)"
    fi
    echo ""
}

# Test 1: Check system health
api_call "GET" "/health" "" "Check System Health"

# Test 2: Get system status
api_call "GET" "/api/v1/system/status" "" "Get System Status"

# Test 3: List existing agents
api_call "GET" "/api/v1/agents" "" "List All Agents"

# Test 4: Create a new agent
AGENT_DATA='{
  "name": "DataProcessorAgent",
  "type": "processing",
  "capabilities": ["data_processing", "file_handling", "batch_operations"],
  "status": "active",
  "metadata": {
    "version": "1.0.0",
    "created_by": "api_example"
  }
}'
api_call "POST" "/api/v1/agents" "$AGENT_DATA" "Create New Agent"

# Test 5: List agents after creation
api_call "GET" "/api/v1/agents" "" "List Agents After Creation"

# Test 6: Create a project
PROJECT_DATA='{
  "name": "Data Processing Pipeline",
  "description": "Automated data processing and analysis pipeline",
  "metadata": {
    "priority": "high",
    "department": "engineering"
  }
}'
api_call "POST" "/api/v1/projects" "$PROJECT_DATA" "Create New Project"

# Test 7: List projects
api_call "GET" "/api/v1/projects" "" "List All Projects"

# Test 8: Create a task
TASK_DATA='{
  "title": "Process Customer Data",
  "description": "Process and analyze customer data files uploaded this week",
  "type": "data_processing",
  "priority": "high",
  "metadata": {
    "estimated_duration": "2 hours",
    "required_resources": ["cpu_intensive", "storage"]
  }
}'
api_call "POST" "/api/v1/tasks" "$TASK_DATA" "Create New Task"

# Test 9: List tasks
api_call "GET" "/api/v1/tasks" "" "List All Tasks"

# Test 10: Get system metrics
api_call "GET" "/api/v1/system/metrics" "" "Get System Metrics"

# Test 11: Test MCP endpoint directly
MCP_INIT_DATA='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
api_call "POST" "/mcp" "$MCP_INIT_DATA" "Test MCP Initialize"

# Test 12: Get MCP tools list
MCP_TOOLS_DATA='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
api_call "POST" "/mcp" "$MCP_TOOLS_DATA" "Get MCP Tools List"

# Test 13: Call MCP get_system_status tool
MCP_STATUS_DATA='{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_system_status","arguments":{}}}'
api_call "POST" "/mcp" "$MCP_STATUS_DATA" "Call MCP System Status Tool"

echo "✅ API Examples completed!"
echo ""
echo "🔍 Additional endpoints to explore:"
echo "   - GET /api/v1/monitoring/stats - Monitoring statistics"
echo "   - GET /api/v1/cache/stats - Cache statistics"
echo "   - GET /api/v1/ai/frameworks - Available AI frameworks"
echo "   - GET /openapi.json - Complete API specification"
echo ""
echo "📚 Full API documentation: $BASE_URL/docs"
