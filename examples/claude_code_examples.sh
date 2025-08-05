#!/bin/bash

# MCP Claude Code Integration Examples
# This script demonstrates various ways to interact with the MCP platform via Claude Code

echo "🚀 MCP Claude Code Integration Examples"
echo "======================================"

# Function to run Claude Code commands with proper permissions
run_claude() {
    local description="$1"
    local command="$2"

    echo ""
    echo "📋 $description"
    echo "Command: $command"
    echo "Output:"
    eval "$command"
    echo ""
}

# Test 1: Check system status
run_claude "Check System Status" \
    'claude --print "Use the get_system_status tool to check the automation platform" --allowedTools "mcp__mcp-automation-api__get_system_status"'

# Test 2: List agents
run_claude "List All Agents" \
    'claude --print "List all available agents in the system" --allowedTools "mcp__mcp-automation-api__list_agents"'

# Test 3: Create a test task
run_claude "Create Test Task" \
    'claude --print "Create a task with title '\''Integration Test'\'', description '\''Testing MCP integration with Claude Code'\'', type '\''test'\'', and priority '\''medium'\''" --allowedTools "mcp__mcp-automation-api__create_task"'

# Test 4: Check status after task creation
run_claude "Check Status After Task Creation" \
    'claude --print "Get updated system status after creating the task" --allowedTools "mcp__mcp-automation-api__get_system_status"'

# Test 5: Using wildcard permissions
run_claude "Using Wildcard Permissions" \
    'claude --print "Show me the current system status and list of agents" --allowedTools "mcp__mcp-automation-api__*"'

# Test 6: Complex task creation
run_claude "Create Complex Task" \
    'claude --print "Create a high-priority data processing task with title '\''Process User Data'\'', description '\''Process and analyze user uploaded data files'\'', type '\''processing'\''" --allowedTools "mcp__mcp-automation-api__*"'

# Test 7: Interactive mode example (commented out - requires user interaction)
echo "💡 Interactive Mode Example (run manually):"
echo "claude \"Use the get_system_status tool to check our automation platform\""
echo "   → This will prompt for permission and save it for future use"
echo ""

# Test 8: Using settings file
echo "⚙️  Settings File Example:"
echo "Create ~/.config/claude/mcp_settings.json with:"
echo '{
  "allowedTools": ["mcp__mcp-automation-api__*"],
  "permissionMode": "acceptEdits"
}'
echo ""
echo "Then use: claude --settings ~/.config/claude/mcp_settings.json --print \"Your command\""
echo ""

echo "✅ Examples completed!"
echo ""
echo "🔗 Useful URLs:"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - MCP UI: http://localhost:3000"
echo "   - System Health: http://localhost:8000/health"
echo ""
echo "📚 For more information, see docs/MCP_CLAUDE_CODE_INTEGRATION.md"
