# MCP Claude Code - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Deploy the Platform
```bash
cd mcp-automation
./scripts/docker-build.sh && ./scripts/deploy-docker.sh
```

### 2. Add MCP Server to Claude Code
```bash
claude mcp add -t http mcp-automation-api http://localhost:8000/mcp
```

### 3. Test the Integration
```bash
claude --print "Use the get_system_status tool" --allowedTools "mcp__mcp-automation-api__*"
```

## ✅ Expected Output
```
System Status:
Total Agents: 0
Active Agents: 0
Idle Agents: 0
Total Tasks: 0
Completed Tasks: 0
Success Rate: 0.0%
```

## 🔧 Common Commands

### Create an Agent
```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "TestAgent", "type": "automation", "status": "active"}'
```

### Create a Task via MCP
```bash
claude --print "Create a task with title 'Test Task', description 'Testing MCP integration', type 'test'" \
  --allowedTools "mcp__mcp-automation-api__create_task"
```

### Check Status
```bash
claude --print "Get current system status" --allowedTools "mcp__mcp-automation-api__get_system_status"
```

## 🔑 Permission Management

### Option 1: Allow All MCP Tools
```bash
claude --allowedTools "mcp__mcp-automation-api__*" --print "Your command here"
```

### Option 2: Create Settings File
```bash
echo '{
  "allowedTools": ["mcp__mcp-automation-api__*"],
  "permissionMode": "acceptEdits"
}' > ~/.config/claude/mcp_settings.json

claude --settings ~/.config/claude/mcp_settings.json --print "Your command here"
```

### Option 3: Interactive Permission
```bash
claude "Use the get_system_status tool"
# Approve when prompted - saves permission for future use
```

## 🌐 Web Interfaces

- **API Documentation**: http://localhost:8000/docs
- **MCP UI**: http://localhost:3000
- **Grafana Monitoring**: http://localhost:3001
- **System Health**: http://localhost:8000/health

## ❌ Troubleshooting

### MCP Server Not Connected?
```bash
# Check platform is running
curl http://localhost:8000/health

# Re-add MCP server
claude mcp remove mcp-automation-api -s local
claude mcp add -t http mcp-automation-api http://localhost:8000/mcp
```

### Permission Denied?
```bash
# Use --allowedTools or --dangerously-skip-permissions
claude --dangerously-skip-permissions --print "Your command"
```

### No Tools Found?
```bash
# Test MCP endpoint directly
curl -X POST http://localhost:8000/mcp -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

---

**Next Steps**: See [MCP_CLAUDE_CODE_INTEGRATION.md](./MCP_CLAUDE_CODE_INTEGRATION.md) for complete documentation.
