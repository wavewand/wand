# Distributed MCP Server with Multi-Agent System

A sophisticated Model Context Protocol (MCP) server that distributes tasks across specialized agents and integrates with multiple services.

## Features

### Multi-Agent Architecture
- **Project Manager Agent**: Coordinates projects, assigns tasks, monitors progress
- **Frontend Agents**: Handle React, Vue, Angular, TypeScript development
- **Backend Agents**: Manage Python, Go, Node.js API development
- **Database Agent**: PostgreSQL, MySQL, MongoDB design and optimization
- **DevOps Agent**: AWS, Docker, Kubernetes, Jenkins, Terraform
- **Integration Agent**: Slack, Git, YouTrack, API integrations
- **QA Agent**: Testing, quality assurance

### Integrations

#### Communication & Collaboration
- **Slack**: Send messages, create channels, thread replies
- **YouTrack**: Create/update issues, project management

#### Development & Version Control
- **Git (GitHub/GitLab)**: Create PRs, issues, manage repositories
- **Jenkins**: Trigger builds, monitor pipelines

#### Infrastructure & Cloud
- **AWS**: EC2, S3, Lambda, RDS management
- **PostgreSQL**: Direct database operations
- **Docker/Kubernetes**: Container orchestration

#### Specialized Hardware
- **Bambu 3D Printers**: Send print jobs, monitor status

#### External Data
- **Web Search**: Internet searches
- **Arbitrary APIs**: Generic HTTP API calls

## Installation

```bash
# Install Python dependencies
pip install -e .

# Install additional integration dependencies
pip install aiohttp asyncpg slack-sdk PyGithub jenkins boto3
```

## Configuration

### Environment Variables

Create a `.env` file with your integration credentials:

```env
# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token

# Git
GITHUB_TOKEN=ghp_your_github_token
GITLAB_TOKEN=glpat-your_gitlab_token

# Jenkins
JENKINS_URL=https://jenkins.example.com
JENKINS_USER=your_username
JENKINS_TOKEN=your_api_token

# YouTrack
YOUTRACK_URL=https://youtrack.example.com
YOUTRACK_TOKEN=perm:your_token

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=production

# AWS
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Bambu 3D Printers
BAMBU_API_KEY=your_api_key
BAMBU_X1_CODE=printer_access_code
BAMBU_P1S_CODE=printer_access_code

# Search API (optional)
SEARCH_API_KEY=your_search_api_key
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "distributed-mcp": {
      "command": "python",
      "args": ["/path/to/mcp-python/enhanced_distributed_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/mcp-python"
      }
    }
  }
}
```

## Usage Examples

### Project Coordination

```python
# Create a full-stack project
await coordinate_project(
    project_name="E-Commerce Platform",
    requirements="Build a scalable e-commerce platform with user auth, product catalog, and payment processing",
    tech_stack={
        "frontend": "React + TypeScript",
        "backend": "Python FastAPI",
        "database": "PostgreSQL",
        "cache": "Redis",
        "deployment": "AWS ECS"
    }
)
```

### Task Distribution

```python
# Distribute a specific task
await distribute_task(
    title="Implement user authentication",
    description="Add JWT-based authentication to the API",
    task_type="backend",
    priority="high"
)
```

### Integrations Usage

```python
# Slack notification
await slack_send(
    channel="#dev-updates",
    message="Deployment completed successfully! 🎉"
)

# Create GitHub PR
await git_create_pr(
    repo="company/project",
    title="Add user authentication",
    body="Implements JWT-based auth as per ticket #123",
    head_branch="feature/auth",
    base_branch="main"
)

# Trigger Jenkins build
await jenkins_build(
    job_name="api-deployment",
    parameters={"ENVIRONMENT": "staging", "BRANCH": "main"}
)

# Database query
await postgres_execute(
    query="SELECT * FROM users WHERE created_at > $1",
    params=["2024-01-01"]
)

# AWS operations
await aws_ec2(
    operation="list",
    instance_type="t3.medium"
)

# 3D printing
await bambu_send_print(
    printer_id="X1-Carbon-01",
    file_path="/models/prototype_v2.3mf",
    material="PETG",
    quality="high"
)
```

### Monitoring

Access real-time status through resources:

- `agents://status` - Current agent status and capabilities
- `agents://workload` - Task distribution across agents
- `tasks://all` - All tasks in the system
- `integrations://status` - Integration configuration status

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   MCP Client (Claude)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│              Enhanced Distributed MCP Server             │
├─────────────────────────────────────────────────────────┤
│                   Task Manager                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Manager   │  │  Frontend   │  │   Backend   │    │
│  │    Agent    │  │   Agents    │  │   Agents    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Database   │  │   DevOps    │  │ Integration │    │
│  │    Agent    │  │    Agent    │  │    Agent    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
├─────────────────────────────────────────────────────────┤
│                    Integrations                          │
│  ┌──────┐ ┌──────┐ ┌────────┐ ┌─────────┐ ┌────────┐  │
│  │Slack │ │ Git  │ │Jenkins │ │YouTrack │ │Postgres│  │
│  └──────┘ └──────┘ └────────┘ └─────────┘ └────────┘  │
│  ┌──────┐ ┌──────┐ ┌────────┐ ┌─────────┐             │
│  │ AWS  │ │Bambu │ │  Web   │ │   API   │             │
│  └──────┘ └──────┘ └────────┘ └─────────┘             │
└─────────────────────────────────────────────────────────┘
```

## Advanced Features

### Automatic Task Assignment
Tasks are automatically assigned to the most suitable agent based on:
- Required capabilities
- Current workload
- Task priority
- Agent specialization

### Task Dependencies
Create tasks with dependencies:
```python
frontend_task = create_task("Build UI", type="frontend")
backend_task = create_task("Create API", type="backend")
frontend_task.dependencies = [backend_task.id]
```

### Performance Metrics
Agents track performance metrics:
- Tasks completed
- Average completion time
- Success rate
- Specialization effectiveness

## Development

### Running Tests
```bash
pytest test_distributed_server.py -v
```

### Adding New Agents
1. Define agent type in `AgentType` enum
2. Create agent instance in `initialize_agents()`
3. Add capability mappings in `find_best_agent()`

### Adding New Integrations
1. Create integration class in `integrations.py`
2. Add configuration in `integrations_config.py`
3. Create MCP tools in `enhanced_distributed_server.py`

## Security Considerations

- Store credentials in environment variables
- Use read-only database users where possible
- Implement rate limiting for API calls
- Validate all external inputs
- Use SSL/TLS for all connections
- Implement proper authentication for Jenkins/YouTrack
- Restrict AWS IAM permissions

## License

MIT