# 🪄 Wand: Magical Multi-Agent MCP Platform

> *"Where AI automation meets enchantment"* ✨

A **spellbinding** Model Context Protocol (MCP) implementation with comprehensive integrations, providing **50+ powerful magical integrations** for AI development workflows. Supports Claude Desktop, Claude Code, and other MCP-compatible clients through mystical stdio and HTTP transports.

```
    🔮 Cast 50+ integration spells across all platforms 🔮
         ⚡ Enhanced error handling with native preservation ⚡
              ✨ Production-ready enchantments ✨
```

## 🚀 Magical Status

- **Protocol Version**: MCP 2025-06-18 (backward compatible)
- **Primary Transport**: stdio (via `./add_to_claude.sh`)
- **Integrations Available**: 50+ comprehensive integrations
- **Client Integration**: Claude Desktop ✅ | Claude Code ✅ | Custom clients ✅
- **Error Handling**: Enhanced with native error preservation

## ✨ Enchanted Features

### 🪄 Comprehensive Integration Arsenal (50+ Integrations)
- **🎥 Media & Content**: Video, Audio, Images, OCR, QR codes
- **🤖 AI & ML**: OpenAI, Anthropic, Cohere, Hugging Face, Local models
- **💬 Communication**: Discord, Telegram, Microsoft Teams, Email, Calendar
- **☁️ Cloud & Storage**: Google Drive, S3, Dropbox, OneDrive
- **🛠️ DevOps**: Docker, Kubernetes, Terraform, Monitoring
- **💼 Business**: CRM, Payments, Project Management, HR tools
- **🔐 Security & Identity**: Enterprise IAM, ServiceNow, SailPoint, Britive, Vault

### 🏗️ Enhanced Error Response Architecture
- **Native Error Preservation**: Complete exception details without abstraction
- **Configuration Validation**: Initialization failures properly propagated
- **Exception Categorization**: Timeout, authentication, connection, and generic errors
- **Structured Logging**: System warnings with comprehensive context
- **Rate Limiting Control**: Configurable per integration (disabled by default)

### 🌐 Multiple Mystical Transports
- **HTTP API**: Full MCP 2025-06-18 implementation with SSE support
- **stdio**: Direct process communication for local clients
- **Session Management**: Secure session handling with cleanup

## 🚀 Quick Spell Casting

### Prerequisites
- Python 3.10 or higher (required for MCP support)
- pip (Python package manager)
- git
- Virtual environment support (venv)

### 1. Setup & Installation
```bash
# Clone and setup
git clone <repository-url>
cd wand

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### Installation Options

Choose the installation that fits your needs:

**Basic Installation (Recommended)**
```bash
# Core server with essential integrations (Slack, GitHub, Docker, AWS)
pip install -r requirements-base.txt
```

**Installation with Audio/Multimedia Support**
```bash
# Includes PyAudio, OpenCV, Whisper, etc.
# Note: Requires system audio libraries
pip install -r requirements-base.txt -r requirements-audio.txt
```

**Installation with AI/ML Support**
```bash
# Includes OpenAI, Anthropic, HuggingFace, etc.
# Note: Large download size due to PyTorch and transformers
pip install -r requirements-base.txt -r requirements-ai.txt
```

**Complete Installation (All Features)**
```bash
# Includes everything - all 55+ integrations
pip install -r requirements-all.txt
```

**Using pip extras (Alternative)**
```bash
# Install from source with optional dependencies
pip install -e .  # Basic installation
pip install -e ".[audio]"  # With audio support
pip install -e ".[ai]"  # With AI/ML support
pip install -e ".[all]"  # Everything
```

#### Environment Configuration
```bash
# Configure environment variables
cp .env.example .env
# Edit .env and configure:
# - OLLAMA_BASE_URL - Your Ollama server URL (default: http://localhost:11434)
# - API keys for any integrations you want to use
# - Database connection string if using PostgreSQL

# Configure Wand
cp config.sample.json config.json
# Edit config.json to match your setup. Replace placeholder paths:
# - {WAND_PATH} - The absolute path to your wand installation
# - {WORKSPACE_PATH} - Your workspace directory
```

### 2. Add to Claude Desktop (Recommended) 🪄
```bash
# One-command setup - adds Wand to Claude Desktop
./add_to_claude.sh

# ✅ Successfully added Wand MCP server to Claude Desktop!
# 📝 Next steps: Restart Claude Desktop to load the server
```

#### Manual Claude Desktop Setup
If you prefer manual setup or the script doesn't work:

1. **Open Claude Desktop Settings**
2. **Navigate to MCP Servers**
3. **Add a new server with these details:**
   - **Name:** `wand`
   - **Command:** `/path/to/wand/venv/bin/python`
   - **Arguments:** `/path/to/wand/wand.py`

#### Troubleshooting Claude Integration
- Ensure you're in the Wand directory when running `./add_to_claude.sh`
- Make sure the virtual environment is set up: `python -m venv venv && source venv/bin/activate && pip install -e .`
- Restart Claude Desktop after adding the server
- Check Claude Desktop logs for connection errors
- Verify the Python path and script path are correct

### 3. Alternative: Manual HTTP Server Setup
```bash
# Start HTTP server
python wand.py http

# Add HTTP MCP server to Claude
claude mcp add wand-http --transport http http://localhost:8001/mcp
```

### 4. Test Your Magic ✨
After restarting Claude Desktop, test the installation:

```bash
# Test imports
./venv/bin/python -c "from integrations.ai_ml.ollama import OllamaIntegration; print('✓ Installation successful')"

# Start the Wand server
./venv/bin/python wand.py stdio
```

Then test the integration:
- Ask Claude: *"Use Wand to check the system status"*
- Try: *"Use Wand to list the available integrations"*
- Or: *"Show me what Wand tools are available"*

#### Troubleshooting Installation

**Module Import Errors**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

**Ollama Connection Issues**
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check `OLLAMA_BASE_URL` in your `.env` file

**Claude Desktop Integration**
- Check logs in the `logs/` directory
- Ensure paths in Claude configuration are absolute, not relative
- Restart Claude Desktop after configuration changes

## ✨ Why the Script Setup is Magical

The `./add_to_claude.sh` script provides the **best experience** because it:

### 🎯 **Optimal Integration Benefits**
- **Direct stdio communication** (faster than HTTP)
- **Extended timeouts** (handles long-running operations)
- **Automatic path detection** (no manual configuration)
- **Enhanced error handling** with detailed diagnostics
- **All 50+ integrations** ready to use immediately

### 🔍 **What the Script Does**
1. **Detects** your Python environment automatically
2. **Configures** the MCP server with proper paths
3. **Adds** Wand to Claude Desktop configuration
4. **Enables** all integrations with enhanced error reporting
5. **Provides** clear next steps

### 🔧 **Extended Timeout Configuration**

For **long-running magical operations** (AI training, large deployments, etc.), configure extended timeouts:

**File:** `/Users/david/wand/settings.json`
```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "env": {
    "BASH_DEFAULT_TIMEOUT_MS": "43200000",
    "BASH_MAX_TIMEOUT_MS": "43200000",
    "MCP_TIMEOUT": "43200000",
    "MCP_TOOL_TIMEOUT": "43200000"
  },
  "permissions": {
    "allow": [
      "mcp__wand__*"
    ]
  }
}
```

**Key Timeout Settings:**
- **43200000ms** = 12 hours (for extensive magical operations)
- **MCP_TIMEOUT**: Overall MCP session timeout
- **MCP_TOOL_TIMEOUT**: Individual spell execution timeout
- **BASH timeouts**: Command execution limits

### 📋 **Complete stdio Configuration**

The working Claude Code configuration (`~/.claude.json`):

```json
{
  "mcpServers": {
    "wand": {
      "command": [
        "/path/to/wand/venv/bin/python",
        "/path/to/wand/wand.py",
        "stdio"
      ],
      "args": [],
      "env": {}
    }
  }
}
```

### 🎭 **stdio vs HTTP Comparison**

| Feature | stdio Mode | HTTP Mode |
|---------|------------|-----------|
| **Performance** | 🚀 Fastest | ⚡ Fast |
| **Timeouts** | 🕐 12+ hours | ⏰ 10 minutes |
| **Setup** | 🔌 One command | 🌐 Server + client |
| **Reliability** | 💎 Highest | 🛡️ High |
| **Use Case** | 🪄 Heavy automation | 🎯 Quick tasks |

## 📚 Magical Grimoires

| Document | Description |
|----------|-------------|
| [📖 Enchanted Documentation Index](docs/README.md) | Complete magical grimoire directory with navigation |
| [🚀 Quick Start](docs/QUICK_START.md) | 5-minute setup guide |
| [🔌 Claude Code Integration](docs/MCP_CLAUDE_CODE_INTEGRATION.md) | Complete integration guide |
| [📊 API Spellbook](docs/API_DOCUMENTATION.md) | All 69 magical tools and enchanted endpoints |
| [🏗️ Architecture](docs/ARCHITECTURE.md) | Complete system design and components |
| [🚀 Deployment](docs/DEPLOYMENT_GUIDE.md) | Production deployment guide |
| [🎨 Wand UI Integration](docs/MCP_UI_INTEGRATION.md) | Magical dashboard and enchanted management APIs |

## 🪄 Available Magical Integrations (50+ Total)

### 🎥 Media & Content Creation
- **FFmpeg** - Video processing and conversion
- **OpenCV** - Computer vision and image processing
- **YouTube** - Video upload and management
- **Twitch** - Streaming platform integration
- **Audio** - Audio processing and manipulation
- **Whisper** - Speech-to-text transcription
- **ElevenLabs** - Text-to-speech synthesis
- **Image** - Image generation and editing
- **OCR** - Optical character recognition
- **QR** - QR code generation and reading
- **Chart** - Data visualization and charting

### 🤖 AI & Machine Learning
- **OpenAI** - GPT models and API integration
- **Anthropic** - Claude model integration
- **Cohere** - Language model services
- **Hugging Face** - Model hub and transformers
- **Replicate** - Cloud AI model hosting
- **Stability AI** - Image generation models
- **Ollama** - Local language model management
- **DeepL** - Advanced translation services

### 💬 Communication & Social
- **Discord** - Bot integration and messaging
- **Telegram** - Bot and messaging automation
- **Microsoft Teams** - Webhook messaging and notifications
- **Email** - SMTP/IMAP email management
- **Calendar** - Calendar integration and scheduling

### ☁️ Cloud Storage & File Management
- **Google Drive** - File storage and sharing
- **Dropbox** - Cloud file synchronization
- **OneDrive** - Microsoft cloud storage
- **S3** - Amazon S3 object storage
- **FTP** - File transfer protocol operations

### 📚 Documentation & Knowledge
- **Notion** - Knowledge management integration
- **Confluence** - Team wiki and documentation
- **GitBook** - Documentation platform
- **Markdown** - Markdown processing and conversion
- **PDF** - PDF generation and manipulation

### 🛠️ DevOps & Infrastructure
- **Docker** - Container management
- **Kubernetes** - Container orchestration
- **Terraform** - Infrastructure as code
- **Prometheus** - Monitoring and metrics
- **Datadog** - Application monitoring
- **Sentry** - Error tracking and monitoring

### 🔍 Testing & Automation
- **Selenium** - Web browser automation
- **Playwright** - Modern web testing
- **Postman** - API testing and development

### 💼 Business & CRM
- **Salesforce** - CRM and sales automation
- **HubSpot** - Marketing and sales platform
- **Pipedrive** - Sales pipeline management
- **Stripe** - Payment processing

### 📋 Project Management
- **Jira** - Issue tracking and project management
- **Asana** - Team task management
- **Trello** - Kanban board management
- **Linear** - Modern issue tracking
- **Monday.com** - Work operating system

### 👥 HR & Productivity
- **Workday** - Human capital management
- **BambooHR** - HR information system
- **Toggl** - Time tracking
- **Harvest** - Time tracking and invoicing

### 🔐 Security & Identity
- **ServiceNow** - IT Service Management and ITSM
- **SailPoint** - Identity Security Cloud and governance
- **Microsoft Entra** - Azure AD identity management
- **Britive** - Privileged access management (PAM)
- **Vault** - Secret management
- **1Password** - Password management
- **Okta** - Identity and access management
- **Auth0** - Authentication as a service
- **Veracode** - Application security testing
- **Snyk** - Vulnerability management
- **SonarQube** - Code quality and security

### 🎮 Entertainment & Gaming
- **Spotify** - Music streaming integration
- **Podcast** - Podcast management and processing
- **Steam** - Gaming platform integration

## 🏢 Enterprise Integration Spotlight

Wand now includes comprehensive **enterprise identity management and communication** tools:

### 🔐 Identity & Access Management
- **`servicenow`** - Create incidents, manage users, query ITSM records
- **`sailpoint`** - Identity governance, access requests, certification campaigns
- **`entra`** - Azure AD user/group management, role assignments
- **`britive`** - Just-in-time privileged access, secret checkout

### 💬 Enterprise Communication
- **`teams`** - Send messages, cards, and notifications via webhooks

**Example Usage:**
```python
# Create ServiceNow incident
servicenow(operation="create_incident",
          short_description="Server outage",
          priority="1")

# Request SailPoint access
sailpoint(operation="request_access",
          identity_id="user123",
          access_profile_ids=["admin_profile"])

# Send Teams notification
teams(operation="send_notification",
      title="Deployment Complete",
      message="v2.1.0 deployed successfully",
      status="success")
```

See [Enterprise Integrations Guide](docs/ENTERPRISE_INTEGRATIONS.md) for complete setup and usage documentation.

## 🪄 Magical Server Modes

### HTTP Mode (Recommended)
```bash
python wand.py http
# Server available at http://localhost:8001/mcp
```

### stdio Mode
```bash
python wand.py stdio
# For direct process communication
```

## 🏗️ Magical Architecture

- **Multi-Agent System**: 3 internal agents with load balancing
- **Execution Backends**: Native, Docker, SSH, Host Agent
- **Security**: Command validation, path restrictions, resource limits
- **Monitoring**: Health checks, performance tracking, audit logging

## 🐳 Docker Support

```bash
# Quick start with Docker
docker build -f scripts/Dockerfile -t wand .
docker run -p 8001:8001 wand
```

## 🔒 Protective Wards

- OAuth 2.1 authentication with resource indicators
- Command allowlist/blocklist with pattern matching
- Path restrictions and resource limits
- Session management with cleanup
- Comprehensive audit logging

## 📊 Magical Performance

- **Response Time**: <200ms average
- **Concurrency**: 10+ simultaneous sessions
- **Memory Usage**: <50MB per backend
- **Scalability**: Horizontal scaling support

## 🧪 Local CI & Testing

Wand includes a comprehensive local CI system that mirrors the GitHub Actions pipeline, helping you catch issues before pushing code.

### Quick Testing
```bash
# Run the full CI pipeline (recommended before every commit)
./ci.sh

# Run only enterprise integration tests
./ci.sh --enterprise

# Run with verbose output for detailed logs
./ci.sh --enterprise --verbose

# Setup environment only
./ci.sh --setup
```

### Available Test Modes
- `./ci.sh` - Full CI pipeline (setup + all checks + tests)
- `./ci.sh --enterprise` - Enterprise integration tests only
- `./ci.sh --basic` - Basic tests (no external dependencies)
- `./ci.sh --tests-only` - All tests without setup
- `./ci.sh --lint-only` - Code linting only
- `./ci.sh --security-only` - Security scans only

### CI Pipeline Features
- **Environment Setup**: Automatic virtual environment management
- **Dependency Management**: Installs required and optional dependencies
- **Enterprise Integration Testing**: Tests for ServiceNow, SailPoint, Microsoft Entra, Britive, Teams
- **Code Quality**: Linting (ruff), type checking (mypy), security scanning (bandit, safety)
- **Smart Dependencies**: Gracefully handles missing optional enterprise packages
- **Colored Output**: Clear, colored logging for easy reading

### Enterprise Integration Tests
The enterprise tests are designed for CI environments:
- **With Dependencies**: Full integration tests with proper mocking
- **Without Dependencies**: Tests automatically skip with descriptive messages
- **Expected Skips**: `pysnc` (ServiceNow), `azure-identity` (Entra), `britive` (PAM)

### Before Committing
```bash
# Always run CI to catch issues early
./ci.sh

# If enterprise tests fail, investigate with verbose output
./ci.sh --enterprise --verbose
```

## 🤝 Join the Magic

### Development Workflow
1. **Setup**: Run `./ci.sh --setup` to prepare your environment
2. **Development**: Use `./ci.sh --tests-only` for quick feedback
3. **Before committing**: Run `./ci.sh` to ensure all checks pass
4. **Code quality**: Use `./ci.sh --lint-only` to fix style issues

### Contributing Steps
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Set up development environment (`./ci.sh --setup`)
4. Make your changes and test (`./ci.sh`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under a proprietary license - see the LICENSE file for details.

## 🆘 Magical Support

- 📖 **Documentation**: Check the `docs/` directory for detailed guides
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Questions**: Start a GitHub Discussion
- 🔧 **Debug**: Enable debug logging with `LOG_LEVEL=DEBUG`

## 🚫 AI Training Policy

This repository **opts out** of AI/ML training. The code and documentation in this repository may not be used for training machine learning models without explicit written permission.

For more details, see:
- [DO_NOT_TRAIN.md](./DO_NOT_TRAIN.md) - Full legal notice
- `.ai-training-opt-out` - Opt-out marker file
- `.noai` - Additional opt-out marker
- `robots.txt` - Crawler directives

---

> 🪄 **May your automation be swift and your magic be strong!** ✨
> *Happy spell casting with Wand!*

```
    🌟 Welcome to the magical realm of automation 🌟
         🔮 Where 50+ integrations await your command 🔮
              ✨ Cast responsibly, automate magically ✨
```
