# 🪄 Wand: Magical Multi-Agent Protocol Server

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
- **💬 Communication**: Discord, Telegram, Email, Calendar
- **☁️ Cloud & Storage**: Google Drive, S3, Dropbox, OneDrive
- **🛠️ DevOps**: Docker, Kubernetes, Terraform, Monitoring
- **💼 Business**: CRM, Payments, Project Management, HR tools
- **🔐 Security**: Identity, Password management, Vulnerability scanning

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

### 1. Setup & Installation
```bash
# Clone and setup
git clone <repository-url>
cd wand
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
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
After restarting Claude Desktop, test the integration:
- Ask Claude: *"Use Wand to check the system status"*
- Try: *"Use Wand to list the available integrations"*
- Or: *"Show me what Wand tools are available"*

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
| [🏗️ Architecture](docs/ARCHITECTURE.md) | System design and components |
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

## 🤝 Join the Magic

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under a proprietary license - see the LICENSE file for details.

## 🆘 Magical Support

- 📖 **Documentation**: Check the `docs/` directory for detailed guides
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Questions**: Start a GitHub Discussion
- 🔧 **Debug**: Enable debug logging with `LOG_LEVEL=DEBUG`

---

> 🪄 **May your automation be swift and your magic be strong!** ✨
> *Happy spell casting with Wand!*

```
    🌟 Welcome to the magical realm of automation 🌟
         🔮 Where 50+ integrations await your command 🔮
              ✨ Cast responsibly, automate magically ✨
```
