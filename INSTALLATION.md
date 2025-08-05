# Wand Installation Guide

## Prerequisites

- Python 3.10 or higher (required for MCP support)
- pip (Python package manager)
- git
- Virtual environment support (venv)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/wavewand/wand.git
cd wand
```

### 2. Set Up Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Wand uses a modular requirements structure. Choose the installation that fits your needs:

#### Basic Installation (Recommended)
```bash
# Core server with essential integrations (Slack, GitHub, Docker, AWS)
pip install -r requirements-base.txt
```

#### Installation with Audio/Multimedia Support
```bash
# Includes PyAudio, OpenCV, Whisper, etc.
# Note: Requires system audio libraries (see requirements-audio.txt for details)
pip install -r requirements-base.txt -r requirements-audio.txt
```

#### Installation with AI/ML Support
```bash
# Includes OpenAI, Anthropic, HuggingFace, etc.
# Note: Large download size due to PyTorch and transformers
pip install -r requirements-base.txt -r requirements-ai.txt
```

#### Complete Installation (All Features)
```bash
# Includes everything - all 50+ integrations
pip install -r requirements-all.txt
```

#### Using pip extras (Alternative)
```bash
# Install from source with optional dependencies
pip install -e .  # Basic installation
pip install -e ".[audio]"  # With audio support
pip install -e ".[ai]"  # With AI/ML support
pip install -e ".[all]"  # Everything
```

### 4. Configure Environment Variables

Copy the example environment file and update with your settings:

```bash
cp .env.example .env
```

Edit `.env` and configure:
- `OLLAMA_BASE_URL` - Your Ollama server URL (default: http://localhost:11434)
- API keys for any integrations you want to use
- Database connection string if using PostgreSQL

### 5. Configure Wand

Copy the sample configuration:

```bash
cp config.sample.json config.json
```

Edit `config.json` to match your setup. Replace placeholder paths:
- `{WAND_PATH}` - The absolute path to your wand installation
- `{WORKSPACE_PATH}` - Your workspace directory

## Integration with Claude Desktop

### Automatic Setup

Run the provided script from the wand directory:

```bash
./add_to_claude.sh
```

### Manual Setup

1. Open Claude Desktop settings
2. Navigate to MCP Servers
3. Add a new server named "wand"
4. Configure the command:
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
   Replace `/path/to/wand` with your actual installation path.

5. Restart Claude Desktop

## Verification

Test the installation:

```bash
# Test imports
./venv/bin/python -c "from integrations.ai_ml.ollama import OllamaIntegration; print('✓ Installation successful')"

# Start the Wand server
./venv/bin/python wand.py stdio
```

## Troubleshooting

### Module Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

### Ollama Connection Issues
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check `OLLAMA_BASE_URL` in your `.env` file

### Claude Desktop Integration
- Check logs in the `logs/` directory
- Ensure paths in Claude configuration are absolute, not relative
- Restart Claude Desktop after configuration changes

## Next Steps

- Read the [Quick Start Guide](docs/QUICK_START.md)
- Check available [Integrations](docs/INTEGRATIONS.md)
- Review the [Architecture](ARCHITECTURE.md)
