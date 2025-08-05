#!/bin/bash
# Script to add Wand MCP server to Claude Desktop

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🪄 Adding Wand MCP Server to Claude Desktop...${NC}"

# Get the absolute path to this directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "${BLUE}Wand directory: ${SCRIPT_DIR}${NC}"

# Check if Python virtual environment exists
if [ ! -f "${SCRIPT_DIR}/venv/bin/python" ]; then
    echo -e "${RED}❌ Python virtual environment not found at ${SCRIPT_DIR}/venv${NC}"
    echo -e "${YELLOW}Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt${NC}"
    exit 1
fi

# Check if main MCP server file exists
if [ ! -f "${SCRIPT_DIR}/wand.py" ]; then
    echo -e "${RED}❌ Wand server file not found: ${SCRIPT_DIR}/wand.py${NC}"
    exit 1
fi

# Construct the command for Claude MCP
PYTHON_PATH="${SCRIPT_DIR}/venv/bin/python"
MCP_SCRIPT="${SCRIPT_DIR}/wand.py"

echo -e "${BLUE}Python path: ${PYTHON_PATH}${NC}"
echo -e "${BLUE}MCP script: ${MCP_SCRIPT}${NC}"

# Add to Claude using claude mcp add
echo -e "${BLUE}Adding to Claude MCP configuration...${NC}"

if command -v claude >/dev/null 2>&1; then
    echo -e "${GREEN}Found Claude CLI, adding MCP server...${NC}"

    # Add the MCP server (stdio transport is default)
    claude mcp add wand "${PYTHON_PATH}" "${MCP_SCRIPT}"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully added Wand MCP server to Claude Desktop!${NC}"
        echo -e "${BLUE}Server name: wand${NC}"
        echo -e "${BLUE}Command: ${PYTHON_PATH} ${MCP_SCRIPT}${NC}"
        echo ""
        echo -e "${YELLOW}📝 Next steps:${NC}"
        echo -e "1. Restart Claude Desktop to load the new server"
        echo -e "2. The server will be available in your conversations"
        echo -e "3. You can test it by asking Claude to use Wand integrations"
        echo ""
        echo -e "${GREEN}🎉 Wand is now ready to use with Claude!${NC}"
    else
        echo -e "${RED}❌ Failed to add MCP server to Claude${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  Claude CLI not found. Please install it first:${NC}"
    echo -e "${BLUE}Visit: https://docs.anthropic.com/claude/docs/claude-code${NC}"
    echo ""
    echo -e "${YELLOW}Or manually add to your Claude MCP configuration:${NC}"
    echo -e "${BLUE}Server name: wand${NC}"
    echo -e "${BLUE}Command: ${PYTHON_PATH}${NC}"
    echo -e "${BLUE}Args: ${MCP_SCRIPT}${NC}"
    exit 1
fi
