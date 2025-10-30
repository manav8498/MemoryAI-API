#!/bin/bash

# Memory AI MCP Server Installation Script
# This script sets up the MCP server for use with Claude Desktop

set -e

echo "🧠 Memory AI MCP Server Installer"
echo "=================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
echo

# Check pip
echo "Checking pip..."
if ! python3 -m pip --version &> /dev/null; then
    echo -e "${RED}❌ pip not found. Installing pip...${NC}"
    python3 -m ensurepip --upgrade
fi
echo -e "${GREEN}✅ pip is available${NC}"
echo

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi
echo

# Check if Memory AI SDK is installed
echo "Verifying Memory AI SDK..."
if python3 -c "import memory_ai_sdk" 2>/dev/null; then
    echo -e "${GREEN}✅ Memory AI SDK installed${NC}"
else
    echo -e "${YELLOW}⚠️  Memory AI SDK not found. Installing...${NC}"
    python3 -m pip install memory-ai-sdk
fi
echo

# Check if MCP SDK is installed
echo "Verifying MCP SDK..."
if python3 -c "import mcp" 2>/dev/null; then
    echo -e "${GREEN}✅ MCP SDK installed${NC}"
else
    echo -e "${YELLOW}⚠️  MCP SDK not found. Installing...${NC}"
    python3 -m pip install 'mcp[cli]'
fi
echo

# Get API configuration
echo "Configuration"
echo "-------------"
echo

read -p "Enter Memory AI API key (or press Enter to skip): " API_KEY
if [ -z "$API_KEY" ]; then
    API_KEY="not_set"
fi

read -p "Enter Memory AI base URL [http://localhost:8000]: " BASE_URL
if [ -z "$BASE_URL" ]; then
    BASE_URL="http://localhost:8000"
fi

# Create .env file
echo "Creating .env file..."
cat > .env << EOF
MEMORY_AI_API_KEY=$API_KEY
MEMORY_AI_BASE_URL=$BASE_URL
EOF
echo -e "${GREEN}✅ .env file created${NC}"
echo

# Detect OS and Claude config location
echo "Detecting Claude Desktop configuration..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
    CLAUDE_CONFIG="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    CLAUDE_CONFIG_DIR="$APPDATA/Claude"
    CLAUDE_CONFIG="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
else
    # Linux
    CLAUDE_CONFIG_DIR="$HOME/.config/Claude"
    CLAUDE_CONFIG="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
fi

# Get absolute path to server.py
SERVER_PATH="$(cd "$(dirname "$0")" && pwd)/server.py"

echo "Claude Desktop config location: $CLAUDE_CONFIG"
echo "Server path: $SERVER_PATH"
echo

# Ask if user wants to configure Claude Desktop
read -p "Configure Claude Desktop? (y/n): " CONFIGURE_CLAUDE

if [[ "$CONFIGURE_CLAUDE" == "y" || "$CONFIGURE_CLAUDE" == "Y" ]]; then
    # Create config directory if it doesn't exist
    mkdir -p "$CLAUDE_CONFIG_DIR"

    # Check if config file exists
    if [ -f "$CLAUDE_CONFIG" ]; then
        echo -e "${YELLOW}⚠️  Claude config already exists${NC}"
        read -p "Backup existing config? (y/n): " BACKUP

        if [[ "$BACKUP" == "y" || "$BACKUP" == "Y" ]]; then
            cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
            echo -e "${GREEN}✅ Backup created${NC}"
        fi
    fi

    # Create or update config
    cat > "$CLAUDE_CONFIG" << EOF
{
  "mcpServers": {
    "memory-ai": {
      "command": "python3",
      "args": [
        "$SERVER_PATH"
      ],
      "env": {
        "MEMORY_AI_API_KEY": "$API_KEY",
        "MEMORY_AI_BASE_URL": "$BASE_URL"
      }
    }
  }
}
EOF

    echo -e "${GREEN}✅ Claude Desktop configured${NC}"
    echo
    echo -e "${YELLOW}⚠️  Please restart Claude Desktop for changes to take effect${NC}"
else
    echo "Skipping Claude Desktop configuration"
    echo
    echo "To configure manually, add this to $CLAUDE_CONFIG:"
    echo
    cat << EOF
{
  "mcpServers": {
    "memory-ai": {
      "command": "python3",
      "args": ["$SERVER_PATH"],
      "env": {
        "MEMORY_AI_API_KEY": "$API_KEY",
        "MEMORY_AI_BASE_URL": "$BASE_URL"
      }
    }
  }
}
EOF
fi
echo

# Test server
echo "Testing server..."
echo -e "${YELLOW}Starting server test (press Ctrl+C to stop)...${NC}"
echo

export MEMORY_AI_API_KEY="$API_KEY"
export MEMORY_AI_BASE_URL="$BASE_URL"

timeout 5 python3 server.py 2>&1 || true

echo
echo -e "${GREEN}✅ Installation complete!${NC}"
echo
echo "Next steps:"
echo "1. Make sure Memory AI API is running at $BASE_URL"
echo "2. Restart Claude Desktop"
echo "3. Look for the 🔌 icon in Claude Desktop"
echo "4. You should see 'memory-ai' server with 50+ tools"
echo
echo "Test with Claude:"
echo '  "List my memory collections"'
echo '  "Create a collection called Test"'
echo
echo "For help, see README.md"
echo

# Make server executable
chmod +x server.py

echo -e "${GREEN}🎉 Setup complete! Enjoy using Memory AI with Claude!${NC}"
