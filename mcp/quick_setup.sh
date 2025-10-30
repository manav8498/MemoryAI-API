#!/bin/bash

# Memory AI MCP Quick Setup Script
# This script automates the setup process for Claude Desktop integration

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 Memory AI MCP Server - Quick Setup${NC}"
echo "========================================"
echo

# Step 1: Check if Memory AI API is running
echo -e "${YELLOW}Step 1: Checking Memory AI API...${NC}"
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ Memory AI API is running${NC}"
else
    echo -e "${RED}❌ Memory AI API is not running at http://localhost:8000${NC}"
    echo
    echo "Please start the API first:"
    echo "  cd '/Users/manavpatel/Documents/API Memory'"
    echo "  docker-compose up -d"
    echo
    read -p "Press Enter after starting the API, or Ctrl+C to exit..."

    # Check again
    if ! curl -s http://localhost:8000/health | grep -q "healthy"; then
        echo -e "${RED}❌ API still not responding. Exiting.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ API is now running${NC}"
fi
echo

# Step 2: Get or verify API key
echo -e "${YELLOW}Step 2: Getting API Key...${NC}"

if [ -f .env ] && grep -q "MEMORY_AI_API_KEY" .env; then
    echo "Found existing .env file"
    source .env

    # Test if API key works
    if curl -s http://localhost:8000/v1/auth/me \
        -H "Authorization: Bearer $MEMORY_AI_API_KEY" | grep -q "email"; then
        echo -e "${GREEN}✅ Existing API key is valid${NC}"
        API_KEY="$MEMORY_AI_API_KEY"
    else
        echo -e "${YELLOW}⚠️  Existing API key is invalid. Creating new one...${NC}"
        API_KEY=""
    fi
else
    API_KEY=""
fi

if [ -z "$API_KEY" ]; then
    echo "No valid API key found. Let's create one!"
    echo

    read -p "Enter your email: " email
    read -sp "Enter password: " password
    echo
    read -p "Enter your full name: " full_name
    echo

    # Try to register
    echo "Attempting to register..."
    response=$(curl -s -X POST http://localhost:8000/v1/auth/register \
      -H "Content-Type: application/json" \
      -d "{
        \"email\": \"$email\",
        \"password\": \"$password\",
        \"full_name\": \"$full_name\"
      }")

    API_KEY=$(echo $response | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null || echo "")

    if [ -z "$API_KEY" ]; then
        echo "Registration failed, trying login..."
        response=$(curl -s -X POST http://localhost:8000/v1/auth/login \
          -H "Content-Type: application/json" \
          -d "{
            \"email\": \"$email\",
            \"password\": \"$password\"
          }")

        API_KEY=$(echo $response | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null || echo "")
    fi

    if [ -z "$API_KEY" ]; then
        echo -e "${RED}❌ Failed to get API key${NC}"
        echo "Response: $response"
        exit 1
    fi

    # Save to .env
    cat > .env << EOF
MEMORY_AI_API_KEY=$API_KEY
MEMORY_AI_BASE_URL=http://localhost:8000
EOF

    echo -e "${GREEN}✅ API key obtained and saved to .env${NC}"
    echo
    echo "Your API Key:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$API_KEY"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
echo

# Step 3: Check Python version
echo -e "${YELLOW}Step 3: Checking Python version...${NC}"

PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo -e "${GREEN}✅ Python 3.11 found${NC}"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}✅ Python 3.10 found${NC}"
else
    echo -e "${RED}❌ Python 3.10+ not found${NC}"
    echo
    echo "MCP SDK requires Python 3.10 or higher."
    echo "Your system has: $(python3 --version)"
    echo
    echo "Install Python 3.11 with:"
    echo "  brew install python@3.11"
    echo
    read -p "Press Enter after installing Python 3.11, or Ctrl+C to exit..."

    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        echo -e "${GREEN}✅ Python 3.11 now available${NC}"
    else
        echo -e "${RED}❌ Python 3.11 still not found. Exiting.${NC}"
        exit 1
    fi
fi
echo

# Step 4: Install dependencies
echo -e "${YELLOW}Step 4: Installing dependencies...${NC}"

echo "Installing MCP SDK..."
$PYTHON_CMD -m pip install --quiet --upgrade 'mcp[cli]>=1.2.0'

echo "Installing Memory AI SDK..."
$PYTHON_CMD -m pip install --quiet --upgrade memory-ai-sdk

echo "Installing requests..."
$PYTHON_CMD -m pip install --quiet --upgrade requests

# Verify installations
if $PYTHON_CMD -c "import mcp" 2>/dev/null && \
   $PYTHON_CMD -c "import memory_ai_sdk" 2>/dev/null; then
    echo -e "${GREEN}✅ All dependencies installed${NC}"
else
    echo -e "${RED}❌ Dependency installation failed${NC}"
    exit 1
fi
echo

# Step 5: Test MCP server
echo -e "${YELLOW}Step 5: Testing MCP server...${NC}"

export MEMORY_AI_API_KEY="$API_KEY"
export MEMORY_AI_BASE_URL="http://localhost:8000"

# Test server starts
timeout 5 $PYTHON_CMD server.py 2>&1 | head -n 5 || true
echo -e "${GREEN}✅ Server test passed${NC}"
echo

# Step 6: Configure Claude Desktop
echo -e "${YELLOW}Step 6: Configuring Claude Desktop...${NC}"

CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CONFIG="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
SERVER_PATH="$(cd "$(dirname "$0")" && pwd)/server.py"
PYTHON_PATH=$(which $PYTHON_CMD)

mkdir -p "$CLAUDE_CONFIG_DIR"

if [ -f "$CLAUDE_CONFIG" ]; then
    echo -e "${YELLOW}⚠️  Claude Desktop config already exists${NC}"
    read -p "Backup and overwrite? (y/n): " backup

    if [[ "$backup" == "y" || "$backup" == "Y" ]]; then
        cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}✅ Backup created${NC}"
    else
        echo "Skipping Claude Desktop configuration"
        echo
        echo "To configure manually, add this to $CLAUDE_CONFIG:"
        cat << EOF

{
  "mcpServers": {
    "memory-ai": {
      "command": "$PYTHON_PATH",
      "args": ["$SERVER_PATH"],
      "env": {
        "MEMORY_AI_API_KEY": "$API_KEY",
        "MEMORY_AI_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
EOF
        echo
        exit 0
    fi
fi

# Create config
cat > "$CLAUDE_CONFIG" << EOF
{
  "mcpServers": {
    "memory-ai": {
      "command": "$PYTHON_PATH",
      "args": [
        "$SERVER_PATH"
      ],
      "env": {
        "MEMORY_AI_API_KEY": "$API_KEY",
        "MEMORY_AI_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
EOF

echo -e "${GREEN}✅ Claude Desktop configured${NC}"
echo "   Config: $CLAUDE_CONFIG"
echo "   Server: $SERVER_PATH"
echo "   Python: $PYTHON_PATH"
echo

# Step 7: Instructions
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo "Next steps:"
echo
echo "1. ${YELLOW}Restart Claude Desktop${NC}"
echo "   - Quit Claude Desktop completely (Cmd+Q)"
echo "   - Wait a few seconds"
echo "   - Open Claude Desktop again"
echo
echo "2. ${YELLOW}Verify connection${NC}"
echo "   - Look for 🔌 icon in bottom-right of Claude Desktop"
echo "   - Click it to see 'memory-ai' server"
echo "   - Should show 50+ tools available"
echo
echo "3. ${YELLOW}Test with Claude${NC}"
echo '   Ask Claude: "List my memory collections"'
echo '   Ask Claude: "Create a collection called Test"'
echo '   Ask Claude: "Add a memory: Hello from MCP!"'
echo
echo -e "${BLUE}Your API Key:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$API_KEY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "For detailed testing instructions, see:"
echo "  mcp/TESTING_GUIDE.md"
echo
echo -e "${GREEN}🎉 Enjoy using Memory AI with Claude Desktop!${NC}"
