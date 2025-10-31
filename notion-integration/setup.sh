#!/bin/bash
# Notion Memory AI Integration - Setup Script

echo "🚀 Setting up Notion Memory AI Integration..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your credentials:"
    echo "   1. NOTION_TOKEN - Get from https://www.notion.so/my-integrations"
    echo "   2. MEMORY_AI_API_KEY - Get from your Memory AI API"
    echo ""
else
    echo "✓ .env file already exists"
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    echo "  Try: pip3 install -r requirements.txt"
    exit 1
fi

echo ""
echo "="*60
echo "✅ Setup Complete!"
echo "="*60
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials"
echo "2. Create Notion integration: https://www.notion.so/my-integrations"
echo "3. Share your Notion pages with the integration"
echo "4. Run: python3.11 notion_memory_bot.py"
echo ""
echo "Read README.md for detailed instructions"
echo ""
