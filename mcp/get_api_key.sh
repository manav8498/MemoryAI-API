#!/bin/bash

# Quick script to register and get API key from Memory AI

echo "🔑 Memory AI - Get API Key"
echo "=========================="
echo

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Memory AI API is not running at http://localhost:8000"
    echo "Please start the API first with:"
    echo "  cd '/Users/manavpatel/Documents/API Memory'"
    echo "  docker-compose up -d"
    echo "  python3 backend/main.py"
    exit 1
fi

echo "✅ API is running at http://localhost:8000"
echo

# Prompt for user details
read -p "Enter your email: " email
read -sp "Enter password: " password
echo
read -p "Enter your full name: " full_name
echo

echo "Registering user..."

# Register user and get API key
response=$(curl -s -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d "{
    \"email\": \"$email\",
    \"password\": \"$password\",
    \"full_name\": \"$full_name\"
  }")

# Extract access token
access_token=$(echo $response | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null)

if [ -z "$access_token" ]; then
    echo "❌ Registration failed. Response:"
    echo $response | python3 -m json.tool 2>/dev/null || echo $response
    echo
    echo "User might already exist. Try logging in..."

    # Try login instead
    response=$(curl -s -X POST http://localhost:8000/v1/auth/login \
      -H "Content-Type: application/json" \
      -d "{
        \"email\": \"$email\",
        \"password\": \"$password\"
      }")

    access_token=$(echo $response | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null)
fi

if [ -n "$access_token" ]; then
    echo
    echo "✅ SUCCESS! Your API Key:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$access_token"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "Save this API key! You'll need it for:"
    echo "1. Claude Desktop configuration"
    echo "2. Testing the MCP server"
    echo "3. Using the Memory AI SDK"
    echo
    echo "Add to your .env file:"
    echo "MEMORY_AI_API_KEY=$access_token"
    echo

    # Optionally save to .env
    read -p "Save to mcp/.env file? (y/n): " save_env
    if [[ "$save_env" == "y" || "$save_env" == "Y" ]]; then
        echo "MEMORY_AI_API_KEY=$access_token" > .env
        echo "MEMORY_AI_BASE_URL=http://localhost:8000" >> .env
        echo "✅ Saved to .env file"
    fi
else
    echo "❌ Failed to get API key"
    echo "Response:"
    echo $response | python3 -m json.tool 2>/dev/null || echo $response
fi
