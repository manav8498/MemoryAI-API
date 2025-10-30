#!/bin/bash
# Quick script to get Memory AI API key for RAG Tutor integration

echo "================================================"
echo "Memory AI - Get API Key for RAG Tutor"
echo "================================================"
echo ""

# Check if Memory AI backend is running
echo "1. Checking if Memory AI backend is running..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Memory AI backend is not running!"
    echo ""
    echo "Start it with:"
    echo "  cd /Users/manavpatel/Documents/API\ Memory"
    echo "  docker-compose up -d"
    exit 1
fi
echo "✅ Memory AI backend is running"
echo ""

# Register or login
echo "2. Creating user account..."
read -p "Enter email: " EMAIL
read -sp "Enter password: " PASSWORD
echo ""

REGISTER_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"full_name\":\"RAG Tutor User\"}")

if echo "$REGISTER_RESPONSE" | grep -q "access_token"; then
    echo "✅ User registered successfully"
    ACCESS_TOKEN=$(echo "$REGISTER_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
else
    # Try login instead
    echo "User exists, logging in..."
    LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/auth/login \
      -H "Content-Type: application/json" \
      -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")

    if echo "$LOGIN_RESPONSE" | grep -q "access_token"; then
        echo "✅ Logged in successfully"
        ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
    else
        echo "❌ Login failed: $LOGIN_RESPONSE"
        exit 1
    fi
fi

echo ""

# Create API key
echo "3. Creating API key..."
API_KEY_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/auth/api-keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"name":"RAG Tutor Integration"}')

if echo "$API_KEY_RESPONSE" | grep -q "\"key\""; then
    API_KEY=$(echo "$API_KEY_RESPONSE" | grep -o '"key":"[^"]*' | cut -d'"' -f4)
    echo "✅ API key created successfully"
    echo ""
    echo "================================================"
    echo "YOUR API KEY:"
    echo ""
    echo "$API_KEY"
    echo ""
    echo "================================================"
    echo ""
    echo "⚠️  SAVE THIS KEY NOW - YOU WON'T SEE IT AGAIN!"
    echo ""
    echo "Next steps:"
    echo "1. Copy the API key above"
    echo "2. Edit: /Users/manavpatel/Downloads/rag-tutor 4/.env"
    echo "3. Set: MEMORY_AI_API_KEY=\"$API_KEY\""
    echo "4. Start your tutor: cd /Users/manavpatel/Downloads/rag-tutor\\ 4 && npm run dev"
    echo ""
else
    echo "❌ Failed to create API key: $API_KEY_RESPONSE"
    exit 1
fi
