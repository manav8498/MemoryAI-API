# 🦜 LangChain Integration Guide for Memory AI

Complete guide to integrating your Memory AI API with LangChain agents and applications.

---

## 📋 Table of Contents

- [Quick Start (5 minutes)](#quick-start)
- [Comprehensive Testing](#comprehensive-testing)
- [Production Deployment](#production-deployment)
- [Real-World Examples](#real-world-examples)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

1. **Memory API running:**
   ```bash
   docker-compose up -d
   curl http://localhost:8000/health
   ```

2. **OpenAI API Key:**
   ```bash
   export OPENAI_API_KEY='sk-...'
   ```

3. **LangChain installed:**
   ```bash
   pip install langchain langchain-openai langchain-core tiktoken
   ```

### Run Your First Agent (5 Minutes)

```bash
# Quick test
python3.11 langchain_quickstart.py
```

This will:
1. ✅ Create a Memory AI account
2. ✅ Set up a collection
3. ✅ Create a LangChain agent with memory tools
4. ✅ Run a test conversation
5. ✅ Demonstrate memory persistence

**Expected Output:**
```
🚀 LangChain + Memory AI Quick Start

1️⃣ Setting up Memory API...
   ✓ Memory API ready

2️⃣ Creating LangChain agent...
   ✓ Agent created

3️⃣ Testing conversation with memory...

👤 User: My name is Bob and I love Python
🤖 Agent: Nice to meet you, Bob! I've made a note that you love Python...

👤 User: What's my name?
🤖 Agent: Your name is Bob!

✅ Success! Your agent has persistent memory.
```

---

## 🧪 Comprehensive Testing

Run the full test suite to verify all features:

```bash
# Set your OpenAI key
export OPENAI_API_KEY='sk-...'

# Run comprehensive tests
python3.11 test_langchain_integration.py
```

### What It Tests

The test suite validates:

1. **Authentication & Setup**
   - User registration/login
   - Collection creation
   - Token management

2. **Memory Storage**
   - Storing user information
   - Storing preferences
   - Metadata handling

3. **Memory Recall**
   - Retrieving specific facts
   - Searching conversations
   - Relevance scoring

4. **Complex Reasoning**
   - Multi-fact synthesis
   - Contextual understanding
   - Inference from memories

5. **Conversation Continuity**
   - Follow-up questions
   - Context maintenance
   - History tracking

6. **Memory Persistence**
   - Direct API verification
   - Long-term storage
   - Data integrity

### Sample Test Output

```
================================================================================
  LANGCHAIN + MEMORY AI INTEGRATION TEST SUITE
================================================================================

📋 Checking prerequisites...
✓ OpenAI API key found
✓ Memory API is running

================================================================================
  STEP 1: Authentication & Setup
================================================================================

🔐 Registering/logging in user: langchain_test@example.com
✓ Authenticated! Token: eyJhbGciOiJIUzI1NiIs...

📚 Creating memory collection...
✓ Collection created: a1b2c3d4-e5f6-7890-abcd-ef1234567890

================================================================================
  STEP 2: Initialize LangChain Agent
================================================================================

🤖 Creating LangChain agent with memory tools...
✓ Agent created successfully!
   Model: gpt-4o-mini
   Tools: search_memory, add_memory

[... detailed test results ...]

================================================================================
  TEST SUMMARY
================================================================================

✅ All tests completed!

📊 Statistics:
   - Collection ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
   - Conversations: 7
   - Memories stored: 12
```

---

## 🏗️ Production Deployment

### Interactive Chatbot

Run a production-ready chatbot with memory:

```bash
# Set environment variables
export OPENAI_API_KEY='sk-...'
export MEMORY_API_EMAIL='your@email.com'
export MEMORY_API_PASSWORD='your-password'

# Run chatbot
python3.11 langchain_chatbot_example.py
```

### Features

- ✅ **Error handling** - Graceful failure recovery
- ✅ **Logging** - Comprehensive logging for debugging
- ✅ **Session management** - Efficient conversation handling
- ✅ **Memory optimization** - Automatic history pruning
- ✅ **Interactive commands** - Built-in utilities

### Interactive Commands

```bash
# In the chatbot:
quit/exit  - Exit the chatbot
clear      - Clear chat history
stats      - Show memory statistics
```

### Sample Session

```
============================================================
  Memory ChatBot - Interactive Mode
============================================================

Type 'quit' or 'exit' to stop

👤 You: My name is Alice and I'm a data scientist

> Entering new AgentExecutor chain...
[Agent uses add_memory tool to store information]

🤖 Bot: Nice to meet you, Alice! I've stored that you're a
data scientist. How can I help you today?

👤 You: What do you know about me?

> Entering new AgentExecutor chain...
[Agent uses search_memory tool to recall]

🤖 Bot: Based on my memory, you're Alice and you work as a
data scientist. Is there anything else you'd like me to know?

👤 You: stats

📊 Statistics:
   Chat history: 2 exchanges
   Collection ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

## 🌟 Real-World Examples

### 1. Customer Support Bot

```python
from langchain_chatbot_example import MemoryChatBot, Config

# Configure for customer support
config = Config()
config.collection_name = "Customer Support Memory"

bot = MemoryChatBot(config)
bot.initialize()

# Bot remembers customer issues across sessions
response = bot.chat("I'm having trouble with my order #12345")
# Bot will search memory for previous issues with this order
```

### 2. Personal Assistant

```python
config = Config()
config.collection_name = "Personal Assistant"

bot = MemoryChatBot(config)
bot.initialize()

# Learns and remembers your preferences
bot.chat("I prefer morning meetings before 11am")
bot.chat("I'm allergic to peanuts")
bot.chat("My favorite restaurant is Sushi Place")

# Later...
response = bot.chat("Schedule a lunch meeting")
# Bot knows your preferences and allergies
```

### 3. Research Assistant

```python
config = Config()
config.collection_name = "Research Notes"

bot = MemoryChatBot(config)
bot.initialize()

# Accumulates research over time
bot.chat("I found that transformers use attention mechanisms")
bot.chat("BERT was introduced in 2018")
bot.chat("GPT-3 has 175B parameters")

# Query accumulated knowledge
response = bot.chat("What do you know about transformer models?")
# Bot synthesizes information from all stored memories
```

---

## 🔧 Advanced Features

### Custom Memory Importance

Control what gets remembered:

```python
from langchain_chatbot_example import MemoryAddTool

# Critical information (0.9-1.0)
tool.memory_client.add_memory(
    "User's email: alice@example.com",
    importance=0.95
)

# Important facts (0.7-0.8)
tool.memory_client.add_memory(
    "User prefers Python over JavaScript",
    importance=0.75
)

# General information (0.5-0.6)
tool.memory_client.add_memory(
    "User mentioned liking coffee",
    importance=0.55
)
```

### Metadata Filtering

Store and search with context:

```python
# Store with metadata
memory_client.add_memory(
    content="Discussed pricing for enterprise plan",
    importance=0.8,
    metadata={
        "category": "sales",
        "customer_id": "cust_123",
        "topic": "pricing",
        "date": "2025-01-15"
    }
)

# Search with filters
results = memory_client.search_memories(
    query="enterprise pricing",
    filters={"category": "sales", "customer_id": "cust_123"}
)
```

### Multi-User Support

Separate memories per user:

```python
class MultiUserChatBot:
    def __init__(self, config: Config):
        self.config = config
        self.user_bots = {}

    def get_bot(self, user_id: str) -> MemoryChatBot:
        if user_id not in self.user_bots:
            config = Config()
            config.collection_name = f"User_{user_id}_Memory"
            bot = MemoryChatBot(config)
            bot.initialize()
            self.user_bots[user_id] = bot
        return self.user_bots[user_id]

    def chat(self, user_id: str, message: str) -> str:
        bot = self.get_bot(user_id)
        return bot.chat(message)
```

### Custom LLM Models

Use different models for different use cases:

```python
# Use GPT-4 for complex reasoning
config = Config()
config.openai_model = "gpt-4"  # More expensive, better quality

# Use GPT-4o-mini for fast responses
config.openai_model = "gpt-4o-mini"  # Cheaper, faster

# Use GPT-4-turbo for balance
config.openai_model = "gpt-4-turbo"  # Good balance
```

### Streaming Responses

For real-time user experience:

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    api_key=config.openai_api_key,
    model=config.openai_model,
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

---

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not set"

**Solution:**
```bash
export OPENAI_API_KEY='sk-your-key-here'

# Or pass directly
OPENAI_API_KEY='sk-...' python3.11 langchain_quickstart.py
```

### Issue: "Memory API not accessible"

**Solution:**
```bash
# Check if API is running
curl http://localhost:8000/health

# If not running, start it
docker-compose up -d

# Check logs
docker-compose logs api
```

### Issue: "Authentication failed"

**Solution:**
```bash
# Update credentials in script or environment variables
export MEMORY_API_EMAIL='your@email.com'
export MEMORY_API_PASSWORD='your-password'

# Or create a new account via API
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "new@example.com",
    "password": "NewPassword123!",
    "full_name": "New User"
  }'
```

### Issue: "No module named 'langchain'"

**Solution:**
```bash
# Install all dependencies
pip install langchain langchain-openai langchain-core tiktoken

# Or use requirements.txt
pip install -r backend/requirements.txt
```

### Issue: "Agent loops without finishing"

**Solution:**
```python
# Reduce max_iterations
config = Config()
config.max_iterations = 3  # Default is 5

# Or make tools return more specific information
```

### Issue: "Memory search returns no results"

**Possible causes:**
1. Memories not indexed yet (wait a few seconds)
2. Query doesn't match stored content
3. Collection ID mismatch

**Solution:**
```python
# Check collection ID
print(f"Using collection: {memory_client.collection_id}")

# List all memories
import requests
response = requests.post(
    f"{MEMORY_API_URL}/search",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "collection_id": collection_id,
        "query": "",  # Empty query returns all
        "limit": 100
    }
)
print(f"Total memories: {len(response.json()['results'])}")
```

---

## 📊 Performance Tips

### 1. Connection Pooling

```python
import requests

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20
)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, limit: int = 5):
    return memory_client.search_memories(query, limit)
```

### 3. Async Operations

```python
import asyncio
import aiohttp

async def async_search(query: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{MEMORY_API_URL}/search",
            headers=headers,
            json={"query": query}
        ) as response:
            return await response.json()
```

---

## 🚀 Next Steps

### 1. Deploy to Production

- Set up proper environment variables
- Add monitoring and logging
- Implement rate limiting
- Set up error alerting

### 2. Extend Functionality

- Add more custom tools
- Implement user authentication
- Create web UI with Streamlit or Gradio
- Add voice interface

### 3. Optimize Performance

- Enable caching
- Use connection pooling
- Implement async operations
- Monitor memory usage

### 4. Scale Up

- Deploy with Docker
- Use Kubernetes for orchestration
- Add load balancing
- Implement Redis for session management

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **LangChain Docs**: https://python.langchain.com/docs/
- **OpenAI API Docs**: https://platform.openai.com/docs/
- **Example Integrations**: `examples/` directory

---

## 🤝 Support

Having issues? Here's how to get help:

1. **Check logs**:
   ```bash
   docker-compose logs api
   ```

2. **Run diagnostics**:
   ```bash
   python3.11 test_langchain_integration.py
   ```

3. **Review examples**:
   ```bash
   ls -l examples/
   ```

4. **Test API directly**:
   ```bash
   curl http://localhost:8000/health
   ```

---

**Happy integrating! 🎉**
