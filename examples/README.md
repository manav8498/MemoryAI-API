# 🤖 AI Agent Integration Examples

This directory contains complete, working examples of integrating the AI Memory API with various AI agent frameworks.

---

## 📁 Files Overview

| File | Description | Framework | Difficulty |
|------|-------------|-----------|------------|
| `integration_simple_agent.py` | Basic agent with memory | None (Pure Python) | ⭐ Beginner |
| `integration_langchain.py` | LangChain agent with memory tools | LangChain | ⭐⭐ Intermediate |
| `integration_autogen.py` | Multi-agent system with shared memory | AutoGen | ⭐⭐⭐ Advanced |
| `INTEGRATION_GUIDE.md` | Complete integration guide | All | 📚 Reference |

---

## 🚀 Quick Start

### 1. Prerequisites

Make sure the Memory API is running:

```bash
# Start the API
cd ..
docker-compose up -d

# Verify it's running
curl http://localhost:8000/health
```

### 2. Create an Account

```bash
# Register a new user
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "SecurePassword123!",
    "full_name": "Test User"
  }'
```

### 3. Update Configuration

Edit the example files and update these variables:

```python
EMAIL = "test@example.com"  # Your email
PASSWORD = "SecurePassword123!"  # Your password
OPENAI_API_KEY = "sk-..."  # Your OpenAI key (for LangChain/AutoGen examples)
```

---

## 📚 Example 1: Simple Agent

**File**: `integration_simple_agent.py`

**What it demonstrates**:
- Basic REST API integration
- Memory search and storage
- Simple conversational agent
- LLM reasoning over memories

**Run it**:
```bash
# Install dependencies
pip install requests

# Update EMAIL and PASSWORD in the file
nano integration_simple_agent.py

# Run
python integration_simple_agent.py
```

**Expected output**:
```
🤖 SIMPLE AI AGENT WITH MEMORY INTEGRATION
==========================================

1️⃣ Authenticating...
✓ Logged in successfully!

2️⃣ Initializing Memory API...
✓ Collection created: abc-123-def

3️⃣ Creating AI Agent...
✓ Agent ready!

4️⃣ Teaching agent some facts...
📚 Learning: The user's name is Alice
✓ Fact stored in memory!
...

5️⃣ Starting conversation...
==========================================

👤 User: What's my name?
🧠 Searching memories for: 'What's my name?'...
💭 Reasoning with 1 relevant memories...
💾 Storing interaction as memory...
🤖 MemoryBot: Your name is Alice!
```

**Key concepts**:
- `MemoryAPI` class wraps REST API calls
- `SimpleAIAgent` searches memory before responding
- Interactions are stored for future reference

---

## 📚 Example 2: LangChain Integration

**File**: `integration_langchain.py`

**What it demonstrates**:
- LangChain tools integration
- Function calling with memory
- Agent decides when to use memory
- Chat history management

**Run it**:
```bash
# Install dependencies
pip install langchain langchain-openai requests

# Update EMAIL, PASSWORD, and OPENAI_API_KEY
nano integration_langchain.py

# Run
python integration_langchain.py
```

**Expected output**:
```
🦜 LANGCHAIN AGENT WITH MEMORY API
==================================

1️⃣ Authenticating with Memory API...
✓ Authenticated!

2️⃣ Creating memory collection...
✓ Collection created: xyz-789

3️⃣ Initializing Memory API wrapper...
✓ Memory API ready!

4️⃣ Creating LangChain agent...
✓ Agent created!

5️⃣ Starting conversation...
==================================

👤 User: My name is Bob and I love Python programming

> Entering new AgentExecutor chain...
Thought: I should store this information in memory
Action: add_memory
Action Input: {"content": "User's name is Bob", "importance": 0.9}
Observation: Memory stored successfully: User's name is Bob
...
```

**Key concepts**:
- `MemorySearchTool` and `MemoryAddTool` as LangChain tools
- Agent autonomously decides when to search/store
- Pydantic models for tool inputs
- Async support

---

## 📚 Example 3: AutoGen Multi-Agent

**File**: `integration_autogen.py`

**What it demonstrates**:
- Multi-agent collaboration
- Shared memory across agents
- Specialized agent roles
- Group chat coordination

**Run it**:
```bash
# Install dependencies
pip install pyautogen requests

# Update EMAIL, PASSWORD, and OPENAI_API_KEY
nano integration_autogen.py

# Run
python integration_autogen.py
```

**Expected output**:
```
🤖 AUTOGEN AGENTS WITH MEMORY API
==================================

1️⃣ Setting up Memory API...
✓ Memory API ready!

2️⃣ Pre-populating knowledge base...
✓ Knowledge base populated!

3️⃣ Single Agent Example...
----------------------------------------------------------------------

👤 User: What do you know about Python?

MemoryBot (to User):
Let me search my memory...
[Searching memory for: Python]
Found: Python is a high-level programming language known for readability
...

4️⃣ Multi-Agent System Example...
----------------------------------------------------------------------

Researcher (to Manager):
I found information about machine learning in our memory...

Writer (to Manager):
Based on the research, here's a clear explanation...
```

**Key concepts**:
- `MemoryEnhancedAgent` wraps AutoGen agents
- Multiple agents share the same memory
- Function registration for memory operations
- Group chat with memory context

---

## 🎯 Which Example Should I Use?

### Use **Simple Agent** if:
- ✅ You're new to AI agents
- ✅ You want to understand the basics
- ✅ You're building a custom agent
- ✅ You don't need a framework

### Use **LangChain** if:
- ✅ You're already using LangChain
- ✅ You want tool-based integration
- ✅ You need function calling
- ✅ You want agent autonomy

### Use **AutoGen** if:
- ✅ You need multiple agents
- ✅ You want agent collaboration
- ✅ You need specialized roles
- ✅ You're building complex systems

---

## 🔧 Customization Guide

### Adding Custom Metadata

```python
# Store memory with custom metadata
memory_api.add_memory(
    content="User prefers dark mode",
    importance=0.8,
    metadata={
        "category": "preference",
        "user_id": "user_123",
        "timestamp": "2024-01-15"
    }
)

# Search with filters
results = memory_api.search(
    query="user preferences",
    filters={"category": "preference"}
)
```

### Using Different LLM Providers

```python
# Use Gemini (default, fast and cost-effective)
result = memory_api.reason_with_memories(
    query="What did I learn?",
    provider="gemini"
)

# Use OpenAI (high quality)
result = memory_api.reason_with_memories(
    query="What did I learn?",
    provider="openai"
)

# Use Claude (strong reasoning)
result = memory_api.reason_with_memories(
    query="What did I learn?",
    provider="anthropic"
)
```

### Implementing Memory Decay

```python
# Add memory with decay
memory_api.add_memory(
    content="Temporary information",
    importance=0.5,
    metadata={
        "decay_rate": 0.01,  # Decays over time
        "created_at": datetime.now()
    }
)
```

---

## 🐛 Troubleshooting

### Error: "Connection refused"

**Problem**: Memory API is not running

**Solution**:
```bash
cd ..
docker-compose up -d
curl http://localhost:8000/health
```

### Error: "Authentication failed"

**Problem**: Wrong credentials

**Solution**:
```bash
# Register a new account
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "new@example.com", "password": "NewPass123!", "full_name": "New User"}'

# Update EMAIL and PASSWORD in the example file
```

### Error: "Collection not found"

**Problem**: Collection was deleted or doesn't exist

**Solution**:
```python
# Create a new collection
collection_id = memory_api.create_collection(
    name="My Collection",
    description="Description"
)
```

### Error: "No module named 'langchain'"

**Problem**: Missing dependencies

**Solution**:
```bash
# Install required packages
pip install langchain langchain-openai pyautogen requests
```

---

## 📖 Additional Resources

- **API Reference**: `/docs/API_REFERENCE.md`
- **Integration Guide**: `INTEGRATION_GUIDE.md`
- **Quick Start**: `/QUICKSTART_GUIDE.md`
- **API Docs**: http://localhost:8000/docs

---

## 💡 Tips for Production

1. **Use environment variables** for credentials:
   ```python
   import os
   EMAIL = os.getenv("MEMORY_API_EMAIL")
   PASSWORD = os.getenv("MEMORY_API_PASSWORD")
   ```

2. **Implement error handling**:
   ```python
   try:
       results = memory_api.search(query)
   except requests.exceptions.RequestException as e:
       logger.error(f"Memory search failed: {e}")
       results = []
   ```

3. **Use connection pooling**:
   ```python
   session = requests.Session()
   session.headers.update({"Authorization": f"Bearer {token}"})
   ```

4. **Implement caching**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def search_memory_cached(query):
       return memory_api.search(query)
   ```

5. **Monitor performance**:
   ```python
   import time
   
   start = time.time()
   results = memory_api.search(query)
   duration = time.time() - start
   logger.info(f"Search took {duration:.2f}s")
   ```

---

## 🤝 Contributing

Have a new integration example? Submit a PR!

**Guidelines**:
- Include clear comments
- Add error handling
- Provide example output
- Update this README

---

## 📝 License

MIT License - See LICENSE file for details

---

Happy integrating! 🚀

