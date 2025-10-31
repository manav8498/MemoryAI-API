# 🚀 LangChain + Memory AI - Quick Reference

One-page cheat sheet for testing and using the integration.

---

## ⚡ Quick Start (Copy & Paste)

```bash
# 1. Start Memory API
docker-compose up -d

# 2. Set OpenAI key
export OPENAI_API_KEY='sk-your-key-here'

# 3. Run quick test
python3.11 langchain_quickstart.py
```

---

## 📁 Files You Need

| File | Purpose | When to Use |
|------|---------|-------------|
| `langchain_quickstart.py` | 5-min demo | First time testing |
| `test_langchain_integration.py` | Full test suite | Validate everything |
| `langchain_chatbot_example.py` | Production bot | Real-world testing |

---

## 🧪 Testing Commands

```bash
# Quick test (2 minutes)
python3.11 langchain_quickstart.py

# Full test (5 minutes)
python3.11 test_langchain_integration.py

# Interactive chatbot
export MEMORY_API_EMAIL='your@email.com'
export MEMORY_API_PASSWORD='your-password'
python3.11 langchain_chatbot_example.py
```

---

## 🔧 Required Environment Variables

```bash
# Minimum required
export OPENAI_API_KEY='sk-...'

# For production chatbot
export MEMORY_API_EMAIL='your@email.com'
export MEMORY_API_PASSWORD='your-password'

# Optional
export MEMORY_API_URL='http://localhost:8000/v1'
export OPENAI_MODEL='gpt-4o-mini'
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "OPENAI_API_KEY not set" | `export OPENAI_API_KEY='sk-...'` |
| "API not accessible" | `docker-compose up -d` |
| "Module not found" | `pip install langchain langchain-openai` |
| "No memories found" | Wait 2-3 seconds after storing |

---

## 📊 Check API Status

```bash
# Health check
curl http://localhost:8000/health

# View services
docker-compose ps

# View API logs
docker-compose logs api
```

---

## 💻 Python Code Snippets

### Minimal Example

```python
from langchain_openai import ChatOpenAI
from memory_client import MemoryClient

# Setup
memory = MemoryClient("http://localhost:8000/v1")
memory.login("user@example.com", "password")
memory.setup_collection("My Memories")

# Store
memory.add_memory("User loves Python", importance=0.8)

# Search
results = memory.search("what does user love?")
print(results[0]['content'])  # "User loves Python"
```

### With LangChain Agent

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

# Create tools
tools = [
    MemorySearchTool(memory_client=memory),
    MemoryAddTool(memory_client=memory)
]

# Create agent
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Chat
response = executor.invoke({"input": "My name is Alice"})
print(response['output'])
```

---

## 🔑 Key Concepts

### Memory Importance Levels

| Level | Score | Use For |
|-------|-------|---------|
| Critical | 0.9-1.0 | User email, core preferences |
| Important | 0.7-0.8 | User facts, key decisions |
| General | 0.5-0.6 | Conversation context |
| Low | 0.3-0.4 | Temporary info |

### Memory Search Scores

| Score | Meaning |
|-------|---------|
| > 0.9 | Exact match |
| 0.7-0.9 | Highly relevant |
| 0.5-0.7 | Somewhat relevant |
| < 0.5 | Weak match |

---

## 🎯 Common Use Cases

### 1. Customer Support Bot
```python
# Store customer info
memory.add_memory("Customer ID: 12345, Issue: login problem", 0.9)

# Recall later
results = memory.search("customer 12345 issues")
```

### 2. Personal Assistant
```python
# Learn preferences
memory.add_memory("User prefers dark mode", 0.8)
memory.add_memory("User is allergic to peanuts", 0.95)

# Apply preferences
results = memory.search("user preferences")
```

### 3. Research Assistant
```python
# Accumulate knowledge
memory.add_memory("Transformers use attention mechanism", 0.7)
memory.add_memory("BERT introduced in 2018", 0.6)

# Query knowledge base
results = memory.search("transformer architecture")
```

---

## 📈 Performance Tips

```python
# Use faster model
model="gpt-4o-mini"  # vs "gpt-4"

# Limit search results
limit=5  # Don't request more than needed

# Set max iterations
max_iterations=3  # Prevent infinite loops

# Use connection pooling
session = requests.Session()
```

---

## 🔍 Debugging Commands

```bash
# Test API directly
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'

# Check memories in collection
curl -X POST http://localhost:8000/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"collection_id":"your-id","query":"","limit":100}'

# View collections
curl http://localhost:8000/v1/collections \
  -H "Authorization: Bearer $TOKEN"
```

---

## 📚 Documentation Links

| Resource | Location |
|----------|----------|
| Full Guide | `LANGCHAIN_INTEGRATION_GUIDE.md` |
| Architecture | `LANGCHAIN_ARCHITECTURE.md` |
| Test Summary | `LANGCHAIN_TESTING_SUMMARY.md` |
| API Docs | http://localhost:8000/docs |

---

## ✅ Testing Checklist

Quick validation before deployment:

- [ ] API health check passes
- [ ] Quick test runs successfully
- [ ] Comprehensive tests pass
- [ ] Interactive chatbot works
- [ ] Memories persist correctly
- [ ] Error handling works
- [ ] Performance is acceptable

---

## 🚀 Deployment Checklist

Before going to production:

- [ ] Environment variables set
- [ ] Error logging configured
- [ ] Monitoring enabled
- [ ] Rate limiting implemented
- [ ] Backup strategy defined
- [ ] Security review complete
- [ ] Load testing done

---

## 💡 Best Practices

### Memory Storage
✅ Store important facts only
✅ Use descriptive content
✅ Add metadata
✅ Set appropriate importance

❌ Don't store every message
❌ Don't use vague content
❌ Don't create duplicates

### Memory Search
✅ Use specific queries
✅ Check relevance scores
✅ Handle empty results
✅ Limit result count

❌ Don't use broad queries
❌ Don't ignore low scores
❌ Don't assume results exist

### Agent Configuration
✅ Set max_iterations
✅ Enable error handling
✅ Use appropriate model
✅ Log agent decisions

❌ Don't allow infinite loops
❌ Don't ignore errors
❌ Don't over-spend on tokens

---

## 🎓 Learning Path

1. **Day 1**: Run `langchain_quickstart.py` ✓
2. **Day 2**: Run `test_langchain_integration.py` ✓
3. **Day 3**: Try `langchain_chatbot_example.py` ✓
4. **Day 4**: Read `LANGCHAIN_INTEGRATION_GUIDE.md` ✓
5. **Day 5**: Build your own use case ✓

---

## 🆘 Get Help

1. Check logs: `docker-compose logs api`
2. Test API: `curl http://localhost:8000/health`
3. Review docs: http://localhost:8000/docs
4. Read guide: `LANGCHAIN_INTEGRATION_GUIDE.md`

---

## 📞 Support Resources

- API Health: http://localhost:8000/health
- API Docs: http://localhost:8000/docs
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

**Keep this page handy for quick reference! 📌**
