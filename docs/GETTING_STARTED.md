# Getting Started with AI Memory API

This guide will help you get started with the AI Memory API in minutes.

## Prerequisites

- Python 3.8+ or Node.js 16+ (for SDK usage)
- An API key (sign up at https://memory-ai.com)

## Quick Start

### 1. Sign Up and Get API Key

```bash
curl -X POST https://api.memory-ai.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "you@example.com",
    "password": "secure_password",
    "full_name": "Your Name"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGc...",
  "user_id": "usr_abc123"
}
```

### 2. Create an API Key

```bash
curl -X POST https://api.memory-ai.com/v1/auth/api-keys \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My First Key"}'
```

**Response:**
```json
{
  "key": "mem_sk_1234567890abcdef",
  "prefix": "mem_sk_1234567"
}
```

⚠️ **Save this API key** - it's only shown once!

### 3. Create Your First Collection

```bash
curl -X POST https://api.memory-ai.com/v1/collections \
  -H "Authorization: Bearer mem_sk_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Knowledge Base",
    "description": "Personal notes and learnings"
  }'
```

**Response:**
```json
{
  "id": "col_abc123",
  "name": "My Knowledge Base",
  "memory_count": 0
}
```

### 4. Add Memories

```bash
curl -X POST https://api.memory-ai.com/v1/memories \
  -H "Authorization: Bearer mem_sk_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_id": "col_abc123",
    "content": "Python is a versatile programming language widely used in AI and machine learning.",
    "importance": 0.8
  }'
```

### 5. Search Your Memories

```bash
curl -X POST https://api.memory-ai.com/v1/search \
  -H "Authorization: Bearer mem_sk_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programming language for AI",
    "limit": 5
  }'
```

### 6. Reason Over Your Memories

```bash
curl -X POST https://api.memory-ai.com/v1/search/reason \
  -H "Authorization: Bearer mem_sk_1234567890abcdef" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What programming languages did I learn about?",
    "provider": "gemini"
  }'
```

---

## Using Python SDK

### Installation

```bash
pip install memory-ai
```

### Example

```python
import asyncio
from memory_ai import MemoryClient

async def main():
    # Initialize client
    client = MemoryClient(api_key="mem_sk_1234567890abcdef")

    # Create collection
    collection = await client.collections.create(
        name="Learning Notes",
        description="Things I'm learning"
    )

    # Add memory
    memory = await client.memories.create(
        collection_id=collection["id"],
        content="Machine learning models learn patterns from data.",
        importance=0.9
    )

    # Search
    results = await client.search(
        query="machine learning",
        limit=5
    )

    for result in results:
        print(f"Score: {result['score']:.2f}")
        print(f"Content: {result['content']}\n")

    # Reason
    answer = await client.reason(
        query="What did I learn about machine learning?",
        provider="gemini"
    )

    print(f"Answer: {answer['answer']}")

    await client.close()

asyncio.run(main())
```

---

## Using TypeScript SDK

### Installation

```bash
npm install @memory-ai/sdk
```

### Example

```typescript
import { MemoryClient } from '@memory-ai/sdk';

const client = new MemoryClient({
  apiKey: 'mem_sk_1234567890abcdef'
});

// Create collection
const collection = await client.collections.create({
  name: 'Learning Notes',
  description: 'Things I\'m learning'
});

// Add memory
const memory = await client.memories.create({
  collection_id: collection.id,
  content: 'Machine learning models learn patterns from data.',
  importance: 0.9
});

// Search
const results = await client.search({
  query: 'machine learning',
  limit: 5
});

results.forEach(result => {
  console.log(`Score: ${result.score.toFixed(2)}`);
  console.log(`Content: ${result.content}\n`);
});

// Reason
const answer = await client.reason({
  query: 'What did I learn about machine learning?',
  provider: 'gemini'
});

console.log(`Answer: ${answer.answer}`);
```

---

## Core Concepts

### Collections

Collections are logical groupings of memories. Use them to:
- Organize memories by topic or project
- Isolate different contexts
- Control access and permissions

**Best Practice**: Create separate collections for different domains (e.g., "Work Notes", "Personal Learning", "Research Papers").

### Memories

Memories are pieces of information stored in the system. Each memory:
- Has text content
- Gets automatically embedded as a vector
- Can have importance score (0-1)
- Can include custom metadata
- Is indexed for fast retrieval

**Best Practice**: Keep memories focused and atomic. Break long documents into smaller chunks.

### Importance Score

The importance score (0-1) affects:
- Search ranking
- Memory decay rate
- Long-term retention

**Guidelines**:
- `0.9-1.0`: Critical information
- `0.7-0.9`: Important information
- `0.5-0.7`: Standard information
- `0.0-0.5`: Low priority information

### Search Types

#### Hybrid Search (Recommended)

Combines three retrieval methods:
1. **Vector similarity**: Semantic understanding
2. **BM25 keyword search**: Exact matches
3. **Knowledge graph**: Entity relationships

```python
results = await client.search(
    query="neural networks",
    search_type="hybrid"  # default
)
```

#### Vector Search

Pure semantic search using embeddings:

```python
results = await client.search(
    query="explain transformers",
    search_type="vector"
)
```

#### BM25 Search

Traditional keyword search:

```python
results = await client.search(
    query="specific technical term",
    search_type="bm25"
)
```

### LLM Reasoning

Get AI-powered answers from your memories:

```python
answer = await client.reason(
    query="Summarize what I learned this week",
    provider="gemini",  # or "openai", "anthropic"
    include_steps=True  # Show reasoning process
)
```

**Provider Comparison**:
- **Gemini**: Fast, supports thinking mode, cost-effective
- **OpenAI**: High quality, well-tested
- **Claude**: Strong reasoning, good for complex queries

---

## Common Use Cases

### Personal Knowledge Base

```python
# Add notes from books
await client.memories.create(
    collection_id=reading_collection,
    content="The Feynman Technique: Explain concepts in simple terms to identify gaps in understanding.",
    source_type="book",
    source_reference="Learning How to Learn",
    importance=0.9
)

# Search your notes
results = await client.search(query="learning techniques")

# Get insights
answer = await client.reason(query="What are the best study methods I've learned?")
```

### Meeting Notes

```python
# Store meeting notes
await client.memories.create(
    collection_id=meetings_collection,
    content="Q4 Planning: Launch new features in January, focus on user retention.",
    metadata={
        "date": "2024-01-15",
        "attendees": ["Alice", "Bob", "Carol"],
        "type": "planning"
    },
    importance=0.8
)

# Find past decisions
results = await client.search(query="Q4 launch plans")
```

### Research Papers

```python
# Add paper summary
await client.memories.create(
    collection_id=research_collection,
    content="Attention Is All You Need introduced the Transformer architecture, using self-attention mechanisms instead of recurrence.",
    source_type="paper",
    source_reference="Vaswani et al., 2017",
    metadata={
        "year": 2017,
        "venue": "NIPS",
        "citations": 50000
    },
    importance=1.0
)

# Find related papers
similar = await client.search_resource.similar(
    memory_id=memory["id"],
    limit=5
)
```

### Code Snippets

```python
# Store useful code
await client.memories.create(
    collection_id=code_collection,
    content="""
    # Python async context manager pattern
    async with MemoryClient(api_key) as client:
        results = await client.search(query)
    """,
    metadata={
        "language": "python",
        "topic": "async",
        "tags": ["patterns", "context-manager"]
    },
    importance=0.7
)
```

---

## Best Practices

### Memory Organization

1. **Use descriptive collection names**: "Python Learnings" not "Notes1"
2. **Add metadata**: Include source, dates, tags
3. **Set appropriate importance**: Don't mark everything as important
4. **Regular cleanup**: Delete outdated memories

### Search Optimization

1. **Start with hybrid search**: Best for most cases
2. **Use filters**: Narrow down by importance, date, metadata
3. **Adjust limit**: More results = better context but slower
4. **Iterate queries**: Refine based on initial results

### Performance

1. **Batch operations**: Create multiple memories in one request (coming soon)
2. **Cache results**: Store frequently accessed memories locally
3. **Use webhooks**: Get notified of changes instead of polling
4. **Set timeouts**: Handle slow responses gracefully

### Security

1. **Rotate API keys**: Generate new keys periodically
2. **Use environment variables**: Never hardcode keys
3. **Limit key permissions**: Create separate keys for different apps
4. **Monitor usage**: Check for unusual activity

---

## Troubleshooting

### Authentication Errors

```
Error: 401 Unauthorized
```

**Solution**: Check that your API key is correct and active:

```bash
curl -H "Authorization: Bearer mem_sk_..." \
  https://api.memory-ai.com/v1/auth/me
```

### Rate Limiting

```
Error: 429 Too Many Requests
```

**Solution**: Implement exponential backoff:

```python
import time
from memory_ai import RateLimitError

max_retries = 3
for attempt in range(max_retries):
    try:
        results = await client.search(query="test")
        break
    except RateLimitError:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
        else:
            raise
```

### Empty Search Results

**Possible causes**:
1. No memories match the query
2. Memories are in a different collection
3. Filters are too restrictive

**Solution**: Try broader search:

```python
# Remove collection filter
results = await client.search(
    query="your query",
    collection_id=None  # Search all collections
)

# Use different search type
results = await client.search(
    query="your query",
    search_type="vector"  # Try semantic search
)
```

---

## Next Steps

1. Read the [complete API reference](API_REFERENCE.md)
2. Explore [example applications](../examples/)
3. Join our [Discord community](https://discord.gg/memory-ai)
4. Check out [advanced guides](ADVANCED.md)

## Support

- **Email**: support@memory-ai.com
- **Discord**: https://discord.gg/memory-ai
- **GitHub Issues**: https://github.com/memory-ai/memory-ai-api/issues

---

Happy building! 🚀
