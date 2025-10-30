# Memory AI Python SDK

[![PyPI version](https://img.shields.io/pypi/v/memory-ai.svg)](https://pypi.org/project/memory-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/memory-ai.svg)](https://pypi.org/project/memory-ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official Python SDK for the AI Memory API - **the only memory API with Reinforcement Learning**.

## 🚀 Features

- ✅ **Complete API Coverage** - All 50+ endpoints supported
- ✅ **Reinforcement Learning** - Train AI agents to optimize memory operations
- ✅ **Temporal Knowledge Graphs** - Track facts over time with bi-temporal model
- ✅ **Procedural Memory** - Store and execute learned skills
- ✅ **Working Memory** - Short-term context buffer for conversations
- ✅ **Memory Consolidation** - Automatic compression of episodic memories
- ✅ **Async/Await** - Full async support with httpx
- ✅ **Type Hints** - Complete type annotations
- ✅ **Clean API** - Pythonic, intuitive interface

## 📦 Installation

```bash
pip install memory-ai
```

## 🔥 Quick Start

```python
import asyncio
from memory_ai import MemoryClient

async def main():
    # Initialize client
    client = MemoryClient(api_key="mem_sk_your_api_key")

    # Create a collection
    collection = await client.collections.create(
        name="My AI Agent",
        description="Personal memory for my chatbot"
    )

    # Store a memory
    memory = await client.memories.create(
        collection_id=collection["id"],
        content="User prefers dark mode and loves Python",
        importance=0.9
    )

    # Search memories
    results = await client.search(
        query="What programming language does user like?",
        collection_id=collection["id"],
        limit=5
    )

    print(results)

    # Close client
    await client.close()

asyncio.run(main())
```

## 📚 Complete Documentation

Visit https://docs.memory-ai.com for full documentation.

## 🔗 Links

- **Documentation**: https://docs.memory-ai.com
- **API Reference**: https://api.memory-ai.com/docs
- **GitHub**: https://github.com/memory-ai/memory-ai
- **Examples**: https://github.com/memory-ai/examples

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

**Built with ❤️ by the Memory AI team**
