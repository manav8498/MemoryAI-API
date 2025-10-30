# AI Memory API

> Universal Memory API with Reinforcement Learning & Advanced Reasoning

A production-ready API that provides persistent, context-aware memory for AI agents with built-in continuous learning through reinforcement learning and neuro-symbolic reasoning.

## 🚀 Features

- **Hybrid Memory System**: Vector search (Milvus) + Knowledge Graph (Neo4j)
- **Autonomous Learning**: Continuous improvement via Reinforcement Learning
- **Neuro-Symbolic Reasoning**: Neural flexibility + Symbolic reliability
- **Universal Integration**: Works with LangChain, LlamaIndex, AutoGen, CrewAI
- **Enterprise-Ready**: SOC 2 compliant, 99.9% uptime, scalable to millions of memories

## 📁 Project Structure

```
API Memory/
├── backend/                 # FastAPI backend
│   ├── api/                # API routes
│   ├── core/               # Core configuration
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   ├── schemas/            # Pydantic schemas
│   └── main.py            # Application entry point
├── sdks/                   # Client SDKs
│   ├── python/            # Python SDK
│   └── typescript/        # TypeScript SDK
├── ml/                     # Machine Learning
│   ├── embeddings/        # Embedding models
│   ├── reasoning/         # Reasoning engine
│   └── rl/               # RL training pipeline
├── infrastructure/         # Deployment configs
│   ├── docker/           # Docker files
│   ├── kubernetes/       # K8s manifests
│   └── terraform/        # Infrastructure as Code
├── docs/                  # Documentation
└── tests/                 # Test suites
```

## 🛠️ Tech Stack

- **API Framework**: FastAPI (Python 3.11+)
- **Vector Database**: Milvus
- **Knowledge Graph**: Neo4j
- **Relational DB**: PostgreSQL
- **Cache**: Redis
- **Message Queue**: Apache Kafka
- **Orchestration**: Kubernetes
- **Observability**: Prometheus + Grafana + Jaeger

## 🚦 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for TypeScript SDK)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/api-memory.git
cd api-memory

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start infrastructure with Docker Compose
docker-compose up -d

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Usage

### Python

```python
from memory_ai import MemoryClient

client = MemoryClient(api_key="mem_sk_live_xxx")

# Add memory
memory = client.memories.create(
    content="User prefers dark mode",
    metadata={"user_id": "user_123"}
)

# Search memories
results = client.memories.search(
    query="What are user preferences?",
    limit=5
)

# Ask with reasoning
response = client.reason(
    query="What theme should I use?",
    user_id="user_123"
)
print(response.answer)
```

### TypeScript

```typescript
import { MemoryAI } from '@memory-ai/sdk';

const client = new MemoryAI({ apiKey: 'mem_sk_live_xxx' });

const memory = await client.memories.create({
  content: 'User prefers dark mode',
  metadata: { userId: 'user_123' }
});

const results = await client.memories.search({
  query: 'What are user preferences?',
  limit: 5
});
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run specific test suite
pytest tests/test_memory_api.py -v
```

## 📦 Deployment

### Docker

```bash
# Build image
docker build -t memory-ai-api:latest -f infrastructure/docker/Dockerfile .

# Run container
docker run -p 8000:8000 --env-file .env memory-ai-api:latest
```

### Kubernetes

```bash
# Apply configs
kubectl apply -f infrastructure/kubernetes/

# Check status
kubectl get pods -n memory-ai
```

## 📊 Monitoring

- **Metrics**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3000 (Grafana)
- **Tracing**: http://localhost:16686 (Jaeger)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [Documentation](https://docs.memory-ai.dev)
- [Discord Community](https://discord.gg/memory-ai)
- [Blog](https://memory-ai.dev/blog)
- [Twitter](https://twitter.com/memoryai_dev)

## 💬 Support

- **Email**: support@memory-ai.dev
- **Discord**: [Join our community](https://discord.gg/memory-ai)
- **GitHub Issues**: [Report bugs](https://github.com/yourusername/api-memory/issues)

---

**Built with ❤️ by the Memory AI Team**
