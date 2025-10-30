# AI Memory API - Complete API Reference

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URL](#base-url)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Authentication](#authentication-endpoints)
  - [Users](#user-endpoints)
  - [Collections](#collection-endpoints)
  - [Memories](#memory-endpoints)
  - [Search](#search-endpoints)

---

## Overview

The AI Memory API provides a powerful memory system for AI agents with:
- **Hybrid Search**: Vector similarity + BM25 + Knowledge graph
- **LLM Reasoning**: Multi-provider support (Gemini, OpenAI, Claude)
- **Entity Extraction**: Automatic NLP processing
- **Reinforcement Learning**: Trajectory logging for continuous improvement

**API Version**: v1
**Protocol**: REST over HTTPS
**Data Format**: JSON

---

## Authentication

All API requests require authentication using one of two methods:

### API Key (Recommended)

Include your API key in the `Authorization` header:

```bash
curl -H "Authorization: Bearer mem_sk_your_api_key" \
  https://api.memory-ai.com/v1/memories
```

### JWT Token

Obtain a JWT token via `/v1/auth/login` and use it in the Authorization header:

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  https://api.memory-ai.com/v1/memories
```

---

## Base URL

**Production**: `https://api.memory-ai.com`
**Development**: `http://localhost:8000`

All endpoints are prefixed with `/v1`.

---

## Rate Limiting

Rate limits vary by subscription tier:

| Tier | Requests per Minute |
|------|---------------------|
| Free | 100 |
| Starter | 1,000 |
| Pro | 10,000 |
| Enterprise | Custom |

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Total requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

When rate limited, the API returns `429 Too Many Requests`.

---

## Error Handling

The API uses standard HTTP status codes and returns errors in the following format:

```json
{
  "error": {
    "type": "validation_error",
    "message": "Request validation failed",
    "details": [
      {
        "loc": ["body", "content"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  }
}
```

### Error Types

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | `bad_request` | Invalid request parameters |
| 401 | `authentication_error` | Invalid or missing credentials |
| 403 | `permission_denied` | Insufficient permissions |
| 404 | `not_found` | Resource not found |
| 422 | `validation_error` | Request validation failed |
| 429 | `rate_limit_exceeded` | Too many requests |
| 500 | `internal_server_error` | Server error |

---

## Endpoints

### Authentication Endpoints

#### POST /v1/auth/register

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "full_name": "John Doe"
}
```

**Response (201 Created):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "usr_abc123"
}
```

---

#### POST /v1/auth/login

Login with email and password.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_id": "usr_abc123"
}
```

---

#### POST /v1/auth/api-keys

Create a new API key (requires authentication).

**Request Body:**
```json
{
  "name": "Production Key"
}
```

**Response (201 Created):**
```json
{
  "id": "key_xyz789",
  "name": "Production Key",
  "key": "mem_sk_1234567890abcdef",
  "prefix": "mem_sk_1234567",
  "created_at": "2024-01-15T10:30:00Z"
}
```

⚠️ **Important**: The full API key is only shown once. Store it securely.

---

#### GET /v1/auth/me

Get current user information.

**Response (200 OK):**
```json
{
  "id": "usr_abc123",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": false,
  "tier": "free",
  "created_at": "2024-01-15T10:30:00Z"
}
```

---

### User Endpoints

#### GET /v1/users/me

Get current user profile.

**Response (200 OK):**
```json
{
  "id": "usr_abc123",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_verified": false,
  "tier": "free"
}
```

---

#### PATCH /v1/users/me

Update current user profile.

**Request Body:**
```json
{
  "full_name": "John Smith"
}
```

**Response (200 OK):**
```json
{
  "id": "usr_abc123",
  "email": "user@example.com",
  "full_name": "John Smith",
  "is_active": true,
  "is_verified": false,
  "tier": "free"
}
```

---

#### DELETE /v1/users/me

Delete current user account (irreversible).

**Response (204 No Content)**

---

### Collection Endpoints

#### POST /v1/collections

Create a new memory collection.

**Request Body:**
```json
{
  "name": "Research Notes",
  "description": "AI research papers and notes",
  "metadata": {
    "category": "research",
    "tags": ["ai", "ml"]
  }
}
```

**Response (201 Created):**
```json
{
  "id": "col_abc123",
  "name": "Research Notes",
  "description": "AI research papers and notes",
  "is_active": true,
  "memory_count": 0,
  "metadata": {
    "category": "research",
    "tags": ["ai", "ml"]
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

---

#### GET /v1/collections

List all collections.

**Query Parameters:**
- `skip` (integer, default: 0): Number of items to skip
- `limit` (integer, default: 100): Maximum items to return

**Response (200 OK):**
```json
[
  {
    "id": "col_abc123",
    "name": "Research Notes",
    "description": "AI research papers and notes",
    "is_active": true,
    "memory_count": 42,
    "metadata": {},
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
]
```

---

#### GET /v1/collections/{collection_id}

Get a specific collection.

**Response (200 OK):**
```json
{
  "id": "col_abc123",
  "name": "Research Notes",
  "description": "AI research papers and notes",
  "is_active": true,
  "memory_count": 42,
  "metadata": {},
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

---

#### PATCH /v1/collections/{collection_id}

Update a collection.

**Request Body:**
```json
{
  "name": "Updated Name",
  "description": "New description",
  "metadata": {
    "updated": true
  }
}
```

**Response (200 OK):**
```json
{
  "id": "col_abc123",
  "name": "Updated Name",
  "description": "New description",
  "is_active": true,
  "memory_count": 42,
  "metadata": {
    "updated": true
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:00:00Z"
}
```

---

#### DELETE /v1/collections/{collection_id}

Delete a collection and all its memories.

**Response (204 No Content)**

---

### Memory Endpoints

#### POST /v1/memories

Create a new memory.

**Request Body:**
```json
{
  "collection_id": "col_abc123",
  "content": "Neural networks are composed of interconnected layers of neurons that process information.",
  "importance": 0.8,
  "source_type": "text",
  "source_reference": "Deep Learning Book, Chapter 6",
  "metadata": {
    "author": "Ian Goodfellow",
    "year": 2016
  }
}
```

**Response (201 Created):**
```json
{
  "id": "mem_xyz789",
  "collection_id": "col_abc123",
  "content": "Neural networks are composed of interconnected layers of neurons that process information.",
  "importance": 0.8,
  "source_type": "text",
  "source_reference": "Deep Learning Book, Chapter 6",
  "access_count": 0,
  "last_accessed_at": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

---

#### GET /v1/memories

List memories.

**Query Parameters:**
- `collection_id` (string, optional): Filter by collection
- `skip` (integer, default: 0): Number of items to skip
- `limit` (integer, default: 100): Maximum items to return

**Response (200 OK):**
```json
[
  {
    "id": "mem_xyz789",
    "collection_id": "col_abc123",
    "content": "Neural networks are composed of interconnected layers...",
    "importance": 0.8,
    "source_type": "text",
    "source_reference": "Deep Learning Book, Chapter 6",
    "access_count": 5,
    "last_accessed_at": "2024-01-15T12:00:00Z",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
]
```

---

#### GET /v1/memories/{memory_id}

Get a specific memory with metadata.

**Response (200 OK):**
```json
{
  "id": "mem_xyz789",
  "collection_id": "col_abc123",
  "content": "Neural networks are composed of interconnected layers of neurons that process information.",
  "importance": 0.8,
  "source_type": "text",
  "source_reference": "Deep Learning Book, Chapter 6",
  "access_count": 5,
  "last_accessed_at": "2024-01-15T12:00:00Z",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "author": "Ian Goodfellow",
    "year": 2016
  }
}
```

---

#### PATCH /v1/memories/{memory_id}

Update a memory.

**Request Body:**
```json
{
  "content": "Updated content",
  "importance": 0.9,
  "metadata": {
    "updated": true
  }
}
```

**Response (200 OK):**
```json
{
  "id": "mem_xyz789",
  "collection_id": "col_abc123",
  "content": "Updated content",
  "importance": 0.9,
  "source_type": "text",
  "source_reference": "Deep Learning Book, Chapter 6",
  "access_count": 5,
  "last_accessed_at": "2024-01-15T12:00:00Z",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T13:00:00Z"
}
```

---

#### DELETE /v1/memories/{memory_id}

Delete a memory.

**Response (204 No Content)**

---

### Search Endpoints

#### POST /v1/search

Search memories using hybrid retrieval.

**Request Body:**
```json
{
  "query": "What are neural networks?",
  "collection_id": "col_abc123",
  "limit": 10,
  "search_type": "hybrid",
  "filters": {
    "min_importance": 0.5
  }
}
```

**Parameters:**
- `query` (string, required): Search query
- `collection_id` (string, optional): Filter by collection
- `limit` (integer, default: 10, max: 100): Number of results
- `search_type` (string, default: "hybrid"): Type of search
  - `hybrid`: Vector + BM25 + Graph (recommended)
  - `vector`: Vector similarity only
  - `bm25`: Keyword search only
  - `graph`: Knowledge graph traversal
- `filters` (object, optional): Additional filters

**Response (200 OK):**
```json
{
  "query": "What are neural networks?",
  "results": [
    {
      "memory_id": "mem_xyz789",
      "content": "Neural networks are composed of interconnected layers of neurons that process information.",
      "score": 0.95,
      "metadata": {
        "importance": 0.8,
        "created_at": "2024-01-15T10:30:00Z",
        "access_count": 5
      },
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "search_type": "hybrid",
  "processing_time_ms": 42.5
}
```

---

#### GET /v1/search/similar/{memory_id}

Find memories similar to a given memory.

**Query Parameters:**
- `limit` (integer, default: 10, max: 100): Number of results

**Response (200 OK):**
```json
{
  "query": "Similar to mem_xyz789",
  "results": [
    {
      "memory_id": "mem_abc456",
      "content": "Deep learning models use multiple layers...",
      "score": 0.87,
      "metadata": {},
      "created_at": "2024-01-15T11:00:00Z"
    }
  ],
  "total": 1,
  "search_type": "vector",
  "processing_time_ms": 15.3
}
```

---

#### POST /v1/search/reason

Perform reasoning over memories using LLM.

**Request Body:**
```json
{
  "query": "Explain the key concepts of neural networks based on my notes",
  "collection_id": "col_abc123",
  "provider": "gemini",
  "include_steps": true
}
```

**Parameters:**
- `query` (string, required): Question or query
- `collection_id` (string, optional): Filter by collection
- `provider` (string, optional, default: "gemini"): LLM provider
  - `gemini`: Google Gemini 2.0 Flash (with thinking mode)
  - `openai`: OpenAI GPT-4
  - `anthropic`: Anthropic Claude 3.5 Sonnet
- `include_steps` (boolean, default: false): Include reasoning steps

**Response (200 OK):**
```json
{
  "answer": "Based on your notes, neural networks are computational models inspired by biological neurons. They consist of interconnected layers that process information through weighted connections. The key concepts include: 1) Layered architecture with input, hidden, and output layers...",
  "sources": [
    {
      "memory_id": "mem_xyz789",
      "content": "Neural networks are composed of interconnected layers...",
      "score": 0.95
    }
  ],
  "metadata": {
    "provider": "gemini",
    "memory_count": 1,
    "timestamp": "2024-01-15T13:00:00Z"
  },
  "reasoning_context": {
    "query": "Explain the key concepts of neural networks based on my notes",
    "memory_count": 1,
    "reasoning_steps": [
      {
        "step": "Retrieved memories",
        "result": 1,
        "timestamp": "2024-01-15T13:00:00.100Z"
      },
      {
        "step": "Generated answer",
        "result": "Based on your notes...",
        "timestamp": "2024-01-15T13:00:01.500Z"
      }
    ],
    "conclusions": [
      {
        "conclusion": "Based on your notes...",
        "confidence": 0.9,
        "timestamp": "2024-01-15T13:00:01.500Z"
      }
    ],
    "metadata": {
      "provider": "gemini",
      "use_thinking": true
    }
  }
}
```

---

## Webhooks

Subscribe to events for real-time notifications (coming soon).

## SDKs

Official SDKs available:
- **Python**: `pip install memory-ai`
- **TypeScript/JavaScript**: `npm install @memory-ai/sdk`

## Support

- **Documentation**: https://docs.memory-ai.com
- **API Status**: https://status.memory-ai.com
- **Support Email**: support@memory-ai.com
- **GitHub**: https://github.com/memory-ai

---

**Last Updated**: January 2024
**API Version**: v1.0.0
