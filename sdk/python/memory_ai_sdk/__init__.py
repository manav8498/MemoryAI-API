"""
AI Memory SDK - Python Client Library

A modern, async-first Python SDK for the AI Memory API.

Example usage:

    from memory_ai import MemoryClient

    # Initialize client
    client = MemoryClient(api_key="mem_sk_...")

    # Create a memory
    memory = await client.memories.create(
        collection_id="col_123",
        content="Important information to remember"
    )

    # Search memories
    results = await client.search(
        query="What was that important thing?",
        limit=5
    )

    # Reason over memories
    answer = await client.reason(
        query="Explain what I learned about AI",
        provider="gemini"
    )
"""

__version__ = "1.0.0"

from memory_ai_sdk.client import MemoryClient
from memory_ai_sdk.exceptions import (
    MemoryAIError,
    AuthenticationError,
    ValidationError,
    NotFoundError,
    RateLimitError,
)


__all__ = [
    "MemoryClient",
    "MemoryAIError",
    "AuthenticationError",
    "ValidationError",
    "NotFoundError",
    "RateLimitError",
]
