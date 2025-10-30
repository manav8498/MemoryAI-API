"""
Automatic Conversation Memory Middleware

Automatically saves important interactions to the "Conversation History" collection.
No manual user action required - memories are created transparently.
"""

from fastapi import Request
from typing import Optional, Dict, Any
from datetime import datetime
import json

from backend.core.logging_config import logger
from backend.core.database import AsyncSessionLocal
from backend.models.collection import Collection
from backend.models.memory import Memory
from sqlalchemy import select


class ConversationMemoryMiddleware:
    """
    Middleware that automatically saves conversation context.

    Captures:
    - Search queries and results
    - Reasoning questions and answers
    - RL training sessions
    - Memory creation/updates
    - Collection operations

    Saves to "Conversation History" collection automatically.
    """

    CONVERSATION_COLLECTION_NAME = "Conversation History"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Capture request
        request = Request(scope, receive)
        path = request.url.path
        method = request.method

        # Only process certain endpoints
        if self._should_save(path, method):
            # Get user ID from request
            user_id = await self._get_user_id(request)

            if user_id:
                # Save after response completes
                async def send_wrapper(message):
                    await send(message)

                    # After response sent, save context
                    if message["type"] == "http.response.body":
                        await self._auto_save_context(
                            user_id=user_id,
                            path=path,
                            method=method,
                            request=request
                        )

                await self.app(scope, receive, send_wrapper)
                return

        # Default: just pass through
        await self.app(scope, receive, send)

    def _should_save(self, path: str, method: str) -> bool:
        """Determine if this request should be auto-saved."""
        important_paths = [
            "/v1/search",
            "/v1/reason",
            "/v1/memories",
            "/v1/rl/train",
            "/v1/rl/evaluate",
            "/v1/collections"
        ]

        return any(p in path for p in important_paths) and method == "POST"

    async def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Try to get from request state (set by auth middleware)
        if hasattr(request.state, "user"):
            return request.state.user.id

        # Try to get from auth header
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                from backend.core.auth import decode_token
                token = auth.split(" ")[1]
                payload = decode_token(token)
                return payload.get("sub")
            except Exception:
                return None

        return None

    async def _auto_save_context(
        self,
        user_id: str,
        path: str,
        method: str,
        request: Request
    ):
        """
        Automatically save conversation context to memory.

        This happens silently in the background.
        """
        try:
            # Get or create Conversation History collection
            collection = await self._get_or_create_collection(user_id)

            if not collection:
                logger.warning("Could not create conversation history collection")
                return

            # Generate summary based on endpoint
            summary = await self._generate_summary(path, request)

            if not summary:
                return  # Nothing important to save

            # Save to database
            async with AsyncSessionLocal() as db:
                memory = Memory(
                    user_id=user_id,
                    collection_id=collection.id,
                    content=summary,
                    importance=0.6,  # Medium importance for auto-saved items
                    metadata={
                        "auto_saved": True,
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": "automatic",
                        "endpoint": path,
                        "method": method
                    }
                )

                db.add(memory)
                await db.commit()

                logger.info(f"Auto-saved conversation context: {summary[:100]}...")

        except Exception as e:
            logger.error(f"Failed to auto-save conversation: {e}")
            # Don't fail the request if auto-save fails

    async def _get_or_create_collection(self, user_id: str) -> Optional[Collection]:
        """Get or create the Conversation History collection."""
        async with AsyncSessionLocal() as db:
            # Check if exists
            result = await db.execute(
                select(Collection).where(
                    Collection.user_id == user_id,
                    Collection.name == self.CONVERSATION_COLLECTION_NAME
                )
            )
            collection = result.scalar_one_or_none()

            if collection:
                return collection

            # Create it
            collection = Collection(
                user_id=user_id,
                name=self.CONVERSATION_COLLECTION_NAME,
                description="Automatically captured conversation history. Contains summaries of searches, reasoning, training, and other important interactions.",
                metadata={"auto_created": True}
            )

            db.add(collection)
            await db.commit()
            await db.refresh(collection)

            logger.info(f"Created Conversation History collection for user {user_id}")
            return collection

    async def _generate_summary(self, path: str, request: Request) -> Optional[str]:
        """Generate human-readable summary of the interaction."""
        try:
            # Get request body
            body = await request.json() if request.method == "POST" else {}

            if "/search" in path:
                query = body.get("query", "")
                return f"Searched for: {query}"

            elif "/reason" in path:
                query = body.get("query", "")
                return f"Reasoned about: {query}"

            elif "/rl/train" in path:
                agent_type = path.split("/")[-1] if "memory-manager" in path or "answer-agent" in path else "agent"
                num_episodes = body.get("num_episodes", "")
                return f"Trained {agent_type} RL agent ({num_episodes} episodes)"

            elif "/rl/evaluate" in path:
                agent_type = body.get("agent_type", "agent")
                return f"Evaluated {agent_type} performance"

            elif "/memories" in path and request.method == "POST":
                content = body.get("content", "")
                if content:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    return f"Added memory: {preview}"

            elif "/collections" in path and request.method == "POST":
                name = body.get("name", "")
                return f"Created collection: {name}"

            return None

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
