"""
Working Memory Buffer.

Short-term memory for active conversational context.
Separate from long-term episodic/semantic memory.
"""
from typing import List, Dict, Any, Optional, Deque
from collections import deque
from datetime import datetime
import asyncio

from backend.core.logging_config import logger
from backend.reasoning.llm_providers import get_llm_provider


class WorkingMemoryBuffer:
    """
    Working memory buffer for conversation context.

    Maintains recent conversation turns and active context
    separate from long-term memory storage.

    Features:
    - Fixed size buffer (FIFO)
    - Automatic summarization when full
    - Context compression
    - Integration with long-term memory
    """

    def __init__(
        self,
        max_size: int = 10,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize working memory buffer.

        Args:
            max_size: Maximum number of items in buffer
            session_id: Optional session identifier
            user_id: Optional user identifier
        """
        self.max_size = max_size
        self.session_id = session_id
        self.user_id = user_id
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=max_size)
        self.compressed_history: List[str] = []
        self.llm = None

    def _get_llm(self):
        """Lazy load LLM provider."""
        if self.llm is None:
            self.llm = get_llm_provider()
        return self.llm

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an item to working memory.

        Args:
            role: Role (user, assistant, system)
            content: Content
            metadata: Optional metadata
        """
        item = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        # If buffer is full, compress oldest item before adding new one
        if len(self.buffer) >= self.max_size:
            oldest = self.buffer[0]  # Will be removed when we append
            self.compressed_history.append(
                f"[{oldest['role']}]: {oldest['content'][:100]}..."
            )

        self.buffer.append(item)
        logger.debug(f"Added to working memory: {role} - {content[:50]}...")

    def get_context(
        self,
        include_compressed: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get current working memory context.

        Args:
            include_compressed: Whether to include compressed history

        Returns:
            List of context items
        """
        context = list(self.buffer)

        if include_compressed and self.compressed_history:
            # Add compressed history as first item
            context.insert(0, {
                "role": "system",
                "content": f"Earlier conversation summary: {' '.join(self.compressed_history[-3:])}",
                "timestamp": None,
                "metadata": {"compressed": True},
            })

        return context

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get n most recent items.

        Args:
            n: Number of items

        Returns:
            Recent items
        """
        return list(self.buffer)[-n:]

    def clear(self) -> None:
        """Clear working memory buffer."""
        self.buffer.clear()
        logger.info("Cleared working memory buffer")

    async def compress(self) -> str:
        """
        Compress working memory to a summary.

        Returns:
            Compressed summary
        """
        if not self.buffer:
            return ""

        try:
            llm = self._get_llm()

            # Build context
            conversation = "\n".join([
                f"{item['role'].upper()}: {item['content']}"
                for item in self.buffer
            ])

            prompt = f"""Please summarize the following conversation, capturing key points and context:

{conversation}

Provide a concise summary (2-3 sentences) that preserves important information:"""

            summary = await llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=150,
            )

            logger.info("Compressed working memory to summary")
            return summary

        except Exception as e:
            logger.error(f"Working memory compression failed: {e}")
            # Fallback: simple truncation
            return " | ".join([
                f"{item['role']}: {item['content'][:30]}..."
                for item in list(self.buffer)[:3]
            ])

    async def consolidate_to_long_term(
        self,
        db,
        collection_id: str,
        importance_threshold: float = 0.5,
    ) -> List[str]:
        """
        Consolidate working memory to long-term storage.

        Analyzes buffer contents and stores important information
        as long-term memories.

        Args:
            db: Database session
            collection_id: Collection to store in
            importance_threshold: Minimum importance to store

        Returns:
            List of created memory IDs
        """
        if not self.buffer:
            return []

        try:
            llm = self._get_llm()

            # Analyze conversation for important information
            conversation = "\n".join([
                f"{item['role'].upper()}: {item['content']}"
                for item in self.buffer
            ])

            prompt = f"""Analyze this conversation and extract important facts that should be remembered long-term.

{conversation}

List important facts to remember (one per line):
- Focus on user preferences, decisions, and key information
- Ignore casual chitchat
- Each fact should be self-contained

Important facts:"""

            response = await llm.generate(
                prompt=prompt,
                temperature=0.3,
            )

            # Parse facts
            facts = [
                line.strip().lstrip("- ")
                for line in response.split("\n")
                if line.strip() and line.strip().startswith("-")
            ]

            # Store as long-term memories
            from backend.services.pipeline.memory_ingestion import get_ingestion_pipeline
            from backend.models.memory import Memory, MemoryMetadata
            import uuid
            import hashlib

            memory_ids = []
            pipeline = get_ingestion_pipeline(db)

            for fact in facts:
                if not fact:
                    continue

                memory_id = str(uuid.uuid4())
                memory = Memory(
                    id=memory_id,
                    collection_id=collection_id,
                    content=fact,
                    content_hash=hashlib.sha256(fact.encode()).hexdigest(),
                    importance=0.7,  # Facts from working memory are important
                    source_type="working_memory_consolidation",
                    created_at=datetime.utcnow(),
                )

                metadata = MemoryMetadata(
                    id=str(uuid.uuid4()),
                    memory_id=memory_id,
                    custom_metadata={
                        "from_working_memory": True,
                        "session_id": self.session_id,
                        "consolidated_at": datetime.utcnow().isoformat(),
                    },
                )

                db.add(memory)
                db.add(metadata)
                await db.commit()

                await pipeline.ingest_memory(
                    memory=memory,
                    metadata=metadata,
                    user_id=self.user_id,
                    collection_id=collection_id,
                )

                memory_ids.append(memory_id)

            logger.info(f"Consolidated {len(memory_ids)} facts from working memory to long-term")
            return memory_ids

        except Exception as e:
            logger.error(f"Working memory consolidation failed: {e}")
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert working memory to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "buffer_size": len(self.buffer),
            "max_size": self.max_size,
            "buffer": list(self.buffer),
            "compressed_history_count": len(self.compressed_history),
        }

    def get_token_count_estimate(self) -> int:
        """
        Estimate total token count in buffer.

        Rough estimate: 1 token ≈ 4 characters
        """
        total_chars = sum(len(item["content"]) for item in self.buffer)
        return total_chars // 4

    def trim_to_token_limit(self, max_tokens: int = 4000) -> None:
        """
        Trim buffer to fit within token limit.

        Args:
            max_tokens: Maximum tokens to keep
        """
        while self.get_token_count_estimate() > max_tokens and len(self.buffer) > 1:
            # Remove oldest item
            oldest = self.buffer.popleft()
            self.compressed_history.append(
                f"[{oldest['role']}]: {oldest['content'][:100]}..."
            )

        logger.info(f"Trimmed working memory to ~{self.get_token_count_estimate()} tokens")


class WorkingMemoryManager:
    """
    Manages working memory buffers for multiple sessions.

    Maintains a separate buffer for each active session.
    """

    def __init__(self):
        self.buffers: Dict[str, WorkingMemoryBuffer] = {}
        self.cleanup_interval = 3600  # Cleanup every hour

    def get_buffer(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        max_size: int = 10,
    ) -> WorkingMemoryBuffer:
        """
        Get or create working memory buffer for session.

        Args:
            session_id: Session ID
            user_id: Optional user ID
            max_size: Buffer size

        Returns:
            WorkingMemoryBuffer instance
        """
        if session_id not in self.buffers:
            self.buffers[session_id] = WorkingMemoryBuffer(
                max_size=max_size,
                session_id=session_id,
                user_id=user_id,
            )

        return self.buffers[session_id]

    def remove_buffer(self, session_id: str) -> bool:
        """Remove buffer for session."""
        if session_id in self.buffers:
            del self.buffers[session_id]
            logger.info(f"Removed working memory buffer for session {session_id}")
            return True
        return False

    async def cleanup_inactive_buffers(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up inactive buffers.

        Args:
            max_age_seconds: Maximum age for buffers

        Returns:
            Number of buffers removed
        """
        # This is a simplified implementation
        # Would check last activity time in production
        removed = 0
        sessions_to_remove = []

        for session_id, buffer in self.buffers.items():
            # Check if buffer is old (simplified - would use actual timestamps)
            if len(buffer.buffer) == 0:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.buffers[session_id]
            removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} inactive working memory buffers")

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all buffers."""
        return {
            "active_buffers": len(self.buffers),
            "total_items": sum(len(b.buffer) for b in self.buffers.values()),
            "sessions": list(self.buffers.keys()),
        }


# Global manager instance
_working_memory_manager: Optional[WorkingMemoryManager] = None


def get_working_memory_manager() -> WorkingMemoryManager:
    """Get global working memory manager instance."""
    global _working_memory_manager

    if _working_memory_manager is None:
        _working_memory_manager = WorkingMemoryManager()

    return _working_memory_manager
