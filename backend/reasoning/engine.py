"""
Reasoning engine for generating insights from retrieved memories.

Combines:
- Retrieved memories (from hybrid search)
- LLM reasoning
- Symbolic validation
- Context management
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.core.logging_config import logger
from backend.core.cache import get_cache_manager
from backend.reasoning.llm_providers import get_llm_provider
from backend.services.hybrid_search import HybridSearchEngine


class ReasoningContext:
    """Context for a reasoning session."""

    def __init__(
        self,
        query: str,
        memories: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None,
    ):
        self.query = query
        self.memories = memories
        self.metadata = metadata or {}
        self.reasoning_steps = []
        self.conclusions = []

    def add_step(self, step: str, result: Any = None):
        """Add a reasoning step."""
        self.reasoning_steps.append({
            "step": step,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def add_conclusion(self, conclusion: str, confidence: float = 1.0):
        """Add a conclusion."""
        self.conclusions.append({
            "conclusion": conclusion,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "memory_count": len(self.memories),
            "reasoning_steps": self.reasoning_steps,
            "conclusions": self.conclusions,
            "metadata": self.metadata,
        }


class ReasoningEngine:
    """
    Reasoning engine that combines memory retrieval with LLM reasoning.
    """

    def __init__(self, db: AsyncSession, provider_name: Optional[str] = None):
        self.db = db
        self.provider_name = provider_name or settings.DEFAULT_LLM_PROVIDER
        self.llm = get_llm_provider(self.provider_name)
        self.search_engine = HybridSearchEngine(db)

    async def reason(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[str] = None,
        use_thinking: bool = True,
        include_steps: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform reasoning over retrieved memories.

        Args:
            query: User query/question
            user_id: User identifier
            collection_id: Optional collection filter
            use_thinking: Whether to use LLM thinking mode
            include_steps: Whether to include reasoning steps in response

        Returns:
            Dictionary with answer, sources, and optional reasoning steps
        """
        try:
            logger.info(f"Starting reasoning for query: {query}")

            # Check cache first
            if settings.ENABLE_RESPONSE_CACHING:
                cache_key = self._get_cache_key(query, user_id, collection_id)
                cached = await self._get_from_cache(cache_key)
                if cached:
                    logger.info("Returning cached reasoning result")
                    return cached

            # Step 1: Retrieve relevant memories
            search_results = await self.search_engine.search(
                query=query,
                user_id=user_id,
                collection_id=collection_id,
                limit=settings.DEFAULT_SEARCH_LIMIT,
                search_type="hybrid",
            )

            memories = [r.to_dict() for r in search_results]

            # Create reasoning context
            context = ReasoningContext(
                query=query,
                memories=memories,
                metadata={
                    "provider": self.provider_name,
                    "use_thinking": use_thinking,
                },
            )

            context.add_step("Retrieved memories", len(memories))

            # Step 2: Build prompt with memories
            prompt = self._build_reasoning_prompt(query, memories)

            # Step 3: Generate reasoning
            if use_thinking:
                llm_response = await self.llm.generate_with_thinking(
                    prompt=prompt,
                    system_prompt=self._get_system_prompt(),
                )
                thinking = llm_response.get("thinking", "")
                answer = llm_response.get("response", "")

                if thinking:
                    context.add_step("LLM thinking", thinking)

            else:
                answer = await self.llm.generate(
                    prompt=prompt,
                    system_prompt=self._get_system_prompt(),
                )

            context.add_step("Generated answer", answer)

            # Step 4: Extract conclusions
            context.add_conclusion(answer, confidence=0.9)

            # Build response
            response = {
                "answer": answer,
                "sources": [
                    {
                        "memory_id": m["memory_id"],
                        "content": m["content"][:200] + "..." if len(m["content"]) > 200 else m["content"],
                        "score": m["score"],
                    }
                    for m in memories[:5]  # Top 5 sources
                ],
                "metadata": {
                    "provider": self.provider_name,
                    "memory_count": len(memories),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            if include_steps:
                response["reasoning_context"] = context.to_dict()

            # Cache result
            if settings.ENABLE_RESPONSE_CACHING:
                await self._cache_result(cache_key, response)

            logger.info("Reasoning completed successfully")
            return response

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your query.",
                "sources": [],
                "error": str(e),
            }

    def _build_reasoning_prompt(
        self,
        query: str,
        memories: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for LLM with retrieved memories."""
        if not memories:
            return (
                f"User query: {query}\n\n"
                "No relevant memories found. Please provide a helpful response "
                "indicating that you don't have enough information to answer this query."
            )

        # Build context from memories
        context_parts = []
        for i, memory in enumerate(memories[:10], 1):  # Top 10 memories
            content = memory["content"]
            score = memory["score"]
            context_parts.append(
                f"[Memory {i}] (relevance: {score:.3f})\n{content}\n"
            )

        context = "\n".join(context_parts)

        prompt = f"""Based on the following relevant memories, please answer the user's query.

RETRIEVED MEMORIES:
{context}

USER QUERY: {query}

Please provide a comprehensive answer based on the retrieved memories.
If the memories don't contain enough information to fully answer the query,
please indicate what information is missing or uncertain.

Your answer:"""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an AI assistant with access to a memory system.
Your role is to provide accurate, helpful answers based on the retrieved memories.

Guidelines:
- Base your answers primarily on the provided memories
- Be clear about what is certain vs. uncertain
- If memories are contradictory, acknowledge this
- If information is missing, say so
- Cite specific memories when making claims
- Be concise but comprehensive"""

    async def summarize_memories(
        self,
        memory_ids: List[str],
        user_id: str,
    ) -> str:
        """
        Generate a summary of multiple memories.

        Args:
            memory_ids: List of memory IDs to summarize
            user_id: User identifier

        Returns:
            Summary text
        """
        try:
            # Fetch memories from database
            from sqlalchemy import select
            from backend.models.memory import Memory

            result = await self.db.execute(
                select(Memory).where(Memory.id.in_(memory_ids))
            )
            memories = result.scalars().all()

            if not memories:
                return "No memories found to summarize."

            # Build summarization prompt
            memory_texts = [m.content for m in memories]
            combined_text = "\n\n".join(
                f"Memory {i+1}:\n{text}"
                for i, text in enumerate(memory_texts)
            )

            prompt = f"""Please provide a concise summary of the following memories,
highlighting key themes and important information:

{combined_text}

Summary:"""

            summary = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that creates concise, informative summaries.",
                temperature=0.5,
            )

            logger.info(f"Generated summary for {len(memories)} memories")
            return summary

        except Exception as e:
            logger.error(f"Memory summarization failed: {e}")
            return "Failed to generate summary."

    async def answer_question(
        self,
        question: str,
        context_memories: List[str],
    ) -> str:
        """
        Answer a question given specific context memories.

        Args:
            question: Question to answer
            context_memories: List of memory contents for context

        Returns:
            Answer text
        """
        try:
            context = "\n\n".join(
                f"Context {i+1}:\n{mem}"
                for i, mem in enumerate(context_memories)
            )

            prompt = f"""Based on the following context, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            answer = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for factual answers
            )

            return answer

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return "I'm unable to answer that question at this time."

    def _get_cache_key(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[str],
    ) -> str:
        """Generate cache key for reasoning result."""
        import hashlib
        key_data = f"{query}:{user_id}:{collection_id or 'all'}:{self.provider_name}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"reasoning:{key_hash}"

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached reasoning result."""
        try:
            cache_manager = await get_cache_manager()
            result = await cache_manager.get(cache_key)
            return result
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache reasoning result."""
        try:
            cache_manager = await get_cache_manager()
            await cache_manager.set(
                cache_key,
                result,
                ttl=settings.REASONING_CACHE_TTL,
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")


def get_reasoning_engine(
    db: AsyncSession,
    provider_name: Optional[str] = None,
) -> ReasoningEngine:
    """
    Get reasoning engine instance.

    Args:
        db: Database session
        provider_name: Optional LLM provider name

    Returns:
        ReasoningEngine instance
    """
    return ReasoningEngine(db, provider_name)
