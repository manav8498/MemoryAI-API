"""
Hybrid search combining vector similarity, BM25, and knowledge graph.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
import math

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from backend.core.config import settings
from backend.core.logging_config import logger
from backend.models.memory import Memory
from backend.models.collection import Collection
from backend.services.vector_store import get_vector_store_client
from backend.services.knowledge_graph import get_knowledge_graph_client
from backend.ml.sparse_search import get_bm25_index
from backend.ml.embeddings.model import get_embedding_generator


class HybridSearchResult:
    """Represents a hybrid search result."""

    def __init__(
        self,
        memory_id: str,
        content: str,
        score: float,
        vector_score: float = 0.0,
        bm25_score: float = 0.0,
        graph_score: float = 0.0,
        metadata: Dict[str, Any] = None,
    ):
        self.memory_id = memory_id
        self.content = content
        self.score = score
        self.vector_score = vector_score
        self.bm25_score = bm25_score
        self.graph_score = graph_score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "score": self.score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "graph_score": self.graph_score,
            "metadata": self.metadata,
        }


class HybridSearchEngine:
    """
    Hybrid search engine combining multiple retrieval methods.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.vector_store = get_vector_store_client()
        self.kg_client = get_knowledge_graph_client()
        self.embedding_generator = get_embedding_generator()

    async def search(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[str] = None,
        limit: int = 10,
        search_type: str = "hybrid",
        alpha: Optional[float] = None,
        filters: Dict[str, Any] = None,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            user_id: User identifier
            collection_id: Optional collection filter
            limit: Number of results
            search_type: Type of search (hybrid, vector, bm25, graph)
            alpha: Hybrid search balance (0=sparse only, 1=dense only)
            filters: Additional filters

        Returns:
            List of search results
        """
        alpha = alpha or settings.HYBRID_SEARCH_ALPHA
        filters = filters or {}

        try:
            if search_type == "vector":
                results = await self._vector_search(query, user_id, collection_id, limit, filters)
            elif search_type == "bm25":
                results = await self._bm25_search(query, user_id, collection_id, limit, filters)
            elif search_type == "graph":
                results = await self._graph_search(query, user_id, collection_id, limit, filters)
            else:  # hybrid
                results = await self._hybrid_search(query, user_id, collection_id, limit, alpha, filters)

            # Apply time decay and importance weighting
            results = await self._apply_ranking_adjustments(results)

            # Re-rank with cross-encoder if enabled
            if settings.ENABLE_CROSS_ENCODER_RERANKING and len(results) > 0:
                results = await self._rerank_with_cross_encoder(query, results)

            return results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def _vector_search(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[str],
        limit: int,
        filters: Dict[str, Any],
    ) -> List[HybridSearchResult]:
        """Perform vector similarity search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.encode_query(query)

            # Search in Milvus
            vector_results = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                user_id=user_id,
                collection_id=collection_id,
                top_k=limit * 2,  # Get more for filtering
                importance_threshold=filters.get("min_importance", 0.0),
            )

            # Fetch memory contents from database
            memory_ids = [r["memory_id"] for r in vector_results]
            if not memory_ids:
                return []

            result = await self.db.execute(
                select(Memory).where(Memory.id.in_(memory_ids))
            )
            memories = {m.id: m for m in result.scalars().all()}

            # Build results
            results = []
            for vr in vector_results:
                memory = memories.get(vr["memory_id"])
                if memory:
                    results.append(HybridSearchResult(
                        memory_id=memory.id,
                        content=memory.content,
                        score=vr["score"],
                        vector_score=vr["score"],
                        metadata={
                            "importance": memory.importance,
                            "created_at": memory.created_at.isoformat(),
                            "access_count": memory.access_count,
                        },
                    ))

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _bm25_search(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[str],
        limit: int,
        filters: Dict[str, Any],
    ) -> List[HybridSearchResult]:
        """Perform BM25 keyword search."""
        try:
            # Get BM25 index
            if collection_id:
                bm25_index = get_bm25_index(collection_id)
            else:
                # For user-wide search, we'd need to aggregate indexes
                # For now, return empty if no collection specified
                logger.warning("BM25 search requires collection_id")
                return []

            # Search
            bm25_results = bm25_index.search(query, top_k=limit * 2)

            if not bm25_results:
                return []

            # Fetch memories
            memory_ids = [r[0] for r in bm25_results]
            result = await self.db.execute(
                select(Memory).where(Memory.id.in_(memory_ids))
            )
            memories = {m.id: m for m in result.scalars().all()}

            # Build results
            results = []
            for memory_id, score in bm25_results:
                memory = memories.get(memory_id)
                if memory:
                    # Normalize BM25 score (simple normalization)
                    normalized_score = min(score / 10.0, 1.0)

                    results.append(HybridSearchResult(
                        memory_id=memory.id,
                        content=memory.content,
                        score=normalized_score,
                        bm25_score=normalized_score,
                        metadata={
                            "importance": memory.importance,
                            "created_at": memory.created_at.isoformat(),
                        },
                    ))

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    async def _graph_search(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[str],
        limit: int,
        filters: Dict[str, Any],
    ) -> List[HybridSearchResult]:
        """Perform knowledge graph search."""
        try:
            # Search for entities matching query
            entities = await self.kg_client.search_entities(
                query=query,
                user_id=user_id,
                limit=5,
            )

            if not entities:
                return []

            # For each entity, find connected memories
            all_memory_ids = set()
            entity_scores = {}

            for entity in entities:
                # Get entity graph would return connected memories
                # For now, we'll score based on mention count
                entity_scores[entity["name"]] = entity.get("mention_count", 1)

            # TODO: Implement full graph traversal to find memories
            # For now, return empty since we don't have the full implementation
            return []

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    async def _hybrid_search(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[str],
        limit: int,
        alpha: float,
        filters: Dict[str, Any],
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and BM25.

        Uses Reciprocal Rank Fusion (RRF) to combine results.
        """
        try:
            # Run vector and BM25 searches in parallel
            vector_task = self._vector_search(query, user_id, collection_id, limit * 2, filters)
            bm25_task = self._bm25_search(query, user_id, collection_id, limit * 2, filters) if collection_id else None

            if bm25_task:
                vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
            else:
                vector_results = await vector_task
                bm25_results = []

            # Reciprocal Rank Fusion
            rrf_scores = {}
            k = 60  # RRF constant

            # Add vector scores
            for rank, result in enumerate(vector_results):
                memory_id = result.memory_id
                rrf_scores[memory_id] = rrf_scores.get(memory_id, 0.0) + alpha / (k + rank + 1)

            # Add BM25 scores
            for rank, result in enumerate(bm25_results):
                memory_id = result.memory_id
                rrf_scores[memory_id] = rrf_scores.get(memory_id, 0.0) + (1 - alpha) / (k + rank + 1)

            # Combine results
            all_results = {r.memory_id: r for r in vector_results + bm25_results}
            combined = []

            for memory_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
                result = all_results[memory_id]
                result.score = rrf_score
                combined.append(result)

            logger.info(f"Hybrid search combined {len(combined)} results")
            return combined

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def _apply_ranking_adjustments(
        self,
        results: List[HybridSearchResult],
    ) -> List[HybridSearchResult]:
        """
        Apply time decay and importance weighting to search results.
        """
        try:
            if settings.ENABLE_MEMORY_DECAY:
                now = datetime.utcnow()

                for result in results:
                    # Parse created_at from metadata
                    created_at_str = result.metadata.get("created_at")
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        days_old = (now - created_at).days

                        # Apply exponential decay
                        decay_rate = settings.DEFAULT_DECAY_RATE
                        decay_factor = math.exp(-decay_rate * days_old)

                        # Apply importance boost
                        importance = result.metadata.get("importance", 0.5)

                        # Adjust score
                        result.score = result.score * decay_factor * (0.5 + importance)

            # Re-sort by adjusted scores
            results.sort(key=lambda r: r.score, reverse=True)
            return results

        except Exception as e:
            logger.error(f"Failed to apply ranking adjustments: {e}")
            return results

    async def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[HybridSearchResult],
    ) -> List[HybridSearchResult]:
        """
        Re-rank results using cross-encoder model.

        Uses sentence-transformers cross-encoder for accurate reranking.
        """
        try:
            from backend.ml.reranking import get_reranker

            if not results:
                return results

            # Convert to format expected by reranker
            results_dicts = [r.to_dict() for r in results]

            # Rerank
            reranker = get_reranker()
            reranked_dicts = await reranker.rerank(
                query=query,
                results=results_dicts,
                top_k=len(results),
                combine_scores=True,
                combination_weight=0.7,  # 70% rerank, 30% original
            )

            # Convert back to HybridSearchResult objects
            reranked_results = []
            for r_dict in reranked_dicts:
                # Find original result to preserve metadata
                original = next(
                    (res for res in results if res.memory_id == r_dict["memory_id"]),
                    None
                )
                if original:
                    # Update score with reranked score
                    original.score = r_dict["score"]
                    reranked_results.append(original)

            logger.info(f"Cross-encoder reranked {len(results)} results")
            return reranked_results

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Return original results on error
            return results


async def search_memories(
    query: str,
    user_id: str,
    db: AsyncSession,
    collection_id: Optional[str] = None,
    limit: int = 10,
    search_type: str = "hybrid",
    filters: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function for searching memories.

    Args:
        query: Search query
        user_id: User identifier
        db: Database session
        collection_id: Optional collection filter
        limit: Number of results
        search_type: Type of search
        filters: Additional filters

    Returns:
        List of search results as dictionaries
    """
    engine = HybridSearchEngine(db)
    results = await engine.search(
        query=query,
        user_id=user_id,
        collection_id=collection_id,
        limit=limit,
        search_type=search_type,
        filters=filters,
    )

    return [r.to_dict() for r in results]
