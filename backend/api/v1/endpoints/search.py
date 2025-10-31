"""
Search endpoints for memory retrieval.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.core.logging_config import logger
from backend.services.hybrid_search import search_memories as hybrid_search
from backend.reasoning.engine import get_reasoning_engine
from backend.rl.trajectory_logger import get_trajectory_logger
from backend.core.cache import get_cache_manager
import hashlib
import json


router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str
    collection_id: Optional[str] = None
    limit: int = 10
    search_type: str = "hybrid"  # hybrid | vector | bm25 | graph
    filters: Dict[str, Any] = {}


class SearchResult(BaseModel):
    """Search result schema."""
    memory_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}
    created_at: datetime


class SearchResponse(BaseModel):
    """Search response schema."""
    query: str
    results: List[SearchResult]
    total: int
    search_type: str
    processing_time_ms: float


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("", response_model=SearchResponse)
async def search_memories_endpoint(
    search_request: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search memories using hybrid retrieval.

    Combines:
    - Dense vector search (Milvus)
    - Sparse BM25 search
    - Knowledge graph traversal (Neo4j)

    Results are re-ranked using cross-encoder and time decay.
    """
    import time
    start_time = time.time()

    logger.info(
        f"Search request from {current_user.email}: "
        f"query='{search_request.query}', type={search_request.search_type}"
    )

    # Generate cache key from search parameters
    cache_key_data = {
        "user_id": str(current_user.id),
        "query": search_request.query,
        "collection_id": search_request.collection_id,
        "limit": search_request.limit,
        "search_type": search_request.search_type,
        "filters": search_request.filters,
    }
    cache_key_str = json.dumps(cache_key_data, sort_keys=True)
    cache_key_hash = hashlib.sha256(cache_key_str.encode()).hexdigest()
    cache_key = f"search:{cache_key_hash}"

    # Try to get from cache
    cache_manager = await get_cache_manager()
    cached_results = await cache_manager.get(cache_key)

    if cached_results is not None:
        logger.info(f"Cache HIT for query: '{search_request.query}' (key: {cache_key})")
        processing_time = (time.time() - start_time) * 1000

        # Convert cached results to SearchResult objects
        results = [
            SearchResult(
                memory_id=r["memory_id"],
                content=r["content"],
                score=r["score"],
                metadata=r.get("metadata", {}),
                created_at=datetime.fromisoformat(r["created_at"])
                if isinstance(r.get("created_at"), str)
                else r.get("created_at", datetime.utcnow()),
            )
            for r in cached_results
        ]

        return SearchResponse(
            query=search_request.query,
            results=results,
            total=len(results),
            search_type=search_request.search_type,
            processing_time_ms=processing_time,
        )

    logger.info(f"Cache MISS for query: '{search_request.query}' (key: {cache_key})")

    # Start RL trajectory logging
    trajectory_logger = get_trajectory_logger()
    trajectory_id = str(uuid.uuid4())
    session_id = f"search_{current_user.id}_{int(time.time())}"

    await trajectory_logger.start_trajectory(
        trajectory_id=trajectory_id,
        user_id=current_user.id,
        session_id=session_id,
        metadata={
            "agent_type": "memory_manager",
            "collection_id": search_request.collection_id,
            "search_type": search_request.search_type,
        },
    )

    # Perform hybrid search
    results_dicts = await hybrid_search(
        query=search_request.query,
        user_id=current_user.id,
        db=db,
        collection_id=search_request.collection_id,
        limit=search_request.limit,
        search_type=search_request.search_type,
        filters=search_request.filters,
    )

    # Convert to response format
    results = [
        SearchResult(
            memory_id=r["memory_id"],
            content=r["content"],
            score=r["score"],
            metadata=r.get("metadata", {}),
            created_at=datetime.fromisoformat(r["metadata"].get("created_at"))
            if r.get("metadata", {}).get("created_at")
            else datetime.utcnow(),
        )
        for r in results_dicts
    ]

    # Store results in cache (TTL: 5 minutes = 300 seconds)
    cache_data = [
        {
            "memory_id": r.memory_id,
            "content": r.content,
            "score": r.score,
            "metadata": r.metadata,
            "created_at": r.created_at.isoformat(),
        }
        for r in results
    ]
    await cache_manager.set(cache_key, cache_data, ttl=300)
    logger.info(f"Stored {len(results)} results in cache with key: {cache_key}")

    processing_time = (time.time() - start_time) * 1000

    # Log trajectory step
    step_id = str(uuid.uuid4())
    await trajectory_logger.log_step(
        trajectory_id=trajectory_id,
        step_id=step_id,
        state={
            "query": search_request.query,
            "collection_id": search_request.collection_id,
            "search_type": search_request.search_type,
            "filters": search_request.filters,
            "user_id": current_user.id,
        },
        action={
            "action_type": "search",
            "results_count": len(results),
            "top_results": [
                {
                    "memory_id": r.memory_id,
                    "score": r.score,
                }
                for r in results[:5]  # Top 5 results
            ],
            "processing_time_ms": processing_time,
        },
        reward=None,  # Will be updated based on user engagement
        metadata={
            "endpoint": "search",
            "search_type": search_request.search_type,
        },
    )

    # End trajectory
    await trajectory_logger.end_trajectory(trajectory_id)

    return SearchResponse(
        query=search_request.query,
        results=results,
        total=len(results),
        search_type=search_request.search_type,
        processing_time_ms=processing_time,
    )


@router.get("/similar/{memory_id}", response_model=SearchResponse)
async def find_similar_memories(
    memory_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(10, ge=1, le=100),
):
    """
    Find similar memories to a given memory.

    Uses vector similarity search.
    """
    import time
    start_time = time.time()

    logger.info(f"Similar search for memory {memory_id} by {current_user.email}")

    # TODO: Implement actual similarity search
    results = []

    processing_time = (time.time() - start_time) * 1000

    return SearchResponse(
        query=f"Similar to {memory_id}",
        results=results,
        total=len(results),
        search_type="vector",
        processing_time_ms=processing_time,
    )


@router.post("/graph")
async def search_knowledge_graph(
    query: str,
    current_user: User = Depends(get_current_user),
    collection_id: Optional[str] = None,
    depth: int = Query(2, ge=1, le=5),
):
    """
    Search the knowledge graph for entities and relationships.

    Returns a subgraph of connected entities.
    """
    logger.info(f"Graph search: '{query}' by {current_user.email}")

    # TODO: Implement Neo4j graph search
    return {
        "query": query,
        "nodes": [],
        "edges": [],
        "depth": depth,
    }


class ReasoningRequest(BaseModel):
    """Reasoning request schema."""
    query: str
    collection_id: Optional[str] = None
    provider: Optional[str] = None  # gemini, openai, anthropic
    include_steps: bool = False


class ReasoningResponse(BaseModel):
    """Reasoning response schema."""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    reasoning_context: Optional[Dict[str, Any]] = None


@router.post("/reason", response_model=ReasoningResponse)
async def reason_with_memories(
    reasoning_request: ReasoningRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Perform reasoning over memories using LLM.

    Retrieves relevant memories and generates a comprehensive answer
    using the configured LLM provider (Gemini, OpenAI, or Claude).
    """
    import time
    start_time = time.time()

    logger.info(
        f"Reasoning request from {current_user.email}: "
        f"query='{reasoning_request.query}'"
    )

    # Start RL trajectory logging
    trajectory_logger = get_trajectory_logger()
    trajectory_id = str(uuid.uuid4())
    session_id = f"reason_{current_user.id}_{int(time.time())}"

    await trajectory_logger.start_trajectory(
        trajectory_id=trajectory_id,
        user_id=current_user.id,
        session_id=session_id,
        metadata={
            "agent_type": "answer_agent",
            "collection_id": reasoning_request.collection_id,
            "provider": reasoning_request.provider,
        },
    )

    # Get reasoning engine
    reasoning_engine = get_reasoning_engine(
        db=db,
        provider_name=reasoning_request.provider,
    )

    # Perform reasoning
    result = await reasoning_engine.reason(
        query=reasoning_request.query,
        user_id=current_user.id,
        collection_id=reasoning_request.collection_id,
        use_thinking=True,
        include_steps=reasoning_request.include_steps,
    )

    processing_time = (time.time() - start_time) * 1000

    # Log trajectory step
    step_id = str(uuid.uuid4())
    await trajectory_logger.log_step(
        trajectory_id=trajectory_id,
        step_id=step_id,
        state={
            "query": reasoning_request.query,
            "collection_id": reasoning_request.collection_id,
            "provider": reasoning_request.provider,
            "user_id": current_user.id,
        },
        action={
            "action_type": "reason",
            "answer_length": len(result["answer"]),
            "sources_count": len(result.get("sources", [])),
            "sources": [
                {
                    "memory_id": s.get("memory_id"),
                    "score": s.get("score"),
                }
                for s in result.get("sources", [])[:5]  # Top 5 sources
            ],
            "provider": reasoning_request.provider,
            "processing_time_ms": processing_time,
        },
        reward=None,  # Will be updated based on user feedback
        metadata={
            "endpoint": "reason",
            "provider": reasoning_request.provider,
        },
    )

    # End trajectory
    await trajectory_logger.end_trajectory(trajectory_id)

    return ReasoningResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        metadata=result.get("metadata", {}),
        reasoning_context=result.get("reasoning_context"),
    )
