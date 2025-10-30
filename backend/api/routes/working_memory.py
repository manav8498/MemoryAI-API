"""
Working Memory API Routes.

Endpoints for managing short-term conversational context.
"""
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.services.working_memory import get_working_memory_manager
from backend.core.logging_config import logger


router = APIRouter(prefix="/working-memory", tags=["Working Memory"])


class AddToWorkingMemoryRequest(BaseModel):
    """Request model for adding to working memory."""
    session_id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ConsolidateRequest(BaseModel):
    """Request model for consolidating to long-term memory."""
    session_id: str
    collection_id: str
    importance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@router.post("/add")
async def add_to_working_memory(
    request: AddToWorkingMemoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Add item to working memory buffer.

    Stores conversational turn in short-term buffer.
    """
    try:
        manager = get_working_memory_manager()

        buffer = manager.get_buffer(
            session_id=request.session_id,
            user_id=current_user.id,
        )

        buffer.add(
            role=request.role,
            content=request.content,
            metadata=request.metadata,
        )

        return {
            "status": "success",
            "message": "Added to working memory",
            "buffer_size": len(buffer.buffer),
            "max_size": buffer.max_size,
        }

    except Exception as e:
        logger.error(f"Failed to add to working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{session_id}")
async def get_working_memory_context(
    session_id: str,
    include_compressed: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get current working memory context for session.

    Returns recent conversation turns and optionally compressed history.
    """
    try:
        manager = get_working_memory_manager()

        buffer = manager.get_buffer(
            session_id=session_id,
            user_id=current_user.id,
        )

        context = buffer.get_context(include_compressed=include_compressed)

        return {
            "status": "success",
            "session_id": session_id,
            "context": context,
            "buffer_size": len(buffer.buffer),
            "token_estimate": buffer.get_token_count_estimate(),
        }

    except Exception as e:
        logger.error(f"Failed to get working memory context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent/{session_id}")
async def get_recent_items(
    session_id: str,
    n: int = Query(default=5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get n most recent items from working memory.

    Returns the last n conversational turns.
    """
    try:
        manager = get_working_memory_manager()

        buffer = manager.get_buffer(
            session_id=session_id,
            user_id=current_user.id,
        )

        recent = buffer.get_recent(n=n)

        return {
            "status": "success",
            "session_id": session_id,
            "count": len(recent),
            "items": recent,
        }

    except Exception as e:
        logger.error(f"Failed to get recent items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compress/{session_id}")
async def compress_working_memory(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compress working memory to summary.

    Uses LLM to create 2-3 sentence summary of buffer.
    """
    try:
        manager = get_working_memory_manager()

        buffer = manager.get_buffer(
            session_id=session_id,
            user_id=current_user.id,
        )

        summary = await buffer.compress()

        return {
            "status": "success",
            "session_id": session_id,
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Failed to compress working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate")
async def consolidate_to_long_term(
    request: ConsolidateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Consolidate working memory to long-term storage.

    Extracts important facts and creates long-term memories.
    """
    try:
        manager = get_working_memory_manager()

        buffer = manager.get_buffer(
            session_id=request.session_id,
            user_id=current_user.id,
        )

        memory_ids = await buffer.consolidate_to_long_term(
            db=db,
            collection_id=request.collection_id,
            importance_threshold=request.importance_threshold,
        )

        return {
            "status": "success",
            "session_id": request.session_id,
            "memories_created": len(memory_ids),
            "memory_ids": memory_ids,
            "message": f"Consolidated {len(memory_ids)} facts to long-term memory",
        }

    except Exception as e:
        logger.error(f"Failed to consolidate working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear/{session_id}")
async def clear_working_memory(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Clear working memory buffer for session.

    Removes all items from buffer (does not affect long-term memory).
    """
    try:
        manager = get_working_memory_manager()

        buffer = manager.get_buffer(
            session_id=session_id,
            user_id=current_user.id,
        )

        buffer.clear()

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Working memory cleared",
        }

    except Exception as e:
        logger.error(f"Failed to clear working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def remove_session_buffer(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Remove entire buffer for session.

    Deletes the buffer from manager (more thorough than clear).
    """
    try:
        manager = get_working_memory_manager()

        success = manager.remove_buffer(session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session buffer not found")

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session buffer removed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove session buffer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_working_memory_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get statistics about all working memory buffers.

    Returns overall stats across all sessions.
    """
    try:
        manager = get_working_memory_manager()
        stats = manager.get_stats()

        return {
            "status": "success",
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Failed to get working memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trim/{session_id}")
async def trim_to_token_limit(
    session_id: str,
    max_tokens: int = Query(default=4000, ge=100, le=20000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Trim working memory to fit within token limit.

    Removes oldest items until under token limit.
    """
    try:
        manager = get_working_memory_manager()

        buffer = manager.get_buffer(
            session_id=session_id,
            user_id=current_user.id,
        )

        buffer.trim_to_token_limit(max_tokens=max_tokens)

        return {
            "status": "success",
            "session_id": session_id,
            "buffer_size": len(buffer.buffer),
            "token_estimate": buffer.get_token_count_estimate(),
            "message": f"Trimmed to ~{buffer.get_token_count_estimate()} tokens",
        }

    except Exception as e:
        logger.error(f"Failed to trim working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))
