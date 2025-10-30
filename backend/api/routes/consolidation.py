"""
Memory Consolidation API Routes.

Endpoints for consolidating episodic to semantic memories.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.services.consolidation import get_memory_consolidator
from backend.core.logging_config import logger


router = APIRouter(prefix="/consolidation", tags=["Memory Consolidation"])


class ConsolidateRequest(BaseModel):
    """Request model for memory consolidation."""
    collection_id: str
    lookback_days: int = Field(default=7, ge=1, le=365)
    utility_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


@router.post("/consolidate")
async def consolidate_memories(
    request: ConsolidateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Consolidate episodic memories to semantic.

    Runs consolidation pipeline: pattern extraction → semantic creation → selective retention.
    """
    try:
        consolidator = get_memory_consolidator(db)

        # Run in background
        background_tasks.add_task(
            consolidator.consolidate_user_memories,
            user_id=current_user.id,
            collection_id=request.collection_id,
            lookback_days=request.lookback_days,
            utility_threshold=request.utility_threshold,
        )

        return {
            "status": "started",
            "message": "Memory consolidation started in background",
            "collection_id": request.collection_id,
            "lookback_days": request.lookback_days,
        }

    except Exception as e:
        logger.error(f"Failed to start consolidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate/{collection_id}/run-now")
async def consolidate_now(
    collection_id: str,
    lookback_days: int = Query(default=7, ge=1, le=365),
    utility_threshold: float = Query(default=0.3, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run consolidation synchronously (blocking).

    Returns results immediately instead of running in background.
    """
    try:
        consolidator = get_memory_consolidator(db)

        result = await consolidator.consolidate_user_memories(
            user_id=current_user.id,
            collection_id=collection_id,
            lookback_days=lookback_days,
            utility_threshold=utility_threshold,
        )

        return {
            "status": "success",
            "result": result,
        }

    except Exception as e:
        logger.error(f"Failed to consolidate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{collection_id}")
async def get_consolidation_stats(
    collection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get consolidation statistics for collection.

    Returns memory counts, retention rates, etc.
    """
    try:
        from sqlalchemy import select, func, and_
        from backend.models.memory import Memory

        # Count episodic memories
        episodic_result = await db.execute(
            select(func.count(Memory.id))
            .where(
                and_(
                    Memory.collection_id == collection_id,
                    Memory.memory_type == "episodic",
                )
            )
        )
        episodic_count = episodic_result.scalar() or 0

        # Count semantic memories
        semantic_result = await db.execute(
            select(func.count(Memory.id))
            .where(
                and_(
                    Memory.collection_id == collection_id,
                    Memory.memory_type == "semantic",
                )
            )
        )
        semantic_count = semantic_result.scalar() or 0

        # Count archived
        archived_result = await db.execute(
            select(func.count(Memory.id))
            .where(
                and_(
                    Memory.collection_id == collection_id,
                    Memory.is_archived == True,
                )
            )
        )
        archived_count = archived_result.scalar() or 0

        return {
            "status": "success",
            "collection_id": collection_id,
            "stats": {
                "episodic_count": episodic_count,
                "semantic_count": semantic_count,
                "archived_count": archived_count,
                "total_count": episodic_count + semantic_count,
                "consolidation_ratio": semantic_count / max(episodic_count + semantic_count, 1),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get consolidation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule/{collection_id}")
async def schedule_consolidation(
    collection_id: str,
    interval_hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Schedule periodic consolidation for collection.

    Sets up automated consolidation at specified interval.
    """
    try:
        # In production, this would integrate with a task scheduler like Celery
        # For now, we'll just return success
        return {
            "status": "success",
            "message": f"Consolidation scheduled every {interval_hours} hours",
            "collection_id": collection_id,
            "interval_hours": interval_hours,
            "note": "Scheduler integration pending",
        }

    except Exception as e:
        logger.error(f"Failed to schedule consolidation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
