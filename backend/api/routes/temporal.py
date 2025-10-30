"""
Temporal Knowledge Graph API Routes.

Endpoints for managing time-varying facts and relationships.
"""
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.services.temporal_graph import get_temporal_graph
from backend.core.logging_config import logger


router = APIRouter(prefix="/temporal", tags=["Temporal Knowledge"])


class AddFactRequest(BaseModel):
    """Request model for adding temporal fact."""
    subject: str
    subject_type: str = "ENTITY"
    predicate: str
    object: str
    object_type: str = "ENTITY"
    valid_from: datetime
    valid_until: Optional[datetime] = None
    observed_at: Optional[datetime] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_memory_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryAtTimeRequest(BaseModel):
    """Request model for point-in-time queries."""
    subject: str
    predicate: Optional[str] = None
    timestamp: datetime


class InvalidateFactRequest(BaseModel):
    """Request model for invalidating facts."""
    subject: str
    predicate: str
    object: str


@router.post("/facts")
async def add_fact(
    request: AddFactRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Add temporal fact to knowledge graph.

    Creates a bi-temporal fact with validity period and observation time.
    """
    try:
        graph = await get_temporal_graph()

        await graph.add_fact(
            subject=request.subject,
            subject_type=request.subject_type,
            predicate=request.predicate,
            object=request.object,
            object_type=request.object_type,
            user_id=current_user.id,
            valid_from=request.valid_from,
            valid_until=request.valid_until,
            observed_at=request.observed_at or datetime.utcnow(),
            confidence=request.confidence,
            source_memory_id=request.source_memory_id,
            metadata=request.metadata or {},
        )

        return {
            "status": "success",
            "message": f"Added fact: {request.subject} {request.predicate} {request.object}",
            "valid_from": request.valid_from.isoformat(),
            "valid_until": request.valid_until.isoformat() if request.valid_until else "ongoing",
        }

    except Exception as e:
        logger.error(f"Failed to add temporal fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facts/query-at-time")
async def query_at_time(
    request: QueryAtTimeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Query facts at specific point in time.

    Returns facts that were valid at the given timestamp.
    """
    try:
        graph = await get_temporal_graph()

        facts = await graph.query_at_time(
            subject=request.subject,
            predicate=request.predicate,
            timestamp=request.timestamp,
        )

        return {
            "status": "success",
            "timestamp": request.timestamp.isoformat(),
            "count": len(facts),
            "facts": [
                {
                    "subject": f["subject"],
                    "predicate": f["predicate"],
                    "object": f["object"],
                    "valid_from": f["valid_from"],
                    "valid_until": f["valid_until"],
                    "confidence": f.get("confidence", 1.0),
                }
                for f in facts
            ],
        }

    except Exception as e:
        logger.error(f"Failed to query at time: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facts/invalidate")
async def invalidate_fact(
    request: InvalidateFactRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Invalidate a fact.

    Sets valid_until to mark the fact as no longer true.
    """
    try:
        graph = await get_temporal_graph()

        await graph.invalidate_fact(
            subject=request.subject,
            predicate=request.predicate,
            object=request.object,
        )

        return {
            "status": "success",
            "message": f"Invalidated: {request.subject} {request.predicate} {request.object}",
        }

    except Exception as e:
        logger.error(f"Failed to invalidate fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/facts/conflicts")
async def get_conflicts(
    subject: str,
    predicate: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get conflicting facts.

    Returns facts with overlapping validity periods that may conflict.
    """
    try:
        graph = await get_temporal_graph()

        conflicts = await graph.resolve_conflicts(
            subject=subject,
            predicate=predicate,
        )

        return {
            "status": "success",
            "subject": subject,
            "predicate": predicate,
            "count": len(conflicts),
            "conflicts": [
                {
                    "object": c["object"],
                    "valid_from": c["valid_from"],
                    "valid_until": c["valid_until"],
                    "confidence": c.get("confidence", 1.0),
                }
                for c in conflicts
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/facts/history")
async def get_fact_history(
    subject: str,
    predicate: str,
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get complete history of facts.

    Returns all facts (current and past) for subject-predicate pair.
    """
    try:
        graph = await get_temporal_graph()

        history = await graph.get_fact_history(
            subject=subject,
            predicate=predicate,
            limit=limit,
        )

        return {
            "status": "success",
            "subject": subject,
            "predicate": predicate,
            "count": len(history),
            "history": [
                {
                    "object": h["object"],
                    "valid_from": h["valid_from"],
                    "valid_until": h["valid_until"],
                    "observed_at": h.get("observed_at"),
                    "confidence": h.get("confidence", 1.0),
                    "is_current": h.get("is_current", False),
                }
                for h in history
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get fact history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/merge")
async def merge_entities(
    entity1: str,
    entity2: str,
    keep_entity: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Merge two entities.

    Combines all relationships from entity2 into entity1 or entity2 (based on keep_entity).
    """
    try:
        if keep_entity not in [entity1, entity2]:
            raise HTTPException(
                status_code=400,
                detail="keep_entity must be one of the entities being merged",
            )

        graph = await get_temporal_graph()

        result = await graph.merge_entities(
            entity1=entity1,
            entity2=entity2,
            keep_entity=keep_entity,
        )

        return {
            "status": "success",
            "message": f"Merged {entity1} and {entity2} into {keep_entity}",
            "relationships_merged": result.get("relationships_merged", 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to merge entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
