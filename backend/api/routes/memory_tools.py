"""
Self-Editing Memory Tools API Routes.

Endpoints for agent-controlled memory manipulation (Letta/MemGPT style).
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.agents.memory_tools import MemoryTools
from backend.core.logging_config import logger


router = APIRouter(prefix="/memory-tools", tags=["Self-Editing Memory"])


class ReplaceMemoryRequest(BaseModel):
    """Request model for replacing memory content."""
    memory_id: str
    old_content: str
    new_content: str
    collection_id: str


class InsertMemoryRequest(BaseModel):
    """Request model for inserting new memory."""
    content: str
    collection_id: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class DeleteMemoryRequest(BaseModel):
    """Request model for deleting memory."""
    memory_id: str
    collection_id: str


class RethinkMemoryRequest(BaseModel):
    """Request model for rethinking memory."""
    memory_id: str
    collection_id: str


class ConsolidateMemoriesRequest(BaseModel):
    """Request model for consolidating memories."""
    memory_ids: List[str]
    collection_id: str


@router.post("/replace")
async def replace_memory_content(
    request: ReplaceMemoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Replace content in existing memory.

    Agent-controlled memory editing.
    """
    try:
        tools = MemoryTools(
            db=db,
            user_id=current_user.id,
            collection_id=request.collection_id,
        )

        result = await tools.memory_replace(
            memory_id=request.memory_id,
            old_content=request.old_content,
            new_content=request.new_content,
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to replace memory"),
            )

        return {
            "status": "success",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to replace memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insert")
async def insert_memory(
    request: InsertMemoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Insert new memory via agent tool.

    Creates memory marked as agent-inserted.
    """
    try:
        tools = MemoryTools(
            db=db,
            user_id=current_user.id,
            collection_id=request.collection_id,
        )

        result = await tools.memory_insert(
            content=request.content,
            importance=request.importance,
            metadata=request.metadata,
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to insert memory"),
            )

        return {
            "status": "success",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to insert memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def delete_memory(
    request: DeleteMemoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete memory via agent tool.

    Removes from all storage systems.
    """
    try:
        tools = MemoryTools(
            db=db,
            user_id=current_user.id,
            collection_id=request.collection_id,
        )

        result = await tools.memory_delete(memory_id=request.memory_id)

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to delete memory"),
            )

        return {
            "status": "success",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rethink")
async def rethink_memory(
    request: RethinkMemoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Reflect on and re-evaluate a memory.

    Uses LLM to analyze memory accuracy and suggest improvements.
    """
    try:
        tools = MemoryTools(
            db=db,
            user_id=current_user.id,
            collection_id=request.collection_id,
        )

        result = await tools.memory_rethink(memory_id=request.memory_id)

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to rethink memory"),
            )

        return {
            "status": "success",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rethink memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_memories(
    query: str,
    collection_id: str,
    limit: int = Query(default=5, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Search memories (for agent self-access).

    Allows agent to search its own memory.
    """
    try:
        tools = MemoryTools(
            db=db,
            user_id=current_user.id,
            collection_id=collection_id,
        )

        result = await tools.memory_search(query=query, limit=limit)

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to search memories"),
            )

        return {
            "status": "success",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate")
async def consolidate_multiple_memories(
    request: ConsolidateMemoriesRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Consolidate multiple memories into one.

    Uses LLM to merge memories, archives originals.
    """
    try:
        if len(request.memory_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 memories to consolidate",
            )

        tools = MemoryTools(
            db=db,
            user_id=current_user.id,
            collection_id=request.collection_id,
        )

        result = await tools.memory_consolidate(memory_ids=request.memory_ids)

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to consolidate memories"),
            )

        return {
            "status": "success",
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to consolidate memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))
