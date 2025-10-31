"""
Memory management endpoints.
"""
import uuid
import hashlib
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.models.collection import Collection
from backend.models.memory import Memory, MemoryMetadata
from backend.core.logging_config import logger


router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class MemoryCreate(BaseModel):
    """Memory creation schema."""
    content: str
    collection_id: str
    importance: float = 0.5
    source_type: str = "text"
    source_reference: Optional[str] = None
    metadata: Dict[str, Any] = {}


class MemoryUpdate(BaseModel):
    """Memory update schema."""
    content: Optional[str] = None
    importance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    """Memory response schema."""
    id: str
    collection_id: str
    content: str
    importance: float
    source_type: str
    source_reference: Optional[str]
    access_count: int
    last_accessed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MemoryWithMetadata(MemoryResponse):
    """Memory response with metadata."""
    metadata: Dict[str, Any] = {}


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_data: MemoryCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new memory.

    The memory will be:
    1. Stored in PostgreSQL
    2. Embedded and stored in Milvus (vector DB)
    3. Processed for entities and stored in Neo4j (knowledge graph)
    """
    # Verify collection belongs to user
    result = await db.execute(
        select(Collection).where(
            Collection.id == memory_data.collection_id,
            Collection.user_id == current_user.id,
        )
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    # Create content hash for deduplication
    content_hash = hashlib.sha256(memory_data.content.encode()).hexdigest()

    # Create memory
    memory = Memory(
        id=str(uuid.uuid4()),
        collection_id=memory_data.collection_id,
        content=memory_data.content,
        content_hash=content_hash,
        importance=memory_data.importance,
        source_type=memory_data.source_type,
        source_reference=memory_data.source_reference,
    )

    # Create metadata
    metadata = MemoryMetadata(
        id=str(uuid.uuid4()),
        memory_id=memory.id,
        custom_metadata=memory_data.metadata,
    )

    db.add(memory)
    db.add(metadata)

    # Update collection memory count atomically to prevent race conditions
    await db.execute(
        update(Collection)
        .where(Collection.id == collection.id)
        .values(memory_count=Collection.memory_count + 1)
    )

    await db.commit()
    await db.refresh(memory)

    logger.info(f"Memory created in collection {collection.name}")

    # Trigger async processing pipeline in background
    try:
        from backend.services.pipeline.memory_ingestion import process_memory_async
        import asyncio

        # Process memory: embeddings → Milvus, entities → Neo4j
        # Note: This runs in background, errors logged but don't block response
        task = asyncio.create_task(process_memory_async(memory.id, str(memory.content)))

        # Add exception handler to catch uncaught exceptions in background task
        def handle_task_exception(task):
            try:
                task.result()  # This will raise if the task had an exception
            except Exception as e:
                logger.error(
                    f"Background task failed for memory {memory.id}: {e}",
                    exc_info=True
                )

        task.add_done_callback(handle_task_exception)
    except Exception as e:
        logger.warning(f"Failed to trigger async processing: {e}")
        # Continue - memory still created in DB

    return memory


@router.get("", response_model=List[MemoryResponse])
async def list_memories(
    collection_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
):
    """
    List memories.

    Optionally filter by collection_id.
    """
    # Build query
    query = select(Memory).join(Collection).where(Collection.user_id == current_user.id)

    if collection_id:
        query = query.where(Memory.collection_id == collection_id)

    query = query.offset(skip).limit(limit).order_by(Memory.created_at.desc())

    result = await db.execute(query)
    memories = result.scalars().all()

    return memories


@router.get("/{memory_id}", response_model=MemoryWithMetadata)
async def get_memory(
    memory_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific memory by ID.
    """
    result = await db.execute(
        select(Memory)
        .join(Collection)
        .where(
            Memory.id == memory_id,
            Collection.user_id == current_user.id,
        )
    )
    memory = result.scalar_one_or_none()

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    # Update access stats
    memory.access_count += 1
    memory.last_accessed_at = datetime.utcnow()
    await db.commit()

    # Get metadata
    result = await db.execute(
        select(MemoryMetadata).where(MemoryMetadata.memory_id == memory_id)
    )
    metadata = result.scalar_one_or_none()

    # Build response
    response = MemoryWithMetadata(
        id=memory.id,
        collection_id=memory.collection_id,
        content=memory.content,
        importance=memory.importance,
        source_type=memory.source_type,
        source_reference=memory.source_reference,
        access_count=memory.access_count,
        last_accessed_at=memory.last_accessed_at,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        metadata=metadata.custom_metadata if metadata else {},
    )

    return response


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    memory_update: MemoryUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a memory.
    """
    result = await db.execute(
        select(Memory)
        .join(Collection)
        .where(
            Memory.id == memory_id,
            Collection.user_id == current_user.id,
        )
    )
    memory = result.scalar_one_or_none()

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    # Update fields
    if memory_update.content is not None:
        memory.content = memory_update.content
        memory.content_hash = hashlib.sha256(memory_update.content.encode()).hexdigest()

    if memory_update.importance is not None:
        memory.importance = memory_update.importance

    if memory_update.metadata is not None:
        result = await db.execute(
            select(MemoryMetadata).where(MemoryMetadata.memory_id == memory_id)
        )
        metadata = result.scalar_one_or_none()
        if metadata:
            metadata.custom_metadata = memory_update.metadata

    await db.commit()
    await db.refresh(memory)

    logger.info(f"Memory updated: {memory_id}")

    # TODO: Re-process updated memory

    return memory


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a memory.

    This will also remove it from Milvus and Neo4j.
    """
    result = await db.execute(
        select(Memory)
        .join(Collection)
        .where(
            Memory.id == memory_id,
            Collection.user_id == current_user.id,
        )
    )
    memory = result.scalar_one_or_none()

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    # Update collection count
    collection_id = memory.collection_id
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()
    if collection:
        collection.memory_count = max(0, collection.memory_count - 1)

    await db.delete(memory)
    await db.commit()

    logger.info(f"Memory deleted: {memory_id}")

    # Delete from Milvus vector store
    try:
        from backend.services.vector_store import delete_memory_from_vector_store
        await delete_memory_from_vector_store(memory_id, collection_id)
        logger.info(f"Memory {memory_id} deleted from Milvus")
    except Exception as e:
        logger.warning(f"Failed to delete memory from Milvus: {e}")

    # Delete from Neo4j knowledge graph
    try:
        from backend.services.knowledge_graph import delete_memory_from_graph
        await delete_memory_from_graph(memory_id)
        logger.info(f"Memory {memory_id} deleted from Neo4j")
    except Exception as e:
        logger.warning(f"Failed to delete memory from Neo4j: {e}")

    return None
