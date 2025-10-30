"""
Collection management endpoints.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.models.collection import Collection
from backend.core.logging_config import logger


router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class CollectionCreate(BaseModel):
    """Collection creation schema."""
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = {}


class CollectionUpdate(BaseModel):
    """Collection update schema."""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CollectionResponse(BaseModel):
    """Collection response schema."""
    id: str
    name: str
    description: Optional[str]
    is_active: bool
    memory_count: int
    metadata: Dict[str, Any] = Field(validation_alias='custom_metadata')
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection_data: CollectionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new memory collection.

    Collections allow you to organize memories into logical groups.
    """
    collection = Collection(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        name=collection_data.name,
        description=collection_data.description,
        metadata=collection_data.metadata,
    )

    db.add(collection)
    await db.commit()
    await db.refresh(collection)

    logger.info(f"Collection created: {collection.name} (user: {current_user.email})")

    return collection


@router.get("", response_model=List[CollectionResponse])
async def list_collections(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
):
    """
    List all collections for the current user.
    """
    result = await db.execute(
        select(Collection)
        .where(Collection.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(Collection.created_at.desc())
    )
    collections = result.scalars().all()

    return collections


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific collection by ID.
    """
    result = await db.execute(
        select(Collection).where(
            Collection.id == collection_id,
            Collection.user_id == current_user.id,
        )
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    return collection


@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str,
    collection_update: CollectionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a collection.
    """
    result = await db.execute(
        select(Collection).where(
            Collection.id == collection_id,
            Collection.user_id == current_user.id,
        )
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    # Update fields
    if collection_update.name is not None:
        collection.name = collection_update.name
    if collection_update.description is not None:
        collection.description = collection_update.description
    if collection_update.metadata is not None:
        collection.metadata = collection_update.metadata

    await db.commit()
    await db.refresh(collection)

    logger.info(f"Collection updated: {collection.name}")

    return collection


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a collection.

    This will cascade delete all memories in the collection.
    """
    result = await db.execute(
        select(Collection).where(
            Collection.id == collection_id,
            Collection.user_id == current_user.id,
        )
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )

    await db.delete(collection)
    await db.commit()

    logger.info(f"Collection deleted: {collection.name}")

    return None
