"""
User management endpoints.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.core.logging_config import logger


router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class UserUpdate(BaseModel):
    """User update schema."""
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    full_name: str
    is_active: bool
    is_verified: bool
    tier: str

    class Config:
        from_attributes = True


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """
    Get current user profile.
    """
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update current user profile.
    """
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name

    await db.commit()
    await db.refresh(current_user)

    logger.info(f"User profile updated: {current_user.email}")

    return current_user


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete current user account.

    This will cascade delete all collections, memories, and API keys.
    """
    await db.delete(current_user)
    await db.commit()

    logger.info(f"User account deleted: {current_user.email}")

    return None
