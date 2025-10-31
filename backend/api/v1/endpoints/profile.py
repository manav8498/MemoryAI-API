"""
User Profile API Endpoints

Similar to SuperMemory's /v4/profile endpoint.
Provides instant access to user profile facts with optional search integration.
"""
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.models.user_profile import ProfileType, ProfileCategory
from backend.services.profile_manager import get_profile_manager
from backend.services.conversation_starters import generate_conversation_starters
from backend.core.logging_config import logger


router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================


class ProfileRequest(BaseModel):
    """Profile retrieval request (similar to SuperMemory's POST /v4/profile)."""
    q: Optional[str] = Field(None, description="Optional search query to combine with profile")
    collection_id: Optional[str] = Field(None, description="Optional collection filter for search")
    include_static: bool = Field(True, description="Include static facts")
    include_dynamic: bool = Field(True, description="Include dynamic facts")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
    search_limit: int = Field(10, ge=1, le=100, description="Number of search results if query provided")


class AddFactRequest(BaseModel):
    """Request to add/update a profile fact."""
    fact_key: str = Field(..., min_length=1, max_length=200)
    fact_value: str = Field(..., min_length=1)
    profile_type: str = Field(..., pattern="^(static|dynamic)$")
    category: str
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    importance: float = Field(0.7, ge=0.0, le=1.0)


class ProfileResponse(BaseModel):
    """Profile response."""
    profile: Dict[str, Any]
    searchResults: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("", response_model=ProfileResponse)
async def get_user_profile_endpoint(
    request: ProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user profile with optional search integration.

    This is the main profile endpoint similar to SuperMemory's POST /v4/profile.

    **Features:**
    - Instant access to user profile facts (static + dynamic)
    - Optional search query integration
    - Combined context for AI assistants
    - Performance: ~50-100ms vs 200-500ms for multiple queries

    **Use Cases:**
    - AI assistants maintaining user preferences
    - Customer support with instant user history
    - Personalized content adaptation
    - Development tools with user context

    **Example:**
    ```python
    # Get profile only
    POST /v1/profile
    {
        "include_static": true,
        "include_dynamic": true
    }

    # Get profile + search results
    POST /v1/profile
    {
        "q": "job applications",
        "collection_id": "...",
        "search_limit": 10
    }
    ```
    """
    try:
        manager = get_profile_manager(db)

        # If query provided, combine profile with search
        if request.q:
            result = await manager.get_profile_with_search(
                user_id=current_user.id,
                query=request.q,
                collection_id=request.collection_id,
                limit=request.search_limit,
            )
            return ProfileResponse(
                profile=result["profile"],
                searchResults=result["searchResults"],
                metadata=result["metadata"],
            )
        else:
            # Profile only
            profile = await manager.get_user_profile(
                user_id=current_user.id,
                include_dynamic=request.include_dynamic,
                include_static=request.include_static,
                min_confidence=request.min_confidence,
            )
            return ProfileResponse(
                profile=profile,
                searchResults=None,
                metadata=profile["metadata"],
            )

    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile retrieval failed: {str(e)}"
        )


@router.post("/facts", status_code=status.HTTP_201_CREATED)
async def add_profile_fact(
    request: AddFactRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Add or update a profile fact.

    Allows manual addition/updating of user profile facts.
    Automatically tracked in profile operations history.
    """
    try:
        manager = get_profile_manager(db)

        # Map strings to enums
        profile_type = ProfileType.STATIC if request.profile_type == "static" else ProfileType.DYNAMIC

        category_map = {
            "role": ProfileCategory.ROLE,
            "expertise": ProfileCategory.EXPERTISE,
            "preference": ProfileCategory.PREFERENCE,
            "education": ProfileCategory.EDUCATION,
            "experience": ProfileCategory.EXPERIENCE,
            "current_project": ProfileCategory.CURRENT_PROJECT,
            "recent_skill": ProfileCategory.RECENT_SKILL,
            "temporary_state": ProfileCategory.TEMPORARY_STATE,
            "goal": ProfileCategory.GOAL,
            "interest": ProfileCategory.INTEREST,
            "communication": ProfileCategory.COMMUNICATION,
        }
        category = category_map.get(request.category, ProfileCategory.OTHER)

        fact = await manager.add_or_update_fact(
            user_id=current_user.id,
            fact_key=request.fact_key,
            fact_value=request.fact_value,
            profile_type=profile_type,
            category=category,
            confidence=request.confidence,
            importance=request.importance,
            trigger_type="user_input",
        )

        return {
            "status": "success",
            "fact_id": fact.id,
            "fact_key": fact.fact_key,
            "message": "Fact added/updated successfully",
        }

    except Exception as e:
        logger.error(f"Failed to add fact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/facts")
async def get_all_facts(
    profile_type: Optional[str] = Query(None, pattern="^(static|dynamic)$"),
    category: Optional[str] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all profile facts with optional filters.

    Query parameters:
    - profile_type: Filter by "static" or "dynamic"
    - category: Filter by category
    - min_confidence: Minimum confidence threshold
    """
    try:
        manager = get_profile_manager(db)

        include_static = profile_type is None or profile_type == "static"
        include_dynamic = profile_type is None or profile_type == "dynamic"

        profile = await manager.get_user_profile(
            user_id=current_user.id,
            include_dynamic=include_dynamic,
            include_static=include_static,
            min_confidence=min_confidence,
        )

        # Filter by category if specified
        if category:
            profile["static"] = [f for f in profile["static"] if f["category"] == category]
            profile["dynamic"] = [f for f in profile["dynamic"] if f["category"] == category]

        return profile

    except Exception as e:
        logger.error(f"Failed to get facts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/facts/{fact_key}")
async def delete_profile_fact(
    fact_key: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a profile fact by key.
    """
    try:
        manager = get_profile_manager(db)

        deleted = await manager.delete_fact(
            user_id=current_user.id,
            fact_key=fact_key,
        )

        if deleted:
            return {
                "status": "success",
                "message": f"Fact '{fact_key}' deleted",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Fact '{fact_key}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete fact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/history")
async def get_profile_history(
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get profile operation history.

    Shows all additions, updates, and removals of profile facts.
    Useful for auditing and understanding profile evolution.
    """
    try:
        manager = get_profile_manager(db)

        history = await manager.get_profile_history(
            user_id=current_user.id,
            limit=limit,
        )

        return {
            "status": "success",
            "history": history,
            "total": len(history),
        }

    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/snapshot")
async def create_profile_snapshot(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a snapshot of the current profile.

    Useful for versioning and rollback capabilities.
    """
    try:
        manager = get_profile_manager(db)

        snapshot = await manager.create_snapshot(
            user_id=current_user.id,
            trigger_reason="user_request",
        )

        return {
            "status": "success",
            "snapshot_id": snapshot.id,
            "created_at": snapshot.created_at.isoformat(),
            "message": "Profile snapshot created",
        }

    except Exception as e:
        logger.error(f"Failed to create snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/stats")
async def get_profile_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get profile statistics.

    Returns counts by type, category, and average confidence.
    """
    try:
        manager = get_profile_manager(db)

        stats = await manager.get_profile_stats(user_id=current_user.id)

        return {
            "status": "success",
            "stats": stats,
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/cleanup")
async def cleanup_old_dynamic_facts(
    older_than_days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Clean up old dynamic facts.

    Dynamic facts are temporary and should be cleaned up periodically.
    This removes dynamic facts older than the specified number of days.
    """
    try:
        manager = get_profile_manager(db)

        removed_count = await manager.cleanup_old_dynamic_facts(
            user_id=current_user.id,
            older_than_days=older_than_days,
        )

        return {
            "status": "success",
            "removed_count": removed_count,
            "message": f"Cleaned up {removed_count} old dynamic facts",
        }

    except Exception as e:
        logger.error(f"Failed to cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/starters")
async def get_conversation_starters(
    count: int = Query(5, ge=1, le=10),
    provider: str = Query("gemini", pattern="^(gemini|openai|anthropic)$"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get personalized conversation starters based on user profile.

    Generates intelligent, context-aware questions and topics based on:
    - User's expertise and interests
    - Current projects and goals
    - Recent activities
    - Professional role

    **Example Response:**
    ```json
    {
        "starters": [
            {
                "question": "How is progress on building Memory AI API going?",
                "context": "Based on your current project",
                "type": "llm_generated"
            },
            ...
        ]
    }
    ```
    """
    try:
        starters = await generate_conversation_starters(
            db=db,
            user_id=current_user.id,
            count=count,
            provider=provider
        )

        return {
            "status": "success",
            "starters": starters,
            "count": len(starters)
        }

    except Exception as e:
        logger.error(f"Failed to generate starters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
