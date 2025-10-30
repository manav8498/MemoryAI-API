"""
Procedural Memory API Routes.

Endpoints for managing learned procedures and skills.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.services.procedural_memory import get_procedural_memory_manager
from backend.core.logging_config import logger


router = APIRouter(prefix="/procedures", tags=["Procedural Memory"])


class CreateProcedureRequest(BaseModel):
    """Request model for creating a procedure."""
    name: str = Field(..., min_length=1, max_length=200)
    condition: Dict[str, Any]
    action: Dict[str, Any]
    description: Optional[str] = None
    category: Optional[str] = None
    collection_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ExecuteProcedureRequest(BaseModel):
    """Request model for executing a procedure."""
    input_state: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None


class LearnProcedureRequest(BaseModel):
    """Request model for learning from examples."""
    name: str
    examples: List[Dict[str, Any]]
    category: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request model for procedure feedback."""
    execution_id: str
    success: bool
    reward: Optional[float] = None
    user_feedback: Optional[float] = Field(None, ge=-1.0, le=1.0)


@router.post("")
async def create_procedure(
    request: CreateProcedureRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new procedure.

    Stores a learned procedure (IF-THEN rule) in procedural memory.
    """
    try:
        manager = get_procedural_memory_manager(db)

        procedure = await manager.create_procedure(
            user_id=current_user.id,
            name=request.name,
            condition=request.condition,
            action=request.action,
            description=request.description,
            category=request.category,
            collection_id=request.collection_id,
            parameters=request.parameters,
        )

        return {
            "status": "success",
            "procedure_id": procedure.id,
            "name": procedure.name,
            "message": "Procedure created successfully",
        }

    except Exception as e:
        logger.error(f"Failed to create procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def get_user_procedures(
    category: Optional[str] = None,
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all procedures for current user.

    Optionally filter by category and active status.
    """
    try:
        manager = get_procedural_memory_manager(db)

        procedures = await manager.get_user_procedures(
            user_id=current_user.id,
            category=category,
            active_only=active_only,
        )

        return {
            "status": "success",
            "count": len(procedures),
            "procedures": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "category": p.category,
                    "success_rate": p.success_rate,
                    "usage_count": p.usage_count,
                    "confidence": p.confidence,
                    "is_active": p.is_active,
                    "created_at": p.created_at.isoformat(),
                }
                for p in procedures
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get procedures: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{procedure_id}")
async def get_procedure(
    procedure_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get specific procedure by ID.

    Returns full procedure details including condition and action.
    """
    try:
        manager = get_procedural_memory_manager(db)

        procedure = await manager.get_procedure(
            procedure_id=procedure_id,
            user_id=current_user.id,
        )

        if not procedure:
            raise HTTPException(status_code=404, detail="Procedure not found")

        return {
            "status": "success",
            "procedure": {
                "id": procedure.id,
                "name": procedure.name,
                "description": procedure.description,
                "category": procedure.category,
                "condition": procedure.condition,
                "action": procedure.action,
                "parameters": procedure.parameters,
                "success_rate": procedure.success_rate,
                "usage_count": procedure.usage_count,
                "confidence": procedure.confidence,
                "strength": procedure.strength,
                "is_active": procedure.is_active,
                "is_verified": procedure.is_verified,
                "created_at": procedure.created_at.isoformat(),
                "last_used_at": procedure.last_used_at.isoformat() if procedure.last_used_at else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{procedure_id}/execute")
async def execute_procedure(
    procedure_id: str,
    request: ExecuteProcedureRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Execute a procedure.

    Runs the procedure with given input state and parameters.
    """
    try:
        manager = get_procedural_memory_manager(db)

        result = await manager.execute_procedure(
            procedure_id=procedure_id,
            user_id=current_user.id,
            input_state=request.input_state,
            parameters=request.parameters,
        )

        return {
            "status": "success",
            "execution_result": result,
        }

    except Exception as e:
        logger.error(f"Failed to execute procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn")
async def learn_procedure(
    request: LearnProcedureRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Learn new procedure from examples.

    Analyzes examples and creates a new procedure.
    """
    try:
        manager = get_procedural_memory_manager(db)

        procedure = await manager.learn_procedure_from_examples(
            user_id=current_user.id,
            examples=request.examples,
            name=request.name,
            category=request.category,
        )

        if not procedure:
            raise HTTPException(status_code=400, detail="Failed to learn procedure from examples")

        return {
            "status": "success",
            "procedure_id": procedure.id,
            "name": procedure.name,
            "message": f"Learned procedure from {len(request.examples)} examples",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to learn procedure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def record_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Record feedback for procedure execution.

    Updates success rate and confidence based on feedback.
    """
    try:
        manager = get_procedural_memory_manager(db)

        success = await manager.record_feedback(
            execution_id=request.execution_id,
            success=request.success,
            reward=request.reward,
            user_feedback=request.user_feedback,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Execution not found")

        return {
            "status": "success",
            "message": "Feedback recorded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match")
async def find_matching_procedures(
    context: Dict[str, Any],
    limit: int = Query(default=10, ge=1, le=50),
    min_confidence: float = Query(default=0.3, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Find procedures matching given context.

    Returns procedures whose conditions match the context.
    """
    try:
        manager = get_procedural_memory_manager(db)

        procedures = await manager.find_matching_procedures(
            user_id=current_user.id,
            context=context,
            limit=limit,
            min_confidence=min_confidence,
        )

        return {
            "status": "success",
            "count": len(procedures),
            "procedures": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "condition": p.condition,
                    "action": p.action,
                    "success_rate": p.success_rate,
                    "confidence": p.confidence,
                }
                for p in procedures
            ],
        }

    except Exception as e:
        logger.error(f"Failed to find matching procedures: {e}")
        raise HTTPException(status_code=500, detail=str(e))
