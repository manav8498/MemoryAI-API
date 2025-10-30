"""
World Model API Routes.

Endpoints for planning and imagination using world models.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.agents.world_model import get_world_model
from backend.core.logging_config import logger


router = APIRouter(prefix="/world-model", tags=["World Model & Planning"])


class ImagineRetrievalRequest(BaseModel):
    """Request model for imagining retrieval."""
    query: str
    memory_state: Dict[str, Any] = Field(default_factory=dict)


class PlanQueryRequest(BaseModel):
    """Request model for planning optimal query."""
    user_intent: str
    num_variations: int = Field(default=5, ge=1, le=10)


class TrainWorldModelRequest(BaseModel):
    """Request model for training world model."""
    trajectories: List[Dict[str, Any]]
    num_epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1, le=128)


@router.post("/imagine")
async def imagine_retrieval(
    request: ImagineRetrievalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Simulate retrieval outcome without executing.

    Uses world model to predict what would be retrieved and quality.
    """
    try:
        world_model = get_world_model()

        predicted_memories, predicted_quality = await world_model.imagine_retrieval(
            query=request.query,
            memory_state=request.memory_state,
        )

        return {
            "status": "success",
            "query": request.query,
            "predicted_memories": predicted_memories,
            "predicted_quality": predicted_quality,
            "message": "Simulated retrieval complete",
        }

    except Exception as e:
        logger.error(f"Failed to imagine retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plan-query")
async def plan_optimal_query(
    request: PlanQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Plan optimal query formulation.

    Generates variations and simulates each to find best query.
    """
    try:
        world_model = get_world_model()

        optimal_query = await world_model.plan_optimal_query(
            user_intent=request.user_intent,
            num_variations=request.num_variations,
        )

        return {
            "status": "success",
            "user_intent": request.user_intent,
            "optimal_query": optimal_query,
            "num_variations_tested": request.num_variations,
            "message": "Query planning complete",
        }

    except Exception as e:
        logger.error(f"Failed to plan query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plan-actions")
async def plan_action_sequence(
    initial_state: List[float],
    goal_state: List[float],
    max_steps: int = Query(default=5, ge=1, le=20),
    beam_width: int = Query(default=3, ge=1, le=10),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Plan sequence of actions to reach goal state.

    Uses beam search in latent space to find optimal action sequence.
    """
    try:
        import torch

        world_model = get_world_model()

        initial_tensor = torch.tensor(initial_state, dtype=torch.float32, device=world_model.device).unsqueeze(0)
        goal_tensor = torch.tensor(goal_state, dtype=torch.float32, device=world_model.device).unsqueeze(0)

        action_sequence = world_model.plan_action_sequence(
            initial_state=initial_tensor,
            goal_state=goal_tensor,
            max_steps=max_steps,
            beam_width=beam_width,
        )

        action_names = ["ADD", "UPDATE", "DELETE", "NOOP"]
        action_sequence_names = [action_names[idx] for idx in action_sequence]

        return {
            "status": "success",
            "action_sequence": action_sequence_names,
            "action_indices": action_sequence,
            "max_steps": max_steps,
            "beam_width": beam_width,
            "message": "Action planning complete",
        }

    except Exception as e:
        logger.error(f"Failed to plan actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_world_model(
    request: TrainWorldModelRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Train world model on trajectories.

    Learns dynamics and reward models from collected data.
    """
    try:
        world_model = get_world_model()

        metrics = world_model.train_world_model(
            trajectories=request.trajectories,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
        )

        return {
            "status": "success",
            "metrics": metrics,
            "trajectories_count": len(request.trajectories),
            "epochs": request.num_epochs,
            "message": "World model training complete",
        }

    except Exception as e:
        logger.error(f"Failed to train world model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_world_model(
    path: str = "models/world_model.pt",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Save world model to disk.

    Saves dynamics and reward models.
    """
    try:
        world_model = get_world_model()
        world_model.save(path)

        return {
            "status": "success",
            "path": path,
            "message": f"World model saved to {path}",
        }

    except Exception as e:
        logger.error(f"Failed to save world model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_world_model(
    path: str = "models/world_model.pt",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Load world model from disk.

    Restores dynamics and reward models.
    """
    try:
        world_model = get_world_model()
        world_model.load(path)

        return {
            "status": "success",
            "path": path,
            "message": f"World model loaded from {path}",
        }

    except Exception as e:
        logger.error(f"Failed to load world model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
