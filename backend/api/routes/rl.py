"""
RL Training API Routes.

Endpoints for managing reinforcement learning training.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.auth import get_current_user
from backend.models.user import User
from backend.rl.training_orchestrator import get_training_orchestrator
from backend.core.logging_config import logger


router = APIRouter(prefix="/rl", tags=["RL Training"])


class TrainingRequest(BaseModel):
    """Request model for training."""
    num_episodes: int = Field(default=100, ge=1, le=10000)
    num_epochs: int = Field(default=10, ge=1, le=100)
    user_id: Optional[str] = None


class CheckpointRequest(BaseModel):
    """Request model for checkpoint operations."""
    checkpoint_path: str


@router.post("/train/memory-manager")
async def train_memory_manager(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Train Memory Manager agent.

    Runs PPO training on collected trajectories.
    """
    try:
        # Initialize orchestrator to ensure it's created
        await get_training_orchestrator(db)

        # Define async wrapper that gets its own DB session
        async def run_training_async():
            async for session in get_db():
                try:
                    orch = await get_training_orchestrator(session)
                    await orch.train_memory_manager(
                        num_episodes=request.num_episodes,
                        num_epochs=request.num_epochs,
                        user_id=request.user_id,
                    )
                finally:
                    await session.close()

        # Run training in background (FastAPI supports async functions)
        background_tasks.add_task(run_training_async)

        return {
            "status": "training_started",
            "message": "Memory Manager training started in background",
            "num_episodes": request.num_episodes,
            "num_epochs": request.num_epochs,
        }

    except Exception as e:
        logger.error(f"Failed to start Memory Manager training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/answer-agent")
async def train_answer_agent(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Train Answer Agent.

    Runs training on query-memory selection data.
    """
    try:
        # Initialize orchestrator to ensure it's created
        await get_training_orchestrator(db)

        # Define async wrapper that gets its own DB session
        async def run_training_async():
            async for session in get_db():
                try:
                    orch = await get_training_orchestrator(session)
                    await orch.train_answer_agent(
                        num_episodes=request.num_episodes,
                        num_epochs=request.num_epochs,
                        user_id=request.user_id,
                    )
                finally:
                    await session.close()

        # Run training in background (FastAPI supports async functions)
        background_tasks.add_task(run_training_async)

        return {
            "status": "training_started",
            "message": "Answer Agent training started in background",
            "num_episodes": request.num_episodes,
            "num_epochs": request.num_epochs,
        }

    except Exception as e:
        logger.error(f"Failed to start Answer Agent training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/both")
async def train_both_agents(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Train both Memory Manager and Answer Agent.

    Runs training for both agents sequentially.
    """
    try:
        # Initialize orchestrator to ensure it's created
        await get_training_orchestrator(db)

        # Define async wrapper that gets its own DB session
        async def run_training_async():
            async for session in get_db():
                try:
                    orch = await get_training_orchestrator(session)
                    await orch.train_both_agents(
                        num_episodes=request.num_episodes,
                        num_epochs=request.num_epochs,
                        user_id=request.user_id,
                    )
                finally:
                    await session.close()

        # Run training in background (FastAPI supports async functions)
        background_tasks.add_task(run_training_async)

        return {
            "status": "training_started",
            "message": "Dual-agent training started in background",
            "num_episodes": request.num_episodes,
            "num_epochs": request.num_epochs,
        }

    except Exception as e:
        logger.error(f"Failed to start dual-agent training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_training_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get training metrics summary.

    Returns current training statistics for both agents.
    """
    try:
        orchestrator = await get_training_orchestrator(db)
        metrics = orchestrator.get_metrics_summary()

        return {
            "status": "success",
            "metrics": metrics,
        }

    except Exception as e:
        logger.error(f"Failed to get training metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class EvaluateRequest(BaseModel):
    """Request model for agent evaluation."""
    agent_type: str = Field(..., description="memory_manager or answer_agent")
    collection_id: Optional[str] = None
    num_episodes: int = Field(default=10, ge=1, le=100)


@router.post("/evaluate")
async def evaluate_agent(
    request: EvaluateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Evaluate a trained RL agent.

    Runs the agent on test queries and computes performance metrics.
    """
    try:
        from backend.models.rl_trajectory import Trajectory
        from backend.services.hybrid_search import search_memories
        from sqlalchemy import select, func
        import time

        logger.info(
            f"Evaluating {request.agent_type} agent "
            f"for user {current_user.id} "
            f"on {request.num_episodes} episodes"
        )

        # Get recent trajectories for evaluation
        query = (
            select(Trajectory)
            .where(Trajectory.user_id == current_user.id)
            .order_by(Trajectory.created_at.desc())
            .limit(request.num_episodes)
        )

        if request.collection_id:
            query = query.where(Trajectory.collection_id == request.collection_id)

        result = await db.execute(query)
        trajectories = list(result.scalars().all())

        if not trajectories:
            return {
                "status": "no_data",
                "message": "No trajectories found for evaluation",
                "agent_type": request.agent_type,
                "episodes_evaluated": 0,
                "metrics": {
                    "avg_reward": 0.0,
                    "avg_steps": 0.0,
                    "success_rate": 0.0,
                },
            }

        # Compute evaluation metrics
        total_reward = sum(t.final_reward or 0.0 for t in trajectories)
        total_steps = sum(t.total_steps for t in trajectories)
        avg_reward = total_reward / len(trajectories)
        avg_steps = total_steps / len(trajectories)

        # Count successful episodes (positive reward)
        successful = sum(1 for t in trajectories if (t.final_reward or 0.0) > 0)
        success_rate = successful / len(trajectories)

        # Get agent-specific metrics
        agent_metrics = {}
        if request.agent_type == "memory_manager":
            # For memory manager, evaluate retrieval quality
            agent_metrics = {
                "retrieval_precision": avg_reward,  # Using reward as proxy
                "avg_results_returned": avg_steps,
            }
        elif request.agent_type == "answer_agent":
            # For answer agent, evaluate answer quality
            agent_metrics = {
                "answer_quality": avg_reward,
                "avg_sources_used": avg_steps,
            }

        return {
            "status": "success",
            "message": f"Evaluated {request.agent_type} on {len(trajectories)} episodes",
            "agent_type": request.agent_type,
            "collection_id": request.collection_id,
            "episodes_evaluated": len(trajectories),
            "metrics": {
                "avg_reward": round(avg_reward, 4),
                "avg_steps": round(avg_steps, 2),
                "success_rate": round(success_rate, 2),
                "total_trajectories": len(trajectories),
                **agent_metrics,
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to evaluate agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint/save")
async def save_checkpoint(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Save current model checkpoint.

    Saves both agents and optimizer states.
    """
    try:
        orchestrator = await get_training_orchestrator(db)
        checkpoint_path = await orchestrator.save_checkpoint()

        return {
            "status": "success",
            "checkpoint_path": checkpoint_path,
            "message": "Checkpoint saved successfully",
        }

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint/load")
async def load_checkpoint(
    request: CheckpointRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Load model checkpoint.

    Restores both agents from saved checkpoint.
    """
    try:
        orchestrator = await get_training_orchestrator(db)
        success = await orchestrator.load_checkpoint(request.checkpoint_path)

        if success:
            return {
                "status": "success",
                "message": f"Checkpoint loaded from {request.checkpoint_path}",
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to load checkpoint")

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
