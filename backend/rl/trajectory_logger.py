"""
Reinforcement Learning trajectory logger.

Logs agent interactions for offline RL training:
- States (queries, contexts)
- Actions (retrieved memories, generated responses)
- Rewards (user feedback, engagement metrics)
- Next states (follow-up queries)

Supports RLHF (Reinforcement Learning from Human Feedback) and
automated reward modeling.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import asyncio

from backend.core.config import settings
from backend.core.logging_config import logger
from backend.ml.embeddings.model import get_embedding_generator


@dataclass
class TrajectoryStep:
    """Single step in an RL trajectory."""
    step_id: str
    timestamp: str

    # State
    state: Dict[str, Any]  # Query, context, user profile

    # Action
    action: Dict[str, Any]  # Retrieved memories, response generated

    # Reward (optional, can be added later)
    reward: Optional[float] = None

    # Next state (for transition)
    next_state: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Trajectory:
    """Complete RL trajectory (sequence of steps)."""
    trajectory_id: str
    user_id: str
    session_id: str
    started_at: str
    ended_at: Optional[str] = None

    steps: List[TrajectoryStep] = None

    # Aggregated metrics
    total_reward: float = 0.0
    episode_length: int = 0

    # Metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.metadata is None:
            self.metadata = {}

    def add_step(self, step: TrajectoryStep):
        """Add a step to the trajectory."""
        self.steps.append(step)
        self.episode_length = len(self.steps)

        if step.reward is not None:
            self.total_reward += step.reward

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "steps": [step.to_dict() for step in self.steps],
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "metadata": self.metadata,
        }


class TrajectoryLogger:
    """
    Logger for RL trajectories.

    Stores trajectories in:
    1. PostgreSQL (for structured querying)
    2. S3/Object storage (for training data archives)
    3. Kafka (for real-time processing)
    """

    def __init__(self):
        self.active_trajectories: Dict[str, Trajectory] = {}
        self._embedding_generator = None

    def _get_embedding_generator(self):
        """Get or create embedding generator (lazy loading)."""
        if self._embedding_generator is None:
            try:
                self._embedding_generator = get_embedding_generator()
            except Exception as e:
                logger.warning(f"Could not load embedding generator: {e}")
                self._embedding_generator = None
        return self._embedding_generator

    async def start_trajectory(
        self,
        trajectory_id: str,
        user_id: str,
        session_id: str,
        metadata: Dict[str, Any] = None,
    ) -> Trajectory:
        """
        Start a new trajectory.

        Args:
            trajectory_id: Unique trajectory identifier
            user_id: User identifier
            session_id: Session identifier
            metadata: Optional metadata

        Returns:
            Trajectory object
        """
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            user_id=user_id,
            session_id=session_id,
            started_at=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )

        self.active_trajectories[trajectory_id] = trajectory

        logger.info(f"Started trajectory {trajectory_id} for user {user_id}")
        return trajectory

    async def log_step(
        self,
        trajectory_id: str,
        step_id: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: Optional[float] = None,
        next_state: Optional[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Log a single step in a trajectory.

        Args:
            trajectory_id: Trajectory identifier
            step_id: Unique step identifier
            state: State representation
            action: Action taken
            reward: Optional reward signal
            next_state: Optional next state
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            if trajectory_id not in self.active_trajectories:
                logger.warning(f"Trajectory {trajectory_id} not found, creating new one")
                # Extract user_id from metadata if available
                user_id = metadata.get("user_id", "unknown") if metadata else "unknown"
                await self.start_trajectory(
                    trajectory_id=trajectory_id,
                    user_id=user_id,
                    session_id=trajectory_id,  # Fallback
                )

            trajectory = self.active_trajectories[trajectory_id]

            # Generate query embedding if query is present in state
            if "query" in state and state["query"]:
                embedding_gen = self._get_embedding_generator()
                if embedding_gen:
                    try:
                        query_embedding = await embedding_gen.encode_query(state["query"])
                        state["query_embedding"] = query_embedding
                        logger.debug(f"Generated query embedding for step {step_id}")
                    except Exception as emb_error:
                        logger.warning(f"Failed to generate embedding: {emb_error}")
                        # Continue without embedding - training can handle missing embeddings

            step = TrajectoryStep(
                step_id=step_id,
                timestamp=datetime.utcnow().isoformat(),
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                metadata=metadata or {},
            )

            trajectory.add_step(step)

            logger.debug(f"Logged step {step_id} for trajectory {trajectory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log step: {e}")
            return False

    async def end_trajectory(
        self,
        trajectory_id: str,
        final_reward: Optional[float] = None,
    ) -> Optional[Trajectory]:
        """
        End a trajectory and persist it.

        Args:
            trajectory_id: Trajectory identifier
            final_reward: Optional final reward

        Returns:
            Completed trajectory
        """
        try:
            if trajectory_id not in self.active_trajectories:
                logger.warning(f"Trajectory {trajectory_id} not found")
                return None

            trajectory = self.active_trajectories[trajectory_id]
            trajectory.ended_at = datetime.utcnow().isoformat()

            if final_reward is not None:
                trajectory.total_reward += final_reward

            # Persist trajectory
            await self._persist_trajectory(trajectory)

            # Remove from active trajectories
            del self.active_trajectories[trajectory_id]

            logger.info(
                f"Ended trajectory {trajectory_id}: "
                f"{trajectory.episode_length} steps, "
                f"total reward: {trajectory.total_reward:.3f}"
            )

            return trajectory

        except Exception as e:
            logger.error(f"Failed to end trajectory: {e}")
            return None

    async def _persist_trajectory(self, trajectory: Trajectory) -> bool:
        """
        Persist trajectory to storage.

        Saves to:
        - PostgreSQL (for querying and training)
        - File logs (for debugging)
        - TODO: S3 (for archival) - optional future enhancement
        - TODO: Kafka (for streaming) - optional future enhancement
        """
        try:
            # Log to file for debugging
            if settings.ENABLE_RL_LOGGING:
                trajectory_json = json.dumps(trajectory.to_dict(), indent=2)
                logger.info(f"Trajectory {trajectory.trajectory_id} completed:\n{trajectory_json}")

            # Save to database
            try:
                from backend.core.database import AsyncSessionLocal
                from backend.models.rl_trajectory import Trajectory as DBTrajectory, TrajectoryStep as DBTrajectoryStep
                import uuid

                async with AsyncSessionLocal() as db:
                    # Create trajectory record
                    db_trajectory = DBTrajectory(
                        id=trajectory.trajectory_id,
                        user_id=trajectory.user_id,
                        collection_id=trajectory.metadata.get("collection_id") if trajectory.metadata else None,
                        agent_type=trajectory.metadata.get("agent_type", "unknown") if trajectory.metadata else "unknown",
                        final_reward=sum(step.reward or 0 for step in trajectory.steps),
                        total_steps=len(trajectory.steps),
                        extra_metadata=trajectory.metadata or {},
                        created_at=datetime.fromisoformat(trajectory.started_at),
                        completed_at=datetime.fromisoformat(trajectory.ended_at) if trajectory.ended_at else None,
                    )
                    db.add(db_trajectory)

                    # Create step records
                    for idx, step in enumerate(trajectory.steps):
                        # Map action type to integer index
                        action_type_map = {
                            "search": 0,
                            "reason": 1,
                            "create": 2,
                            "update": 3,
                        }
                        action_type = step.action.get("action_type", "search") if isinstance(step.action, dict) else "search"
                        action_index = action_type_map.get(action_type, 0)

                        # Merge action dict and metadata into info
                        info_data = {
                            **(step.metadata or {}),
                            "action_details": step.action if isinstance(step.action, dict) else {},
                        }

                        db_step = DBTrajectoryStep(
                            id=str(uuid.uuid4()),
                            trajectory_id=trajectory.trajectory_id,
                            step_number=idx,
                            state=step.state,
                            action=action_index,  # Integer action index
                            reward=step.reward,
                            next_state=step.next_state,
                            value=None,  # Can be computed later during training
                            log_prob=None,  # Can be computed later during training
                            done=1 if idx == len(trajectory.steps) - 1 else 0,
                            info=info_data,  # Store full action details here
                            created_at=datetime.fromisoformat(step.timestamp),
                        )
                        db.add(db_step)

                    await db.commit()
                    logger.info(f"Trajectory {trajectory.trajectory_id} saved to database with {len(trajectory.steps)} steps")

            except Exception as db_error:
                logger.error(f"Failed to save trajectory to database: {db_error}")
                # Continue anyway - logging is not critical

            # Optional: Upload to S3 for long-term archival
            # This would be useful for large-scale training data storage
            # if settings.S3_BUCKET_TRAJECTORIES:
            #     await self._upload_to_s3(trajectory)

            # Optional: Send to Kafka for real-time processing
            # This would enable streaming RL training
            # if settings.KAFKA_TOPIC_TRAJECTORIES:
            #     await self._send_to_kafka(trajectory)

            return True

        except Exception as e:
            logger.error(f"Failed to persist trajectory: {e}")
            return False

    async def add_reward(
        self,
        trajectory_id: str,
        step_id: str,
        reward: float,
    ) -> bool:
        """
        Add reward to a specific step (for delayed rewards).

        Args:
            trajectory_id: Trajectory identifier
            step_id: Step identifier
            reward: Reward value

        Returns:
            True if successful
        """
        try:
            if trajectory_id not in self.active_trajectories:
                logger.warning(f"Trajectory {trajectory_id} not found")
                return False

            trajectory = self.active_trajectories[trajectory_id]

            # Find and update step
            for step in trajectory.steps:
                if step.step_id == step_id:
                    old_reward = step.reward or 0.0
                    step.reward = reward

                    # Update total reward
                    trajectory.total_reward += (reward - old_reward)

                    logger.debug(f"Added reward {reward} to step {step_id}")
                    return True

            logger.warning(f"Step {step_id} not found in trajectory {trajectory_id}")
            return False

        except Exception as e:
            logger.error(f"Failed to add reward: {e}")
            return False


class RewardCalculator:
    """
    Calculate rewards for RL from various signals.
    """

    @staticmethod
    def calculate_engagement_reward(
        clicked: bool = False,
        time_spent_seconds: float = 0.0,
        followed_up: bool = False,
    ) -> float:
        """
        Calculate reward based on user engagement.

        Args:
            clicked: Whether user clicked on result
            time_spent_seconds: Time spent viewing result
            followed_up: Whether user asked follow-up question

        Returns:
            Reward value (0-1)
        """
        reward = 0.0

        if clicked:
            reward += 0.3

        # Time-based reward (capped at 60 seconds)
        reward += min(time_spent_seconds / 60.0, 1.0) * 0.4

        if followed_up:
            reward += 0.3

        return min(reward, 1.0)

    @staticmethod
    def calculate_quality_reward(
        relevance_score: float = 0.5,
        diversity_score: float = 0.5,
        freshness_score: float = 0.5,
    ) -> float:
        """
        Calculate reward based on result quality metrics.

        Args:
            relevance_score: Relevance to query (0-1)
            diversity_score: Diversity of results (0-1)
            freshness_score: Recency of memories (0-1)

        Returns:
            Reward value (0-1)
        """
        # Weighted combination
        reward = (
            relevance_score * 0.6 +
            diversity_score * 0.2 +
            freshness_score * 0.2
        )

        return reward

    @staticmethod
    def calculate_user_feedback_reward(
        thumbs_up: Optional[bool] = None,
        rating: Optional[int] = None,  # 1-5 stars
        flagged: bool = False,
    ) -> float:
        """
        Calculate reward from explicit user feedback.

        Args:
            thumbs_up: Explicit positive/negative feedback
            rating: Star rating (1-5)
            flagged: Whether user flagged as problematic

        Returns:
            Reward value (-1 to 1)
        """
        if flagged:
            return -1.0

        if thumbs_up is not None:
            return 1.0 if thumbs_up else -0.5

        if rating is not None:
            # Convert 1-5 scale to -1 to 1 scale
            return (rating - 3) / 2.0

        return 0.0


# Global trajectory logger
_trajectory_logger: Optional[TrajectoryLogger] = None


def get_trajectory_logger() -> TrajectoryLogger:
    """
    Get trajectory logger instance.

    Returns:
        TrajectoryLogger instance
    """
    global _trajectory_logger

    if _trajectory_logger is None:
        _trajectory_logger = TrajectoryLogger()

    return _trajectory_logger
