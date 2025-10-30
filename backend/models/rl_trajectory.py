"""
RL Trajectory Models.

Stores trajectories and steps for reinforcement learning training.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Float, Text, JSON
from sqlalchemy.orm import relationship

from backend.core.database import Base


class Trajectory(Base):
    """
    RL trajectory (episode).

    Stores a complete episode of interactions for training.
    """

    __tablename__ = "trajectories"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    collection_id = Column(String(36), ForeignKey("collections.id", ondelete="CASCADE"), nullable=True, index=True)

    # Episode metadata
    agent_type = Column(String(50), nullable=False)  # "memory_manager" or "answer_agent"
    final_reward = Column(Float, nullable=True)
    total_steps = Column(Integer, default=0, nullable=False)

    # Context
    extra_metadata = Column(JSON, default=dict, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="trajectories")
    collection = relationship("Collection", back_populates="trajectories")
    steps = relationship("TrajectoryStep", back_populates="trajectory", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Trajectory {self.id} (agent: {self.agent_type}, reward: {self.final_reward})>"


class TrajectoryStep(Base):
    """
    Single step in a trajectory.

    Records state, action, reward at each timestep.
    """

    __tablename__ = "trajectory_steps"

    id = Column(String(36), primary_key=True, index=True)
    trajectory_id = Column(
        String(36),
        ForeignKey("trajectories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Step information
    step_number = Column(Integer, nullable=False)

    # RL tuple: (state, action, reward, next_state)
    state = Column(JSON, nullable=False)
    action = Column(Integer, nullable=False)  # Action index (0-3 for memory operations)
    reward = Column(Float, nullable=True)
    next_state = Column(JSON, nullable=True)

    # Value estimates
    value = Column(Float, nullable=True)  # Value function estimate
    log_prob = Column(Float, nullable=True)  # Action log probability

    # Additional context
    done = Column(Integer, default=0, nullable=False)  # 1 if terminal state
    info = Column(JSON, default=dict, nullable=False)  # Extra information

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    trajectory = relationship("Trajectory", back_populates="steps")

    def __repr__(self) -> str:
        return f"<TrajectoryStep {self.step_number} (action: {self.action}, reward: {self.reward})>"
