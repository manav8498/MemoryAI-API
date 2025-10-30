"""
RL Training Orchestration.

Coordinates training of Memory Manager and Answer Agent using PPO.
Integrates trajectory collection, training loops, and model checkpointing.
"""
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

import torch
import torch.nn.functional as F

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from backend.core.logging_config import logger
from backend.core.database import get_db
from backend.rl.memory_manager import MemoryManagerAgent, MemoryOperation, MemoryState
from backend.rl.answer_agent import AnswerAgent
from backend.rl.ppo import PPOTrainer, PPOConfig, RolloutBuffer
from backend.models.rl_trajectory import Trajectory, TrajectoryStep
from backend.models.memory import Memory
from backend.services.hybrid_search import search_memories


class TrainingOrchestrator:
    """
    Orchestrates RL training for memory system agents.

    Features:
    - Dual-agent training (Memory Manager + Answer Agent)
    - Trajectory collection from production data
    - PPO training with checkpointing
    - Metrics tracking and logging
    - Automatic model saving
    """

    def __init__(
        self,
        db: AsyncSession,
        memory_manager: MemoryManagerAgent,
        answer_agent: AnswerAgent,
        config: Optional[PPOConfig] = None,
        checkpoint_dir: str = "checkpoints/rl",
    ):
        self.db = db
        self.memory_manager = memory_manager
        self.answer_agent = answer_agent
        self.config = config or PPOConfig()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create PPO trainers
        self.memory_manager_trainer = PPOTrainer(
            policy=memory_manager.policy,
            config=self.config,
        )

        self.answer_agent_trainer = PPOTrainer(
            policy=answer_agent.policy,
            config=self.config,
        )

        # Training metrics
        self.training_metrics: Dict[str, List[float]] = {
            "memory_manager_reward": [],
            "memory_manager_loss": [],
            "answer_agent_reward": [],
            "answer_agent_loss": [],
        }

    async def collect_memory_manager_trajectories(
        self,
        num_episodes: int = 100,
        user_id: Optional[str] = None,
    ) -> RolloutBuffer:
        """
        Collect trajectories for Memory Manager by replaying production data.

        Args:
            num_episodes: Number of episodes to collect
            user_id: Optional user to filter by

        Returns:
            RolloutBuffer with collected trajectories
        """
        try:
            buffer = RolloutBuffer(
                buffer_size=num_episodes * 10,
                device=self.memory_manager.device,
            )

            logger.info(f"Collecting {num_episodes} Memory Manager trajectories")

            # Get recent trajectories from database
            query = select(Trajectory).order_by(Trajectory.created_at.desc()).limit(num_episodes)
            if user_id:
                query = query.where(Trajectory.user_id == user_id)

            result = await self.db.execute(query)
            trajectories = list(result.scalars().all())

            episodes_collected = 0

            for trajectory in trajectories:
                if episodes_collected >= num_episodes:
                    break

                # Get steps
                steps_result = await self.db.execute(
                    select(TrajectoryStep)
                    .where(TrajectoryStep.trajectory_id == trajectory.id)
                    .order_by(TrajectoryStep.step_number)
                )
                steps = list(steps_result.scalars().all())

                if not steps:
                    continue

                # Reconstruct episode
                for step in steps:
                    # Create MemoryState from logged trajectory data
                    state = MemoryState(
                        extracted_info=step.state.get("query", ""),  # Use query as extracted info
                        query_context=step.state.get("query"),
                        existing_memories=[],  # No existing memories in simple search/reason
                        user_id=trajectory.user_id,
                        collection_id=trajectory.collection_id or "",
                        metadata=step.state.get("filters", {}),
                    )

                    # Encode state to tensor using the agent's encoding method
                    state_tensor = await self.memory_manager.encode_state(state)

                    # Get action (already an integer from database)
                    action = step.action
                    action_tensor = torch.tensor([action], dtype=torch.long, device=self.memory_manager.device)

                    # Get reward
                    reward = step.reward or 0.0

                    # Create tensors for value and log_prob (placeholder values)
                    value_tensor = torch.tensor([0.0], dtype=torch.float32, device=self.memory_manager.device)
                    log_prob_tensor = torch.tensor([0.0], dtype=torch.float32, device=self.memory_manager.device)

                    # Add to buffer
                    buffer.add(
                        state=state_tensor,
                        action=action_tensor,
                        reward=reward,
                        value=value_tensor,
                        log_prob=log_prob_tensor,
                        done=step.step_number == len(steps) - 1,
                    )

                episodes_collected += 1

            logger.info(f"Collected {episodes_collected} episodes with {len(buffer.states)} steps")
            return buffer

        except Exception as e:
            logger.error(f"Failed to collect Memory Manager trajectories: {e}")
            # Return empty buffer
            return RolloutBuffer(
                buffer_size=1,
                device=self.memory_manager.device,
            )

    async def collect_answer_agent_trajectories(
        self,
        num_episodes: int = 100,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect trajectories for Answer Agent from production queries.

        Args:
            num_episodes: Number of episodes to collect
            user_id: Optional user to filter by

        Returns:
            List of episode data
        """
        try:
            logger.info(f"Collecting {num_episodes} Answer Agent trajectories")

            # Get recent successful queries
            query = (
                select(Trajectory)
                .where(Trajectory.final_reward > 0.5)
                .order_by(Trajectory.created_at.desc())
                .limit(num_episodes)
            )
            if user_id:
                query = query.where(Trajectory.user_id == user_id)

            result = await self.db.execute(query)
            trajectories = list(result.scalars().all())

            episodes = []

            for trajectory in trajectories:
                # Get query
                query_text = trajectory.metadata.get("query", "")
                if not query_text:
                    continue

                collection_id = trajectory.metadata.get("collection_id")
                if not collection_id:
                    continue

                # Search for candidate memories
                try:
                    results = await search_memories(
                        query=query_text,
                        user_id=trajectory.user_id,
                        db=self.db,
                        collection_id=collection_id,
                        limit=20,
                    )

                    if len(results) < 2:
                        continue

                    # Determine which memories were actually used (top 5)
                    used_memory_ids = trajectory.metadata.get("used_memory_ids", [])

                    # Create episode
                    episode = {
                        "query": query_text,
                        "candidates": results[:20],
                        "selected_ids": used_memory_ids[:5],
                        "reward": trajectory.final_reward,
                        "user_id": trajectory.user_id,
                        "collection_id": collection_id,
                    }

                    episodes.append(episode)

                except Exception as e:
                    logger.debug(f"Skipping trajectory {trajectory.id}: {e}")
                    continue

            logger.info(f"Collected {len(episodes)} Answer Agent episodes")
            return episodes

        except Exception as e:
            logger.error(f"Failed to collect Answer Agent trajectories: {e}")
            return []

    async def train_memory_manager(
        self,
        num_episodes: int = 100,
        num_epochs: int = 10,
        user_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Train Memory Manager agent.

        Args:
            num_episodes: Number of episodes to collect
            num_epochs: Training epochs per collection
            user_id: Optional user to filter by

        Returns:
            Training metrics
        """
        try:
            logger.info("Starting Memory Manager training")
            start_time = time.time()

            # Collect trajectories
            rollout_buffer = await self.collect_memory_manager_trajectories(
                num_episodes=num_episodes,
                user_id=user_id,
            )

            if len(rollout_buffer.states) == 0:
                logger.warning("No trajectories collected for Memory Manager")
                return {"error": "No data"}

            # Compute advantages and returns
            # For completed episodes, last_value is 0 (terminal state)
            last_value = torch.tensor([0.0], dtype=torch.float32, device=self.memory_manager.device)
            rollout_buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=0.99,
                gae_lambda=0.95
            )

            # Train
            total_loss = 0.0
            for epoch in range(num_epochs):
                # Get batch from buffer
                batch = rollout_buffer.get(batch_size=self.config.batch_size)

                # Compute loss
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Forward pass
                action_logits, values = self.memory_manager.policy(states)

                # Get log probs for taken actions
                log_probs = F.log_softmax(action_logits, dim=-1)
                action_indices = actions.argmax(dim=-1, keepdim=True)
                new_log_probs = log_probs.gather(1, action_indices).squeeze(-1)

                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                )
                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages,
                ).mean()

                # Value loss - values might be [batch_size, num_actions] or [batch_size, 1]
                # We need to reduce it to [batch_size] to match returns
                if values.dim() > 1:
                    values = values.mean(dim=-1)  # Average across actions if multi-dimensional
                values_flat = values.squeeze()  # Remove all singleton dimensions
                returns_flat = returns.squeeze()  # Remove all singleton dimensions
                value_loss = F.mse_loss(values_flat, returns_flat)

                # Entropy bonus
                probs = F.softmax(action_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Backward
                self.memory_manager_trainer.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.memory_manager.policy.parameters(),
                    self.config.max_grad_norm,
                )
                self.memory_manager_trainer.optimizer.step()

                total_loss += loss.item()

            # Calculate average loss (1 batch per epoch)
            avg_loss = total_loss / max(1, num_epochs)
            avg_reward = torch.tensor(rollout_buffer.rewards).mean().item()

            # Update metrics
            self.training_metrics["memory_manager_reward"].append(avg_reward)
            self.training_metrics["memory_manager_loss"].append(avg_loss)

            elapsed_time = time.time() - start_time

            metrics = {
                "avg_reward": avg_reward,
                "avg_loss": avg_loss,
                "episodes": num_episodes,
                "total_steps": len(rollout_buffer.states),
                "training_time": elapsed_time,
            }

            logger.info(f"Memory Manager training complete: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Memory Manager training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    async def train_answer_agent(
        self,
        num_episodes: int = 100,
        num_epochs: int = 10,
        user_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Train Answer Agent.

        Args:
            num_episodes: Number of episodes to collect
            num_epochs: Training epochs
            user_id: Optional user to filter by

        Returns:
            Training metrics
        """
        try:
            logger.info("Starting Answer Agent training")
            start_time = time.time()

            # Collect episodes
            episodes = await self.collect_answer_agent_trajectories(
                num_episodes=num_episodes,
                user_id=user_id,
            )

            if not episodes:
                logger.warning("No episodes collected for Answer Agent")
                return {"error": "No data"}

            total_loss = 0.0
            total_reward = 0.0
            num_batches = 0

            for epoch in range(num_epochs):
                for episode in episodes:
                    # Get selection and reward
                    query = episode["query"]
                    candidates = episode["candidates"]
                    selected_ids = episode["selected_ids"]
                    reward = episode["reward"]

                    # Run selection through agent
                    selected_memories = await self.answer_agent.select_memories(
                        query=query,
                        candidate_memories=candidates,
                        deterministic=False,
                    )

                    # Compute loss based on whether we selected the right memories
                    # This is a simplified version - would use full PPO in production
                    selected_ids_set = set(selected_ids)
                    actual_ids = [m.id for m in selected_memories]
                    overlap = len(set(actual_ids) & selected_ids_set)

                    # Loss is negative reward weighted by overlap
                    loss_value = -reward * (overlap / max(len(selected_ids), 1))

                    total_loss += abs(loss_value)
                    total_reward += reward
                    num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            avg_reward = total_reward / max(len(episodes), 1)

            # Update metrics
            self.training_metrics["answer_agent_reward"].append(avg_reward)
            self.training_metrics["answer_agent_loss"].append(avg_loss)

            elapsed_time = time.time() - start_time

            metrics = {
                "avg_reward": avg_reward,
                "avg_loss": avg_loss,
                "episodes": len(episodes),
                "training_time": elapsed_time,
            }

            logger.info(f"Answer Agent training complete: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Answer Agent training failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    async def train_both_agents(
        self,
        num_episodes: int = 100,
        num_epochs: int = 10,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train both Memory Manager and Answer Agent.

        Args:
            num_episodes: Number of episodes per agent
            num_epochs: Training epochs
            user_id: Optional user to filter by

        Returns:
            Combined metrics
        """
        logger.info("Starting dual-agent training")

        # Train Memory Manager
        mm_metrics = await self.train_memory_manager(
            num_episodes=num_episodes,
            num_epochs=num_epochs,
            user_id=user_id,
        )

        # Train Answer Agent
        aa_metrics = await self.train_answer_agent(
            num_episodes=num_episodes,
            num_epochs=num_epochs,
            user_id=user_id,
        )

        combined_metrics = {
            "memory_manager": mm_metrics,
            "answer_agent": aa_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Save checkpoint
        await self.save_checkpoint(metrics=combined_metrics)

        return combined_metrics

    async def save_checkpoint(
        self,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save model checkpoint.

        Args:
            metrics: Optional metrics to save with checkpoint

        Returns:
            Checkpoint path
        """
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.pt"

            checkpoint = {
                "memory_manager_state_dict": self.memory_manager.policy.state_dict(),
                "answer_agent_state_dict": self.answer_agent.policy.state_dict(),
                "memory_manager_optimizer": self.memory_manager_trainer.optimizer.state_dict(),
                "answer_agent_optimizer": self.answer_agent_trainer.optimizer.state_dict(),
                "training_metrics": self.training_metrics,
                "config": {
                    "learning_rate": self.config.learning_rate,
                    "clip_epsilon": self.config.clip_epsilon,
                    "batch_size": self.config.batch_size,
                },
                "timestamp": timestamp,
                "metrics": metrics,
            }

            torch.save(checkpoint, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return ""

    async def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            True if successful
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.memory_manager.device)

            self.memory_manager.policy.load_state_dict(checkpoint["memory_manager_state_dict"])
            self.answer_agent.policy.load_state_dict(checkpoint["answer_agent_state_dict"])

            self.memory_manager_trainer.optimizer.load_state_dict(checkpoint["memory_manager_optimizer"])
            self.answer_agent_trainer.optimizer.load_state_dict(checkpoint["answer_agent_optimizer"])

            self.training_metrics = checkpoint.get("training_metrics", self.training_metrics)

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        import numpy as np

        return {
            "memory_manager": {
                "avg_reward": np.mean(self.training_metrics["memory_manager_reward"]) if self.training_metrics["memory_manager_reward"] else 0.0,
                "avg_loss": np.mean(self.training_metrics["memory_manager_loss"]) if self.training_metrics["memory_manager_loss"] else 0.0,
                "total_updates": len(self.training_metrics["memory_manager_reward"]),
            },
            "answer_agent": {
                "avg_reward": np.mean(self.training_metrics["answer_agent_reward"]) if self.training_metrics["answer_agent_reward"] else 0.0,
                "avg_loss": np.mean(self.training_metrics["answer_agent_loss"]) if self.training_metrics["answer_agent_loss"] else 0.0,
                "total_updates": len(self.training_metrics["answer_agent_reward"]),
            },
        }


async def create_training_orchestrator(
    db: AsyncSession,
    checkpoint_path: Optional[str] = None,
) -> TrainingOrchestrator:
    """
    Create and initialize training orchestrator.

    Args:
        db: Database session
        checkpoint_path: Optional checkpoint to load

    Returns:
        TrainingOrchestrator instance
    """
    from backend.rl.memory_manager import get_memory_manager_agent
    from backend.rl.answer_agent import get_answer_agent

    # Create agents
    memory_manager = get_memory_manager_agent(db)
    answer_agent = get_answer_agent(db)

    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        db=db,
        memory_manager=memory_manager,
        answer_agent=answer_agent,
    )

    # Load checkpoint if provided
    if checkpoint_path:
        await orchestrator.load_checkpoint(checkpoint_path)

    return orchestrator


# Global instance
_orchestrator: Optional[TrainingOrchestrator] = None


async def get_training_orchestrator(db: AsyncSession) -> TrainingOrchestrator:
    """Get global training orchestrator instance."""
    global _orchestrator

    if _orchestrator is None:
        _orchestrator = await create_training_orchestrator(db)

    return _orchestrator
