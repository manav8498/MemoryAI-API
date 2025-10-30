"""
World Model and Planning for Memory Operations.

Simulates outcomes of memory operations before executing them.
Based on Dreamer-style world models for AI agents.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from backend.core.logging_config import logger
from backend.ml.embeddings.model import get_embedding_generator


class LatentDynamicsModel(nn.Module):
    """
    Learns dynamics in latent space.

    Predicts: next_state = f(current_state, action)
    """

    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 4,  # ADD, UPDATE, DELETE, NOOP
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # State + action encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Next state predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state given current state and action.

        Args:
            state: Current state [batch_size, state_dim]
            action: Action (one-hot) [batch_size, action_dim]

        Returns:
            Predicted next state [batch_size, state_dim]
        """
        # Concatenate state and action
        combined = torch.cat([state, action], dim=-1)

        # Encode
        encoded = self.encoder(combined)

        # Predict next state
        next_state = self.predictor(encoded)

        return next_state


class RewardPredictor(nn.Module):
    """
    Predicts reward for a state-action pair.

    Learns: reward = r(state, action)
    """

    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict reward.

        Args:
            state: State [batch_size, state_dim]
            action: Action [batch_size, action_dim]

        Returns:
            Predicted reward [batch_size, 1]
        """
        combined = torch.cat([state, action], dim=-1)
        reward = self.predictor(combined)
        return reward


class MemoryWorldModel:
    """
    World model for planning memory operations.

    Features:
    - Simulate retrieval outcomes
    - Plan optimal action sequences
    - Imagine future states
    - Tree search for planning
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.embedding_generator = get_embedding_generator()

        # Initialize models
        self.dynamics_model = LatentDynamicsModel().to(device)
        self.reward_model = RewardPredictor().to(device)

    async def imagine_retrieval(
        self,
        query: str,
        memory_state: Dict[str, Any],
    ) -> Tuple[List[str], float]:
        """
        Simulate what memories would be retrieved without actually retrieving.

        Args:
            query: Query string
            memory_state: Current memory state

        Returns:
            (predicted_memories, predicted_quality)
        """
        try:
            # Encode query
            query_embedding = await self.embedding_generator.encode_query(query)
            state_tensor = torch.tensor(
                query_embedding,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            # Simulate different actions
            action_scores = []
            for action_idx in range(4):  # ADD, UPDATE, DELETE, NOOP
                action_tensor = torch.zeros(1, 4, device=self.device)
                action_tensor[0, action_idx] = 1.0

                # Predict next state
                with torch.no_grad():
                    next_state = self.dynamics_model(state_tensor, action_tensor)
                    predicted_reward = self.reward_model(state_tensor, action_tensor)

                action_scores.append({
                    "action": action_idx,
                    "next_state": next_state,
                    "reward": predicted_reward.item(),
                })

            # Return best predicted outcome
            best = max(action_scores, key=lambda x: x["reward"])

            # Simplified prediction of what would be retrieved
            predicted_memories = [f"simulated_memory_{i}" for i in range(5)]
            predicted_quality = best["reward"]

            logger.debug(f"Imagined retrieval: quality={predicted_quality:.3f}")
            return predicted_memories, predicted_quality

        except Exception as e:
            logger.error(f"Imagination failed: {e}")
            return [], 0.0

    async def plan_optimal_query(
        self,
        user_intent: str,
        num_variations: int = 5,
    ) -> str:
        """
        Find best query formulation through planning.

        Args:
            user_intent: User's intent
            num_variations: Number of query variations to try

        Returns:
            Optimal query
        """
        try:
            # Generate query variations (would use LLM in production)
            variations = [
                user_intent,
                f"Find information about {user_intent}",
                f"What do you know about {user_intent}?",
                f"Recall memories related to {user_intent}",
                f"Search for {user_intent}",
            ][:num_variations]

            best_query = user_intent
            best_quality = -float('inf')

            # Simulate each variation
            for query in variations:
                _, quality = await self.imagine_retrieval(query, {})

                if quality > best_quality:
                    best_quality = quality
                    best_query = query

            logger.info(f"Planned optimal query: {best_query} (quality: {best_quality:.3f})")
            return best_query

        except Exception as e:
            logger.error(f"Query planning failed: {e}")
            return user_intent

    def plan_action_sequence(
        self,
        initial_state: torch.Tensor,
        goal_state: torch.Tensor,
        max_steps: int = 5,
        beam_width: int = 3,
    ) -> List[int]:
        """
        Plan sequence of actions to reach goal state.

        Uses beam search in imagined latent space.

        Args:
            initial_state: Starting state
            goal_state: Target state
            max_steps: Maximum planning depth
            beam_width: Beam search width

        Returns:
            List of action indices
        """
        try:
            # Beam search
            beams = [(initial_state, [], 0.0)]  # (state, actions, cumulative_reward)

            for step in range(max_steps):
                new_beams = []

                for state, actions, cum_reward in beams:
                    # Try each action
                    for action_idx in range(4):
                        action_tensor = torch.zeros(1, 4, device=self.device)
                        action_tensor[0, action_idx] = 1.0

                        with torch.no_grad():
                            # Predict next state and reward
                            next_state = self.dynamics_model(state, action_tensor)
                            reward = self.reward_model(state, action_tensor).item()

                            # Distance to goal
                            distance = torch.norm(next_state - goal_state).item()
                            reward -= distance * 0.1  # Penalize distance from goal

                            new_beams.append((
                                next_state,
                                actions + [action_idx],
                                cum_reward + reward,
                            ))

                # Keep top beam_width beams
                beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

            # Return best action sequence
            best_sequence = beams[0][1]
            logger.info(f"Planned action sequence: {best_sequence}")
            return best_sequence

        except Exception as e:
            logger.error(f"Action planning failed: {e}")
            return []

    def train_world_model(
        self,
        trajectories: List[Dict[str, Any]],
        num_epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Train world model on collected trajectories.

        Args:
            trajectories: List of (state, action, next_state, reward) tuples
            num_epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training metrics
        """
        try:
            optimizer_dynamics = torch.optim.Adam(
                self.dynamics_model.parameters(),
                lr=1e-4,
            )
            optimizer_reward = torch.optim.Adam(
                self.reward_model.parameters(),
                lr=1e-4,
            )

            dynamics_losses = []
            reward_losses = []

            for epoch in range(num_epochs):
                # Sample batch
                batch_indices = np.random.choice(
                    len(trajectories),
                    min(batch_size, len(trajectories)),
                    replace=False,
                )

                batch = [trajectories[i] for i in batch_indices]

                # Prepare tensors
                states = torch.stack([torch.tensor(t["state"]) for t in batch]).to(self.device)
                actions = torch.stack([torch.tensor(t["action"]) for t in batch]).to(self.device)
                next_states = torch.stack([torch.tensor(t["next_state"]) for t in batch]).to(self.device)
                rewards = torch.tensor([t["reward"] for t in batch], dtype=torch.float32).to(self.device).unsqueeze(1)

                # Train dynamics model
                optimizer_dynamics.zero_grad()
                predicted_next_states = self.dynamics_model(states, actions)
                dynamics_loss = nn.functional.mse_loss(predicted_next_states, next_states)
                dynamics_loss.backward()
                optimizer_dynamics.step()

                # Train reward model
                optimizer_reward.zero_grad()
                predicted_rewards = self.reward_model(states, actions)
                reward_loss = nn.functional.mse_loss(predicted_rewards, rewards)
                reward_loss.backward()
                optimizer_reward.step()

                dynamics_losses.append(dynamics_loss.item())
                reward_losses.append(reward_loss.item())

            metrics = {
                "dynamics_loss": np.mean(dynamics_losses),
                "reward_loss": np.mean(reward_losses),
            }

            logger.info(f"World model training complete: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"World model training failed: {e}")
            return {"error": str(e)}

    def save(self, path: str):
        """Save world model."""
        torch.save({
            "dynamics_model": self.dynamics_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
        }, path)
        logger.info(f"Saved world model to {path}")

    def load(self, path: str):
        """Load world model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.dynamics_model.load_state_dict(checkpoint["dynamics_model"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        logger.info(f"Loaded world model from {path}")


# Global instance
_world_model: Optional[MemoryWorldModel] = None


def get_world_model() -> MemoryWorldModel:
    """Get world model instance."""
    global _world_model

    if _world_model is None:
        _world_model = MemoryWorldModel()

    return _world_model
