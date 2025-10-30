"""
Proximal Policy Optimization (PPO) Trainer for Memory Agents.

Implements the PPO algorithm for training memory manager and answer agents.
Based on stable-baselines3 and Memory-R1 approaches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque

from backend.core.logging_config import logger
from backend.rl.policy_network import MemoryPolicyNetwork, AnswerPolicyNetwork


@dataclass
class PPOConfig:
    """Configuration for PPO trainer."""

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Experience replay
    buffer_size: int = 2048
    minibatch_size: int = 64

    # Misc
    normalize_advantages: bool = True
    clip_value_loss: bool = True


class RolloutBuffer:
    """
    Buffer for storing trajectories during rollout.

    Stores states, actions, rewards, values, log_probs for PPO training.
    """

    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.reset()

    def reset(self):
        """Reset buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.pos = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool,
    ):
        """Add a transition to the buffer."""
        if len(self.states) < self.buffer_size:
            self.states.append(state.cpu())
            self.actions.append(action.cpu())
            self.rewards.append(reward)
            self.values.append(value.cpu())
            self.log_probs.append(log_prob.cpu())
            self.dones.append(done)
        else:
            # Overwrite oldest
            idx = self.pos % self.buffer_size
            self.states[idx] = state.cpu()
            self.actions[idx] = action.cpu()
            self.rewards[idx] = reward
            self.values[idx] = value.cpu()
            self.log_probs[idx] = log_prob.cpu()
            self.dones[idx] = done

        self.pos += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Convert to tensors
        values = torch.stack(self.values).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        # Append last value
        values = torch.cat([values, last_value.unsqueeze(0)])

        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

        # Compute returns
        returns = advantages + values[:-1]

        self.advantages = advantages
        self.returns = returns

    def get(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer.

        Args:
            batch_size: If provided, sample a random batch

        Returns:
            Dictionary with states, actions, advantages, returns, log_probs
        """
        size = len(self.states)

        if batch_size is None or batch_size >= size:
            # Return all
            indices = list(range(size))
        else:
            # Sample random batch
            indices = np.random.choice(size, batch_size, replace=False)

        return {
            "states": torch.stack([self.states[i] for i in indices]).to(self.device),
            "actions": torch.stack([self.actions[i] for i in indices]).to(self.device),
            "old_log_probs": torch.stack([self.log_probs[i] for i in indices]).to(self.device),
            "advantages": self.advantages[indices],
            "returns": self.returns[indices],
        }

    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """
    PPO Trainer for memory agents.

    Implements the full PPO training loop with clipped objectives.
    """

    def __init__(
        self,
        policy: nn.Module,
        config: Optional[PPOConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy = policy.to(device)
        self.config = config or PPOConfig()
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.config.buffer_size,
            device=device,
        )

        # Training stats
        self.training_stats = {
            "total_steps": 0,
            "total_episodes": 0,
            "policy_loss": deque(maxlen=100),
            "value_loss": deque(maxlen=100),
            "entropy": deque(maxlen=100),
            "returns": deque(maxlen=100),
        }

        logger.info(f"Initialized PPO trainer on device: {device}")

    def collect_rollout(
        self,
        env,
        num_steps: int,
        deterministic: bool = False,
    ) -> Dict[str, float]:
        """
        Collect a rollout of experience.

        Args:
            env: Environment to collect from
            num_steps: Number of steps to collect
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary with rollout statistics
        """
        self.policy.eval()
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        for step in range(num_steps):
            # Get action from policy
            with torch.no_grad():
                output = self.policy.get_action_and_value(
                    state_tensor,
                    deterministic=deterministic,
                )

            action = output["action"].cpu().numpy()[0]
            value = output["value"]
            log_prob = output["log_prob"]

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Store in buffer
            self.buffer.add(
                state=state_tensor.squeeze(0),
                action=output["action"],
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
            )

            # Update stats
            current_episode_reward += reward
            current_episode_length += 1
            self.training_stats["total_steps"] += 1

            # Check if episode done
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                self.training_stats["total_episodes"] += 1

                current_episode_reward = 0
                current_episode_length = 0

                state = env.reset()
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Compute last value for GAE
        with torch.no_grad():
            last_value = self.policy.get_action_and_value(state_tensor)["value"]

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        return {
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step using data in buffer.

        Returns:
            Dictionary with training metrics
        """
        self.policy.train()

        policy_losses = []
        value_losses = []
        entropies = []

        # Get all data from buffer
        all_data = self.buffer.get()

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = all_data["advantages"]
            all_data["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs over the data
        for epoch in range(self.config.num_epochs):
            # Create minibatches
            batch_size = len(self.buffer)
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_states = all_data["states"][mb_indices]
                mb_actions = all_data["actions"][mb_indices]
                mb_old_log_probs = all_data["old_log_probs"][mb_indices]
                mb_advantages = all_data["advantages"][mb_indices]
                mb_returns = all_data["returns"][mb_indices]

                # Forward pass through policy
                output = self.policy.get_action_and_value(mb_states, action=mb_actions)

                # Compute PPO loss
                ratio = torch.exp(output["log_prob"] - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                if self.config.clip_value_loss:
                    value_pred_clipped = all_data["returns"][mb_indices] + torch.clamp(
                        output["value"] - all_data["returns"][mb_indices],
                        -self.config.clip_epsilon,
                        self.config.clip_epsilon,
                    )
                    value_loss_clipped = F.mse_loss(value_pred_clipped, mb_returns)
                    value_loss_unclipped = F.mse_loss(output["value"], mb_returns)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                else:
                    value_loss = F.mse_loss(output["value"], mb_returns)

                # Entropy bonus
                entropy = output["entropy"].mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # Track stats
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        # Update stats
        self.training_stats["policy_loss"].append(np.mean(policy_losses))
        self.training_stats["value_loss"].append(np.mean(value_losses))
        self.training_stats["entropy"].append(np.mean(entropies))

        # Clear buffer
        self.buffer.reset()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }

    def save(self, path: str):
        """Save policy and optimizer state."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "config": self.config,
        }, path)
        logger.info(f"Saved PPO trainer to {path}")

    def load(self, path: str):
        """Load policy and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]
        logger.info(f"Loaded PPO trainer from {path}")


def create_ppo_trainer(
    policy: nn.Module,
    config: Optional[PPOConfig] = None,
    device: Optional[str] = None,
) -> PPOTrainer:
    """
    Factory function to create PPO trainer.

    Args:
        policy: Policy network to train
        config: PPO configuration
        device: Device to use (cuda/cpu)

    Returns:
        PPO trainer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return PPOTrainer(policy=policy, config=config, device=device)
