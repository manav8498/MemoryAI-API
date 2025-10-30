"""
Policy Network for Memory Management RL.

Neural network that learns when to ADD, UPDATE, DELETE, or keep (NOOP) memories.
Based on Memory-R1 architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np

from backend.core.logging_config import logger


class MemoryPolicyNetwork(nn.Module):
    """
    Policy network for memory operations.

    Takes as input:
    - Current memory state representation
    - New information (extracted from dialogue)
    - Query context

    Outputs:
    - Action logits: [ADD, UPDATE, DELETE, NOOP]
    - Value estimate: Expected future reward
    """

    def __init__(
        self,
        state_dim: int = 768,  # Embedding dimension
        hidden_dim: int = 512,
        action_dim: int = 4,  # ADD, UPDATE, DELETE, NOOP
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # State encoder
        encoder_layers = []
        input_dim = state_dim
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through policy network.

        Args:
            state: State tensor [batch_size, state_dim]
            return_value: Whether to return value estimate

        Returns:
            action_logits: [batch_size, action_dim]
            value: [batch_size, 1] if return_value else None
        """
        # Encode state
        encoded = self.encoder(state)

        # Get action logits
        action_logits = self.policy_head(encoded)

        # Get value estimate if requested
        value = self.value_head(encoded) if return_value else None

        return action_logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample action from policy and return log prob + value.

        Args:
            state: State tensor
            action: If provided, compute log prob of this action
            deterministic: If True, take argmax instead of sampling

        Returns:
            Dictionary with action, log_prob, entropy, value
        """
        action_logits, value = self.forward(state, return_value=True)

        # Create categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # Sample or use provided action
        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        # Get log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value.squeeze(-1),
            "probs": probs,
        }


class AnswerPolicyNetwork(nn.Module):
    """
    Policy network for answer agent.

    Decides which memories to select from retrieved candidates
    for answering the query.
    """

    def __init__(
        self,
        state_dim: int = 768,
        hidden_dim: int = 512,
        max_candidates: int = 60,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_candidates = max_candidates

        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Memory encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Selection head (binary: select or not)
        self.selection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        query: torch.Tensor,
        memories: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through answer policy network.

        Args:
            query: Query embedding [batch_size, state_dim]
            memories: Memory embeddings [batch_size, num_memories, state_dim]
            memory_mask: Mask for padded memories [batch_size, num_memories]

        Returns:
            selection_logits: [batch_size, num_memories]
            value: [batch_size]
        """
        batch_size = query.size(0)

        # Encode query
        query_encoded = self.query_encoder(query)  # [B, H]
        query_encoded = query_encoded.unsqueeze(1)  # [B, 1, H]

        # Encode memories
        memories_encoded = self.memory_encoder(memories)  # [B, M, H]

        # Cross-attention: query attends to memories
        attended, _ = self.cross_attention(
            query=query_encoded,
            key=memories_encoded,
            value=memories_encoded,
            key_padding_mask=~memory_mask if memory_mask is not None else None,
        )

        # Selection logits for each memory
        # Use attended query to score each memory
        selection_scores = []
        for i in range(memories.size(1)):
            # Combine attended query with each memory
            combined = memories_encoded[:, i, :] + attended.squeeze(1)
            score = self.selection_head(combined)
            selection_scores.append(score)

        selection_logits = torch.cat(selection_scores, dim=1)  # [B, M]

        # Value estimate from attended representation
        value = self.value_head(attended.squeeze(1)).squeeze(-1)  # [B]

        return selection_logits, value

    def select_memories(
        self,
        query: torch.Tensor,
        memories: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Select top-k memories based on policy.

        Args:
            query: Query embedding
            memories: Memory embeddings
            memory_mask: Padding mask
            top_k: Number of memories to select
            threshold: Probability threshold for selection

        Returns:
            Dictionary with selected indices, probabilities, value
        """
        selection_logits, value = self.forward(query, memories, memory_mask)

        # Convert to probabilities
        probs = torch.sigmoid(selection_logits)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

        # Apply threshold
        selected_mask = top_probs > threshold

        return {
            "indices": top_indices,
            "probabilities": top_probs,
            "mask": selected_mask,
            "value": value,
        }


def create_policy_network(
    network_type: str = "memory_manager",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create policy networks.

    Args:
        network_type: Type of network ("memory_manager" or "answer_agent")
        **kwargs: Arguments passed to network constructor

    Returns:
        Policy network
    """
    if network_type == "memory_manager":
        return MemoryPolicyNetwork(**kwargs)
    elif network_type == "answer_agent":
        return AnswerPolicyNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# Example usage
if __name__ == "__main__":
    # Test memory manager network
    batch_size = 4
    state_dim = 768

    memory_net = MemoryPolicyNetwork(state_dim=state_dim)
    state = torch.randn(batch_size, state_dim)

    result = memory_net.get_action_and_value(state)
    print("Memory Manager Output:")
    print(f"  Action: {result['action']}")
    print(f"  Log Prob: {result['log_prob']}")
    print(f"  Value: {result['value']}")
    print(f"  Probs: {result['probs']}")

    # Test answer agent network
    answer_net = AnswerPolicyNetwork(state_dim=state_dim)
    query = torch.randn(batch_size, state_dim)
    memories = torch.randn(batch_size, 60, state_dim)
    memory_mask = torch.ones(batch_size, 60, dtype=torch.bool)

    selection = answer_net.select_memories(query, memories, memory_mask, top_k=10)
    print("\nAnswer Agent Output:")
    print(f"  Selected indices: {selection['indices'][:, :5]}")  # Show first 5
    print(f"  Selection probs: {selection['probabilities'][:, :5]}")
    print(f"  Value: {selection['value']}")
