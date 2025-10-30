"""
Memory Manager Agent.

Learns when to ADD, UPDATE, DELETE, or keep (NOOP) memories using RL.
Based on Memory-R1 architecture.
"""
import torch
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.logging_config import logger
from backend.models.memory import Memory, MemoryMetadata
from backend.rl.policy_network import MemoryPolicyNetwork
from backend.ml.embeddings.model import get_embedding_generator


class MemoryOperation(Enum):
    """Memory operations that the agent can perform."""
    ADD = 0
    UPDATE = 1
    DELETE = 2
    NOOP = 3  # No operation


@dataclass
class MemoryState:
    """Represents the current memory state."""
    extracted_info: str  # New information from dialogue
    query_context: Optional[str]  # Current query
    existing_memories: List[Dict[str, Any]]  # Current memory state
    user_id: str
    collection_id: str
    metadata: Dict[str, Any]


class MemoryManagerAgent:
    """
    Memory Manager Agent using RL.

    Decides when to:
    - ADD: Create new memory
    - UPDATE: Modify existing memory
    - DELETE: Remove outdated memory
    - NOOP: No operation needed
    """

    def __init__(
        self,
        policy_network: MemoryPolicyNetwork,
        db: AsyncSession,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy = policy_network.to(device)
        self.db = db
        self.device = device
        self.embedding_generator = get_embedding_generator()

        # Expose dimensions from policy network for training
        self.state_dim = policy_network.state_dim
        self.action_dim = policy_network.action_dim

    async def encode_state(self, state: MemoryState) -> torch.Tensor:
        """
        Encode memory state into vector representation.

        Args:
            state: Current memory state

        Returns:
            State embedding tensor
        """
        # Combine extracted info and query context
        text_parts = [state.extracted_info]
        if state.query_context:
            text_parts.append(f"Query: {state.query_context}")

        # Add recent memory context
        if state.existing_memories:
            recent = state.existing_memories[:5]  # Last 5 memories
            memory_text = " | ".join([m.get("content", "")[:100] for m in recent])
            text_parts.append(f"Recent memories: {memory_text}")

        combined_text = " ".join(text_parts)

        # Generate embedding
        embedding = await self.embedding_generator.encode_query(combined_text)

        return torch.tensor(embedding, dtype=torch.float32, device=self.device)

    async def select_action(
        self,
        state: MemoryState,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Select memory operation based on current state.

        Args:
            state: Current memory state
            deterministic: If True, take argmax action

        Returns:
            Dictionary with action, log_prob, value, etc.
        """
        self.policy.eval()

        # Encode state
        state_tensor = await self.encode_state(state)
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

        # Get action from policy
        with torch.no_grad():
            output = self.policy.get_action_and_value(
                state_tensor,
                deterministic=deterministic,
            )

        action_idx = output["action"].item()
        operation = MemoryOperation(action_idx)

        return {
            "operation": operation,
            "action_idx": action_idx,
            "log_prob": output["log_prob"].item(),
            "value": output["value"].item(),
            "probabilities": output["probs"].cpu().numpy()[0],
        }

    async def execute_operation(
        self,
        operation: MemoryOperation,
        state: MemoryState,
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the selected memory operation.

        Args:
            operation: Operation to perform
            state: Current state
            memory_id: Memory ID for UPDATE/DELETE operations

        Returns:
            Result of operation
        """
        try:
            if operation == MemoryOperation.ADD:
                result = await self._add_memory(state)

            elif operation == MemoryOperation.UPDATE:
                if memory_id is None:
                    # Find most similar memory to update
                    memory_id = await self._find_similar_memory(state)

                result = await self._update_memory(memory_id, state)

            elif operation == MemoryOperation.DELETE:
                if memory_id is None:
                    # Find memory to delete (least important or contradictory)
                    memory_id = await self._find_memory_to_delete(state)

                result = await self._delete_memory(memory_id, state)

            else:  # NOOP
                result = {
                    "operation": "NOOP",
                    "success": True,
                    "message": "No operation performed",
                }

            logger.info(f"Executed {operation.name}: {result.get('message')}")
            return result

        except Exception as e:
            logger.error(f"Failed to execute operation {operation.name}: {e}")
            return {
                "operation": operation.name,
                "success": False,
                "error": str(e),
            }

    async def _add_memory(self, state: MemoryState) -> Dict[str, Any]:
        """Create new memory."""
        from backend.services.pipeline.memory_ingestion import get_ingestion_pipeline
        import uuid
        from datetime import datetime

        # Create memory object
        memory_id = str(uuid.uuid4())
        memory = Memory(
            id=memory_id,
            collection_id=state.collection_id,
            content=state.extracted_info,
            content_hash=hash(state.extracted_info),
            importance=0.7,  # Default importance
            created_at=datetime.utcnow(),
        )

        # Create metadata
        metadata = MemoryMetadata(
            id=str(uuid.uuid4()),
            memory_id=memory_id,
            custom_metadata=state.metadata,
        )

        # Add to database
        self.db.add(memory)
        self.db.add(metadata)
        await self.db.commit()

        # Process through ingestion pipeline
        pipeline = get_ingestion_pipeline(self.db)
        await pipeline.ingest_memory(
            memory=memory,
            metadata=metadata,
            user_id=state.user_id,
            collection_id=state.collection_id,
        )

        return {
            "operation": "ADD",
            "success": True,
            "memory_id": memory_id,
            "message": f"Added new memory: {state.extracted_info[:50]}...",
        }

    async def _update_memory(
        self,
        memory_id: str,
        state: MemoryState,
    ) -> Dict[str, Any]:
        """Update existing memory."""
        # Fetch memory
        result = await self.db.execute(
            select(Memory).where(Memory.id == memory_id)
        )
        memory = result.scalar_one_or_none()

        if not memory:
            return {
                "operation": "UPDATE",
                "success": False,
                "error": f"Memory {memory_id} not found",
            }

        # Update content (merge or replace)
        old_content = memory.content
        memory.content = f"{old_content} | Updated: {state.extracted_info}"
        memory.importance = min(memory.importance + 0.1, 1.0)  # Boost importance

        await self.db.commit()

        # Re-process embeddings
        from backend.services.pipeline.memory_ingestion import get_ingestion_pipeline
        pipeline = get_ingestion_pipeline(self.db)

        metadata_result = await self.db.execute(
            select(MemoryMetadata).where(MemoryMetadata.memory_id == memory_id)
        )
        metadata = metadata_result.scalar_one()

        await pipeline.ingest_memory(
            memory=memory,
            metadata=metadata,
            user_id=state.user_id,
            collection_id=state.collection_id,
        )

        return {
            "operation": "UPDATE",
            "success": True,
            "memory_id": memory_id,
            "message": f"Updated memory {memory_id}",
        }

    async def _delete_memory(
        self,
        memory_id: str,
        state: MemoryState,
    ) -> Dict[str, Any]:
        """Delete memory."""
        from backend.services.pipeline.memory_ingestion import get_ingestion_pipeline

        pipeline = get_ingestion_pipeline(self.db)
        success = await pipeline.delete_memory(memory_id, state.collection_id)

        if success:
            # Delete from database
            result = await self.db.execute(
                select(Memory).where(Memory.id == memory_id)
            )
            memory = result.scalar_one_or_none()
            if memory:
                await self.db.delete(memory)
                await self.db.commit()

        return {
            "operation": "DELETE",
            "success": success,
            "memory_id": memory_id,
            "message": f"Deleted memory {memory_id}" if success else "Failed to delete",
        }

    async def _find_similar_memory(self, state: MemoryState) -> Optional[str]:
        """Find most similar memory for UPDATE operation."""
        from backend.services.vector_store import get_vector_store_client

        # Generate embedding for new info
        embedding = await self.embedding_generator.encode_query(state.extracted_info)

        # Search for similar memories
        vector_store = get_vector_store_client()
        results = await vector_store.search_similar(
            query_embedding=embedding,
            user_id=state.user_id,
            collection_id=state.collection_id,
            top_k=1,
        )

        return results[0]["memory_id"] if results else None

    async def _find_memory_to_delete(self, state: MemoryState) -> Optional[str]:
        """Find memory to delete (least important or contradictory)."""
        # Get all memories for user
        result = await self.db.execute(
            select(Memory)
            .where(Memory.collection_id == state.collection_id)
            .order_by(Memory.importance.asc())
            .limit(1)
        )
        memory = result.scalar_one_or_none()

        return memory.id if memory else None


async def process_dialogue_turn(
    dialogue_text: str,
    query_context: Optional[str],
    user_id: str,
    collection_id: str,
    memory_manager: MemoryManagerAgent,
    db: AsyncSession,
) -> Dict[str, Any]:
    """
    Process a dialogue turn with memory manager.

    Args:
        dialogue_text: New dialogue text
        query_context: Optional query context
        user_id: User ID
        collection_id: Collection ID
        memory_manager: Memory manager agent
        db: Database session

    Returns:
        Result of memory operation
    """
    # Get existing memories
    result = await db.execute(
        select(Memory)
        .where(Memory.collection_id == collection_id)
        .order_by(Memory.created_at.desc())
        .limit(10)
    )
    existing_memories = [
        {"id": m.id, "content": m.content, "importance": m.importance}
        for m in result.scalars().all()
    ]

    # Create state
    state = MemoryState(
        extracted_info=dialogue_text,
        query_context=query_context,
        existing_memories=existing_memories,
        user_id=user_id,
        collection_id=collection_id,
        metadata={},
    )

    # Select action
    action_result = await memory_manager.select_action(state, deterministic=False)

    # Execute operation
    exec_result = await memory_manager.execute_operation(
        operation=action_result["operation"],
        state=state,
    )

    # Combine results
    return {
        **action_result,
        **exec_result,
    }


def get_memory_manager_agent(db: AsyncSession) -> MemoryManagerAgent:
    """
    Get Memory Manager Agent instance.

    Args:
        db: Database session

    Returns:
        MemoryManagerAgent instance
    """
    from backend.core.config import settings

    policy_network = MemoryPolicyNetwork(
        state_dim=settings.EMBEDDING_DIMENSION,
        hidden_dim=256,
        action_dim=4,  # ADD, UPDATE, DELETE, NOOP
    )

    return MemoryManagerAgent(
        policy_network=policy_network,
        db=db,
    )
