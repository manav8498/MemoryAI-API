"""
Answer Agent.

Learns to pre-select relevant memories from retrieved candidates.
Based on Memory-R1 architecture.
"""
import torch
from typing import Dict, List, Any, Optional
import numpy as np

from backend.core.logging_config import logger
from backend.rl.policy_network import AnswerPolicyNetwork
from backend.ml.embeddings.model import get_embedding_generator


class AnswerAgent:
    """
    Answer Agent using RL.

    Given a query and candidate memories (e.g., top-60 from RAG),
    selects the most relevant subset for answering.
    """

    def __init__(
        self,
        policy_network: AnswerPolicyNetwork,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        top_k: int = 10,
        selection_threshold: float = 0.5,
    ):
        self.policy = policy_network.to(device)
        self.device = device
        self.top_k = top_k
        self.selection_threshold = selection_threshold
        self.embedding_generator = get_embedding_generator()

        # Expose dimensions from policy network for training
        self.state_dim = policy_network.state_dim

    async def select_memories(
        self,
        query: str,
        candidate_memories: List[Dict[str, Any]],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Select relevant memories from candidates.

        Args:
            query: User query
            candidate_memories: List of candidate memory dictionaries
            deterministic: If True, use deterministic selection

        Returns:
            Dictionary with selected memories and metadata
        """
        self.policy.eval()

        if not candidate_memories:
            return {
                "selected_memories": [],
                "selected_indices": [],
                "selection_scores": [],
                "value": 0.0,
            }

        try:
            # Encode query
            query_embedding = await self.embedding_generator.encode_query(query)
            query_tensor = torch.tensor(
                query_embedding,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)  # [1, dim]

            # Encode candidate memories
            memory_texts = [m.get("content", "") for m in candidate_memories]
            memory_embeddings = []

            for text in memory_texts:
                emb = await self.embedding_generator.encode_document(text)
                memory_embeddings.append(emb)

            # Stack into tensor [1, num_memories, dim]
            memories_tensor = torch.tensor(
                memory_embeddings,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            # Create mask (all valid)
            num_memories = len(candidate_memories)
            memory_mask = torch.ones(1, num_memories, dtype=torch.bool, device=self.device)

            # Get selection from policy
            with torch.no_grad():
                selection = self.policy.select_memories(
                    query=query_tensor,
                    memories=memories_tensor,
                    memory_mask=memory_mask,
                    top_k=min(self.top_k, num_memories),
                    threshold=self.selection_threshold,
                )

            # Extract results
            selected_indices = selection["indices"][0].cpu().numpy()
            selection_probs = selection["probabilities"][0].cpu().numpy()
            selection_mask = selection["mask"][0].cpu().numpy()
            value = selection["value"][0].item()

            # Filter by threshold
            valid_selections = []
            for idx, prob, is_selected in zip(selected_indices, selection_probs, selection_mask):
                if is_selected:
                    valid_selections.append({
                        "index": int(idx),
                        "probability": float(prob),
                        "memory": candidate_memories[idx],
                    })

            # Sort by probability
            valid_selections.sort(key=lambda x: x["probability"], reverse=True)

            logger.info(
                f"Answer agent selected {len(valid_selections)} memories "
                f"from {len(candidate_memories)} candidates"
            )

            return {
                "selected_memories": [s["memory"] for s in valid_selections],
                "selected_indices": [s["index"] for s in valid_selections],
                "selection_scores": [s["probability"] for s in valid_selections],
                "value": value,
                "num_candidates": len(candidate_memories),
                "num_selected": len(valid_selections),
            }

        except Exception as e:
            logger.error(f"Answer agent selection failed: {e}")
            # Fallback: return top-k by original scores
            return {
                "selected_memories": candidate_memories[:self.top_k],
                "selected_indices": list(range(min(self.top_k, len(candidate_memories)))),
                "selection_scores": [1.0] * min(self.top_k, len(candidate_memories)),
                "value": 0.0,
                "error": str(e),
            }

    async def evaluate_selection(
        self,
        query: str,
        selected_memories: List[Dict[str, Any]],
        ground_truth_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate quality of memory selection.

        Args:
            query: User query
            selected_memories: Selected memories
            ground_truth_answer: Optional ground truth for evaluation

        Returns:
            Evaluation metrics
        """
        metrics = {
            "num_selected": len(selected_memories),
            "coverage": 0.0,
            "relevance": 0.0,
            "diversity": 0.0,
        }

        if not selected_memories:
            return metrics

        try:
            # Compute coverage (how much info is captured)
            selected_texts = [m.get("content", "") for m in selected_memories]
            combined_text = " ".join(selected_texts)
            metrics["coverage"] = min(len(combined_text.split()) / 500, 1.0)  # Normalize

            # Compute relevance (similarity to query)
            query_emb = await self.embedding_generator.encode_query(query)

            relevance_scores = []
            for text in selected_texts:
                mem_emb = await self.embedding_generator.encode_document(text)
                # Cosine similarity
                similarity = np.dot(query_emb, mem_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(mem_emb) + 1e-8
                )
                relevance_scores.append(similarity)

            metrics["relevance"] = float(np.mean(relevance_scores))

            # Compute diversity (how different memories are from each other)
            if len(selected_memories) > 1:
                memory_embs = []
                for text in selected_texts:
                    emb = await self.embedding_generator.encode_document(text)
                    memory_embs.append(emb)

                # Pairwise similarities
                similarities = []
                for i in range(len(memory_embs)):
                    for j in range(i + 1, len(memory_embs)):
                        sim = np.dot(memory_embs[i], memory_embs[j]) / (
                            np.linalg.norm(memory_embs[i]) * np.linalg.norm(memory_embs[j]) + 1e-8
                        )
                        similarities.append(sim)

                # Diversity = 1 - mean similarity
                metrics["diversity"] = 1.0 - float(np.mean(similarities))

            logger.debug(f"Selection evaluation: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Selection evaluation failed: {e}")
            return metrics


async def answer_with_agent(
    query: str,
    candidate_memories: List[Dict[str, Any]],
    answer_agent: AnswerAgent,
    llm_provider,
) -> Dict[str, Any]:
    """
    Answer query using answer agent + LLM.

    Args:
        query: User query
        candidate_memories: Candidate memories from retrieval
        answer_agent: Answer agent instance
        llm_provider: LLM provider for generation

    Returns:
        Answer and metadata
    """
    # Select relevant memories
    selection_result = await answer_agent.select_memories(
        query=query,
        candidate_memories=candidate_memories,
        deterministic=False,
    )

    selected_memories = selection_result["selected_memories"]

    # Build prompt with selected memories
    if not selected_memories:
        context = "No relevant memories found."
    else:
        context_parts = []
        for i, memory in enumerate(selected_memories, 1):
            content = memory.get("content", "")
            score = selection_result["selection_scores"][i - 1]
            context_parts.append(f"[Memory {i}] (relevance: {score:.3f})\n{content}\n")

        context = "\n".join(context_parts)

    prompt = f"""Based on the following relevant memories, please answer the user's query.

SELECTED MEMORIES:
{context}

USER QUERY: {query}

Please provide a comprehensive answer based on the selected memories.

Your answer:"""

    # Generate answer using LLM
    answer = await llm_provider.generate(
        prompt=prompt,
        temperature=0.7,
    )

    # Evaluate selection
    eval_metrics = await answer_agent.evaluate_selection(
        query=query,
        selected_memories=selected_memories,
    )

    return {
        "answer": answer,
        "selected_memories": selected_memories,
        "num_candidates": selection_result["num_candidates"],
        "num_selected": selection_result["num_selected"],
        "value_estimate": selection_result["value"],
        "evaluation": eval_metrics,
    }


def get_answer_agent(db=None) -> AnswerAgent:
    """
    Get Answer Agent instance.

    Args:
        db: Database session (optional, for compatibility)

    Returns:
        AnswerAgent instance with default policy network
    """
    from backend.rl.policy_network import AnswerPolicyNetwork
    from backend.core.config import settings

    policy_network = AnswerPolicyNetwork(
        state_dim=settings.EMBEDDING_DIMENSION,
        hidden_dim=256,
    )

    return AnswerAgent(policy_network=policy_network)
