"""
Cross-Encoder Reranking for search results.

Provides accurate reranking of retrieved memories using cross-encoder models.
Typically provides 10-20% accuracy improvement over pure vector search.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from backend.core.logging_config import logger
from backend.core.config import settings


@dataclass
class RerankResult:
    """Result from reranking."""
    memory_id: str
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    rank: int


class CrossEncoderReranker:
    """
    Cross-encoder based reranker.

    Uses sentence-transformers cross-encoder models to rerank search results
    by computing relevance scores for (query, document) pairs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to run on (cuda/cpu)
            use_cache: Whether to cache model
        """
        self.model_name = model_name
        self.device = device
        self.use_cache = use_cache
        self._model = None

    def _load_model(self):
        """Lazy load cross-encoder model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )

            logger.info(f"Loaded cross-encoder model: {self.model_name}")
            return self._model

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            return None

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        combine_scores: bool = True,
        combination_weight: float = 0.7,
    ) -> List[RerankResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Search query
            results: List of search results with 'content' and 'score'
            top_k: Number of results to return (None = all)
            combine_scores: Whether to combine original + rerank scores
            combination_weight: Weight for rerank score (0-1)

        Returns:
            Reranked results
        """
        if not results:
            return []

        model = self._load_model()
        if model is None:
            # Fallback: return original results
            logger.warning("Cross-encoder not available, returning original results")
            return [
                RerankResult(
                    memory_id=r.get("memory_id", ""),
                    content=r.get("content", ""),
                    original_score=r.get("score", 0.0),
                    rerank_score=r.get("score", 0.0),
                    final_score=r.get("score", 0.0),
                    rank=i,
                )
                for i, r in enumerate(results)
            ]

        try:
            # Create (query, document) pairs
            pairs = []
            for result in results:
                content = result.get("content", "")
                # Truncate long content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                pairs.append([query, content])

            # Get rerank scores
            rerank_scores = model.predict(pairs, show_progress_bar=False)

            # Normalize rerank scores to 0-1
            rerank_scores = np.array(rerank_scores)
            if len(rerank_scores) > 1:
                min_score = rerank_scores.min()
                max_score = rerank_scores.max()
                if max_score > min_score:
                    rerank_scores = (rerank_scores - min_score) / (max_score - min_score)

            # Create rerank results
            reranked = []
            for i, (result, rerank_score) in enumerate(zip(results, rerank_scores)):
                original_score = result.get("score", 0.0)

                # Combine scores if requested
                if combine_scores:
                    final_score = (
                        combination_weight * rerank_score
                        + (1 - combination_weight) * original_score
                    )
                else:
                    final_score = rerank_score

                reranked.append(
                    RerankResult(
                        memory_id=result.get("memory_id", ""),
                        content=result.get("content", ""),
                        original_score=float(original_score),
                        rerank_score=float(rerank_score),
                        final_score=float(final_score),
                        rank=i,
                    )
                )

            # Sort by final score
            reranked.sort(key=lambda x: x.final_score, reverse=True)

            # Update ranks
            for i, result in enumerate(reranked):
                result.rank = i

            # Limit to top-k
            if top_k is not None:
                reranked = reranked[:top_k]

            logger.info(
                f"Reranked {len(results)} results, "
                f"returning top {len(reranked)}"
            )

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original results on error
            return [
                RerankResult(
                    memory_id=r.get("memory_id", ""),
                    content=r.get("content", ""),
                    original_score=r.get("score", 0.0),
                    rerank_score=r.get("score", 0.0),
                    final_score=r.get("score", 0.0),
                    rank=i,
                )
                for i, r in enumerate(results)
            ]


class TwoStageReranker:
    """
    Two-stage retrieval and reranking.

    Stage 1: Fast bi-encoder retrieval (top-100)
    Stage 2: Accurate cross-encoder reranking (top-10)
    """

    def __init__(
        self,
        cross_encoder: CrossEncoderReranker,
        stage1_k: int = 100,
        stage2_k: int = 10,
    ):
        self.cross_encoder = cross_encoder
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k

    async def retrieve_and_rerank(
        self,
        query: str,
        retrieval_fn,
        **retrieval_kwargs,
    ) -> List[RerankResult]:
        """
        Two-stage retrieve and rerank.

        Args:
            query: Search query
            retrieval_fn: Async function that performs retrieval
            **retrieval_kwargs: Arguments to retrieval function

        Returns:
            Reranked results
        """
        # Stage 1: Retrieve top-K with bi-encoder
        retrieval_kwargs["limit"] = self.stage1_k
        stage1_results = await retrieval_fn(query=query, **retrieval_kwargs)

        logger.info(f"Stage 1: Retrieved {len(stage1_results)} candidates")

        # Stage 2: Rerank with cross-encoder
        reranked = await self.cross_encoder.rerank(
            query=query,
            results=stage1_results,
            top_k=self.stage2_k,
        )

        logger.info(f"Stage 2: Reranked to top {len(reranked)}")

        return reranked


# Global reranker instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: Optional[str] = None,
) -> CrossEncoderReranker:
    """
    Get cross-encoder reranker instance.

    Args:
        model_name: Optional model name override

    Returns:
        CrossEncoderReranker instance
    """
    global _reranker

    if _reranker is None or (model_name and model_name != _reranker.model_name):
        _reranker = CrossEncoderReranker(
            model_name=model_name or settings.RERANKER_MODEL
        )

    return _reranker


async def rerank_search_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query
        results: Search results
        top_k: Number of results to return

    Returns:
        Reranked results as dictionaries
    """
    reranker = get_reranker()
    reranked = await reranker.rerank(
        query=query,
        results=results,
        top_k=top_k,
    )

    # Convert back to dict format
    return [
        {
            "memory_id": r.memory_id,
            "content": r.content,
            "score": r.final_score,
            "original_score": r.original_score,
            "rerank_score": r.rerank_score,
            "rank": r.rank,
        }
        for r in reranked
    ]


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_reranking():
        # Example results
        query = "What is machine learning?"
        results = [
            {
                "memory_id": "1",
                "content": "Machine learning is a subset of artificial intelligence...",
                "score": 0.85,
            },
            {
                "memory_id": "2",
                "content": "Deep learning uses neural networks with many layers...",
                "score": 0.75,
            },
            {
                "memory_id": "3",
                "content": "Python is a popular programming language...",
                "score": 0.65,
            },
        ]

        reranker = get_reranker()
        reranked = await reranker.rerank(query, results, top_k=2)

        for r in reranked:
            print(f"Rank {r.rank}: {r.memory_id} (score: {r.final_score:.3f})")

    asyncio.run(test_reranking())
