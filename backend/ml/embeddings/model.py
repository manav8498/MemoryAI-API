"""
Embedding model for converting text to dense vectors.
"""
from typing import List, Optional, Union
import asyncio
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from backend.core.config import settings
from backend.core.logging_config import logger


# Global embedding model instance
_embedding_model: Optional[SentenceTransformer] = None


async def load_embedding_model() -> None:
    """
    Load the embedding model into memory.
    Called during application startup.
    """
    global _embedding_model

    try:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")

        # Load model
        _embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
        )

        # Warm up model with a test encoding
        _ = _embedding_model.encode(
            ["This is a test sentence."],
            convert_to_numpy=True,
        )

        logger.info(
            f"Embedding model loaded successfully "
            f"(dimension: {_embedding_model.get_sentence_embedding_dimension()})"
        )

    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def get_embedding_model() -> SentenceTransformer:
    """
    Get the global embedding model instance.

    Returns:
        SentenceTransformer model

    Raises:
        RuntimeError: If model not loaded
    """
    global _embedding_model

    if _embedding_model is None:
        raise RuntimeError("Embedding model not loaded. Call load_embedding_model() first.")

    return _embedding_model


class EmbeddingGenerator:
    """
    High-level interface for generating embeddings.
    """

    def __init__(self):
        self.model = get_embedding_model()
        self.dimension = self.model.get_sentence_embedding_dimension()

    async def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode text into embeddings.

        Args:
            text: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar for batches

        Returns:
            Numpy array of embeddings (shape: [n_texts, dimension])
        """
        try:
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    text,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    batch_size=settings.EMBEDDING_BATCH_SIZE,
                ),
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise

    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Encode a batch of texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size (defaults to settings)

        Returns:
            List of embeddings as lists
        """
        try:
            batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

            embeddings = await self.encode_text(
                texts,
                normalize=True,
                show_progress=len(texts) > 100,
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise

    async def encode_query(self, query: str) -> List[float]:
        """
        Encode a search query.

        For some models, queries may need different encoding than documents.

        Args:
            query: Query text

        Returns:
            Query embedding as list
        """
        try:
            # For BGE models, add instruction prefix for queries
            if "bge" in settings.EMBEDDING_MODEL_NAME.lower():
                query = f"Represent this sentence for searching relevant passages: {query}"

            embedding = await self.encode_text(query, normalize=True)

            # Handle both single and batch results
            if len(embedding.shape) == 1:
                return embedding.tolist()
            else:
                return embedding[0].tolist()

        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise

    async def encode_document(self, document: str) -> List[float]:
        """
        Encode a document for storage.

        Args:
            document: Document text

        Returns:
            Document embedding as list
        """
        try:
            embedding = await self.encode_text(document, normalize=True)

            # Handle both single and batch results
            if len(embedding.shape) == 1:
                return embedding.tolist()
            else:
                return embedding[0].tolist()

        except Exception as e:
            logger.error(f"Failed to encode document: {e}")
            raise

    def compute_similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        try:
            # Convert to numpy if needed
            if isinstance(embedding1, list):
                embedding1 = np.array(embedding1)
            if isinstance(embedding2, list):
                embedding2 = np.array(embedding2)

            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )

            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0

    async def compute_batch_similarity(
        self,
        query_embedding: Union[List[float], np.ndarray],
        document_embeddings: List[Union[List[float], np.ndarray]],
    ) -> List[float]:
        """
        Compute similarity between query and multiple documents.

        Args:
            query_embedding: Query embedding
            document_embeddings: List of document embeddings

        Returns:
            List of similarity scores
        """
        try:
            # Convert to numpy
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)

            doc_matrix = np.array(
                [emb if isinstance(emb, np.ndarray) else np.array(emb)
                 for emb in document_embeddings]
            )

            # Compute cosine similarity
            similarities = np.dot(doc_matrix, query_embedding) / (
                np.linalg.norm(doc_matrix, axis=1) * np.linalg.norm(query_embedding)
            )

            return similarities.tolist()

        except Exception as e:
            logger.error(f"Failed to compute batch similarity: {e}")
            return [0.0] * len(document_embeddings)


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get embedding generator instance.

    Usage:
        generator = get_embedding_generator()
        embedding = await generator.encode_query("search query")
    """
    return EmbeddingGenerator()
