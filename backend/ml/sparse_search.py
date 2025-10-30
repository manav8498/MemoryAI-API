"""
Sparse search using BM25 for keyword-based retrieval.
"""
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict

from backend.core.logging_config import logger


class BM25Index:
    """
    BM25 index for sparse keyword search.
    """

    def __init__(self):
        self.index = None
        self.documents = []
        self.document_ids = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_documents(
        self,
        documents: List[str],
        document_ids: List[str],
    ) -> None:
        """
        Add documents to the BM25 index.

        Args:
            documents: List of document texts
            document_ids: List of corresponding document IDs
        """
        try:
            self.documents.extend(documents)
            self.document_ids.extend(document_ids)

            # Tokenize new documents
            new_tokenized = [self._tokenize(doc) for doc in documents]
            self.tokenized_corpus.extend(new_tokenized)

            # Rebuild index
            self.index = BM25Okapi(self.tokenized_corpus)

            logger.debug(f"Added {len(documents)} documents to BM25 index")

        except Exception as e:
            logger.error(f"Failed to add documents to BM25 index: {e}")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Search for documents using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document_id, score) tuples
        """
        try:
            if not self.index or not self.document_ids:
                logger.warning("BM25 index is empty")
                return []

            # Tokenize query
            tokenized_query = self._tokenize(query)

            # Get scores
            scores = self.index.get_scores(tokenized_query)

            # Get top-k results
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            results = [
                (self.document_ids[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0  # Only include non-zero scores
            ]

            logger.debug(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            document_id: Document ID to remove

        Returns:
            True if successful
        """
        try:
            if document_id not in self.document_ids:
                return False

            # Find index
            idx = self.document_ids.index(document_id)

            # Remove from all lists
            del self.document_ids[idx]
            del self.documents[idx]
            del self.tokenized_corpus[idx]

            # Rebuild index
            if self.tokenized_corpus:
                self.index = BM25Okapi(self.tokenized_corpus)
            else:
                self.index = None

            logger.debug(f"Removed document {document_id} from BM25 index")
            return True

        except Exception as e:
            logger.error(f"Failed to remove document from BM25 index: {e}")
            return False

    def get_size(self) -> int:
        """Get number of documents in index."""
        return len(self.document_ids)


# Global BM25 indexes per collection
_bm25_indexes: Dict[str, BM25Index] = {}


def get_bm25_index(collection_id: str) -> BM25Index:
    """
    Get or create BM25 index for a collection.

    Args:
        collection_id: Collection identifier

    Returns:
        BM25Index instance
    """
    global _bm25_indexes

    if collection_id not in _bm25_indexes:
        _bm25_indexes[collection_id] = BM25Index()

    return _bm25_indexes[collection_id]


def clear_bm25_index(collection_id: str) -> None:
    """
    Clear BM25 index for a collection.

    Args:
        collection_id: Collection identifier
    """
    global _bm25_indexes

    if collection_id in _bm25_indexes:
        del _bm25_indexes[collection_id]
        logger.info(f"Cleared BM25 index for collection {collection_id}")
