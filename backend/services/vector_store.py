"""
Vector database (Milvus) integration for semantic search.
"""
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
import numpy as np
import re

from backend.core.config import settings
from backend.core.logging_config import logger


# Global Milvus connection status
_milvus_connected = False


def sanitize_milvus_value(value: str, value_type: str = "id") -> str:
    """
    Sanitize values for Milvus query expressions to prevent injection.

    Args:
        value: The value to sanitize
        value_type: Type of value ("id" for UUIDs, "string" for general strings)

    Returns:
        Sanitized value safe for use in expressions

    Raises:
        ValueError: If value contains unsafe characters
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected string, got {type(value)}")

    if value_type == "id":
        # UUIDs should only contain alphanumeric characters and hyphens
        if not re.match(r'^[a-zA-Z0-9-]+$', value):
            raise ValueError(f"Invalid ID format: {value}")
    elif value_type == "string":
        # Escape special characters for string values
        # Disallow quotes and backslashes to prevent injection
        if any(char in value for char in ['"', "'", '\\', '\n', '\r']):
            raise ValueError(f"Invalid characters in string value")

    return value


async def init_vector_store() -> None:
    """
    Initialize connection to Milvus vector database.
    Called during application startup.
    """
    global _milvus_connected

    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
        )

        _milvus_connected = True
        logger.info("Connected to Milvus vector database")

        # Create default collection if it doesn't exist
        await _ensure_default_collection()

    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise


async def close_vector_store() -> None:
    """
    Close connection to Milvus.
    Called during application shutdown.
    """
    global _milvus_connected

    try:
        connections.disconnect(alias="default")
        _milvus_connected = False
        logger.info("Disconnected from Milvus")
    except Exception as e:
        logger.error(f"Error disconnecting from Milvus: {e}")


def _get_collection_name(user_id: str, collection_id: str) -> str:
    """
    Generate Milvus collection name for user/collection.

    Format: {prefix}_{user_id}_{collection_id}
    """
    return f"{settings.MILVUS_COLLECTION_PREFIX}_{user_id}_{collection_id}"


async def _ensure_default_collection() -> None:
    """
    Ensure the default memory collection exists in Milvus.
    """
    collection_name = f"{settings.MILVUS_COLLECTION_PREFIX}_memories"

    if utility.has_collection(collection_name):
        logger.info(f"Collection {collection_name} already exists")
        return

    # Define schema
    fields = [
        FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="collection_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.EMBEDDING_DIMENSION,
        ),
        FieldSchema(name="importance", dtype=DataType.FLOAT),
        FieldSchema(name="created_timestamp", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Memory embeddings for semantic search",
    )

    # Create collection
    collection = Collection(name=collection_name, schema=schema)

    # Create index for vector field
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024},
    }

    collection.create_index(field_name="embedding", index_params=index_params)

    logger.info(f"Created Milvus collection: {collection_name}")


class VectorStoreClient:
    """
    Client for interacting with Milvus vector database.
    """

    def __init__(self):
        self.collection_name = f"{settings.MILVUS_COLLECTION_PREFIX}_memories"
        self.collection = Collection(self.collection_name)
        # Load collection into memory once during initialization
        # This prevents expensive load() calls on every search
        try:
            self.collection.load()
            logger.debug(f"Loaded collection {self.collection_name} into memory")
        except Exception as e:
            logger.warning(f"Could not pre-load collection: {e}")

    async def insert_memory(
        self,
        memory_id: str,
        user_id: str,
        collection_id: str,
        embedding: List[float],
        importance: float,
        created_timestamp: int,
    ) -> bool:
        """
        Insert a memory embedding into Milvus.

        Args:
            memory_id: Unique memory identifier
            user_id: User identifier
            collection_id: Collection identifier
            embedding: Dense vector embedding
            importance: Memory importance score
            created_timestamp: Unix timestamp of creation

        Returns:
            True if successful
        """
        try:
            entities = [
                [memory_id],
                [user_id],
                [collection_id],
                [embedding],
                [importance],
                [created_timestamp],
            ]

            self.collection.insert(entities)
            self.collection.flush()

            logger.debug(f"Inserted memory {memory_id} into Milvus")
            return True

        except Exception as e:
            logger.error(f"Failed to insert memory into Milvus: {e}")
            return False

    async def search_similar(
        self,
        query_embedding: List[float],
        user_id: str,
        collection_id: Optional[str] = None,
        top_k: int = 10,
        importance_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity.

        Args:
            query_embedding: Query vector embedding
            user_id: User identifier for filtering
            collection_id: Optional collection filter
            top_k: Number of results to return
            importance_threshold: Minimum importance score

        Returns:
            List of search results with memory_id, score, and metadata
        """
        try:
            # Build search expression with sanitized values
            user_id = sanitize_milvus_value(user_id, "id")
            expr = f'user_id == "{user_id}"'

            if collection_id:
                collection_id = sanitize_milvus_value(collection_id, "id")
                expr += f' && collection_id == "{collection_id}"'

            if importance_threshold > 0:
                # Numeric values are safe, but validate it's actually a number
                if not isinstance(importance_threshold, (int, float)):
                    raise ValueError("importance_threshold must be numeric")
                expr += f" && importance >= {importance_threshold}"

            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }

            # Collection is already loaded in __init__, no need to load again

            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["memory_id", "collection_id", "importance", "created_timestamp"],
            )

            # Parse results
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "memory_id": hit.entity.get("memory_id"),
                        "collection_id": hit.entity.get("collection_id"),
                        "importance": hit.entity.get("importance"),
                        "created_timestamp": hit.entity.get("created_timestamp"),
                        "score": hit.score,
                        "distance": hit.distance,
                    })

            logger.info(f"Vector search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from Milvus.

        Args:
            memory_id: Memory identifier to delete

        Returns:
            True if successful
        """
        try:
            memory_id = sanitize_milvus_value(memory_id, "id")
            expr = f'memory_id == "{memory_id}"'
            self.collection.delete(expr)
            self.collection.flush()

            logger.debug(f"Deleted memory {memory_id} from Milvus")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory from Milvus: {e}")
            return False

    async def delete_collection_memories(self, collection_id: str) -> bool:
        """
        Delete all memories from a collection.

        Args:
            collection_id: Collection identifier

        Returns:
            True if successful
        """
        try:
            collection_id = sanitize_milvus_value(collection_id, "id")
            expr = f'collection_id == "{collection_id}"'
            self.collection.delete(expr)
            self.collection.flush()

            logger.info(f"Deleted all memories from collection {collection_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection memories from Milvus: {e}")
            return False

    async def count_memories(self, user_id: str, collection_id: Optional[str] = None) -> int:
        """
        Count memories for a user/collection.

        Args:
            user_id: User identifier
            collection_id: Optional collection filter

        Returns:
            Number of memories
        """
        try:
            user_id = sanitize_milvus_value(user_id, "id")
            expr = f'user_id == "{user_id}"'

            if collection_id:
                collection_id = sanitize_milvus_value(collection_id, "id")
                expr += f' && collection_id == "{collection_id}"'

            # Query for count (collection already loaded in __init__)
            count = self.collection.query(expr=expr, output_fields=["count(*)"])

            return count[0]["count(*)"] if count else 0

        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0


def get_vector_store_client() -> VectorStoreClient:
    """
    Get vector store client instance.

    Usage:
        client = get_vector_store_client()
        results = await client.search_similar(query_embedding, user_id)
    """
    return VectorStoreClient()


async def delete_memory_from_vector_store(memory_id: str, collection_id: str) -> bool:
    """
    Helper function to delete a memory from the vector store.

    Args:
        memory_id: Memory identifier
        collection_id: Collection identifier

    Returns:
        True if successful
    """
    client = get_vector_store_client()
    return await client.delete_memory(memory_id)
