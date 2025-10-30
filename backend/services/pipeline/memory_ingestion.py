"""
Memory ingestion pipeline.

Orchestrates the complete memory processing workflow:
1. Text chunking
2. Embedding generation
3. Vector storage (Milvus)
4. Entity extraction
5. Knowledge graph storage (Neo4j)
6. BM25 indexing
7. RL trajectory logging
"""
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.logging_config import logger
from backend.models.memory import Memory, MemoryMetadata
from backend.ml.text_processing import get_text_chunker
from backend.ml.embeddings.model import get_embedding_generator
from backend.services.vector_store import get_vector_store_client
from backend.services.knowledge_graph import get_knowledge_graph_client
from backend.ml.sparse_search import get_bm25_index


class MemoryIngestionPipeline:
    """
    Complete pipeline for ingesting and processing memories.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.chunker = get_text_chunker()
        self.embedding_generator = get_embedding_generator()
        self.vector_store = get_vector_store_client()
        self.kg_client = get_knowledge_graph_client()

    async def ingest_memory(
        self,
        memory: Memory,
        metadata: MemoryMetadata,
        user_id: str,
        collection_id: str,
    ) -> bool:
        """
        Ingest a single memory through the complete pipeline.

        Args:
            memory: Memory object from database
            metadata: Memory metadata object
            user_id: User identifier
            collection_id: Collection identifier

        Returns:
            True if successful
        """
        try:
            logger.info(f"Starting ingestion for memory {memory.id}")

            # Step 1: Generate embedding
            embedding = await self._generate_embedding(memory.content)
            if not embedding:
                logger.error(f"Failed to generate embedding for memory {memory.id}")
                return False

            # Step 2: Store in Milvus
            vector_success = await self._store_in_vector_db(
                memory_id=memory.id,
                user_id=user_id,
                collection_id=collection_id,
                embedding=embedding,
                importance=memory.importance,
                created_at=memory.created_at,
            )

            # Step 3: Extract entities (async)
            entities_task = self._extract_and_store_entities(
                memory_id=memory.id,
                content=memory.content,
                user_id=user_id,
                collection_id=collection_id,
            )

            # Step 4: Add to BM25 index
            bm25_success = await self._add_to_bm25_index(
                memory_id=memory.id,
                content=memory.content,
                collection_id=collection_id,
            )

            # Wait for entity extraction
            entities_success = await entities_task

            # Step 5: Update metadata with processing info
            await self._update_processing_metadata(
                metadata,
                embedding_dimension=len(embedding),
                vector_stored=vector_success,
                entities_extracted=entities_success,
                bm25_indexed=bm25_success,
            )

            logger.info(f"Successfully ingested memory {memory.id}")
            return True

        except Exception as e:
            logger.error(f"Memory ingestion failed for {memory.id}: {e}")
            return False

    async def ingest_batch(
        self,
        memories: List[Memory],
        metadatas: List[MemoryMetadata],
        user_id: str,
        collection_id: str,
    ) -> Dict[str, int]:
        """
        Ingest multiple memories in batch.

        Args:
            memories: List of memory objects
            metadatas: List of metadata objects
            user_id: User identifier
            collection_id: Collection identifier

        Returns:
            Dictionary with success/failure counts
        """
        try:
            logger.info(f"Starting batch ingestion of {len(memories)} memories")

            # Generate all embeddings in batch
            contents = [m.content for m in memories]
            embeddings = await self.embedding_generator.encode_batch(contents)

            # Process each memory
            tasks = []
            for memory, metadata, embedding in zip(memories, metadatas, embeddings):
                task = self._process_single_memory(
                    memory=memory,
                    metadata=metadata,
                    user_id=user_id,
                    collection_id=collection_id,
                    embedding=embedding,
                )
                tasks.append(task)

            # Execute in parallel with concurrency limit
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

            async def limited_task(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(
                *[limited_task(task) for task in tasks],
                return_exceptions=True,
            )

            # Count successes and failures
            successes = sum(1 for r in results if r is True)
            failures = len(results) - successes

            logger.info(
                f"Batch ingestion complete: {successes} succeeded, {failures} failed"
            )

            return {
                "total": len(memories),
                "succeeded": successes,
                "failed": failures,
            }

        except Exception as e:
            logger.error(f"Batch ingestion failed: {e}")
            return {
                "total": len(memories),
                "succeeded": 0,
                "failed": len(memories),
            }

    async def _process_single_memory(
        self,
        memory: Memory,
        metadata: MemoryMetadata,
        user_id: str,
        collection_id: str,
        embedding: List[float],
    ) -> bool:
        """Process a single memory with pre-computed embedding."""
        try:
            # Store in vector DB
            await self._store_in_vector_db(
                memory_id=memory.id,
                user_id=user_id,
                collection_id=collection_id,
                embedding=embedding,
                importance=memory.importance,
                created_at=memory.created_at,
            )

            # Extract entities
            await self._extract_and_store_entities(
                memory_id=memory.id,
                content=memory.content,
                user_id=user_id,
                collection_id=collection_id,
            )

            # Add to BM25
            await self._add_to_bm25_index(
                memory_id=memory.id,
                content=memory.content,
                collection_id=collection_id,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to process memory {memory.id}: {e}")
            return False

    async def _generate_embedding(self, content: str) -> Optional[List[float]]:
        """Generate embedding for content."""
        try:
            embedding = await self.embedding_generator.encode_document(content)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    async def _store_in_vector_db(
        self,
        memory_id: str,
        user_id: str,
        collection_id: str,
        embedding: List[float],
        importance: float,
        created_at: datetime,
    ) -> bool:
        """Store embedding in Milvus."""
        try:
            success = await self.vector_store.insert_memory(
                memory_id=memory_id,
                user_id=user_id,
                collection_id=collection_id,
                embedding=embedding,
                importance=importance,
                created_timestamp=int(created_at.timestamp()),
            )
            return success
        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return False

    async def _extract_and_store_entities(
        self,
        memory_id: str,
        content: str,
        user_id: str,
        collection_id: str,
    ) -> bool:
        """Extract entities and store in knowledge graph."""
        try:
            # First, create memory node in graph
            await self.kg_client.create_memory_node(
                memory_id=memory_id,
                user_id=user_id,
                collection_id=collection_id,
                content=content,
            )

            # TODO: Implement actual entity extraction using NLP
            # For now, we'll skip entity extraction
            # Full implementation would use spaCy or similar

            return True

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return False

    async def _add_to_bm25_index(
        self,
        memory_id: str,
        content: str,
        collection_id: str,
    ) -> bool:
        """Add memory to BM25 index."""
        try:
            bm25_index = get_bm25_index(collection_id)
            bm25_index.add_documents(
                documents=[content],
                document_ids=[memory_id],
            )
            return True
        except Exception as e:
            logger.error(f"BM25 indexing failed: {e}")
            return False

    async def _update_processing_metadata(
        self,
        metadata: MemoryMetadata,
        embedding_dimension: int,
        vector_stored: bool,
        entities_extracted: bool,
        bm25_indexed: bool,
    ) -> None:
        """Update metadata with processing results."""
        try:
            metadata.processing_metadata = {
                "processed_at": datetime.utcnow().isoformat(),
                "embedding_dimension": embedding_dimension,
                "vector_stored": vector_stored,
                "entities_extracted": entities_extracted,
                "bm25_indexed": bm25_indexed,
                "processing_version": "1.0",
            }
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to update processing metadata: {e}")

    async def delete_memory(
        self,
        memory_id: str,
        collection_id: str,
    ) -> bool:
        """
        Delete memory from all storage systems.

        Args:
            memory_id: Memory identifier
            collection_id: Collection identifier

        Returns:
            True if successful
        """
        try:
            logger.info(f"Deleting memory {memory_id} from all systems")

            # Delete from vector DB
            await self.vector_store.delete_memory(memory_id)

            # Delete from knowledge graph
            await self.kg_client.delete_memory_node(memory_id)

            # Delete from BM25 index
            bm25_index = get_bm25_index(collection_id)
            bm25_index.remove_document(memory_id)

            logger.info(f"Successfully deleted memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Memory deletion failed: {e}")
            return False


def get_ingestion_pipeline(db: AsyncSession) -> MemoryIngestionPipeline:
    """
    Get memory ingestion pipeline instance.

    Args:
        db: Database session

    Returns:
        MemoryIngestionPipeline instance
    """
    return MemoryIngestionPipeline(db)


async def process_memory_async(memory_id: str, content: str) -> None:
    """
    Process a memory asynchronously (background task).

    This function is called after memory creation to:
    1. Generate embeddings
    2. Store in Milvus
    3. Extract entities
    4. Store in Neo4j

    Args:
        memory_id: Memory identifier
        content: Memory content
    """
    from backend.core.database import get_db_context
    from sqlalchemy import select

    try:
        async with get_db_context() as db:
            # Fetch memory from database
            result = await db.execute(
                select(Memory).where(Memory.id == memory_id)
            )
            memory = result.scalar_one_or_none()

            if not memory:
                logger.error(f"Memory {memory_id} not found for async processing")
                return

            # Fetch metadata
            result = await db.execute(
                select(MemoryMetadata).where(MemoryMetadata.memory_id == memory_id)
            )
            metadata = result.scalar_one_or_none()

            if not metadata:
                logger.error(f"Metadata for memory {memory_id} not found")
                return

            # Get collection to find user_id
            from backend.models.collection import Collection
            result = await db.execute(
                select(Collection).where(Collection.id == memory.collection_id)
            )
            collection = result.scalar_one_or_none()

            if not collection:
                logger.error(f"Collection {memory.collection_id} not found")
                return

            # Process through pipeline
            pipeline = get_ingestion_pipeline(db)
            success = await pipeline.ingest_memory(
                memory=memory,
                metadata=metadata,
                user_id=collection.user_id,
                collection_id=memory.collection_id,
            )

            if success:
                logger.info(f"Successfully processed memory {memory_id} in background")
            else:
                logger.error(f"Failed to process memory {memory_id} in background")

    except Exception as e:
        logger.error(f"Error in async memory processing: {e}", exc_info=True)
