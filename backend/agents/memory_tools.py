"""
Self-Editing Memory Tools for AI Agents.

LangChain-style tools that allow agents to modify their own memories.
Based on Letta/MemGPT architecture for active memory management.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib

from langchain.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.logging_config import logger
from backend.models.memory import Memory, MemoryMetadata
from backend.services.pipeline.memory_ingestion import get_ingestion_pipeline
from backend.services.vector_store import get_vector_store_client
from backend.reasoning.llm_providers import get_llm_provider


class MemoryTools:
    """Collection of memory manipulation tools for agents."""

    def __init__(self, db: AsyncSession, user_id: str, collection_id: str):
        self.db = db
        self.user_id = user_id
        self.collection_id = collection_id
        self.llm = get_llm_provider()

    async def memory_replace(
        self,
        memory_id: str,
        old_content: str,
        new_content: str,
    ) -> Dict[str, Any]:
        """
        Replace content in an existing memory.

        Args:
            memory_id: ID of memory to modify
            old_content: Content to replace
            new_content: New content

        Returns:
            Result dictionary
        """
        try:
            # Fetch memory
            result = await self.db.execute(
                select(Memory).where(Memory.id == memory_id)
            )
            memory = result.scalar_one_or_none()

            if not memory:
                return {
                    "success": False,
                    "error": f"Memory {memory_id} not found",
                }

            # Verify ownership
            if memory.collection_id != self.collection_id:
                return {
                    "success": False,
                    "error": "Not authorized to modify this memory",
                }

            # Replace content
            old_full_content = memory.content
            if old_content not in old_full_content:
                return {
                    "success": False,
                    "error": f"Content '{old_content}' not found in memory",
                }

            memory.content = old_full_content.replace(old_content, new_content)
            memory.content_hash = hashlib.sha256(memory.content.encode()).hexdigest()

            # Update metadata to track modification
            metadata_result = await self.db.execute(
                select(MemoryMetadata).where(MemoryMetadata.memory_id == memory_id)
            )
            metadata = metadata_result.scalar_one_or_none()

            if metadata:
                if "modifications" not in metadata.custom_metadata:
                    metadata.custom_metadata["modifications"] = []

                metadata.custom_metadata["modifications"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "old_content": old_content,
                    "new_content": new_content,
                    "type": "replace",
                })

            await self.db.commit()

            # Re-process embeddings
            pipeline = get_ingestion_pipeline(self.db)
            await pipeline.ingest_memory(
                memory=memory,
                metadata=metadata,
                user_id=self.user_id,
                collection_id=self.collection_id,
            )

            logger.info(f"Replaced content in memory {memory_id}")
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Replaced '{old_content}' with '{new_content}'",
            }

        except Exception as e:
            logger.error(f"memory_replace failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_insert(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Insert new memory.

        Args:
            content: Memory content
            importance: Importance score (0-1)
            metadata: Optional metadata

        Returns:
            Result dictionary
        """
        try:
            import uuid

            memory_id = str(uuid.uuid4())
            memory = Memory(
                id=memory_id,
                collection_id=self.collection_id,
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                importance=importance,
                source_type="agent_insert",
                created_at=datetime.utcnow(),
            )

            mem_metadata = MemoryMetadata(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                custom_metadata=metadata or {},
            )
            mem_metadata.custom_metadata["created_by"] = "agent_tool"
            mem_metadata.custom_metadata["created_at"] = datetime.utcnow().isoformat()

            self.db.add(memory)
            self.db.add(mem_metadata)
            await self.db.commit()

            # Process through ingestion pipeline
            pipeline = get_ingestion_pipeline(self.db)
            await pipeline.ingest_memory(
                memory=memory,
                metadata=mem_metadata,
                user_id=self.user_id,
                collection_id=self.collection_id,
            )

            logger.info(f"Agent inserted new memory: {memory_id}")
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Inserted new memory: {content[:50]}...",
            }

        except Exception as e:
            logger.error(f"memory_insert failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_delete(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            Result dictionary
        """
        try:
            # Fetch memory
            result = await self.db.execute(
                select(Memory).where(Memory.id == memory_id)
            )
            memory = result.scalar_one_or_none()

            if not memory:
                return {
                    "success": False,
                    "error": f"Memory {memory_id} not found",
                }

            # Verify ownership
            if memory.collection_id != self.collection_id:
                return {
                    "success": False,
                    "error": "Not authorized to delete this memory",
                }

            # Delete from all storage systems
            pipeline = get_ingestion_pipeline(self.db)
            await pipeline.delete_memory(memory_id, self.collection_id)

            # Delete from database
            await self.db.delete(memory)
            await self.db.commit()

            logger.info(f"Agent deleted memory: {memory_id}")
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Deleted memory {memory_id}",
            }

        except Exception as e:
            logger.error(f"memory_delete failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_rethink(self, memory_id: str) -> Dict[str, Any]:
        """
        Re-evaluate and potentially update a memory.

        Uses LLM to reconsider the memory and suggest improvements.

        Args:
            memory_id: ID of memory to rethink

        Returns:
            Result dictionary with suggestions
        """
        try:
            # Fetch memory
            result = await self.db.execute(
                select(Memory).where(Memory.id == memory_id)
            )
            memory = result.scalar_one_or_none()

            if not memory:
                return {
                    "success": False,
                    "error": f"Memory {memory_id} not found",
                }

            # Ask LLM to reconsider
            prompt = f"""Please reconsider the following memory and provide your analysis:

MEMORY CONTENT:
{memory.content}

MEMORY METADATA:
- Created: {memory.created_at}
- Importance: {memory.importance}
- Access Count: {memory.access_count}

Please analyze:
1. Is this memory still accurate and useful?
2. Should it be updated, merged with other memories, or deleted?
3. What improvements would you suggest?
4. What is the confidence in this memory's accuracy?

Provide your analysis and recommendations:"""

            analysis = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,
            )

            # Update metadata with reflection
            metadata_result = await self.db.execute(
                select(MemoryMetadata).where(MemoryMetadata.memory_id == memory_id)
            )
            metadata = metadata_result.scalar_one_or_none()

            if metadata:
                if "reflections" not in metadata.custom_metadata:
                    metadata.custom_metadata["reflections"] = []

                metadata.custom_metadata["reflections"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "analysis": analysis,
                })

                await self.db.commit()

            logger.info(f"Agent reflected on memory: {memory_id}")
            return {
                "success": True,
                "memory_id": memory_id,
                "analysis": analysis,
                "message": "Memory reflection complete",
            }

        except Exception as e:
            logger.error(f"memory_rethink failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_search(
        self,
        query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Search memories (for agent to access its own memory).

        Args:
            query: Search query
            limit: Number of results

        Returns:
            Search results
        """
        try:
            from backend.services.hybrid_search import search_memories

            results = await search_memories(
                query=query,
                user_id=self.user_id,
                db=self.db,
                collection_id=self.collection_id,
                limit=limit,
            )

            return {
                "success": True,
                "results": results,
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"memory_search failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def memory_consolidate(
        self,
        memory_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Consolidate multiple memories into one.

        Args:
            memory_ids: List of memory IDs to consolidate

        Returns:
            Result dictionary
        """
        try:
            # Fetch memories
            result = await self.db.execute(
                select(Memory).where(Memory.id.in_(memory_ids))
            )
            memories = list(result.scalars().all())

            if len(memories) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 memories to consolidate",
                }

            # Combine contents and ask LLM to consolidate
            combined = "\n\n".join([
                f"[Memory {i+1}]\n{m.content}"
                for i, m in enumerate(memories)
            ])

            prompt = f"""Please consolidate the following memories into a single, coherent memory:

{combined}

Create a consolidated memory that:
1. Captures all important information
2. Removes redundancy
3. Resolves any conflicts
4. Is concise and clear

Consolidated memory:"""

            consolidated_content = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,
            )

            # Create new consolidated memory
            new_memory_result = await self.memory_insert(
                content=f"[CONSOLIDATED] {consolidated_content}",
                importance=max(m.importance for m in memories),
                metadata={
                    "consolidated_from": memory_ids,
                    "consolidation_date": datetime.utcnow().isoformat(),
                },
            )

            # Archive original memories
            for memory_id in memory_ids:
                result = await self.db.execute(
                    select(MemoryMetadata).where(MemoryMetadata.memory_id == memory_id)
                )
                metadata = result.scalar_one_or_none()
                if metadata:
                    metadata.custom_metadata["archived"] = True
                    metadata.custom_metadata["consolidated_into"] = new_memory_result["memory_id"]

            await self.db.commit()

            logger.info(f"Consolidated {len(memory_ids)} memories")
            return {
                "success": True,
                "new_memory_id": new_memory_result["memory_id"],
                "archived_memory_ids": memory_ids,
                "message": f"Consolidated {len(memory_ids)} memories",
            }

        except Exception as e:
            logger.error(f"memory_consolidate failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# LangChain tool wrappers for use in agent chains
def create_langchain_memory_tools(
    db: AsyncSession,
    user_id: str,
    collection_id: str,
) -> List:
    """
    Create LangChain-compatible tool instances.

    Args:
        db: Database session
        user_id: User ID
        collection_id: Collection ID

    Returns:
        List of tool instances
    """
    tools = MemoryTools(db, user_id, collection_id)

    @tool
    async def replace_memory_content(memory_id: str, old_content: str, new_content: str) -> str:
        """Replace content in an existing memory. Use when you need to correct or update a memory."""
        result = await tools.memory_replace(memory_id, old_content, new_content)
        return str(result)

    @tool
    async def insert_new_memory(content: str, importance: float = 0.5) -> str:
        """Insert a new memory. Use when you learn something new that should be remembered."""
        result = await tools.memory_insert(content, importance)
        return str(result)

    @tool
    async def delete_memory(memory_id: str) -> str:
        """Delete a memory. Use when a memory is outdated, incorrect, or no longer needed."""
        result = await tools.memory_delete(memory_id)
        return str(result)

    @tool
    async def reflect_on_memory(memory_id: str) -> str:
        """Reflect on and re-evaluate a memory. Use to check if a memory is still accurate."""
        result = await tools.memory_rethink(memory_id)
        return str(result)

    @tool
    async def search_my_memories(query: str, limit: int = 5) -> str:
        """Search your own memories. Use to recall relevant information."""
        result = await tools.memory_search(query, limit)
        return str(result)

    @tool
    async def consolidate_memories(memory_ids: str) -> str:
        """Consolidate multiple memories into one. memory_ids should be comma-separated."""
        ids = [id.strip() for id in memory_ids.split(",")]
        result = await tools.memory_consolidate(ids)
        return str(result)

    return [
        replace_memory_content,
        insert_new_memory,
        delete_memory,
        reflect_on_memory,
        search_my_memories,
        consolidate_memories,
    ]
