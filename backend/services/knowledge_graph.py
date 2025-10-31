"""
Knowledge graph (Neo4j) integration for entity relationships.
"""
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable
import re

from backend.core.config import settings
from backend.core.logging_config import logger


def sanitize_relationship_type(relationship_type: str) -> str:
    """
    Sanitize relationship type for Neo4j Cypher queries.

    Neo4j relationship types must follow naming conventions:
    - Only uppercase letters, numbers, and underscores
    - Cannot start with a number
    - Used in MERGE/CREATE statements where parameterization isn't supported

    Args:
        relationship_type: The relationship type to sanitize

    Returns:
        Sanitized relationship type

    Raises:
        ValueError: If relationship type contains invalid characters
    """
    if not isinstance(relationship_type, str):
        raise ValueError(f"Expected string, got {type(relationship_type)}")

    if not relationship_type:
        raise ValueError("Relationship type cannot be empty")

    # Neo4j relationship types should be uppercase with underscores
    # Allow alphanumeric and underscores only, must not start with a number
    if not re.match(r'^[A-Z_][A-Z0-9_]*$', relationship_type):
        raise ValueError(
            f"Invalid relationship type: {relationship_type}. "
            "Must contain only uppercase letters, numbers, and underscores, "
            "and cannot start with a number."
        )

    # Additional length check to prevent extremely long relationship types
    if len(relationship_type) > 100:
        raise ValueError("Relationship type too long (max 100 characters)")

    return relationship_type


# Global Neo4j driver
_neo4j_driver: Optional[AsyncDriver] = None


async def init_knowledge_graph() -> None:
    """
    Initialize connection to Neo4j knowledge graph with proper connection pooling.
    Called during application startup.
    """
    global _neo4j_driver

    try:
        _neo4j_driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            # Connection pool configuration to prevent connection exhaustion
            max_connection_pool_size=50,  # Maximum connections in pool
            connection_acquisition_timeout=60.0,  # Timeout waiting for connection (seconds)
            max_connection_lifetime=3600,  # Close connections after 1 hour
            keep_alive=True,  # Send keep-alive messages
            connection_timeout=30.0,  # Timeout for establishing connection
        )

        # Verify connection
        await _neo4j_driver.verify_connectivity()

        logger.info("Connected to Neo4j knowledge graph with connection pooling")

        # Create indexes
        await _create_indexes()

    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise


async def close_knowledge_graph() -> None:
    """
    Close connection to Neo4j.
    Called during application shutdown.
    """
    global _neo4j_driver

    if _neo4j_driver:
        await _neo4j_driver.close()
        logger.info("Disconnected from Neo4j")


async def _create_indexes() -> None:
    """
    Create Neo4j indexes for performance.
    """
    global _neo4j_driver

    if not _neo4j_driver:
        return

    indexes = [
        "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_id)",
        "CREATE INDEX user_id_index IF NOT EXISTS FOR (m:Memory) ON (m.user_id)",
        "CREATE INDEX collection_id_index IF NOT EXISTS FOR (m:Memory) ON (m.collection_id)",
        "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    ]

    async with _neo4j_driver.session() as session:
        for index_query in indexes:
            try:
                await session.run(index_query)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

    logger.info("Neo4j indexes created")


class KnowledgeGraphClient:
    """
    Client for interacting with Neo4j knowledge graph.
    """

    def __init__(self):
        global _neo4j_driver
        self.driver = _neo4j_driver

        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")

    async def create_memory_node(
        self,
        memory_id: str,
        user_id: str,
        collection_id: str,
        content: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Create a memory node in the knowledge graph.

        Args:
            memory_id: Unique memory identifier
            user_id: User identifier
            collection_id: Collection identifier
            content: Memory content (preview)
            metadata: Additional metadata

        Returns:
            True if successful
        """
        try:
            async with self.driver.session() as session:
                query = """
                MERGE (m:Memory {memory_id: $memory_id})
                SET m.user_id = $user_id,
                    m.collection_id = $collection_id,
                    m.content_preview = $content_preview,
                    m.created_at = datetime()
                RETURN m
                """

                content_preview = content[:200] if len(content) > 200 else content

                await session.run(
                    query,
                    memory_id=memory_id,
                    user_id=user_id,
                    collection_id=collection_id,
                    content_preview=content_preview,
                )

            logger.debug(f"Created memory node {memory_id} in knowledge graph")
            return True

        except Exception as e:
            logger.error(f"Failed to create memory node: {e}")
            return False

    async def add_entity(
        self,
        memory_id: str,
        entity_name: str,
        entity_type: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Add an entity and link it to a memory.

        Args:
            memory_id: Memory identifier
            entity_name: Entity name (e.g., "John Smith")
            entity_type: Entity type (e.g., "PERSON", "ORG", "LOCATION")
            metadata: Additional entity metadata

        Returns:
            True if successful
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (m:Memory {memory_id: $memory_id})
                MERGE (e:Entity {name: $entity_name, type: $entity_type})
                MERGE (m)-[:MENTIONS]->(e)
                RETURN e
                """

                await session.run(
                    query,
                    memory_id=memory_id,
                    entity_name=entity_name,
                    entity_type=entity_type,
                )

            logger.debug(f"Added entity {entity_name} ({entity_type}) to memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            return False

    async def create_relationship(
        self,
        source_entity: str,
        target_entity: str,
        relationship_type: str,
        memory_id: str,
    ) -> bool:
        """
        Create a relationship between two entities.

        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            relationship_type: Type of relationship (e.g., "WORKS_AT", "LOCATED_IN")
            memory_id: Memory where relationship was mentioned

        Returns:
            True if successful
        """
        try:
            # Sanitize relationship type to prevent Cypher injection
            # Cannot use parameterization for relationship types in MERGE statements
            relationship_type = sanitize_relationship_type(relationship_type)

            async with self.driver.session() as session:
                query = f"""
                MATCH (source:Entity {{name: $source_entity}})
                MATCH (target:Entity {{name: $target_entity}})
                MERGE (source)-[r:{relationship_type}]->(target)
                SET r.mentioned_in = COALESCE(r.mentioned_in, []) + $memory_id
                RETURN r
                """

                await session.run(
                    query,
                    source_entity=source_entity,
                    target_entity=target_entity,
                    memory_id=memory_id,
                )

            logger.debug(f"Created relationship: {source_entity} -{relationship_type}-> {target_entity}")
            return True

        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False

    async def find_related_memories(
        self,
        memory_id: str,
        max_depth: int = 2,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find memories related through the knowledge graph.

        Args:
            memory_id: Source memory identifier
            max_depth: Maximum traversal depth
            limit: Maximum number of results

        Returns:
            List of related memory IDs with connection paths
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH path = (m1:Memory {memory_id: $memory_id})-[*1..$max_depth]-(m2:Memory)
                WHERE m1 <> m2
                RETURN DISTINCT m2.memory_id AS memory_id,
                       length(path) AS distance,
                       [node in nodes(path) | labels(node)[0]] AS path_types
                ORDER BY distance ASC
                LIMIT $limit
                """

                result = await session.run(
                    query,
                    memory_id=memory_id,
                    max_depth=max_depth,
                    limit=limit,
                )

                related = []
                async for record in result:
                    related.append({
                        "memory_id": record["memory_id"],
                        "distance": record["distance"],
                        "path_types": record["path_types"],
                    })

            logger.info(f"Found {len(related)} related memories for {memory_id}")
            return related

        except Exception as e:
            logger.error(f"Failed to find related memories: {e}")
            return []

    async def search_entities(
        self,
        query: str,
        user_id: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for entities in the knowledge graph.

        Args:
            query: Search query
            user_id: User identifier for filtering
            entity_type: Optional entity type filter
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        try:
            async with self.driver.session() as session:
                # Build query with parameterized entity_type
                type_filter = "AND e.type = $entity_type" if entity_type else ""

                cypher_query = f"""
                MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
                WHERE m.user_id = $user_id
                  AND toLower(e.name) CONTAINS toLower($query)
                  {type_filter}
                RETURN DISTINCT e.name AS name,
                       e.type AS type,
                       count(m) AS mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
                """

                # Build parameters dict
                params = {
                    "user_id": user_id,
                    "query": query,
                    "limit": limit,
                }
                if entity_type:
                    params["entity_type"] = entity_type

                result = await session.run(cypher_query, **params)

                entities = []
                async for record in result:
                    entities.append({
                        "name": record["name"],
                        "type": record["type"],
                        "mention_count": record["mention_count"],
                    })

            logger.info(f"Entity search for '{query}' returned {len(entities)} results")
            return entities

        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []

    async def get_entity_graph(
        self,
        entity_name: str,
        user_id: str,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Get subgraph around an entity.

        Args:
            entity_name: Entity name
            user_id: User identifier for filtering
            depth: Traversal depth

        Returns:
            Dictionary with nodes and edges
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH path = (e1:Entity {name: $entity_name})-[*1..$depth]-(e2:Entity)
                WHERE EXISTS {
                    MATCH (m:Memory)-[:MENTIONS]->(e1)
                    WHERE m.user_id = $user_id
                }
                RETURN DISTINCT
                    [node in nodes(path) | {name: node.name, type: node.type}] AS nodes,
                    [rel in relationships(path) | type(rel)] AS edges
                LIMIT 100
                """

                result = await session.run(
                    query,
                    entity_name=entity_name,
                    user_id=user_id,
                    depth=depth,
                )

                all_nodes = []
                all_edges = []

                async for record in result:
                    all_nodes.extend(record["nodes"])
                    all_edges.extend(record["edges"])

                # Deduplicate
                unique_nodes = {node["name"]: node for node in all_nodes}.values()

                return {
                    "nodes": list(unique_nodes),
                    "edges": all_edges,
                    "center_entity": entity_name,
                }

        except Exception as e:
            logger.error(f"Failed to get entity graph: {e}")
            return {"nodes": [], "edges": [], "center_entity": entity_name}

    async def delete_memory_node(self, memory_id: str) -> bool:
        """
        Delete a memory node and its relationships.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (m:Memory {memory_id: $memory_id})
                DETACH DELETE m
                """

                await session.run(query, memory_id=memory_id)

            logger.debug(f"Deleted memory node {memory_id} from knowledge graph")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory node: {e}")
            return False


def get_knowledge_graph_client() -> KnowledgeGraphClient:
    """
    Get knowledge graph client instance.

    Usage:
        client = get_knowledge_graph_client()
        await client.create_memory_node(memory_id, user_id, collection_id, content)
    """
    return KnowledgeGraphClient()


async def delete_memory_from_graph(memory_id: str) -> bool:
    """
    Helper function to delete a memory from the knowledge graph.

    Args:
        memory_id: Memory identifier

    Returns:
        True if successful
    """
    client = get_knowledge_graph_client()
    return await client.delete_memory_node(memory_id)
