"""
Temporal Knowledge Graph with Bi-temporal Model.

Tracks:
1. Valid Time: When a fact was true in the real world (t_valid, t_invalid)
2. Transaction Time: When the system learned about it (observed_at)

Based on Zep/Graphiti architecture for +18.5% accuracy improvement.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable

from backend.core.config import settings
from backend.core.logging_config import logger


@dataclass
class TemporalFact:
    """Represents a temporal fact in the knowledge graph."""
    subject: str  # Entity
    predicate: str  # Relationship type
    object: str  # Target entity
    valid_from: datetime  # When fact became true
    valid_until: Optional[datetime]  # When fact became false (None = still true)
    observed_at: datetime  # When we learned about it
    confidence: float = 1.0  # Confidence score
    source_memory_id: Optional[str] = None  # Provenance
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TemporalKnowledgeGraph:
    """
    Bi-temporal knowledge graph for AI agent memory.

    Features:
    - Validity intervals: (t_valid_from, t_valid_until)
    - Transaction time: observed_at
    - Point-in-time queries
    - Temporal conflict resolution
    - Provenance tracking
    """

    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    @classmethod
    async def create(cls) -> "TemporalKnowledgeGraph":
        """Create and initialize temporal knowledge graph."""
        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

        await driver.verify_connectivity()
        logger.info("Connected to temporal knowledge graph")

        instance = cls(driver)
        await instance._create_indexes()

        return instance

    async def _create_indexes(self):
        """Create indexes for temporal queries."""
        indexes = [
            # Temporal indexes
            "CREATE INDEX fact_valid_from IF NOT EXISTS FOR ()-[r:FACT]->() ON (r.valid_from)",
            "CREATE INDEX fact_valid_until IF NOT EXISTS FOR ()-[r:FACT]->() ON (r.valid_until)",
            "CREATE INDEX fact_observed_at IF NOT EXISTS FOR ()-[r:FACT]->() ON (r.observed_at)",

            # Entity indexes
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_user IF NOT EXISTS FOR (e:Entity) ON (e.user_id)",
        ]

        async with self.driver.session() as session:
            for index_query in indexes:
                try:
                    await session.run(index_query)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

        logger.info("Temporal graph indexes created")

    async def add_fact(
        self,
        subject: str,
        subject_type: str,
        predicate: str,
        object: str,
        object_type: str,
        user_id: str,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        observed_at: Optional[datetime] = None,
        confidence: float = 1.0,
        source_memory_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Add a temporal fact to the knowledge graph.

        Args:
            subject: Subject entity name
            subject_type: Subject entity type (PERSON, ORG, etc.)
            predicate: Relationship type (WORKS_AT, LOCATED_IN, etc.)
            object: Object entity name
            object_type: Object entity type
            user_id: User ID for isolation
            valid_from: When fact became true
            valid_until: When fact became false (None = still true)
            observed_at: When we learned about it
            confidence: Confidence score
            source_memory_id: Source memory for provenance
            metadata: Additional metadata

        Returns:
            True if successful
        """
        try:
            # Default timestamps
            if valid_from is None:
                valid_from = datetime.now(timezone.utc)
            if observed_at is None:
                observed_at = datetime.now(timezone.utc)

            async with self.driver.session() as session:
                query = """
                // Create or merge subject entity
                MERGE (subj:Entity {name: $subject, type: $subject_type, user_id: $user_id})

                // Create or merge object entity
                MERGE (obj:Entity {name: $object, type: $object_type, user_id: $user_id})

                // Create temporal fact relationship
                CREATE (subj)-[r:FACT {
                    predicate: $predicate,
                    valid_from: datetime($valid_from),
                    valid_until: CASE WHEN $valid_until IS NOT NULL
                                  THEN datetime($valid_until)
                                  ELSE NULL END,
                    observed_at: datetime($observed_at),
                    confidence: $confidence,
                    source_memory_id: $source_memory_id,
                    metadata: $metadata
                }]->(obj)

                RETURN r
                """

                await session.run(
                    query,
                    subject=subject,
                    subject_type=subject_type,
                    predicate=predicate,
                    object=object,
                    object_type=object_type,
                    user_id=user_id,
                    valid_from=valid_from.isoformat(),
                    valid_until=valid_until.isoformat() if valid_until else None,
                    observed_at=observed_at.isoformat(),
                    confidence=confidence,
                    source_memory_id=source_memory_id,
                    metadata=metadata or {},
                )

            logger.info(f"Added temporal fact: {subject} -{predicate}-> {object}")
            return True

        except Exception as e:
            logger.error(f"Failed to add temporal fact: {e}")
            return False

    async def query_at_time(
        self,
        subject: str,
        predicate: str,
        user_id: str,
        timestamp: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query what was true at a specific point in time.

        Args:
            subject: Subject entity
            predicate: Relationship type
            user_id: User ID
            timestamp: Point in time to query (None = now)

        Returns:
            List of facts that were valid at that time
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            async with self.driver.session() as session:
                query = """
                MATCH (subj:Entity {name: $subject, user_id: $user_id})-[r:FACT {predicate: $predicate}]->(obj:Entity)
                WHERE datetime($timestamp) >= r.valid_from
                  AND (r.valid_until IS NULL OR datetime($timestamp) < r.valid_until)
                RETURN obj.name AS object,
                       obj.type AS object_type,
                       r.valid_from AS valid_from,
                       r.valid_until AS valid_until,
                       r.observed_at AS observed_at,
                       r.confidence AS confidence,
                       r.source_memory_id AS source_memory_id
                ORDER BY r.confidence DESC
                """

                result = await session.run(
                    query,
                    subject=subject,
                    predicate=predicate,
                    user_id=user_id,
                    timestamp=timestamp.isoformat(),
                )

                facts = []
                async for record in result:
                    facts.append({
                        "object": record["object"],
                        "object_type": record["object_type"],
                        "valid_from": record["valid_from"],
                        "valid_until": record["valid_until"],
                        "observed_at": record["observed_at"],
                        "confidence": record["confidence"],
                        "source_memory_id": record["source_memory_id"],
                    })

                logger.info(
                    f"Point-in-time query returned {len(facts)} facts for "
                    f"{subject} -{predicate}-> ? at {timestamp}"
                )
                return facts

        except Exception as e:
            logger.error(f"Point-in-time query failed: {e}")
            return []

    async def invalidate_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        user_id: str,
        invalid_from: Optional[datetime] = None,
    ) -> bool:
        """
        Mark a fact as no longer valid.

        Args:
            subject: Subject entity
            predicate: Relationship type
            object: Object entity
            user_id: User ID
            invalid_from: When fact became invalid

        Returns:
            True if successful
        """
        try:
            if invalid_from is None:
                invalid_from = datetime.now(timezone.utc)

            async with self.driver.session() as session:
                query = """
                MATCH (subj:Entity {name: $subject, user_id: $user_id})-[r:FACT {predicate: $predicate}]->(obj:Entity {name: $object, user_id: $user_id})
                WHERE r.valid_until IS NULL
                SET r.valid_until = datetime($invalid_from)
                RETURN count(r) AS updated_count
                """

                result = await session.run(
                    query,
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    user_id=user_id,
                    invalid_from=invalid_from.isoformat(),
                )

                record = await result.single()
                updated_count = record["updated_count"] if record else 0

                logger.info(f"Invalidated {updated_count} facts")
                return updated_count > 0

        except Exception as e:
            logger.error(f"Failed to invalidate fact: {e}")
            return False

    async def get_history(
        self,
        subject: str,
        predicate: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get full history of a fact over time.

        Args:
            subject: Subject entity
            predicate: Relationship type
            user_id: User ID

        Returns:
            List of all facts across time, ordered chronologically
        """
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (subj:Entity {name: $subject, user_id: $user_id})-[r:FACT {predicate: $predicate}]->(obj:Entity)
                RETURN obj.name AS object,
                       obj.type AS object_type,
                       r.valid_from AS valid_from,
                       r.valid_until AS valid_until,
                       r.observed_at AS observed_at,
                       r.confidence AS confidence,
                       r.source_memory_id AS source_memory_id
                ORDER BY r.valid_from ASC
                """

                result = await session.run(
                    query,
                    subject=subject,
                    predicate=predicate,
                    user_id=user_id,
                )

                history = []
                async for record in result:
                    history.append({
                        "object": record["object"],
                        "object_type": record["object_type"],
                        "valid_from": record["valid_from"],
                        "valid_until": record["valid_until"],
                        "observed_at": record["observed_at"],
                        "confidence": record["confidence"],
                        "source_memory_id": record["source_memory_id"],
                    })

                logger.info(f"Retrieved history with {len(history)} entries")
                return history

        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    async def resolve_conflicts(
        self,
        subject: str,
        predicate: str,
        user_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect and resolve conflicting facts.

        Args:
            subject: Subject entity
            predicate: Relationship type
            user_id: User ID

        Returns:
            List of resolved facts
        """
        try:
            # Get all facts
            history = await self.get_history(subject, predicate, user_id)

            if len(history) <= 1:
                return history  # No conflicts

            # Detect overlapping validity periods
            conflicts = []
            for i, fact1 in enumerate(history):
                for fact2 in history[i + 1:]:
                    # Check if validity periods overlap
                    f1_start = fact1["valid_from"]
                    f1_end = fact1["valid_until"]
                    f2_start = fact2["valid_from"]
                    f2_end = fact2["valid_until"]

                    # Simple overlap check (would need proper interval logic)
                    overlaps = (
                        (f1_end is None or f2_start < f1_end) and
                        (f2_end is None or f1_start < f2_end)
                    )

                    if overlaps and fact1["object"] != fact2["object"]:
                        conflicts.append((fact1, fact2))

            if conflicts:
                logger.warning(f"Found {len(conflicts)} temporal conflicts")

                # Resolution strategy: Keep fact with higher confidence
                # and more recent observation time
                resolved = []
                for fact1, fact2 in conflicts:
                    if fact1["confidence"] >= fact2["confidence"]:
                        resolved.append(fact1)
                    else:
                        resolved.append(fact2)

                return resolved

            return history

        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return []

    async def close(self):
        """Close connection to Neo4j."""
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from temporal knowledge graph")


# Global instance
_temporal_graph: Optional[TemporalKnowledgeGraph] = None


async def get_temporal_graph() -> TemporalKnowledgeGraph:
    """
    Get temporal knowledge graph instance.

    Returns:
        TemporalKnowledgeGraph instance
    """
    global _temporal_graph

    if _temporal_graph is None:
        _temporal_graph = await TemporalKnowledgeGraph.create()

    return _temporal_graph
