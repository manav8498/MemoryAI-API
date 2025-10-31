"""
Profile Manager Service - User Profile Management

Manages user profiles with operations for retrieving, updating, and organizing facts.
Similar to SuperMemory's profile API functionality.
"""
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, delete, func

from backend.core.logging_config import logger
from backend.models.user_profile import (
    UserProfileFact,
    ProfileOperation,
    ProfileSnapshot,
    ProfileType,
    ProfileCategory,
)
from backend.services.hybrid_search import search_memories


class ProfileManager:
    """
    Manages user profile operations.

    Features:
    - Retrieve static/dynamic profiles
    - Combine profiles with search results
    - Update and delete facts
    - Profile snapshots and versioning
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_profile(
        self,
        user_id: str,
        include_dynamic: bool = True,
        include_static: bool = True,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get user profile facts.

        Args:
            user_id: User ID
            include_dynamic: Include dynamic facts
            include_static: Include static facts
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with static and dynamic facts
        """
        profile = {
            "static": [],
            "dynamic": [],
            "metadata": {
                "user_id": user_id,
                "retrieved_at": datetime.utcnow().isoformat(),
            }
        }

        # Build query conditions
        conditions = [
            UserProfileFact.user_id == user_id,
            UserProfileFact.confidence >= min_confidence,
        ]

        # Filter by profile type
        type_conditions = []
        if include_static:
            type_conditions.append(UserProfileFact.profile_type == ProfileType.STATIC.value)
        if include_dynamic:
            type_conditions.append(UserProfileFact.profile_type == ProfileType.DYNAMIC.value)

        if type_conditions:
            conditions.append(or_(*type_conditions))

        # Query facts
        result = await self.db.execute(
            select(UserProfileFact)
            .where(and_(*conditions))
            .order_by(UserProfileFact.importance.desc(), UserProfileFact.confidence.desc())
        )
        facts = result.scalars().all()

        # Organize by type
        for fact in facts:
            fact_dict = {
                "id": fact.id,
                "category": fact.category,  # Already a string in DB
                "key": fact.fact_key,
                "value": fact.fact_value,
                "confidence": fact.confidence,
                "importance": fact.importance,
                "verified": fact.verified,
                "created_at": fact.created_at.isoformat(),
                "updated_at": fact.updated_at.isoformat(),
            }

            if fact.profile_type == ProfileType.STATIC.value:
                profile["static"].append(fact_dict)
            else:
                profile["dynamic"].append(fact_dict)

            # Update access tracking
            fact.last_accessed_at = datetime.utcnow()
            try:
                count = int(fact.access_count)
                fact.access_count = str(count + 1)
            except:
                fact.access_count = "1"

        await self.db.commit()

        # Add stats to metadata
        profile["metadata"]["static_count"] = len(profile["static"])
        profile["metadata"]["dynamic_count"] = len(profile["dynamic"])
        profile["metadata"]["total_facts"] = len(facts)

        logger.info(f"Retrieved profile for user {user_id}: {len(facts)} facts")
        return profile

    async def get_profile_with_search(
        self,
        user_id: str,
        query: str,
        collection_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get user profile combined with search results.

        This is similar to SuperMemory's profile + search combo mode.

        Args:
            user_id: User ID
            query: Search query
            collection_id: Optional collection filter
            limit: Number of search results

        Returns:
            Dictionary with profile and search results
        """
        # Get profile
        profile = await self.get_user_profile(user_id=user_id)

        # Get search results
        search_results = await search_memories(
            query=query,
            user_id=user_id,
            db=self.db,
            collection_id=collection_id,
            limit=limit,
        )

        # Combine
        return {
            "profile": profile,
            "searchResults": {
                "results": search_results,
                "total": len(search_results),
                "query": query,
            },
            "metadata": {
                "combined_retrieval": True,
                "profile_facts": profile["metadata"]["total_facts"],
                "search_results": len(search_results),
                "retrieved_at": datetime.utcnow().isoformat(),
            }
        }

    async def add_or_update_fact(
        self,
        user_id: str,
        fact_key: str,
        fact_value: str,
        profile_type: ProfileType,
        category: ProfileCategory,
        confidence: float = 0.9,  # User-provided facts are highly confident
        importance: float = 0.7,
        trigger_type: str = "user_input",
    ) -> UserProfileFact:
        """
        Add or update a profile fact.

        Args:
            user_id: User ID
            fact_key: Fact key
            fact_value: Fact value
            profile_type: Static or dynamic
            category: Fact category
            confidence: Confidence score
            importance: Importance score
            trigger_type: How this was triggered

        Returns:
            UserProfileFact object
        """
        # Check if exists
        result = await self.db.execute(
            select(UserProfileFact).where(
                UserProfileFact.user_id == user_id,
                UserProfileFact.fact_key == fact_key,
            )
        )
        existing_fact = result.scalar_one_or_none()

        if existing_fact:
            # Update
            operation = ProfileOperation(
                id=str(uuid.uuid4()),
                profile_fact_id=existing_fact.id,
                user_id=user_id,
                operation_type="update",
                old_value=existing_fact.fact_value,
                new_value=fact_value,
                confidence_change=confidence - existing_fact.confidence,
                trigger_type=trigger_type,
            )
            self.db.add(operation)

            existing_fact.fact_value = fact_value
            existing_fact.confidence = confidence
            existing_fact.importance = importance
            existing_fact.updated_at = datetime.utcnow()

            await self.db.commit()
            await self.db.refresh(existing_fact)

            logger.info(f"Updated fact: {fact_key}")
            return existing_fact
        else:
            # Create
            new_fact = UserProfileFact(
                id=str(uuid.uuid4()),
                user_id=user_id,
                profile_type=profile_type,
                category=category,
                fact_key=fact_key,
                fact_value=fact_value,
                confidence=confidence,
                importance=importance,
                source_memory_ids=[],
                extraction_metadata={
                    "creation_date": datetime.utcnow().isoformat(),
                    "trigger_type": trigger_type,
                },
                verified="user_confirmed" if trigger_type == "user_input" else "auto",
                access_count="0",
            )
            self.db.add(new_fact)

            operation = ProfileOperation(
                id=str(uuid.uuid4()),
                profile_fact_id=new_fact.id,
                user_id=user_id,
                operation_type="add",
                new_value=fact_value,
                confidence_change=confidence,
                trigger_type=trigger_type,
            )
            self.db.add(operation)

            await self.db.commit()
            await self.db.refresh(new_fact)

            logger.info(f"Created fact: {fact_key}")
            return new_fact

    async def delete_fact(
        self,
        user_id: str,
        fact_key: str,
    ) -> bool:
        """
        Delete a profile fact.

        Args:
            user_id: User ID
            fact_key: Fact key to delete

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            select(UserProfileFact).where(
                UserProfileFact.user_id == user_id,
                UserProfileFact.fact_key == fact_key,
            )
        )
        fact = result.scalar_one_or_none()

        if fact:
            # Track operation
            operation = ProfileOperation(
                id=str(uuid.uuid4()),
                profile_fact_id=fact.id,
                user_id=user_id,
                operation_type="remove",
                old_value=fact.fact_value,
                trigger_type="user_request",
            )
            self.db.add(operation)

            # Delete fact
            await self.db.delete(fact)
            await self.db.commit()

            logger.info(f"Deleted fact: {fact_key}")
            return True
        else:
            logger.warning(f"Fact not found for deletion: {fact_key}")
            return False

    async def get_profile_history(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get profile operation history.

        Args:
            user_id: User ID
            limit: Number of operations to return

        Returns:
            List of operations
        """
        result = await self.db.execute(
            select(ProfileOperation)
            .where(ProfileOperation.user_id == user_id)
            .order_by(ProfileOperation.created_at.desc())
            .limit(limit)
        )
        operations = result.scalars().all()

        history = []
        for op in operations:
            history.append({
                "id": op.id,
                "operation_type": op.operation_type,
                "old_value": op.old_value,
                "new_value": op.new_value,
                "confidence_change": op.confidence_change,
                "trigger_type": op.trigger_type,
                "created_at": op.created_at.isoformat(),
            })

        return history

    async def create_snapshot(
        self,
        user_id: str,
        trigger_reason: str = "manual",
    ) -> ProfileSnapshot:
        """
        Create a snapshot of the current profile.

        Args:
            user_id: User ID
            trigger_reason: Why snapshot was created

        Returns:
            ProfileSnapshot object
        """
        # Get current profile
        profile = await self.get_user_profile(user_id=user_id)

        # Create snapshot
        snapshot = ProfileSnapshot(
            id=str(uuid.uuid4()),
            user_id=user_id,
            static_facts=profile["static"],
            dynamic_facts=profile["dynamic"],
            snapshot_metadata={
                "trigger_reason": trigger_reason,
                "total_facts": profile["metadata"]["total_facts"],
                "static_count": profile["metadata"]["static_count"],
                "dynamic_count": profile["metadata"]["dynamic_count"],
            },
        )
        self.db.add(snapshot)
        await self.db.commit()
        await self.db.refresh(snapshot)

        logger.info(f"Created profile snapshot for user {user_id}")
        return snapshot

    async def cleanup_old_dynamic_facts(
        self,
        user_id: str,
        older_than_days: int = 30,
    ) -> int:
        """
        Clean up old dynamic facts.

        Dynamic facts are temporary and should be cleaned up periodically.

        Args:
            user_id: User ID
            older_than_days: Remove dynamic facts older than this

        Returns:
            Number of facts removed
        """
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

        result = await self.db.execute(
            delete(UserProfileFact).where(
                and_(
                    UserProfileFact.user_id == user_id,
                    UserProfileFact.profile_type == ProfileType.DYNAMIC.value,
                    UserProfileFact.updated_at < cutoff_date,
                )
            )
        )
        await self.db.commit()

        removed_count = result.rowcount
        logger.info(f"Cleaned up {removed_count} old dynamic facts for user {user_id}")
        return removed_count

    async def get_profile_stats(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Get profile statistics.

        Args:
            user_id: User ID

        Returns:
            Statistics dictionary
        """
        # Count by type
        static_count = await self.db.scalar(
            select(func.count()).where(
                and_(
                    UserProfileFact.user_id == user_id,
                    UserProfileFact.profile_type == ProfileType.STATIC.value,
                )
            )
        )

        dynamic_count = await self.db.scalar(
            select(func.count()).where(
                and_(
                    UserProfileFact.user_id == user_id,
                    UserProfileFact.profile_type == ProfileType.DYNAMIC.value,
                )
            )
        )

        # Count by category
        category_result = await self.db.execute(
            select(
                UserProfileFact.category,
                func.count().label("count")
            )
            .where(UserProfileFact.user_id == user_id)
            .group_by(UserProfileFact.category)
        )
        categories = {row[0]: row[1] for row in category_result}  # category is already a string

        # Average confidence
        avg_confidence = await self.db.scalar(
            select(func.avg(UserProfileFact.confidence))
            .where(UserProfileFact.user_id == user_id)
        )

        return {
            "user_id": user_id,
            "total_facts": (static_count or 0) + (dynamic_count or 0),
            "static_facts": static_count or 0,
            "dynamic_facts": dynamic_count or 0,
            "categories": categories,
            "average_confidence": float(avg_confidence) if avg_confidence else 0.0,
        }


def get_profile_manager(db: AsyncSession) -> ProfileManager:
    """Get profile manager instance."""
    return ProfileManager(db=db)
