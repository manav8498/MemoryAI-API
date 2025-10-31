"""
Profile Extractor Service - Automatic User Profile Fact Extraction

Automatically extracts user facts from memories using LLM analysis.
Similar to SuperMemory's automatic profiling system.
"""
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.logging_config import logger
from backend.models.memory import Memory
from backend.models.user_profile import (
    UserProfileFact,
    ProfileOperation,
    ProfileType,
    ProfileCategory,
)
from backend.reasoning.llm_providers import get_llm_provider


class ProfileExtractor:
    """
    Extracts user profile facts from memory content using LLM analysis.

    Features:
    - Automatic fact extraction from memories
    - Static vs Dynamic classification
    - Confidence scoring
    - Duplicate detection and merging
    - Profile operations tracking
    """

    def __init__(self, db: AsyncSession, provider_name: str = "gemini"):
        self.db = db
        self.llm = get_llm_provider(provider_name)

    async def extract_facts_from_memory(
        self,
        memory: Memory,
        user_id: str,
    ) -> List[UserProfileFact]:
        """
        Extract user profile facts from a memory.

        Args:
            memory: Memory object to extract from
            user_id: User ID

        Returns:
            List of extracted UserProfileFact objects
        """
        try:
            logger.info(f"Extracting profile facts from memory {memory.id}")

            # Build extraction prompt
            prompt = self._build_extraction_prompt(memory.content)

            # Call LLM
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for factual extraction
                max_tokens=1000,
            )

            # Parse response
            facts = self._parse_llm_response(response)

            # Create UserProfileFact objects
            profile_facts = []
            for fact_data in facts:
                profile_fact = await self._create_profile_fact(
                    user_id=user_id,
                    memory_id=memory.id,
                    fact_data=fact_data,
                )
                if profile_fact:
                    profile_facts.append(profile_fact)

            logger.info(f"Extracted {len(profile_facts)} facts from memory {memory.id}")
            return profile_facts

        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    def _build_extraction_prompt(self, content: str) -> str:
        """Build LLM prompt for fact extraction."""
        return f"""Analyze the following content and extract facts about the USER (not about general topics).

Content:
{content}

Extract user facts in the following format (one per line):
TYPE | CATEGORY | KEY | VALUE | CONFIDENCE

Where:
- TYPE: "static" (long-term facts) or "dynamic" (recent/temporary facts)
- CATEGORY: role, expertise, preference, education, experience, current_project, recent_skill, temporary_state, goal, interest, communication, other
- KEY: Specific attribute name (e.g., "current_role", "expertise_python", "preference_editor")
- VALUE: The fact value/description
- CONFIDENCE: 0.0-1.0 (how confident you are in this fact)

Rules:
1. Only extract facts ABOUT THE USER (their role, skills, preferences, projects, etc.)
2. Don't extract general information or facts about other people/companies
3. Static facts: long-term attributes like role, expertise, education
4. Dynamic facts: recent activities, current projects, temporary states
5. Be specific in KEY names
6. VALUE should be concise but informative

Examples:
static | role | current_role | Software Engineer | 0.9
static | expertise | expertise_python | Expert in Python and FastAPI | 0.8
dynamic | current_project | project_memory_ai | Building a Memory AI API | 0.9
static | preference | preference_editor | Prefers VSCode over other editors | 0.7

Extract facts (skip the header):"""

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response into structured fact data.

        Args:
            response: Raw LLM response

        Returns:
            List of fact dictionaries
        """
        facts = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TYPE"):
                continue

            # Parse format: TYPE | CATEGORY | KEY | VALUE | CONFIDENCE
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                try:
                    profile_type = parts[0].lower()
                    category = parts[1].lower()
                    key = parts[2]
                    value = parts[3]
                    confidence = float(parts[4])

                    # Validate
                    if profile_type not in ["static", "dynamic"]:
                        continue
                    if not key or not value:
                        continue
                    if confidence < 0.0 or confidence > 1.0:
                        confidence = 0.7  # Default

                    facts.append({
                        "profile_type": profile_type,
                        "category": category,
                        "key": key,
                        "value": value,
                        "confidence": confidence,
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse fact line: {line} - {e}")
                    continue

        return facts

    async def _create_profile_fact(
        self,
        user_id: str,
        memory_id: str,
        fact_data: Dict[str, Any],
    ) -> Optional[UserProfileFact]:
        """
        Create or update a profile fact in the database.

        Args:
            user_id: User ID
            memory_id: Source memory ID
            fact_data: Parsed fact data

        Returns:
            UserProfileFact object or None
        """
        try:
            # Check if similar fact already exists
            existing_fact = await self._find_existing_fact(
                user_id=user_id,
                fact_key=fact_data["key"],
            )

            if existing_fact:
                # Update existing fact
                return await self._update_existing_fact(
                    existing_fact=existing_fact,
                    memory_id=memory_id,
                    new_value=fact_data["value"],
                    new_confidence=fact_data["confidence"],
                )
            else:
                # Create new fact
                return await self._create_new_fact(
                    user_id=user_id,
                    memory_id=memory_id,
                    fact_data=fact_data,
                )

        except Exception as e:
            logger.error(f"Failed to create profile fact: {e}")
            return None

    async def _find_existing_fact(
        self,
        user_id: str,
        fact_key: str,
    ) -> Optional[UserProfileFact]:
        """Find existing fact by key."""
        result = await self.db.execute(
            select(UserProfileFact).where(
                UserProfileFact.user_id == user_id,
                UserProfileFact.fact_key == fact_key,
            )
        )
        return result.scalar_one_or_none()

    async def _update_existing_fact(
        self,
        existing_fact: UserProfileFact,
        memory_id: str,
        new_value: str,
        new_confidence: float,
    ) -> UserProfileFact:
        """Update an existing profile fact."""
        # Track operation
        operation = ProfileOperation(
            id=str(uuid.uuid4()),
            profile_fact_id=existing_fact.id,
            user_id=existing_fact.user_id,
            operation_type="update",
            old_value=existing_fact.fact_value,
            new_value=new_value,
            confidence_change=new_confidence - existing_fact.confidence,
            trigger_memory_id=memory_id,
            trigger_type="auto_extraction",
            operation_metadata={},
        )
        self.db.add(operation)

        # Update fact
        existing_fact.fact_value = new_value
        existing_fact.confidence = max(existing_fact.confidence, new_confidence)  # Take higher confidence
        if memory_id not in existing_fact.source_memory_ids:
            existing_fact.source_memory_ids = existing_fact.source_memory_ids + [memory_id]
        existing_fact.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(existing_fact)

        logger.info(f"Updated profile fact: {existing_fact.fact_key}")
        return existing_fact

    async def _create_new_fact(
        self,
        user_id: str,
        memory_id: str,
        fact_data: Dict[str, Any],
    ) -> UserProfileFact:
        """Create a new profile fact."""
        # Map category string to enum
        category_map = {
            "role": ProfileCategory.ROLE,
            "expertise": ProfileCategory.EXPERTISE,
            "preference": ProfileCategory.PREFERENCE,
            "education": ProfileCategory.EDUCATION,
            "experience": ProfileCategory.EXPERIENCE,
            "current_project": ProfileCategory.CURRENT_PROJECT,
            "recent_skill": ProfileCategory.RECENT_SKILL,
            "temporary_state": ProfileCategory.TEMPORARY_STATE,
            "goal": ProfileCategory.GOAL,
            "interest": ProfileCategory.INTEREST,
            "communication": ProfileCategory.COMMUNICATION,
        }
        category = category_map.get(fact_data["category"], ProfileCategory.OTHER)

        # Create fact
        profile_fact = UserProfileFact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            profile_type=ProfileType.STATIC.value if fact_data["profile_type"] == "static" else ProfileType.DYNAMIC.value,
            category=category.value,  # Store enum value as string
            fact_key=fact_data["key"],
            fact_value=fact_data["value"],
            confidence=fact_data["confidence"],
            importance=0.5,  # Default
            source_memory_ids=[memory_id],
            extraction_metadata={
                "extraction_date": datetime.utcnow().isoformat(),
                "llm_provider": self.llm.provider_name,
            },
            verified="auto",
            access_count="0",
        )
        self.db.add(profile_fact)

        # Track operation
        operation = ProfileOperation(
            id=str(uuid.uuid4()),
            profile_fact_id=profile_fact.id,
            user_id=user_id,
            operation_type="add",
            old_value=None,
            new_value=fact_data["value"],
            confidence_change=fact_data["confidence"],
            trigger_memory_id=memory_id,
            trigger_type="auto_extraction",
            operation_metadata={},
        )
        self.db.add(operation)

        await self.db.commit()
        await self.db.refresh(profile_fact)

        logger.info(f"Created new profile fact: {profile_fact.fact_key}")
        return profile_fact


# Global instance
_profile_extractor: Optional[ProfileExtractor] = None


def get_profile_extractor(
    db: AsyncSession,
    provider_name: str = "gemini",
) -> ProfileExtractor:
    """Get or create profile extractor instance."""
    return ProfileExtractor(db=db, provider_name=provider_name)
