"""
Conversation Starters Service - Profile-Driven Question Generation

Generates intelligent conversation starters based on user profile:
- Leverages expertise to suggest relevant topics
- Connects to current projects for contextual questions
- Uses interests and goals to spark engagement
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from backend.core.logging_config import logger
from backend.models.user_profile import UserProfileFact
from backend.reasoning.llm_providers import get_llm_provider


class ConversationStartersService:
    """
    Generates personalized conversation starters based on user profile.

    Features:
    - LLM-powered question generation
    - Context-aware based on profile facts
    - Prioritizes recent activities and projects
    - Fallback templates for various scenarios
    """

    def __init__(self, db: AsyncSession, provider_name: str = "gemini"):
        self.db = db
        self.llm = get_llm_provider(provider_name)

    async def generate_starters(
        self,
        user_id: str,
        count: int = 5,
        use_llm: bool = True
    ) -> List[Dict[str, str]]:
        """
        Generate conversation starters for a user.

        Args:
            user_id: User ID
            count: Number of starters to generate
            use_llm: Whether to use LLM (fallback to templates if False)

        Returns:
            List of starter dictionaries with 'question' and 'context'
        """
        try:
            logger.info(f"Generating {count} conversation starters for user {user_id}")

            # Get user profile facts
            profile_facts = await self._get_profile_facts(user_id)

            if not profile_facts:
                # No profile, return generic starters
                return self._get_generic_starters(count)

            # Use LLM to generate contextual starters
            if use_llm and len(profile_facts) > 0:
                starters = await self._generate_with_llm(profile_facts, count)
                if starters:
                    return starters

            # Fallback to template-based generation
            return self._generate_from_templates(profile_facts, count)

        except Exception as e:
            logger.error(f"Failed to generate starters: {e}")
            return self._get_generic_starters(count)

    async def _get_profile_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get relevant profile facts for starter generation.

        Args:
            user_id: User ID

        Returns:
            List of profile facts
        """
        # Get high-importance facts
        result = await self.db.execute(
            select(UserProfileFact)
            .where(UserProfileFact.user_id == user_id)
            .order_by(
                UserProfileFact.importance.desc(),
                UserProfileFact.updated_at.desc()
            )
            .limit(10)
        )
        facts = result.scalars().all()

        return [
            {
                "category": fact.category,
                "key": fact.fact_key,
                "value": fact.fact_value,
                "type": fact.profile_type
            }
            for fact in facts
        ]

    async def _generate_with_llm(
        self,
        profile_facts: List[Dict[str, Any]],
        count: int
    ) -> Optional[List[Dict[str, str]]]:
        """
        Generate starters using LLM.

        Args:
            profile_facts: User profile facts
            count: Number of starters to generate

        Returns:
            List of starters or None if generation fails
        """
        try:
            # Build context from profile
            profile_context = self._build_profile_context(profile_facts)

            # Create prompt
            prompt = f"""Based on this user's profile, generate {count} engaging conversation starters that are:
1. Highly relevant to their work, interests, or current projects
2. Specific and actionable
3. Open-ended to encourage detailed responses
4. Natural and conversational

User Profile:
{profile_context}

Generate exactly {count} conversation starters in this format:
STARTER 1 | <question>
STARTER 2 | <question>
...

Make each starter unique and focused on different aspects of the profile."""

            # Call LLM
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.7,  # Moderate creativity
                max_tokens=500
            )

            # Parse response
            starters = []
            lines = response.strip().split("\n")

            for line in lines:
                if "|" in line:
                    parts = line.split("|", 1)
                    if len(parts) == 2:
                        question = parts[1].strip()
                        # Remove quotes if present
                        question = question.strip('"\'')

                        if len(question) > 20:  # Valid question
                            starters.append({
                                "question": question,
                                "context": "Generated based on your profile",
                                "type": "llm_generated"
                            })

            if len(starters) >= count:
                return starters[:count]

            # If not enough, supplement with templates
            if starters:
                template_starters = self._generate_from_templates(
                    profile_facts,
                    count - len(starters)
                )
                starters.extend(template_starters)
                return starters[:count]

            return None

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _generate_from_templates(
        self,
        profile_facts: List[Dict[str, Any]],
        count: int
    ) -> List[Dict[str, str]]:
        """
        Generate starters from templates based on profile facts.

        Args:
            profile_facts: User profile facts
            count: Number of starters to generate

        Returns:
            List of starter dictionaries
        """
        starters = []

        # Organize facts by category
        by_category = {}
        for fact in profile_facts:
            category = fact["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(fact)

        # Generate starters by category
        for category, facts in by_category.items():
            if len(starters) >= count:
                break

            fact = facts[0]  # Use first (most important) fact
            value = fact["value"]

            starter = self._get_template_for_category(category, value)
            if starter:
                starters.append(starter)

        # Fill remaining with generic if needed
        while len(starters) < count:
            generic = random.choice(self._get_generic_starters(1))
            if generic not in starters:
                starters.append(generic)

        return starters[:count]

    def _get_template_for_category(
        self,
        category: str,
        value: str
    ) -> Optional[Dict[str, str]]:
        """
        Get conversation starter template for a category.

        Args:
            category: Profile fact category
            value: Fact value

        Returns:
            Starter dictionary or None
        """
        templates = {
            "current_project": [
                f"What challenges are you currently facing with {value}?",
                f"How is the progress on {value} going?",
                f"What have you learned while working on {value}?",
            ],
            "expertise": [
                f"What's your take on recent developments in {value}?",
                f"Can you share insights about best practices in {value}?",
                f"What trends are you seeing in {value}?",
            ],
            "goal": [
                f"What steps are you taking towards {value}?",
                f"How can I help you achieve {value}?",
                f"What's the biggest obstacle to {value}?",
            ],
            "interest": [
                f"What got you interested in {value}?",
                f"What resources would you recommend for learning about {value}?",
                f"How do you stay updated on {value}?",
            ],
            "recent_skill": [
                f"How has learning {value} been going?",
                f"What project are you applying {value} to?",
                f"What surprised you most about {value}?",
            ],
            "role": [
                f"What's the most exciting aspect of being a {value}?",
                f"What does a typical day look like for a {value}?",
                f"What skills are most valuable for a {value}?",
            ],
        }

        category_templates = templates.get(category)
        if category_templates:
            question = random.choice(category_templates)
            return {
                "question": question,
                "context": f"Based on your {category}: {value}",
                "type": "template_based"
            }

        return None

    def _build_profile_context(self, profile_facts: List[Dict[str, Any]]) -> str:
        """
        Build a readable context string from profile facts.

        Args:
            profile_facts: Profile facts

        Returns:
            Formatted context string
        """
        context_lines = []

        for fact in profile_facts[:5]:  # Limit to avoid token overflow
            category = fact["category"].replace("_", " ").title()
            value = fact["value"]
            context_lines.append(f"- {category}: {value}")

        return "\n".join(context_lines)

    def _get_generic_starters(self, count: int) -> List[Dict[str, str]]:
        """
        Get generic conversation starters (fallback).

        Args:
            count: Number of starters

        Returns:
            List of generic starters
        """
        generic = [
            {
                "question": "What are you working on today?",
                "context": "General check-in",
                "type": "generic"
            },
            {
                "question": "What's on your mind?",
                "context": "Open-ended",
                "type": "generic"
            },
            {
                "question": "How can I help you today?",
                "context": "Assistance offer",
                "type": "generic"
            },
            {
                "question": "What would you like to explore?",
                "context": "Discovery",
                "type": "generic"
            },
            {
                "question": "Tell me about your current priorities.",
                "context": "Focus check",
                "type": "generic"
            },
        ]

        # Shuffle and return
        random.shuffle(generic)
        return generic[:count]


async def generate_conversation_starters(
    db: AsyncSession,
    user_id: str,
    count: int = 5,
    provider: str = "gemini"
) -> List[Dict[str, str]]:
    """
    Convenience function to generate conversation starters.

    Args:
        db: Database session
        user_id: User ID
        count: Number of starters
        provider: LLM provider

    Returns:
        List of conversation starters
    """
    service = ConversationStartersService(db, provider)
    return await service.generate_starters(user_id, count)


# Example usage
if __name__ == "__main__":
    import asyncio
    from backend.core.database import get_db_context

    async def test_starters():
        async with get_db_context() as db:
            # Example user ID (replace with real user)
            user_id = "b1bc75e9-237d-4b5c-aff8-7daa1e07c5a6"

            service = ConversationStartersService(db)
            starters = await service.generate_starters(user_id, count=5)

            print("\n💬 Generated Conversation Starters:\n")
            for i, starter in enumerate(starters, 1):
                print(f"{i}. {starter['question']}")
                print(f"   Context: {starter['context']}")
                print(f"   Type: {starter['type']}\n")

    asyncio.run(test_starters())
