"""
Profile-Based Reranking for Personalized Search Results

Boosts search results based on user profile facts:
- Prioritizes content matching user's expertise
- Boosts results related to current projects
- Considers user preferences and goals
"""
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

from backend.core.logging_config import logger


@dataclass
class ProfileBoostConfig:
    """Configuration for profile-based boosting"""
    expertise_boost: float = 1.5      # Boost for expertise matches
    project_boost: float = 1.4        # Boost for current project matches
    interest_boost: float = 1.3       # Boost for interest matches
    goal_boost: float = 1.2           # Boost for goal matches
    preference_boost: float = 1.1     # Boost for preference matches
    recency_decay: float = 0.9        # Decay factor for old dynamic facts


class ProfileReranker:
    """
    Reranks search results based on user profile.

    Boosts results that are relevant to user's:
    - Expertise areas
    - Current projects
    - Interests and goals
    - Preferences
    """

    def __init__(self, config: Optional[ProfileBoostConfig] = None):
        """
        Initialize profile reranker.

        Args:
            config: Boosting configuration
        """
        self.config = config or ProfileBoostConfig()

    def rerank_with_profile(
        self,
        results: List[Dict[str, Any]],
        profile: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on user profile.

        Args:
            results: Search results to rerank
            profile: User profile with static and dynamic facts
            top_k: Number of results to return

        Returns:
            Reranked results with profile boost scores
        """
        if not results or not profile:
            return results

        # Extract relevant profile information
        profile_keywords = self._extract_profile_keywords(profile)

        # Score each result
        boosted_results = []
        for result in results:
            original_score = result.get("score", 0.0)
            content = result.get("content", "").lower()

            # Calculate profile boost
            boost_factor = self._calculate_boost_factor(
                content=content,
                keywords=profile_keywords
            )

            # Apply boost
            boosted_score = original_score * boost_factor

            # Add boosted result
            boosted_result = result.copy()
            boosted_result["original_score"] = original_score
            boosted_result["profile_boost"] = boost_factor
            boosted_result["score"] = boosted_score

            boosted_results.append(boosted_result)

        # Sort by boosted score
        boosted_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top-k
        if top_k:
            boosted_results = boosted_results[:top_k]

        logger.info(
            f"Profile-reranked {len(results)} results "
            f"with {len(profile_keywords)} profile keywords"
        )

        return boosted_results

    def _extract_profile_keywords(
        self,
        profile: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Extract keywords from profile organized by category.

        Args:
            profile: User profile

        Returns:
            Dictionary mapping categories to keyword lists
        """
        keywords = {
            "expertise": [],
            "projects": [],
            "interests": [],
            "goals": [],
            "preferences": []
        }

        # Process static facts
        for fact in profile.get("static", []):
            category = fact.get("category", "")
            value = fact.get("value", "").lower()

            if category in ["expertise", "role", "experience"]:
                keywords["expertise"].extend(self._tokenize(value))
            elif category == "preference":
                keywords["preferences"].extend(self._tokenize(value))
            elif category in ["interest"]:
                keywords["interests"].extend(self._tokenize(value))
            elif category == "goal":
                keywords["goals"].extend(self._tokenize(value))

        # Process dynamic facts (current projects, recent skills)
        for fact in profile.get("dynamic", []):
            category = fact.get("category", "")
            value = fact.get("value", "").lower()

            if category == "current_project":
                keywords["projects"].extend(self._tokenize(value))
            elif category == "recent_skill":
                keywords["expertise"].extend(self._tokenize(value))
            elif category == "goal":
                keywords["goals"].extend(self._tokenize(value))

        # Remove duplicates and filter
        for key in keywords:
            keywords[key] = list(set(keywords[key]))
            # Remove very short or common words
            keywords[key] = [
                kw for kw in keywords[key]
                if len(kw) > 3 and kw not in self._get_stop_words()
            ]

        return keywords

    def _calculate_boost_factor(
        self,
        content: str,
        keywords: Dict[str, List[str]]
    ) -> float:
        """
        Calculate boost factor based on keyword matches.

        Args:
            content: Content to check
            keywords: Profile keywords by category

        Returns:
            Boost multiplier (>= 1.0)
        """
        boost = 1.0

        # Check expertise matches
        for keyword in keywords.get("expertise", []):
            if keyword in content:
                boost *= self.config.expertise_boost
                break  # Apply once per category

        # Check project matches
        for keyword in keywords.get("projects", []):
            if keyword in content:
                boost *= self.config.project_boost
                break

        # Check interest matches
        for keyword in keywords.get("interests", []):
            if keyword in content:
                boost *= self.config.interest_boost
                break

        # Check goal matches
        for keyword in keywords.get("goals", []):
            if keyword in content:
                boost *= self.config.goal_boost
                break

        # Check preference matches
        for keyword in keywords.get("preferences", []):
            if keyword in content:
                boost *= self.config.preference_boost
                break

        return boost

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into keywords.

        Args:
            text: Text to tokenize

        Returns:
            List of keywords
        """
        # Remove special characters and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()

        # Filter short tokens
        tokens = [t for t in tokens if len(t) > 2]

        return tokens

    def _get_stop_words(self) -> set:
        """Common stop words to filter out"""
        return {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "her", "was", "one", "our", "out", "day", "get",
            "has", "him", "his", "how", "man", "new", "now", "old",
            "see", "two", "way", "who", "boy", "did", "its", "let",
            "put", "say", "she", "too", "use", "this", "that", "with",
            "from", "have", "they", "what", "will", "about", "using"
        }


# Global instance
_profile_reranker: Optional[ProfileReranker] = None


def get_profile_reranker(config: Optional[ProfileBoostConfig] = None) -> ProfileReranker:
    """
    Get profile reranker instance.

    Args:
        config: Optional configuration

    Returns:
        ProfileReranker instance
    """
    global _profile_reranker

    if _profile_reranker is None or config:
        _profile_reranker = ProfileReranker(config=config)

    return _profile_reranker


async def rerank_with_profile(
    results: List[Dict[str, Any]],
    profile: Dict[str, Any],
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank with profile.

    Args:
        results: Search results
        profile: User profile
        top_k: Number of results to return

    Returns:
        Profile-boosted results
    """
    reranker = get_profile_reranker()
    return reranker.rerank_with_profile(results, profile, top_k)


# Example usage
if __name__ == "__main__":
    # Example profile
    profile = {
        "static": [
            {"category": "expertise", "value": "Python programming and FastAPI"},
            {"category": "role", "value": "Senior Software Engineer"},
            {"category": "interest", "value": "Machine learning and AI"},
        ],
        "dynamic": [
            {"category": "current_project", "value": "Building Memory AI API"},
            {"category": "recent_skill", "value": "Vector databases like Milvus"},
        ]
    }

    # Example results
    results = [
        {
            "memory_id": "1",
            "content": "A guide to Python FastAPI development",
            "score": 0.75
        },
        {
            "memory_id": "2",
            "content": "Introduction to Milvus vector database",
            "score": 0.70
        },
        {
            "memory_id": "3",
            "content": "Java Spring Boot tutorial",
            "score": 0.80
        }
    ]

    reranker = get_profile_reranker()
    boosted = reranker.rerank_with_profile(results, profile, top_k=3)

    print("\n📊 Profile-Based Reranking Results:\n")
    for r in boosted:
        print(f"ID: {r['memory_id']}")
        print(f"  Original Score: {r['original_score']:.3f}")
        print(f"  Profile Boost: {r['profile_boost']:.3f}x")
        print(f"  Final Score: {r['score']:.3f}\n")
