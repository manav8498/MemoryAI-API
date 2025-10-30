"""
Cognitive processor for entity extraction and relationship inference.

Uses NLP to extract:
- Named entities (persons, organizations, locations, etc.)
- Keywords and concepts
- Temporal information
- Relationships between entities
"""
from typing import List, Dict, Any, Tuple, Optional
import re
from datetime import datetime
from collections import Counter

from backend.core.logging_config import logger


class Entity:
    """Represents an extracted entity."""

    def __init__(self, text: str, entity_type: str, start: int, end: int):
        self.text = text
        self.type = entity_type
        self.start = start
        self.end = end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "start": self.start,
            "end": self.end,
        }


class Relationship:
    """Represents a relationship between entities."""

    def __init__(
        self,
        source: str,
        target: str,
        relation_type: str,
        confidence: float = 1.0,
    ):
        self.source = source
        self.target = target
        self.type = relation_type
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "confidence": self.confidence,
        }


class CognitiveProcessor:
    """
    Cognitive processor for extracting structured information from text.
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.nlp = None

        # Try to load spaCy model
        if use_spacy:
            try:
                import spacy
                # Try to load English model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model: en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model not found, using fallback extraction")
                    self.nlp = None
            except ImportError:
                logger.warning("spaCy not available, using fallback extraction")
                self.nlp = None

    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text and extract structured information.

        Args:
            text: Text to process

        Returns:
            Dictionary with extracted entities, keywords, temporal info, etc.
        """
        try:
            # Extract entities
            entities = await self.extract_entities(text)

            # Extract keywords
            keywords = await self.extract_keywords(text)

            # Extract temporal information
            temporal_info = await self.extract_temporal_info(text)

            # Infer relationships
            relationships = await self.infer_relationships(text, entities)

            return {
                "entities": [e.to_dict() for e in entities],
                "keywords": keywords,
                "temporal_info": temporal_info,
                "relationships": [r.to_dict() for r in relationships],
            }

        except Exception as e:
            logger.error(f"Cognitive processing failed: {e}")
            return {
                "entities": [],
                "keywords": [],
                "temporal_info": {},
                "relationships": [],
            }

    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Text to process

        Returns:
            List of Entity objects
        """
        try:
            if self.nlp:
                return await self._extract_entities_spacy(text)
            else:
                return await self._extract_entities_fallback(text)

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    async def _extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                entities.append(Entity(
                    text=ent.text,
                    entity_type=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                ))

            logger.debug(f"Extracted {len(entities)} entities using spaCy")
            return entities

        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
            return []

    async def _extract_entities_fallback(self, text: str) -> List[Entity]:
        """Fallback entity extraction using regex patterns."""
        entities = []

        # Extract potential person names (capitalized words)
        person_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        for match in re.finditer(person_pattern, text):
            entities.append(Entity(
                text=match.group(1),
                entity_type="PERSON",
                start=match.start(),
                end=match.end(),
            ))

        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="EMAIL",
                start=match.start(),
                end=match.end(),
            ))

        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="URL",
                start=match.start(),
                end=match.end(),
            ))

        # Extract phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="PHONE",
                start=match.start(),
                end=match.end(),
            ))

        logger.debug(f"Extracted {len(entities)} entities using fallback")
        return entities

    async def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract keywords from text.

        Args:
            text: Text to process
            top_k: Number of keywords to return

        Returns:
            List of keywords
        """
        try:
            # Simple keyword extraction using word frequency
            # Remove common words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                'does', 'did', 'will', 'would', 'could', 'should', 'may',
                'might', 'must', 'can', 'this', 'that', 'these', 'those',
                'it', 'its', 'itself', 'i', 'me', 'my', 'myself', 'we', 'our',
                'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                'hers', 'herself', 'they', 'them', 'their', 'theirs',
                'themselves', 'what', 'which', 'who', 'when', 'where', 'why',
                'how',
            }

            # Tokenize and count
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            word_counts = Counter(w for w in words if w not in stop_words)

            # Get top keywords
            keywords = [word for word, count in word_counts.most_common(top_k)]

            logger.debug(f"Extracted {len(keywords)} keywords")
            return keywords

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    async def extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """
        Extract temporal information from text.

        Args:
            text: Text to process

        Returns:
            Dictionary with temporal information
        """
        try:
            temporal_info = {
                "dates": [],
                "times": [],
                "periods": [],
            }

            # Extract dates (simple patterns)
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b',         # YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            ]

            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                temporal_info["dates"].extend(matches)

            # Extract times
            time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'
            temporal_info["times"] = re.findall(time_pattern, text)

            # Extract time periods
            period_keywords = ['morning', 'afternoon', 'evening', 'night', 'today', 'yesterday', 'tomorrow']
            temporal_info["periods"] = [
                word for word in period_keywords
                if word in text.lower()
            ]

            logger.debug(f"Extracted temporal info: {len(temporal_info['dates'])} dates, "
                        f"{len(temporal_info['times'])} times")
            return temporal_info

        except Exception as e:
            logger.error(f"Temporal extraction failed: {e}")
            return {"dates": [], "times": [], "periods": []}

    async def infer_relationships(
        self,
        text: str,
        entities: List[Entity],
    ) -> List[Relationship]:
        """
        Infer relationships between entities.

        Args:
            text: Original text
            entities: List of extracted entities

        Returns:
            List of Relationship objects
        """
        try:
            relationships = []

            # Simple relationship patterns
            relationship_patterns = [
                (r'\b(\w+)\s+works\s+(?:at|for)\s+(\w+)\b', 'WORKS_AT'),
                (r'\b(\w+)\s+(?:is|was)\s+(?:the\s+)?CEO\s+of\s+(\w+)\b', 'CEO_OF'),
                (r'\b(\w+)\s+founded\s+(\w+)\b', 'FOUNDED'),
                (r'\b(\w+)\s+(?:lives|lived)\s+in\s+(\w+)\b', 'LIVES_IN'),
                (r'\b(\w+)\s+(?:knows|knew)\s+(\w+)\b', 'KNOWS'),
            ]

            for pattern, rel_type in relationship_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    relationships.append(Relationship(
                        source=match.group(1),
                        target=match.group(2),
                        relation_type=rel_type,
                        confidence=0.7,  # Lower confidence for simple pattern matching
                    ))

            logger.debug(f"Inferred {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.error(f"Relationship inference failed: {e}")
            return []

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis
        """
        try:
            # Simple sentiment analysis using keyword matching
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'happy', 'joy', 'pleased', 'satisfied', 'positive',
            }
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad',
                'angry', 'disappointed', 'frustrated', 'negative', 'poor',
            }

            words = re.findall(r'\b[a-z]+\b', text.lower())

            positive_count = sum(1 for w in words if w in positive_words)
            negative_count = sum(1 for w in words if w in negative_words)
            total_words = len(words)

            if total_words == 0:
                return {"sentiment": "neutral", "score": 0.0}

            # Calculate sentiment score (-1 to 1)
            score = (positive_count - negative_count) / max(total_words, 1)

            if score > 0.1:
                sentiment = "positive"
            elif score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "score": score,
                "positive_count": positive_count,
                "negative_count": negative_count,
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "score": 0.0}


# Global processor instance
_cognitive_processor: Optional[CognitiveProcessor] = None


def get_cognitive_processor() -> CognitiveProcessor:
    """
    Get cognitive processor instance.

    Returns:
        CognitiveProcessor instance
    """
    global _cognitive_processor

    if _cognitive_processor is None:
        _cognitive_processor = CognitiveProcessor()

    return _cognitive_processor
