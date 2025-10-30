"""
Memory model for storing user memories and associated metadata.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Float, Text, JSON, Boolean
from sqlalchemy.orm import relationship

from backend.core.database import Base


class Memory(Base):
    """
    Memory model for storing semantic memories.

    Each memory represents a chunk of information with:
    - Text content stored in PostgreSQL
    - Vector embedding stored in Milvus
    - Entity/relationship graph stored in Neo4j
    - Metadata for filtering and ranking

    Attributes:
        id: Unique memory identifier (UUID)
        collection_id: Parent collection ID
        content: Memory text content
        content_hash: SHA256 hash for deduplication
        chunk_index: Index if memory is part of chunked document
        source_type: Type of source (text, document, url, etc.)
        source_reference: Reference to original source
        importance: Importance score (0-1)
        access_count: Number of times accessed
        last_accessed_at: Last access timestamp
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    __tablename__ = "memories"

    id = Column(String(36), primary_key=True, index=True)
    collection_id = Column(
        String(36),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)

    # Chunking info
    chunk_index = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)

    # Source tracking
    source_type = Column(String(50), default="text", nullable=False)
    source_reference = Column(String(500), nullable=True)

    # Memory attributes
    importance = Column(Float, default=0.5, nullable=False)
    decay_rate = Column(Float, default=0.001, nullable=False)

    # Usage stats
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    collection = relationship("Collection", back_populates="memories")
    extended_metadata = relationship(
        "MemoryMetadata",
        back_populates="memory",
        cascade="all, delete-orphan",
        uselist=False,
    )

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Memory {self.id}: {preview}>"


class MemoryMetadata(Base):
    """
    Extended metadata for memories.

    Stores additional information extracted during ingestion:
    - Entities and keywords
    - Temporal information
    - Custom metadata
    - Processing results

    Attributes:
        id: Unique metadata ID
        memory_id: Associated memory ID
        entities: Extracted entities (JSON)
        keywords: Extracted keywords
        temporal_context: Temporal information
        custom_metadata: User-provided metadata
        processing_metadata: System processing info
    """

    __tablename__ = "memory_metadata"

    id = Column(String(36), primary_key=True, index=True)
    memory_id = Column(
        String(36),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Extracted information
    entities = Column(JSON, default=list, nullable=False)  # [{"text": "John", "type": "PERSON"}]
    keywords = Column(JSON, default=list, nullable=False)  # ["machine learning", "neural networks"]

    # Temporal context
    temporal_context = Column(JSON, default=dict, nullable=False)  # {"date": "2024-01-01", "period": "morning"}

    # Custom metadata
    custom_metadata = Column(JSON, default=dict, nullable=False)  # User-provided metadata

    # Processing metadata
    processing_metadata = Column(JSON, default=dict, nullable=False)  # {"model": "bge-m3", "processed_at": "..."}

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    memory = relationship("Memory", back_populates="extended_metadata")

    def __repr__(self) -> str:
        return f"<MemoryMetadata for {self.memory_id}>"
