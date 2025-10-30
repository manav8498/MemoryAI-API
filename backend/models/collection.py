"""
Collection model for organizing memories.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Integer, Text, JSON
from sqlalchemy.orm import relationship

from backend.core.database import Base


class Collection(Base):
    """
    Collection model for organizing memories into logical groups.

    Collections allow users to separate memories by project, context, or use case.
    Each collection has its own vector space in Milvus and subgraph in Neo4j.

    Attributes:
        id: Unique collection identifier
        user_id: Owner user ID
        name: Collection name
        description: Collection description
        is_active: Whether collection is active
        memory_count: Cached count of memories in collection
        metadata: Additional metadata (JSON)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    __tablename__ = "collections"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Collection data
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Stats
    memory_count = Column(Integer, default=0, nullable=False)

    # Metadata
    custom_metadata = Column(JSON, default=dict, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="collections")
    memories = relationship("Memory", back_populates="collection", cascade="all, delete-orphan")
    trajectories = relationship("Trajectory", back_populates="collection", cascade="all, delete-orphan")
    procedures = relationship("Procedure", back_populates="collection", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Collection {self.name} ({self.memory_count} memories)>"
