"""
User Profile Models - Automatic User Profiling System

Similar to SuperMemory's user profiles feature:
- Automatically extracts and maintains facts about users
- Static profile: long-term facts (role, expertise, preferences)
- Dynamic profile: recent facts (current projects, temporary states)
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Text, JSON, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
import enum

from backend.core.database import Base


class ProfileType(str, enum.Enum):
    """Profile fact types."""
    STATIC = "static"    # Long-term, rarely changing facts
    DYNAMIC = "dynamic"  # Recent, temporary facts


class ProfileCategory(str, enum.Enum):
    """Categories of user facts."""
    ROLE = "role"                        # Professional role
    EXPERTISE = "expertise"              # Areas of expertise
    PREFERENCE = "preference"            # User preferences
    EDUCATION = "education"              # Educational background
    EXPERIENCE = "experience"            # Work experience
    CURRENT_PROJECT = "current_project"  # Active projects
    RECENT_SKILL = "recent_skill"        # Recently acquired skills
    TEMPORARY_STATE = "temporary_state"  # Temporary conditions
    GOAL = "goal"                        # User goals
    INTEREST = "interest"                # Interests and hobbies
    COMMUNICATION = "communication"      # Communication style
    OTHER = "other"                      # Other facts


class UserProfileFact(Base):
    """
    Individual user profile facts extracted from memories.

    Automatically maintained collection of facts about the user,
    enabling instant context retrieval without searching through all memories.

    Attributes:
        id: Unique fact identifier
        user_id: User this fact belongs to
        profile_type: Static (long-term) or Dynamic (recent)
        category: Type of fact (role, expertise, preference, etc.)
        fact_key: Specific attribute name
        fact_value: The actual value/description
        confidence: Confidence score (0.0-1.0)
        importance: Importance score (0.0-1.0)
        source_memory_ids: Memory IDs this was extracted from
        extraction_metadata: How and when this was extracted
        verified: Whether this fact has been verified
        last_accessed_at: Last time this fact was used
        access_count: How many times accessed
        created_at: When fact was first created
        updated_at: Last update timestamp
    """

    __tablename__ = "user_profile_facts"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Profile classification
    profile_type = Column(
        String(20),
        nullable=False,
        default=ProfileType.DYNAMIC.value,
        index=True,
    )
    category = Column(
        String(50),
        nullable=False,
        default=ProfileCategory.OTHER.value,
        index=True,
    )

    # Fact content
    fact_key = Column(String(200), nullable=False, index=True)  # e.g., "current_role", "expertise_area"
    fact_value = Column(Text, nullable=False)  # The actual fact content

    # Confidence and importance
    confidence = Column(Float, default=0.7, nullable=False)  # How confident we are in this fact
    importance = Column(Float, default=0.5, nullable=False)  # How important this fact is

    # Source tracking
    source_memory_ids = Column(JSON, default=list, nullable=False)  # Which memories this came from
    extraction_metadata = Column(JSON, default=dict, nullable=False)  # Extraction details

    # Verification and usage
    verified = Column(String(20), default="auto", nullable=False)  # auto, user_confirmed, user_corrected
    last_accessed_at = Column(DateTime, nullable=True)
    access_count = Column(String(36), default="0", nullable=False)  # Changed to String to match schema

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    user = relationship("User", backref="profile_facts")
    operations = relationship(
        "ProfileOperation",
        back_populates="profile_fact",
        cascade="all, delete-orphan",
    )

    # Composite index for fast lookups
    __table_args__ = (
        Index("idx_user_profile_lookup", "user_id", "profile_type", "category"),
        Index("idx_user_fact_key", "user_id", "fact_key"),
    )

    def __repr__(self) -> str:
        return f"<UserProfileFact {self.fact_key}={self.fact_value[:50]}... (conf: {self.confidence:.2f})>"


class ProfileOperation(Base):
    """
    Tracks all operations on user profiles for audit and learning.

    Records additions, updates, and removals of profile facts,
    enabling profile versioning and operation history.

    Attributes:
        id: Unique operation ID
        profile_fact_id: Associated profile fact
        user_id: User this operation affects
        operation_type: Type of operation (add, update, remove)
        old_value: Previous value (for updates/removals)
        new_value: New value (for adds/updates)
        confidence_change: Change in confidence score
        trigger_memory_id: Memory that triggered this operation
        trigger_type: What triggered this operation
        metadata: Additional operation metadata
        created_at: When operation occurred
    """

    __tablename__ = "profile_operations"

    id = Column(String(36), primary_key=True, index=True)
    profile_fact_id = Column(
        String(36),
        ForeignKey("user_profile_facts.id", ondelete="CASCADE"),
        nullable=True,  # Null for 'add' operations before fact is created
        index=True,
    )
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Operation details
    operation_type = Column(String(20), nullable=False)  # add, update, remove, verify
    old_value = Column(Text, nullable=True)  # Previous value
    new_value = Column(Text, nullable=True)  # New value
    confidence_change = Column(Float, nullable=True)  # Change in confidence

    # Trigger information
    trigger_memory_id = Column(String(36), nullable=True)  # Which memory triggered this
    trigger_type = Column(String(50), nullable=True)  # auto_extraction, user_input, consolidation

    # Additional context
    operation_metadata = Column(JSON, default=dict, nullable=False)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    profile_fact = relationship("UserProfileFact", back_populates="operations")
    user = relationship("User")

    def __repr__(self) -> str:
        return f"<ProfileOperation {self.operation_type} at {self.created_at}>"


class ProfileSnapshot(Base):
    """
    Periodic snapshots of complete user profiles.

    Enables profile versioning and rollback capabilities.
    Useful for analyzing how user understanding evolves over time.

    Attributes:
        id: Unique snapshot ID
        user_id: User this snapshot belongs to
        static_facts: Snapshot of static profile
        dynamic_facts: Snapshot of dynamic profile
        metadata: Snapshot metadata (trigger, stats, etc.)
        created_at: When snapshot was taken
    """

    __tablename__ = "profile_snapshots"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Profile data at snapshot time
    static_facts = Column(JSON, nullable=False)  # Static profile facts
    dynamic_facts = Column(JSON, nullable=False)  # Dynamic profile facts

    # Snapshot metadata
    snapshot_metadata = Column(JSON, default=dict, nullable=False)  # Stats, trigger reason, etc.

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User")

    def __repr__(self) -> str:
        return f"<ProfileSnapshot user={self.user_id} at {self.created_at}>"
