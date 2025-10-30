"""
User model for authentication and multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Enum
from sqlalchemy.orm import relationship
import enum

from backend.core.database import Base


class UserTier(str, enum.Enum):
    """User subscription tiers."""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    """
    User model for multi-tenant memory storage.

    Attributes:
        id: Unique user identifier
        email: User email (unique)
        hashed_password: Bcrypt hashed password
        full_name: User's full name
        is_active: Whether user account is active
        is_verified: Whether email is verified
        tier: Subscription tier
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))

    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    tier = Column(Enum(UserTier), default=UserTier.FREE, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    collections = relationship("Collection", back_populates="user", cascade="all, delete-orphan")
    trajectories = relationship("Trajectory", back_populates="user", cascade="all, delete-orphan")
    procedures = relationship("Procedure", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.email}>"
