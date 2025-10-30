"""
API Key model for authentication.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship

from backend.core.database import Base


class APIKey(Base):
    """
    API Key model for secure API access.

    Attributes:
        id: Unique key identifier
        user_id: Associated user ID
        key_hash: SHA256 hash of the API key
        name: Human-readable key name
        prefix: First 7 characters of key (for identification)
        is_active: Whether key is active
        last_used_at: Last usage timestamp
        expires_at: Expiration timestamp (optional)
        created_at: Key creation timestamp
    """

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Key data
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    prefix = Column(String(20), nullable=False)  # e.g., "mem_sk_abc1234567"

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Usage tracking
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<APIKey {self.prefix}... ({self.name})>"
