"""
Procedural Memory Models.

Stores learned procedures, skills, and production rules.
Third type of long-term memory (episodic, semantic, procedural).
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Float, Text, JSON, Boolean
from sqlalchemy.orm import relationship

from backend.core.database import Base


class Procedure(Base):
    """
    Procedural memory: learned procedures and skills.

    Stores IF-THEN rules, algorithms, and behavioral patterns
    that the agent has learned from experience.
    """

    __tablename__ = "procedures"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    collection_id = Column(String(36), ForeignKey("collections.id", ondelete="CASCADE"), nullable=True, index=True)

    # Procedure identification
    name = Column(String(200), nullable=False, index=True)  # "calculate_tip", "chess_opening"
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True, index=True)  # "math", "navigation", "social"

    # Condition → Action structure
    condition = Column(JSON, nullable=False)  # {"context": "restaurant", "trigger": "bill_received"}
    action = Column(JSON, nullable=False)  # {"operation": "multiply", "factor": 0.15, "steps": [...]}

    # Learning metadata
    success_rate = Column(Float, default=0.5, nullable=False)  # How often this procedure succeeds
    usage_count = Column(Integer, default=0, nullable=False)  # How many times used
    learned_from = Column(JSON, default=list, nullable=False)  # Memory IDs this was learned from

    # Confidence and strength
    confidence = Column(Float, default=0.5, nullable=False)  # Confidence in this procedure
    strength = Column(Float, default=0.5, nullable=False)  # How strongly encoded (like procedural memory)

    # Parameters
    parameters = Column(JSON, default=dict, nullable=False)  # Configurable parameters
    constraints = Column(JSON, default=dict, nullable=False)  # Constraints on when to apply

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)  # Has been verified to work

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="procedures")
    collection = relationship("Collection", back_populates="procedures")

    def __repr__(self) -> str:
        return f"<Procedure {self.name} (success_rate: {self.success_rate:.2f}, uses: {self.usage_count})>"


class ProcedureExecution(Base):
    """
    Tracks executions of procedures for learning.

    Records when procedures are used and their outcomes,
    enabling reinforcement learning of procedural memory.
    """

    __tablename__ = "procedure_executions"

    id = Column(String(36), primary_key=True, index=True)
    procedure_id = Column(
        String(36),
        ForeignKey("procedures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Execution context
    input_state = Column(JSON, nullable=False)  # State when procedure was triggered
    output_state = Column(JSON, nullable=True)  # Resulting state
    parameters_used = Column(JSON, nullable=False)  # Actual parameters used

    # Outcome
    success = Column(Boolean, nullable=True)  # Whether execution succeeded
    execution_time_ms = Column(Integer, nullable=True)  # How long it took
    error_message = Column(Text, nullable=True)  # Error if failed

    # Feedback
    user_feedback = Column(Float, nullable=True)  # User rating of outcome (-1 to 1)
    reward = Column(Float, nullable=True)  # RL reward signal

    # Timestamps
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    procedure = relationship("Procedure")

    def __repr__(self) -> str:
        return f"<ProcedureExecution {self.id} (success: {self.success})>"


class ProcedureTemplate(Base):
    """
    Templates for common procedure types.

    Pre-defined procedure structures that can be instantiated
    with specific parameters.
    """

    __tablename__ = "procedure_templates"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True, index=True)

    # Template structure
    condition_schema = Column(JSON, nullable=False)  # Schema for conditions
    action_schema = Column(JSON, nullable=False)  # Schema for actions
    parameter_schema = Column(JSON, nullable=False)  # Required/optional parameters

    # Examples
    examples = Column(JSON, default=list, nullable=False)  # Example instantiations

    # Metadata
    is_builtin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ProcedureTemplate {self.name}>"
