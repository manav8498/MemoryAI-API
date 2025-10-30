"""
Procedural Memory Service.

Manages learned procedures, skills, and production rules.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from backend.core.logging_config import logger
from backend.models.procedural_memory import Procedure, ProcedureExecution, ProcedureTemplate


class ProceduralMemoryManager:
    """
    Manager for procedural memory operations.

    Handles:
    - Procedure creation and retrieval
    - Procedure execution tracking
    - Success rate updates
    - Procedure selection based on context
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_procedure(
        self,
        user_id: str,
        name: str,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        description: Optional[str] = None,
        category: Optional[str] = None,
        collection_id: Optional[str] = None,
        learned_from: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Procedure:
        """
        Create a new procedure.

        Args:
            user_id: User ID
            name: Procedure name
            condition: Condition dict
            action: Action dict
            description: Optional description
            category: Optional category
            collection_id: Optional collection
            learned_from: Memory IDs this was learned from
            parameters: Optional parameters

        Returns:
            Created procedure
        """
        try:
            procedure = Procedure(
                id=str(uuid.uuid4()),
                user_id=user_id,
                collection_id=collection_id,
                name=name,
                description=description,
                category=category,
                condition=condition,
                action=action,
                learned_from=learned_from or [],
                parameters=parameters or {},
                created_at=datetime.utcnow(),
            )

            self.db.add(procedure)
            await self.db.commit()
            await self.db.refresh(procedure)

            logger.info(f"Created procedure: {name} ({procedure.id})")
            return procedure

        except Exception as e:
            logger.error(f"Failed to create procedure: {e}")
            raise

    async def get_procedure(
        self,
        procedure_id: str,
        user_id: str,
    ) -> Optional[Procedure]:
        """Get a specific procedure."""
        result = await self.db.execute(
            select(Procedure).where(
                and_(
                    Procedure.id == procedure_id,
                    Procedure.user_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def find_matching_procedures(
        self,
        user_id: str,
        context: Dict[str, Any],
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> List[Procedure]:
        """
        Find procedures that match the given context.

        Args:
            user_id: User ID
            context: Current context
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching procedures
        """
        try:
            # Get active procedures with sufficient confidence
            result = await self.db.execute(
                select(Procedure)
                .where(
                    and_(
                        Procedure.user_id == user_id,
                        Procedure.is_active == True,
                        Procedure.confidence >= min_confidence,
                    )
                )
                .order_by(Procedure.success_rate.desc(), Procedure.usage_count.desc())
                .limit(limit * 2)  # Get more for filtering
            )
            procedures = list(result.scalars().all())

            # Filter by context matching
            matching = []
            for procedure in procedures:
                if self._matches_context(procedure.condition, context):
                    matching.append(procedure)
                    if len(matching) >= limit:
                        break

            logger.info(f"Found {len(matching)} matching procedures for context")
            return matching

        except Exception as e:
            logger.error(f"Failed to find matching procedures: {e}")
            return []

    def _matches_context(
        self,
        condition: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """
        Check if condition matches context.

        Simple implementation - would be more sophisticated in production.
        """
        # Check if all required keys in condition are present in context
        for key, value in condition.items():
            if key not in context:
                return False

            # Exact match or contains
            if isinstance(value, str):
                if value.lower() not in str(context[key]).lower():
                    return False
            elif context[key] != value:
                return False

        return True

    async def execute_procedure(
        self,
        procedure_id: str,
        user_id: str,
        input_state: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a procedure and track the execution.

        Args:
            procedure_id: Procedure ID
            user_id: User ID
            input_state: Current state
            parameters: Execution parameters

        Returns:
            Execution result
        """
        start_time = time.time()

        try:
            # Get procedure
            procedure = await self.get_procedure(procedure_id, user_id)
            if not procedure:
                return {
                    "success": False,
                    "error": f"Procedure {procedure_id} not found",
                }

            # Execute action
            # This is a simplified execution - real implementation would
            # interpret the action dict and execute the steps
            output_state = await self._execute_action(
                procedure.action,
                input_state,
                parameters or procedure.parameters,
            )

            success = output_state is not None
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log execution
            execution = ProcedureExecution(
                id=str(uuid.uuid4()),
                procedure_id=procedure_id,
                input_state=input_state,
                output_state=output_state,
                parameters_used=parameters or procedure.parameters,
                success=success,
                execution_time_ms=execution_time_ms,
                executed_at=datetime.utcnow(),
            )

            self.db.add(execution)

            # Update procedure stats
            procedure.usage_count += 1
            procedure.last_used_at = datetime.utcnow()
            if success:
                procedure.last_success_at = datetime.utcnow()

            await self.db.commit()

            logger.info(f"Executed procedure {procedure.name}: success={success}")

            return {
                "success": success,
                "output_state": output_state,
                "execution_time_ms": execution_time_ms,
                "procedure_name": procedure.name,
            }

        except Exception as e:
            logger.error(f"Procedure execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _execute_action(
        self,
        action: Dict[str, Any],
        input_state: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the action specified in the procedure.

        Simplified implementation - would be more sophisticated in production.
        """
        try:
            operation = action.get("operation")
            output_state = input_state.copy()

            if operation == "multiply":
                # Example: multiply a value by a factor
                key = action.get("key", "value")
                factor = action.get("factor", parameters.get("factor", 1.0))
                if key in input_state:
                    output_state[key] = input_state[key] * factor

            elif operation == "add":
                key = action.get("key", "value")
                amount = action.get("amount", parameters.get("amount", 0))
                if key in input_state:
                    output_state[key] = input_state[key] + amount

            elif operation == "transform":
                # Custom transformation
                transform_fn = action.get("function")
                # Would evaluate transform function here
                pass

            elif operation == "sequence":
                # Execute a sequence of steps
                steps = action.get("steps", [])
                for step in steps:
                    output_state = await self._execute_action(step, output_state, parameters)
                    if output_state is None:
                        return None

            return output_state

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return None

    async def record_feedback(
        self,
        execution_id: str,
        success: bool,
        reward: Optional[float] = None,
        user_feedback: Optional[float] = None,
    ) -> bool:
        """
        Record feedback for a procedure execution.

        Args:
            execution_id: Execution ID
            success: Whether it succeeded
            reward: RL reward signal
            user_feedback: User rating

        Returns:
            True if successful
        """
        try:
            # Update execution
            result = await self.db.execute(
                select(ProcedureExecution).where(ProcedureExecution.id == execution_id)
            )
            execution = result.scalar_one_or_none()

            if not execution:
                return False

            execution.success = success
            execution.reward = reward
            execution.user_feedback = user_feedback

            # Update procedure success rate
            procedure_result = await self.db.execute(
                select(Procedure).where(Procedure.id == execution.procedure_id)
            )
            procedure = procedure_result.scalar_one_or_none()

            if procedure:
                # Recalculate success rate
                executions_result = await self.db.execute(
                    select(func.count(), func.sum(ProcedureExecution.success.cast(Integer)))
                    .where(ProcedureExecution.procedure_id == procedure.id)
                )
                total, successes = executions_result.one()

                if total > 0:
                    procedure.success_rate = (successes or 0) / total

                # Update confidence based on feedback
                if reward is not None:
                    # Exponential moving average
                    alpha = 0.1
                    procedure.confidence = (
                        alpha * max(0, min(1, (reward + 1) / 2))  # Normalize reward to 0-1
                        + (1 - alpha) * procedure.confidence
                    )

            await self.db.commit()

            logger.info(f"Recorded feedback for execution {execution_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    async def learn_procedure_from_examples(
        self,
        user_id: str,
        examples: List[Dict[str, Any]],
        name: str,
        category: Optional[str] = None,
    ) -> Optional[Procedure]:
        """
        Learn a new procedure from examples.

        Args:
            user_id: User ID
            examples: List of example executions
            name: Procedure name
            category: Optional category

        Returns:
            Learned procedure or None
        """
        try:
            # Analyze examples to extract condition and action patterns
            # This is simplified - real implementation would use ML/LLM

            if not examples:
                return None

            # Extract common condition pattern
            condition = self._extract_condition_pattern(examples)

            # Extract action pattern
            action = self._extract_action_pattern(examples)

            # Create procedure
            procedure = await self.create_procedure(
                user_id=user_id,
                name=name,
                condition=condition,
                action=action,
                category=category,
                learned_from=[],  # Would include memory IDs
            )

            logger.info(f"Learned new procedure from {len(examples)} examples")
            return procedure

        except Exception as e:
            logger.error(f"Failed to learn procedure: {e}")
            return None

    def _extract_condition_pattern(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common condition pattern from examples."""
        # Simplified - would analyze common fields in input states
        if not examples:
            return {}

        first = examples[0].get("input_state", {})
        common_keys = set(first.keys())

        for example in examples[1:]:
            common_keys &= set(example.get("input_state", {}).keys())

        return {key: "any" for key in common_keys}

    def _extract_action_pattern(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common action pattern from examples."""
        # Simplified - would analyze transformations
        return {"operation": "custom", "steps": []}

    async def get_user_procedures(
        self,
        user_id: str,
        category: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Procedure]:
        """Get all procedures for a user."""
        conditions = [Procedure.user_id == user_id]

        if category:
            conditions.append(Procedure.category == category)
        if active_only:
            conditions.append(Procedure.is_active == True)

        result = await self.db.execute(
            select(Procedure)
            .where(and_(*conditions))
            .order_by(Procedure.success_rate.desc())
        )

        return list(result.scalars().all())


def get_procedural_memory_manager(db: AsyncSession) -> ProceduralMemoryManager:
    """Get procedural memory manager instance."""
    return ProceduralMemoryManager(db)
