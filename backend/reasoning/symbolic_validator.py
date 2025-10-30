"""
Symbolic validator for applying logical rules and constraints.

Provides rule-based validation and reasoning to complement neural approaches.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re

from backend.core.logging_config import logger


class RuleType(str, Enum):
    """Types of validation rules."""
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    LENGTH = "length"
    CUSTOM = "custom"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    TEMPORAL = "temporal"


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    rule_id: str
    rule_type: RuleType
    condition: Any
    message: str
    severity: str = "error"  # error, warning, info


class ValidationResult:
    """Result of validation."""

    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.info = []

    def add_error(self, message: str):
        """Add error."""
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add warning."""
        self.warnings.append(message)

    def add_info(self, message: str):
        """Add info."""
        self.info.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


class SymbolicValidator:
    """
    Symbolic validator for rule-based reasoning.
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.custom_validators: Dict[str, Callable] = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self.rules.append(rule)
        logger.debug(f"Added validation rule: {rule.rule_id}")

    def add_custom_validator(self, name: str, validator: Callable) -> None:
        """
        Add custom validator function.

        Args:
            name: Validator name
            validator: Function that takes (value) and returns (bool, str)
        """
        self.custom_validators[name] = validator

    async def validate(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate text against all rules.

        Args:
            text: Text to validate
            context: Optional context for validation

        Returns:
            ValidationResult
        """
        result = ValidationResult()
        context = context or {}

        for rule in self.rules:
            try:
                is_valid = await self._apply_rule(text, rule, context)

                if not is_valid:
                    if rule.severity == "error":
                        result.add_error(rule.message)
                    elif rule.severity == "warning":
                        result.add_warning(rule.message)
                    else:
                        result.add_info(rule.message)

            except Exception as e:
                logger.error(f"Rule {rule.rule_id} failed: {e}")
                result.add_error(f"Validation error: {rule.rule_id}")

        return result

    async def _apply_rule(
        self,
        text: str,
        rule: ValidationRule,
        context: Dict[str, Any],
    ) -> bool:
        """Apply a single rule."""
        if rule.rule_type == RuleType.CONTAINS:
            return self._validate_contains(text, rule.condition)

        elif rule.rule_type == RuleType.NOT_CONTAINS:
            return not self._validate_contains(text, rule.condition)

        elif rule.rule_type == RuleType.REGEX:
            return self._validate_regex(text, rule.condition)

        elif rule.rule_type == RuleType.LENGTH:
            return self._validate_length(text, rule.condition)

        elif rule.rule_type == RuleType.CUSTOM:
            return await self._validate_custom(text, rule.condition, context)

        elif rule.rule_type == RuleType.LOGICAL_AND:
            return all(
                await self._apply_rule(text, sub_rule, context)
                for sub_rule in rule.condition
            )

        elif rule.rule_type == RuleType.LOGICAL_OR:
            return any(
                await self._apply_rule(text, sub_rule, context)
                for sub_rule in rule.condition
            )

        else:
            logger.warning(f"Unknown rule type: {rule.rule_type}")
            return True

    def _validate_contains(self, text: str, pattern: str) -> bool:
        """Check if text contains pattern."""
        return pattern.lower() in text.lower()

    def _validate_regex(self, text: str, pattern: str) -> bool:
        """Check if text matches regex pattern."""
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return False

    def _validate_length(self, text: str, condition: Dict[str, int]) -> bool:
        """Validate text length."""
        length = len(text)

        if "min" in condition and length < condition["min"]:
            return False

        if "max" in condition and length > condition["max"]:
            return False

        return True

    async def _validate_custom(
        self,
        text: str,
        validator_name: str,
        context: Dict[str, Any],
    ) -> bool:
        """Run custom validator."""
        if validator_name not in self.custom_validators:
            logger.warning(f"Custom validator not found: {validator_name}")
            return True

        validator = self.custom_validators[validator_name]

        try:
            result = validator(text, context)
            if isinstance(result, tuple):
                return result[0]
            return bool(result)
        except Exception as e:
            logger.error(f"Custom validator {validator_name} failed: {e}")
            return False

    def create_content_policy_rules(self) -> None:
        """Create common content policy validation rules."""
        # No PII
        self.add_rule(ValidationRule(
            rule_id="no_ssn",
            rule_type=RuleType.REGEX,
            condition=r'\b\d{3}-\d{2}-\d{4}\b',
            message="Content may contain SSN (Social Security Number)",
            severity="warning",
        ))

        self.add_rule(ValidationRule(
            rule_id="no_credit_card",
            rule_type=RuleType.REGEX,
            condition=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            message="Content may contain credit card number",
            severity="warning",
        ))

        # Length constraints
        self.add_rule(ValidationRule(
            rule_id="min_length",
            rule_type=RuleType.LENGTH,
            condition={"min": 10},
            message="Content is too short (minimum 10 characters)",
            severity="warning",
        ))

        self.add_rule(ValidationRule(
            rule_id="max_length",
            rule_type=RuleType.LENGTH,
            condition={"max": 100000},
            message="Content exceeds maximum length",
            severity="error",
        ))

        logger.info("Created content policy validation rules")

    def validate_memory_quality(self, text: str) -> Dict[str, Any]:
        """
        Validate memory content quality.

        Returns:
            Dictionary with quality metrics
        """
        quality = {
            "score": 1.0,
            "issues": [],
            "suggestions": [],
        }

        # Check length
        if len(text) < 20:
            quality["score"] -= 0.2
            quality["issues"].append("Very short content")
            quality["suggestions"].append("Add more detail for better context")

        # Check for gibberish (very simple check)
        words = text.split()
        if len(words) > 0:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length > 15 or avg_word_length < 3:
                quality["score"] -= 0.1
                quality["issues"].append("Unusual word patterns")

        # Check for repeated content
        if len(set(words)) < len(words) * 0.3:
            quality["score"] -= 0.15
            quality["issues"].append("High repetition detected")

        # Check for code blocks (might indicate technical content)
        if "```" in text or "def " in text or "function " in text:
            quality["suggestions"].append("Consider adding explanation for code snippets")

        quality["score"] = max(0.0, min(1.0, quality["score"]))

        return quality


# Global validator instance
_symbolic_validator: Optional[SymbolicValidator] = None


def get_symbolic_validator() -> SymbolicValidator:
    """
    Get symbolic validator instance.

    Returns:
        SymbolicValidator instance
    """
    global _symbolic_validator

    if _symbolic_validator is None:
        _symbolic_validator = SymbolicValidator()
        # Add default rules
        if settings.ENABLE_SYMBOLIC_VALIDATION:
            _symbolic_validator.create_content_policy_rules()

    return _symbolic_validator


def validate_llm_output(
    output: str,
    expected_format: Optional[str] = None,
) -> ValidationResult:
    """
    Validate LLM output for consistency and format.

    Args:
        output: LLM generated text
        expected_format: Expected format (json, markdown, etc.)

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    # Check if empty
    if not output or not output.strip():
        result.add_error("Output is empty")
        return result

    # Format-specific validation
    if expected_format == "json":
        try:
            import json
            json.loads(output)
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON: {e}")

    elif expected_format == "markdown":
        # Basic markdown validation
        if not any(marker in output for marker in ["#", "**", "-", "1."]):
            result.add_warning("Output doesn't appear to be formatted as Markdown")

    # Check for hallucination indicators
    hallucination_phrases = [
        "i don't have access",
        "i cannot access",
        "as an ai",
        "i apologize, but",
    ]

    for phrase in hallucination_phrases:
        if phrase in output.lower():
            result.add_info(f"Possible hallucination indicator: '{phrase}'")

    return result
