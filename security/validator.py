"""
Input Validation and Security

Provides comprehensive input validation, sanitization, and security checks
to prevent common attacks and ensure data integrity.
"""

import functools
import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from utils.error_handling import ErrorCategory, ErrorSeverity, MCPError


class ValidationRule(str, Enum):
    """Built-in validation rules."""

    REQUIRED = "required"
    EMAIL = "email"
    URL = "url"
    ALPHANUMERIC = "alphanumeric"
    NUMERIC = "numeric"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    REGEX = "regex"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    JSON_VALID = "json_valid"
    NO_SQL_INJECTION = "no_sql_injection"
    NO_XSS = "no_xss"
    NO_SCRIPT_TAGS = "no_script_tags"
    SAFE_FILENAME = "safe_filename"
    SAFE_PATH = "safe_path"


class ValidationError(MCPError):
    """Validation error."""

    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details={"field": field, "value": str(value) if value else None},
        )
        self.field = field
        self.value = value


@dataclass
class ValidationResult:
    """Result of validation."""

    is_valid: bool
    errors: List[str]
    sanitized_value: Any = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SecurityValidator:
    """Security-focused validation utilities."""

    # Common patterns for security checks
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"('|(\\')|(\")|(\\\"))",
        r"(;|\|\||&&)",
    ]

    XSS_PATTERNS = [
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe\b",
        r"<object\b",
        r"<embed\b",
    ]

    PATH_TRAVERSAL_PATTERNS = [r"\.\./", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e%5c"]

    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check for SQL injection patterns."""
        value_lower = value.lower()

        for pattern in SecurityValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def check_xss(value: str) -> bool:
        """Check for XSS patterns."""
        value_lower = value.lower()

        for pattern in SecurityValidator.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def check_path_traversal(value: str) -> bool:
        """Check for path traversal patterns."""
        value_lower = value.lower()

        for pattern in SecurityValidator.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def sanitize_html(value: str) -> str:
        """Sanitize HTML content."""
        return html.escape(value, quote=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\.+', '.', sanitized)  # Replace multiple dots
        sanitized = sanitized.strip('. ')  # Remove leading/trailing dots and spaces

        # Limit length
        if len(sanitized) > 255:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:250] + ('.' + ext if ext else '')

        return sanitized or 'unnamed_file'

    @staticmethod
    def generate_content_hash(content: str) -> str:
        """Generate hash for content verification."""
        return hashlib.sha256(content.encode()).hexdigest()


class InputValidator:
    """Comprehensive input validation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Built-in validation patterns
        self.patterns = {
            ValidationRule.EMAIL: r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            ValidationRule.URL: r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
            ValidationRule.ALPHANUMERIC: r'^[a-zA-Z0-9]+$',
            ValidationRule.NUMERIC: r'^\d+$',
        }

        # Custom validators
        self.custom_validators: Dict[str, Callable] = {}

    def add_custom_validator(self, name: str, validator: Callable[[Any], bool]):
        """Add custom validator function."""
        self.custom_validators[name] = validator
        self.logger.info(f"Added custom validator: {name}")

    def validate_field(
        self, value: Any, rules: List[Union[ValidationRule, str, Dict[str, Any]]], field_name: str = "field"
    ) -> ValidationResult:
        """Validate a single field against rules."""
        errors = []
        warnings = []
        sanitized_value = value

        for rule in rules:
            try:
                result = self._apply_rule(value, rule, field_name)
                if not result.is_valid:
                    errors.extend(result.errors)
                if result.warnings:
                    warnings.extend(result.warnings)
                if result.sanitized_value is not None:
                    sanitized_value = result.sanitized_value
            except Exception as e:
                self.logger.error(f"Error applying validation rule {rule}: {e}")
                errors.append(f"Validation error for {field_name}")

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_value, warnings=warnings
        )

    def validate_data(
        self, data: Dict[str, Any], schema: Dict[str, List[Union[ValidationRule, str, Dict[str, Any]]]]
    ) -> Dict[str, ValidationResult]:
        """Validate data against schema."""
        results = {}

        for field_name, rules in schema.items():
            value = data.get(field_name)
            results[field_name] = self.validate_field(value, rules, field_name)

        return results

    def _apply_rule(
        self, value: Any, rule: Union[ValidationRule, str, Dict[str, Any]], field_name: str
    ) -> ValidationResult:
        """Apply single validation rule."""

        # Handle dict rules (with parameters)
        if isinstance(rule, dict):
            rule_name = rule.get("rule")
            params = {k: v for k, v in rule.items() if k != "rule"}
        else:
            rule_name = rule
            params = {}

        # Convert string to enum if possible
        if isinstance(rule_name, str):
            try:
                rule_name = ValidationRule(rule_name)
            except ValueError:
                # Check custom validators
                if rule_name in self.custom_validators:
                    is_valid = self.custom_validators[rule_name](value)
                    return ValidationResult(
                        is_valid=is_valid,
                        errors=[] if is_valid else [f"{field_name} failed custom validation: {rule_name}"],
                    )
                else:
                    return ValidationResult(is_valid=False, errors=[f"Unknown validation rule: {rule_name}"])

        # Apply built-in rules
        return self._apply_builtin_rule(value, rule_name, params, field_name)

    def _apply_builtin_rule(
        self, value: Any, rule: ValidationRule, params: Dict[str, Any], field_name: str
    ) -> ValidationResult:
        """Apply built-in validation rule."""

        if rule == ValidationRule.REQUIRED:
            is_valid = value is not None and value != ""
            return ValidationResult(is_valid=is_valid, errors=[] if is_valid else [f"{field_name} is required"])

        # Skip other validations if value is None/empty and not required
        if value is None or value == "":
            return ValidationResult(is_valid=True, errors=[])

        value_str = str(value)

        if rule == ValidationRule.EMAIL:
            is_valid = bool(re.match(self.patterns[rule], value_str))
            return ValidationResult(
                is_valid=is_valid, errors=[] if is_valid else [f"{field_name} must be a valid email address"]
            )

        elif rule == ValidationRule.URL:
            is_valid = bool(re.match(self.patterns[rule], value_str))
            return ValidationResult(is_valid=is_valid, errors=[] if is_valid else [f"{field_name} must be a valid URL"])

        elif rule == ValidationRule.ALPHANUMERIC:
            is_valid = bool(re.match(self.patterns[rule], value_str))
            return ValidationResult(
                is_valid=is_valid, errors=[] if is_valid else [f"{field_name} must contain only letters and numbers"]
            )

        elif rule == ValidationRule.NUMERIC:
            is_valid = bool(re.match(self.patterns[rule], value_str))
            return ValidationResult(is_valid=is_valid, errors=[] if is_valid else [f"{field_name} must be numeric"])

        elif rule == ValidationRule.MIN_LENGTH:
            min_len = params.get("min", 0)
            is_valid = len(value_str) >= min_len
            return ValidationResult(
                is_valid=is_valid, errors=[] if is_valid else [f"{field_name} must be at least {min_len} characters"]
            )

        elif rule == ValidationRule.MAX_LENGTH:
            max_len = params.get("max", float('inf'))
            is_valid = len(value_str) <= max_len
            sanitized = value_str[:max_len] if not is_valid else value_str
            return ValidationResult(
                is_valid=is_valid,
                errors=[] if is_valid else [f"{field_name} must be no more than {max_len} characters"],
                sanitized_value=sanitized,
                warnings=[] if is_valid else [f"{field_name} was truncated to {max_len} characters"],
            )

        elif rule == ValidationRule.REGEX:
            pattern = params.get("pattern", ".*")
            is_valid = bool(re.match(pattern, value_str))
            return ValidationResult(
                is_valid=is_valid, errors=[] if is_valid else [f"{field_name} does not match required pattern"]
            )

        elif rule == ValidationRule.IN_LIST:
            allowed_values = params.get("values", [])
            is_valid = value in allowed_values
            return ValidationResult(
                is_valid=is_valid, errors=[] if is_valid else [f"{field_name} must be one of: {allowed_values}"]
            )

        elif rule == ValidationRule.NOT_IN_LIST:
            forbidden_values = params.get("values", [])
            is_valid = value not in forbidden_values
            return ValidationResult(
                is_valid=is_valid, errors=[] if is_valid else [f"{field_name} cannot be one of: {forbidden_values}"]
            )

        elif rule == ValidationRule.JSON_VALID:
            try:
                json.loads(value_str)
                is_valid = True
            except (json.JSONDecodeError, TypeError):
                is_valid = False

            return ValidationResult(is_valid=is_valid, errors=[] if is_valid else [f"{field_name} must be valid JSON"])

        elif rule == ValidationRule.NO_SQL_INJECTION:
            has_sql_injection = SecurityValidator.check_sql_injection(value_str)
            return ValidationResult(
                is_valid=not has_sql_injection,
                errors=[] if not has_sql_injection else [f"{field_name} contains potentially malicious SQL patterns"],
            )

        elif rule == ValidationRule.NO_XSS:
            has_xss = SecurityValidator.check_xss(value_str)
            sanitized = SecurityValidator.sanitize_html(value_str)
            return ValidationResult(
                is_valid=not has_xss,
                errors=[] if not has_xss else [f"{field_name} contains potentially malicious script content"],
                sanitized_value=sanitized,
                warnings=[] if not has_xss else [f"{field_name} was sanitized to remove script content"],
            )

        elif rule == ValidationRule.NO_SCRIPT_TAGS:
            has_script = bool(re.search(r'<script\b', value_str, re.IGNORECASE))
            return ValidationResult(
                is_valid=not has_script, errors=[] if not has_script else [f"{field_name} cannot contain script tags"]
            )

        elif rule == ValidationRule.SAFE_FILENAME:
            original_filename = value_str
            sanitized_filename = SecurityValidator.sanitize_filename(original_filename)
            is_safe = original_filename == sanitized_filename

            return ValidationResult(
                is_valid=True,  # We can always sanitize
                errors=[],
                sanitized_value=sanitized_filename,
                warnings=[] if is_safe else [f"{field_name} was sanitized for safe file storage"],
            )

        elif rule == ValidationRule.SAFE_PATH:
            has_traversal = SecurityValidator.check_path_traversal(value_str)
            return ValidationResult(
                is_valid=not has_traversal,
                errors=[]
                if not has_traversal
                else [f"{field_name} contains potentially malicious path traversal patterns"],
            )

        # Default case
        return ValidationResult(is_valid=True, errors=[])


# Global validator instance
input_validator = InputValidator()


def validate_input(schema: Dict[str, List[Union[ValidationRule, str, Dict[str, Any]]]]):
    """Decorator to validate input data."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract data to validate (could be from request body, query params, etc.)
            # This would need to be integrated with the specific framework being used

            # For now, assume data is in kwargs
            data = kwargs.get("data", {})

            # Validate data
            validation_results = input_validator.validate_data(data, schema)

            # Check for validation errors
            errors = []
            sanitized_data = {}

            for field, result in validation_results.items():
                if not result.is_valid:
                    errors.extend([f"{field}: {error}" for error in result.errors])

                # Use sanitized value if available
                if result.sanitized_value is not None:
                    sanitized_data[field] = result.sanitized_value
                else:
                    sanitized_data[field] = data.get(field)

            if errors:
                raise ValidationError(f"Validation failed: {'; '.join(errors)}", details={"validation_errors": errors})

            # Replace original data with sanitized data
            kwargs["data"] = sanitized_data

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def sanitize_input(func):
    """Decorator to automatically sanitize common inputs."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Sanitize string inputs
        for key, value in kwargs.items():
            if isinstance(value, str):
                # Basic HTML sanitization
                kwargs[key] = SecurityValidator.sanitize_html(value)

        return await func(*args, **kwargs)

    return wrapper
