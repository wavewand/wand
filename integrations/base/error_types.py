"""
Enhanced error response structures for integration failures
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(Enum):
    """Simplified error category"""

    INTEGRATION_ERROR = "integration_error"


class ErrorSeverity(Enum):
    """Error severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class EnhancedError:
    """Structured error with comprehensive information"""

    message: str
    code: str
    category: ErrorCategory
    severity: ErrorSeverity = ErrorSeverity.ERROR
    retry_suggested: bool = False
    retry_after_seconds: Optional[int] = None
    troubleshooting: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response"""
        return {
            "message": self.message,
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity.value,
            "retry_suggested": self.retry_suggested,
            "retry_after_seconds": self.retry_after_seconds,
            "troubleshooting": self.troubleshooting,
            "details": self.details or {},
        }


def create_enhanced_error_response(
    integration_name: str,
    operation: str,
    error: EnhancedError,
    response_time_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create standardized enhanced error response"""
    return {
        "success": False,
        "error": error.to_dict(),
        "integration": integration_name,
        "operation": operation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "response_time_ms": response_time_ms,
    }


# Common error creation function


def create_integration_error(
    message: str,
    code: str = "INTEGRATION_ERROR",
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    retry_suggested: bool = False,
    retry_after_seconds: Optional[int] = None,
    troubleshooting: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    native_error: Optional[Any] = None,
) -> EnhancedError:
    """Create integration error with all parameters including native error preservation"""
    enhanced_details = details or {}

    # Preserve native error information
    if native_error is not None:
        enhanced_details["native_error"] = {
            "type": type(native_error).__name__,
            "message": str(native_error),
            "module": getattr(type(native_error), '__module__', 'unknown'),
        }

        # Preserve exception attributes
        if hasattr(native_error, '__dict__'):
            native_attrs = {}
            for key, value in native_error.__dict__.items():
                try:
                    # Only include serializable attributes
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        native_attrs[key] = value
                    else:
                        native_attrs[key] = str(value)
                except BaseException:
                    native_attrs[key] = f"<non-serializable {type(value).__name__}>"

            if native_attrs:
                enhanced_details["native_error"]["attributes"] = native_attrs

        # For HTTP exceptions, preserve response details
        if hasattr(native_error, 'response'):
            response = native_error.response
            response_info = {}
            if hasattr(response, 'status_code'):
                response_info["status_code"] = response.status_code
            if hasattr(response, 'headers'):
                try:
                    response_info["headers"] = dict(response.headers)
                except BaseException:
                    response_info["headers"] = str(response.headers)
            if hasattr(response, 'text'):
                try:
                    response_info["body"] = response.text
                except BaseException:
                    response_info["body"] = "<unable to read response body>"
            if hasattr(response, 'url'):
                response_info["url"] = str(response.url)

            if response_info:
                enhanced_details["native_error"]["response"] = response_info

    return EnhancedError(
        message=message,
        code=code,
        category=ErrorCategory.INTEGRATION_ERROR,
        severity=severity,
        retry_suggested=retry_suggested,
        retry_after_seconds=retry_after_seconds,
        troubleshooting=troubleshooting,
        details=enhanced_details,
    )
