from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import ValidationException
from pydantic import ValidationError
from datetime import datetime
import traceback

from app.models.responses import ErrorResponse, ValidationErrorResponse, ValidationErrorDetail
from app.services.logger_service import get_logger_service


logger_service = get_logger_service()


class RAGBotException(Exception):
    """Base exception for RAG Bot application."""

    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR",
                 status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class DatabaseException(RAGBotException):
    """Exception for database-related errors."""

    def __init__(self, message: str, operation: str = "unknown",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details={
                "operation": operation,
                **(details or {})
            }
        )


class LLMException(RAGBotException):
    """Exception for LLM API errors."""

    def __init__(self, message: str, model: str = "unknown",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            status_code=502,
            details={
                "model": model,
                **(details or {})
            }
        )


class VectorSearchException(RAGBotException):
    """Exception for vector search errors."""

    def __init__(self, message: str, query: str = "",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_SEARCH_ERROR",
            status_code=500,
            details={
                "query_length": len(query),
                **(details or {})
            }
        )


class RateLimitException(RAGBotException):
    """Exception for rate limiting."""

    def __init__(self, message: str = "Rate limit exceeded",
                 retry_after: int = 60,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={
                "retry_after": retry_after,
                **(details or {})
            }
        )


class AuthenticationException(RAGBotException):
    """Exception for authentication errors."""

    def __init__(self, message: str = "Authentication failed",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details
        )


class AuthorizationException(RAGBotException):
    """Exception for authorization errors."""

    def __init__(self, message: str = "Access denied",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details
        )


class ValidationException(RAGBotException):
    """Exception for input validation errors."""

    def __init__(self, message: str, field: str = "unknown",
                 value: Any = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details={
                "field": field,
                "value": str(value) if value is not None else None,
                **(details or {})
            }
        )


class ResourceNotFoundException(RAGBotException):
    """Exception for resource not found errors."""

    def __init__(self, message: str, resource_type: str = "resource",
                 resource_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=404,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                **(details or {})
            }
        )


class ExternalServiceException(RAGBotException):
    """Exception for external service errors."""

    def __init__(self, message: str, service: str = "unknown",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={
                "service": service,
                **(details or {})
            }
        )


async def ragbot_exception_handler(request: Request, exc: RAGBotException) -> JSONResponse:
    """Handle custom RAG Bot exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')

    # Log the exception
    logger_service.set_request_context(
        request_id=request_id,
        endpoint=str(request.url.path),
        method=request.method
    )

    logger_service.log_error(exc, {
        "error_code": exc.error_code,
        "status_code": exc.status_code,
        "endpoint": str(request.url.path),
        "method": request.method
    })

    # Create error response
    error_response = ErrorResponse(
        error=exc.error_code,
        message=exc.message,
        details=exc.details,
        request_id=request_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers={"X-Request-ID": request_id}
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')

    # Log the exception
    logger_service.set_request_context(
        request_id=request_id,
        endpoint=str(request.url.path),
        method=request.method
    )

    logger_service.get_logger("http").warning(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        status_code=exc.status_code,
        detail=exc.detail,
        endpoint=str(request.url.path)
    )

    # Create error response
    error_response = ErrorResponse(
        error="HTTP_ERROR",
        message=exc.detail,
        details={"status_code": exc.status_code},
        request_id=request_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict(),
        headers={"X-Request-ID": request_id}
    )


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic validation exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')

    # Log the validation error
    logger_service.set_request_context(
        request_id=request_id,
        endpoint=str(request.url.path),
        method=request.method
    )

    logger_service.get_logger("validation").warning(
        "Request validation failed",
        error_count=len(exc.errors()),
        endpoint=str(request.url.path)
    )

    # Format validation errors
    validation_errors: List[ValidationErrorDetail] = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        validation_errors.append(
            ValidationErrorDetail(
                field=field,
                message=error["msg"],
                value=error.get("input")
            )
        )

    # Create validation error response
    error_response = ValidationErrorResponse(
        details=validation_errors
    )

    return JSONResponse(
        status_code=422,
        content=error_response.dict(),
        headers={"X-Request-ID": request_id}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')

    # Log the unexpected exception
    logger_service.set_request_context(
        request_id=request_id,
        endpoint=str(request.url.path),
        method=request.method
    )

    logger_service.log_error(exc, {
        "error_type": type(exc).__name__,
        "endpoint": str(request.url.path),
        "method": request.method,
        "traceback": traceback.format_exc()
    })

    # Create generic error response (don't expose internal details)
    error_response = ErrorResponse(
        error="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred. Please try again later.",
        details={"error_type": type(exc).__name__} if logger_service.get_logger().level.name == "DEBUG" else {},
        request_id=request_id
    )

    return JSONResponse(
        status_code=500,
        content=error_response.dict(),
        headers={"X-Request-ID": request_id}
    )


def setup_exception_handlers(app):
    """Setup all exception handlers for the FastAPI app."""

    # Custom RAG Bot exceptions
    app.add_exception_handler(RAGBotException, ragbot_exception_handler)

    # FastAPI HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Pydantic validation exceptions
    app.add_exception_handler(ValidationError, validation_exception_handler)

    # General exceptions (catch-all)
    app.add_exception_handler(Exception, general_exception_handler)


# Utility functions for raising specific exceptions
def raise_database_error(message: str, operation: str = "unknown", **details):
    """Raise a database exception."""
    raise DatabaseException(message, operation, details)


def raise_llm_error(message: str, model: str = "unknown", **details):
    """Raise an LLM exception."""
    raise LLMException(message, model, details)


def raise_validation_error(message: str, field: str = "unknown", value: Any = None, **details):
    """Raise a validation exception."""
    raise ValidationException(message, field, value, details)


def raise_not_found_error(message: str, resource_type: str = "resource",
                         resource_id: str = "", **details):
    """Raise a resource not found exception."""
    raise ResourceNotFoundException(message, resource_type, resource_id, details)


def raise_rate_limit_error(message: str = "Rate limit exceeded", retry_after: int = 60, **details):
    """Raise a rate limit exception."""
    raise RateLimitException(message, retry_after, details)


def raise_auth_error(message: str = "Authentication failed", **details):
    """Raise an authentication exception."""
    raise AuthenticationException(message, details)


def raise_external_service_error(message: str, service: str = "unknown", **details):
    """Raise an external service exception."""
    raise ExternalServiceException(message, service, details)