import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from datetime import datetime

from app.config import get_settings


settings = get_settings()


class LoggerService:
    """Enhanced logging service with structured logging and correlation."""

    def __init__(self):
        self._setup_logger()
        self._request_context: Dict[str, Any] = {}

    def _setup_logger(self):
        """Configure loguru with structured logging."""
        logger.remove()

        def _add_default_context(record):
            record["extra"].setdefault("request_id", "no_req")
            record["extra"].setdefault("user_id", None)
            record["extra"].setdefault("endpoint", None)

        logger.configure(extra={"request_id": "no_req"}, patcher=_add_default_context)

        log_to_stdout = getattr(settings, "log_to_stdout", False)
        logs_dir = Path("logs")
        if not log_to_stdout:
            logs_dir.mkdir(exist_ok=True)

        def json_formatter(record):
            """Format log records as JSON."""
            log_entry = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "logger": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
                "module": record["module"],
            }

            extra = record["extra"]
            if "request_id" in extra:
                log_entry["request_id"] = extra["request_id"]
            if "user_id" in extra and extra["user_id"] is not None:
                log_entry["user_id"] = extra["user_id"]
            if "endpoint" in extra and extra["endpoint"] is not None:
                log_entry["endpoint"] = extra["endpoint"]

            for key, value in record["extra"].items():
                if key not in log_entry:
                    log_entry[key] = value

            return json.dumps(log_entry, default=str)

        if not log_to_stdout:
            if settings.environment == "production":
                logger.add(
                    logs_dir / "app.json",
                    format=json_formatter,
                    level=settings.log_level,
                    rotation="1 day",
                    retention="30 days",
                    compression="gz",
                    serialize=True
                )
            else:
                logger.add(
                    logs_dir / "app.log",
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {extra[request_id]} | {message}",
                    level=settings.log_level,
                    rotation="1 day",
                    retention="7 days"
                )

        console_target = sys.stdout if log_to_stdout else sys.stderr
        console_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[request_id]} | {message}"
            if log_to_stdout
            else "<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{extra[request_id]}</cyan> | <level>{message}</level>"
        )
        if log_to_stdout or settings.environment == "development":
            logger.add(
                console_target,
                format=console_format,
                level=settings.log_level,
                colorize=not log_to_stdout
            )

        if not log_to_stdout:
            logger.add(
                logs_dir / "errors.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {extra[request_id]} | {message}",
                level="ERROR",
                rotation="1 day",
                retention="30 days"
            )

    def set_request_context(self, request_id: str, user_id: Optional[str] = None,
                           endpoint: Optional[str] = None, **kwargs):
        """Set request context for correlation."""
        self._request_context = {
            "request_id": request_id,
            "user_id": user_id,
            "endpoint": endpoint,
            **kwargs
        }

    def clear_request_context(self):
        """Clear request context."""
        self._request_context = {}

    def get_logger(self, name: Optional[str] = None):
        """Get logger with current context."""
        # Ensure request_id exists even if not set
        context = {"request_id": "no_req", **self._request_context}
        log = logger.bind(**context)
        if name:
            log = log.bind(logger_name=name)
        return log

    def log_request(self, method: str, path: str, status_code: int,
                   duration_ms: float, **kwargs):
        """Log HTTP request with structured data."""
        log_data = {
            "event_type": "http_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            **kwargs
        }

        log = self.get_logger("http")
        if status_code >= 500:
            log.error("HTTP request failed", **log_data)
        elif status_code >= 400:
            log.warning("HTTP request error", **log_data)
        else:
            log.info("HTTP request completed", **log_data)

    def log_rag_query(self, query: str, chunks_retrieved: int,
                     processing_time_ms: float, **kwargs):
        """Log RAG query processing."""
        log_data = {
            "event_type": "rag_query",
            "query_length": len(query),
            "chunks_retrieved": chunks_retrieved,
            "processing_time_ms": round(processing_time_ms, 2),
            **kwargs
        }
        self.get_logger("rag").info("RAG query processed", **log_data)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log error with context and stack trace."""
        log_data = {
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            **(context or {})
        }
        self.get_logger("error").opt(exception=error).error("Application error occurred", **log_data)

    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics."""
        log_data = {
            "event_type": "performance",
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            **kwargs
        }

        log = self.get_logger("performance")
        if duration_ms > 5000:  # 5 seconds
            log.warning("Slow operation detected", **log_data)
        else:
            log.info("Operation completed", **log_data)

    def log_database_operation(self, operation: str, collection: str,
                             duration_ms: float, success: bool, **kwargs):
        """Log database operations."""
        log_data = {
            "event_type": "database",
            "operation": operation,
            "collection": collection,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            **kwargs
        }

        log = self.get_logger("database")
        if not success:
            log.error("Database operation failed", **log_data)
        elif duration_ms > 1000:  # 1 second
            log.warning("Slow database operation", **log_data)
        else:
            log.info("Database operation completed", **log_data)

    def log_llm_request(self, model: str, tokens_used: int, duration_ms: float,
                       success: bool, **kwargs):
        """Log LLM API requests."""
        log_data = {
            "event_type": "llm_request",
            "model": model,
            "tokens_used": tokens_used,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            **kwargs
        }

        log = self.get_logger("llm")
        if not success:
            log.error("LLM request failed", **log_data)
        else:
            log.info("LLM request completed", **log_data)

    def log_security_event(self, event_type: str, severity: str = "INFO", **kwargs):
        """Log security-related events."""
        log_data = {
            "event_type": "security",
            "security_event": event_type,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        log = self.get_logger("security")
        if severity == "CRITICAL":
            log.critical("Security event", **log_data)
        elif severity == "HIGH":
            log.error("Security event", **log_data)
        elif severity == "MEDIUM":
            log.warning("Security event", **log_data)
        else:
            log.info("Security event", **log_data)


# Global logger service instance
logger_service = LoggerService()


def get_logger_service() -> LoggerService:
    """Get the global logger service instance."""
    return logger_service