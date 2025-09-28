from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from loguru import logger
from typing import Any, Dict

from app.config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI application startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v1.0.0")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Host: {settings.host}:{settings.port}")
    logger.info(f"Log level: {settings.log_level}")

    # Start background tasks
    from app.services.metrics_service import get_metrics_service
    metrics_service = get_metrics_service()
    metrics_service.start_background_tasks()
    logger.info("Background tasks started")

    yield  # Run the application
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        description="Discord RAG Bot Backend API",
        version="1.0.0",
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        debug=settings.debug,
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log all HTTP requests and responses."""
        start_time = time.time()

        # Log request
        logger.info(
            f"[{request.state.request_id if hasattr(request.state, 'request_id') else 'unknown'}] "
            f"Started {request.method} {request.url.path} | "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log response
        logger.info(
            f"[{request.state.request_id if hasattr(request.state, 'request_id') else 'unknown'}] "
            f"Completed {request.method} {request.url.path} | "
            f"status={response.status_code} | "
            f"duration={duration_ms:.2f}ms"
        )

        return response

    # Add request ID middleware
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Add unique request ID to each request."""
        import uuid
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, 'request_id', 'unknown')

        logger.error(
            f"[{request_id}] Unhandled exception in {request.method} {request.url.path}: {str(exc)}",
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
        )

    # Root endpoint
    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint with basic service information."""
        return {
            "service": settings.app_name,
            "version": "1.0.0",
            "environment": settings.environment,
            "status": "running"
        }

    # Include API routes
    from app.api.routes import health
    app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])

    # Future API routes
    # from app.api.routes import rag, feedback, ingest, metrics
    # app.include_router(rag.router, prefix=settings.api_prefix, tags=["rag"])
    # app.include_router(feedback.router, prefix=settings.api_prefix, tags=["feedback"])
    # app.include_router(ingest.router, prefix=settings.api_prefix, tags=["ingest"])
    # app.include_router(metrics.router, prefix=settings.api_prefix, tags=["metrics"])

    return app


# Create the FastAPI app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logger.remove()
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="30 days",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=settings.log_level,
        format="{time:HH:mm:ss} | {level} | {message}"
    )

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )