import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, StatusEnum
from app.services.rag_service import get_rag_service, RAGService
from app.services.logger_service import get_logger_service
from app.services.metrics_service import get_metrics_service
from app.utils.exceptions import (
    RAGBotException,
    raise_validation_error,
    raise_external_service_error,
    raise_rate_limit_error
)

router = APIRouter()
logger_service = get_logger_service()
metrics_service = get_metrics_service()


@router.post("/rag-query", response_model=QueryResponse, tags=["rag"])
async def rag_query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    http_request: Request = None
) -> QueryResponse:
    """
    Process a user query through the RAG pipeline.

    This endpoint:
    1. Validates the user query
    2. Generates embeddings for the query
    3. Searches for relevant document chunks
    4. Generates a contextual response using LLM
    5. Returns the answer with source attribution

    Args:
        request: Query request containing user question and parameters
        rag_service: Injected RAG service instance
        http_request: FastAPI request object for metadata

    Returns:
        QueryResponse: Answer with sources and processing metadata

    Raises:
        HTTPException: For various error conditions (400, 429, 500, 502)
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown') if http_request else 'unknown'

    try:
        # Set logging context
        logger_service.set_request_context(
            request_id=request_id,
            user_id=request.user_id,
            endpoint="/api/rag-query"
        )

        # Validate request
        if not request.query.strip():
            raise_validation_error(
                message="Query cannot be empty",
                field="query",
                value=request.query
            )

        if len(request.query) > 2000:
            raise_validation_error(
                message="Query too long (max 2000 characters)",
                field="query",
                value=len(request.query)
            )

        # Log query processing start
        logger_service.get_logger("rag_endpoint").info(
            f"Processing RAG query for user {request.user_id}: "
            f"'{request.query[:100]}{'...' if len(request.query) > 100 else ''}'"
        )

        # Check for rate limiting (simple implementation)
        # In production, you'd use Redis or similar for distributed rate limiting
        current_time = time.time()
        user_requests = getattr(rag_query, '_user_requests', {})
        user_last_request = user_requests.get(request.user_id, 0)

        if current_time - user_last_request < 1.0:  # 1 second between requests
            raise_rate_limit_error(
                message="Please wait before making another request",
                retry_after=1
            )

        # Update rate limiting tracker
        if not hasattr(rag_query, '_user_requests'):
            rag_query._user_requests = {}
        rag_query._user_requests[request.user_id] = current_time

        # Process the query through RAG pipeline
        response = await rag_service.process_query(request)

        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Record metrics
        metrics_service.record_request(
            method="POST",
            endpoint="/api/rag-query",
            status_code=200,
            duration_ms=processing_time_ms,
            user_id=request.user_id
        )

        # Log successful processing
        logger_service.log_rag_query(
            query=request.query,
            chunks_retrieved=len(response.sources),
            processing_time_ms=processing_time_ms,
            user_id=request.user_id,
            query_id=response.metadata.query_id if response.metadata else "unknown",
            tokens_used=response.metadata.llm_tokens_used if response.metadata else 0
        )

        logger_service.get_logger("rag_endpoint").info(
            f"RAG query completed successfully for user {request.user_id} "
            f"(duration: {processing_time_ms:.2f}ms, chunks: {len(response.sources)})"
        )

        return response

    except RAGBotException:
        # Re-raise custom exceptions (they'll be handled by exception handlers)
        raise

    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000

        # Record error metrics
        metrics_service.record_request(
            method="POST",
            endpoint="/api/rag-query",
            status_code=500,
            duration_ms=processing_time_ms,
            user_id=request.user_id,
            error_type=type(e).__name__
        )

        # Log the error
        logger_service.log_error(e, {
            "operation": "rag_query_endpoint",
            "user_id": request.user_id,
            "query_length": len(request.query) if request.query else 0,
            "processing_time_ms": processing_time_ms
        })

        # Determine appropriate error response
        if "embedding" in str(e).lower():
            raise_external_service_error(
                message="Embedding service temporarily unavailable",
                service="embedding_service"
            )
        elif "vector" in str(e).lower() or "mongodb" in str(e).lower():
            raise_external_service_error(
                message="Search service temporarily unavailable",
                service="vector_database"
            )
        elif "openai" in str(e).lower() or "llm" in str(e).lower():
            raise_external_service_error(
                message="AI service temporarily unavailable",
                service="llm_service"
            )
        else:
            # Generic internal server error
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred while processing your query",
                    "request_id": request_id
                }
            )

    finally:
        logger_service.clear_request_context()


@router.get("/rag-query/health", tags=["rag"])
async def rag_health_check(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Check the health of all RAG pipeline components.

    Returns:
        Dictionary with health status of each component
    """
    try:
        start_time = time.time()

        # Get health status from RAG service
        health_status = await rag_service.health_check()

        duration_ms = (time.time() - start_time) * 1000

        # Log health check
        logger_service.log_performance(
            operation="rag_health_check",
            duration_ms=duration_ms,
            components_checked=len(health_status)
        )

        # Determine overall status
        overall_healthy = health_status.get("overall", False)
        status_code = 200 if overall_healthy else 503

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": time.time(),
            "components": health_status,
            "response_time_ms": duration_ms
        }

    except Exception as e:
        logger_service.log_error(e, {"operation": "rag_health_check_endpoint"})

        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": "Health check failed",
                "response_time_ms": (time.time() - start_time) * 1000
            }
        )


@router.get("/rag-query/stats", tags=["rag"])
async def rag_stats(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get statistics and information about the RAG service.

    Returns:
        Service statistics and component information
    """
    try:
        start_time = time.time()

        # Get service statistics
        stats = await rag_service.get_service_stats()

        duration_ms = (time.time() - start_time) * 1000

        # Add endpoint metadata
        stats["endpoint_info"] = {
            "response_time_ms": duration_ms,
            "timestamp": time.time(),
            "available_endpoints": [
                "/api/rag-query",
                "/api/rag-query/health",
                "/api/rag-query/stats"
            ]
        }

        logger_service.log_performance(
            operation="rag_stats",
            duration_ms=duration_ms
        )

        return stats

    except Exception as e:
        logger_service.log_error(e, {"operation": "rag_stats_endpoint"})

        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATS_RETRIEVAL_ERROR",
                "message": "Failed to retrieve RAG service statistics"
            }
        )


# Additional utility endpoints for testing and debugging

@router.post("/rag-query/test", tags=["rag", "testing"])
async def test_rag_components(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Test individual RAG components with sample data.
    Useful for debugging and development.

    Returns:
        Test results for each component
    """
    try:
        test_results = {}
        test_text = "This is a test query for the AI Engineering Bootcamp system."

        # Test chunker
        try:
            chunks = await rag_service.chunker.chunk_text(test_text)
            test_results["chunker"] = {
                "status": "healthy",
                "chunks_created": len(chunks),
                "test_passed": len(chunks) > 0
            }
        except Exception as e:
            test_results["chunker"] = {
                "status": "unhealthy",
                "error": str(e),
                "test_passed": False
            }

        # Test embedder
        try:
            embedding = await rag_service.embedder.embed_text(test_text)
            test_results["embedder"] = {
                "status": "healthy",
                "embedding_dimensions": len(embedding),
                "test_passed": len(embedding) > 0
            }
        except Exception as e:
            test_results["embedder"] = {
                "status": "unhealthy",
                "error": str(e),
                "test_passed": False
            }

        # Test vector database (simple connection test)
        try:
            db_healthy = await rag_service.vector_db.health_check()
            test_results["vector_db"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "connection_test": db_healthy,
                "test_passed": db_healthy
            }
        except Exception as e:
            test_results["vector_db"] = {
                "status": "unhealthy",
                "error": str(e),
                "test_passed": False
            }

        # Test LLM client
        try:
            llm_healthy = await rag_service.llm_client.health_check()
            test_results["llm_client"] = {
                "status": "healthy" if llm_healthy else "unhealthy",
                "connection_test": llm_healthy,
                "test_passed": llm_healthy
            }
        except Exception as e:
            test_results["llm_client"] = {
                "status": "unhealthy",
                "error": str(e),
                "test_passed": False
            }

        # Overall test result
        all_passed = all(result.get("test_passed", False) for result in test_results.values())
        test_results["overall"] = {
            "all_tests_passed": all_passed,
            "summary": f"{sum(1 for r in test_results.values() if r.get('test_passed', False))}/{len(test_results)} components healthy"
        }

        return test_results

    except Exception as e:
        logger_service.log_error(e, {"operation": "test_rag_components"})

        raise HTTPException(
            status_code=500,
            detail={
                "error": "COMPONENT_TEST_ERROR",
                "message": "Failed to test RAG components"
            }
        )