import time
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.requests import IngestRequest, DocumentRequest
from app.models.responses import IngestResponse, StatusEnum
from app.services.rag_service import get_rag_service, RAGService
from app.services.logger_service import get_logger_service
from app.services.metrics_service import get_metrics_service
from app.utils.exceptions import (
    RAGBotException,
    raise_validation_error,
    raise_external_service_error
)

router = APIRouter()
logger_service = get_logger_service()
metrics_service = get_metrics_service()


@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service),
    http_request: Request = None
) -> IngestResponse:
    """
    Ingest documents into the knowledge base.

    This endpoint:
    1. Validates the input documents
    2. Chunks documents into smaller pieces
    3. Generates embeddings for each chunk
    4. Stores documents and embeddings in vector database
    5. Returns ingestion results

    Args:
        request: Ingest request containing documents and parameters
        background_tasks: FastAPI background tasks for cleanup
        rag_service: Injected RAG service instance
        http_request: FastAPI request object for metadata

    Returns:
        IngestResponse: Processing results and metadata

    Raises:
        HTTPException: For various error conditions (400, 413, 500, 502)
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown') if http_request else 'unknown'

    try:
        # Set logging context
        logger_service.set_request_context(
            request_id=request_id,
            endpoint="/api/ingest"
        )

        # Validate request
        if not request.documents:
            raise_validation_error(
                message="No documents provided for ingestion",
                field="documents",
                value=len(request.documents)
            )

        if len(request.documents) > 100:
            raise_validation_error(
                message="Too many documents (max 100 per request)",
                field="documents",
                value=len(request.documents)
            )

        # Validate individual documents
        total_content_size = 0
        for i, doc in enumerate(request.documents):
            if not doc.title.strip():
                raise_validation_error(
                    message=f"Document {i+1} has empty title",
                    field=f"documents[{i}].title",
                    value=doc.title
                )

            if not doc.content.strip():
                raise_validation_error(
                    message=f"Document {i+1} has empty content",
                    field=f"documents[{i}].content",
                    value="<empty>"
                )

            total_content_size += len(doc.content)

        # Check total payload size (10MB limit)
        if total_content_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "PAYLOAD_TOO_LARGE",
                    "message": "Total document content exceeds 10MB limit",
                    "total_size_bytes": total_content_size
                }
            )

        # Validate chunk parameters
        if request.chunk_overlap >= request.chunk_size:
            raise_validation_error(
                message="Chunk overlap must be less than chunk size",
                field="chunk_overlap",
                value=request.chunk_overlap
            )

        # Log ingestion start
        logger_service.get_logger("ingest_endpoint").info(
            f"Starting document ingestion: {len(request.documents)} documents, "
            f"total size: {total_content_size:,} bytes"
        )

        # Process documents through RAG pipeline
        response = await rag_service.ingest_documents(request)

        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Record metrics
        status_code = 200 if response.status == StatusEnum.SUCCESS else 207  # Multi-status for partial success
        metrics_service.record_request(
            method="POST",
            endpoint="/api/ingest",
            status_code=status_code,
            duration_ms=processing_time_ms
        )

        # Log successful processing
        successful_docs = response.metadata.successful_documents if response.metadata else 0
        total_docs = response.metadata.total_documents if response.metadata else len(request.documents)
        total_chunks = response.metadata.total_chunks if response.metadata else 0

        logger_service.get_logger("ingest_endpoint").info(
            f"Document ingestion completed: {successful_docs}/{total_docs} documents, "
            f"{total_chunks} chunks created (duration: {processing_time_ms:.2f}ms)"
        )

        # Add background task for cleanup if needed
        if response.metadata and response.metadata.errors:
            background_tasks.add_task(
                _log_ingestion_errors,
                request_id,
                response.metadata.errors
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
            endpoint="/api/ingest",
            status_code=500,
            duration_ms=processing_time_ms,
            error_type=type(e).__name__
        )

        # Log the error
        logger_service.log_error(e, {
            "operation": "ingest_documents_endpoint",
            "document_count": len(request.documents) if request.documents else 0,
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
                message="Database service temporarily unavailable",
                service="vector_database"
            )
        else:
            # Generic internal server error
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred during document ingestion",
                    "request_id": request_id
                }
            )

    finally:
        logger_service.clear_request_context()


@router.post("/ingest/validate", tags=["ingest"])
async def validate_documents(
    documents: List[DocumentRequest]
) -> Dict[str, Any]:
    """
    Validate documents before ingestion without actually processing them.

    Args:
        documents: List of documents to validate

    Returns:
        Validation results and recommendations
    """
    try:
        validation_results = {
            "valid": True,
            "document_count": len(documents),
            "total_content_size": 0,
            "estimated_chunks": 0,
            "warnings": [],
            "errors": []
        }

        if not documents:
            validation_results["valid"] = False
            validation_results["errors"].append("No documents provided")
            return validation_results

        if len(documents) > 100:
            validation_results["valid"] = False
            validation_results["errors"].append("Too many documents (max 100 per request)")

        for i, doc in enumerate(documents):
            doc_size = len(doc.content)
            validation_results["total_content_size"] += doc_size

            # Validate title
            if not doc.title.strip():
                validation_results["valid"] = False
                validation_results["errors"].append(f"Document {i+1} has empty title")

            # Validate content
            if not doc.content.strip():
                validation_results["valid"] = False
                validation_results["errors"].append(f"Document {i+1} has empty content")
            elif doc_size < 50:
                validation_results["warnings"].append(f"Document {i+1} has very short content ({doc_size} chars)")
            elif doc_size > 100000:  # 100KB
                validation_results["warnings"].append(f"Document {i+1} is very large ({doc_size:,} chars)")

            # Estimate chunks (rough calculation)
            estimated_doc_chunks = max(1, doc_size // 500)  # Assuming 500 char chunks
            validation_results["estimated_chunks"] += estimated_doc_chunks

        # Check total size
        if validation_results["total_content_size"] > 10 * 1024 * 1024:
            validation_results["valid"] = False
            validation_results["errors"].append("Total content exceeds 10MB limit")
        elif validation_results["total_content_size"] > 5 * 1024 * 1024:
            validation_results["warnings"].append("Large payload size may take longer to process")

        # Add recommendations
        validation_results["recommendations"] = []
        if validation_results["estimated_chunks"] > 1000:
            validation_results["recommendations"].append("Consider splitting into smaller batches for better performance")

        if validation_results["warnings"]:
            validation_results["recommendations"].append("Review warnings for potential issues")

        return validation_results

    except Exception as e:
        logger_service.log_error(e, {"operation": "validate_documents_endpoint"})

        raise HTTPException(
            status_code=500,
            detail={
                "error": "VALIDATION_ERROR",
                "message": "Failed to validate documents"
            }
        )


@router.get("/ingest/status/{ingestion_id}", tags=["ingest"])
async def get_ingestion_status(
    ingestion_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get the status of a previous ingestion operation.
    Note: This is a placeholder - full implementation would require
    storing ingestion status in a database or cache.

    Args:
        ingestion_id: The ingestion ID to check

    Returns:
        Ingestion status information
    """
    # This is a simplified implementation
    # In production, you'd store ingestion status in Redis or database
    return {
        "ingestion_id": ingestion_id,
        "status": "completed",  # placeholder
        "message": "Ingestion status tracking not fully implemented",
        "note": "This endpoint requires implementation of persistent status storage"
    }


@router.delete("/ingest/document/{document_id}", tags=["ingest"])
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Delete a document and all its chunks from the knowledge base.

    Args:
        document_id: The document ID to delete
        rag_service: Injected RAG service instance

    Returns:
        Deletion results
    """
    try:
        if not document_id.strip():
            raise_validation_error(
                message="Document ID cannot be empty",
                field="document_id",
                value=document_id
            )

        logger_service.get_logger("ingest_endpoint").info(
            f"Deleting document: {document_id}"
        )

        # Delete from vector database
        result = await rag_service.vector_db.delete_document(document_id)

        logger_service.get_logger("ingest_endpoint").info(
            f"Document deletion completed: {document_id}, "
            f"chunks deleted: {result.get('deleted_count', 0)}"
        )

        return {
            "status": "success",
            "document_id": document_id,
            "chunks_deleted": result.get("deleted_count", 0),
            "message": f"Successfully deleted document {document_id}"
        }

    except RAGBotException:
        raise

    except Exception as e:
        logger_service.log_error(e, {
            "operation": "delete_document_endpoint",
            "document_id": document_id
        })

        raise HTTPException(
            status_code=500,
            detail={
                "error": "DELETION_ERROR",
                "message": f"Failed to delete document {document_id}"
            }
        )


@router.get("/ingest/stats", tags=["ingest"])
async def get_ingestion_stats(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get statistics about the document collection.

    Returns:
        Collection statistics and information
    """
    try:
        start_time = time.time()

        # Get collection statistics
        stats = await rag_service.vector_db.get_collection_stats()

        duration_ms = (time.time() - start_time) * 1000

        # Add endpoint metadata
        stats["query_time_ms"] = duration_ms
        stats["timestamp"] = time.time()

        logger_service.log_performance(
            operation="get_collection_stats",
            duration_ms=duration_ms
        )

        return stats

    except Exception as e:
        logger_service.log_error(e, {"operation": "get_ingestion_stats_endpoint"})

        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATS_RETRIEVAL_ERROR",
                "message": "Failed to retrieve collection statistics"
            }
        )


# Background task functions

async def _log_ingestion_errors(request_id: str, errors: List[str]):
    """
    Background task to log ingestion errors for analysis.

    Args:
        request_id: The request ID for correlation
        errors: List of error messages
    """
    logger_service.set_request_context(request_id=request_id)

    logger_service.get_logger("ingest_background").warning(
        f"Ingestion completed with {len(errors)} errors",
        error_count=len(errors),
        errors=errors[:5]  # Log first 5 errors to avoid spam
    )

    logger_service.clear_request_context()