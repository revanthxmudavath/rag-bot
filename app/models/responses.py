from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import datetime as dt
from enum import Enum


class StatusEnum(str, Enum):
    """Status enumeration for API responses."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PROCESSING = "processing"


class SourceChunk(BaseModel):
    """Source chunk information for RAG responses."""

    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    similarity_score: float = Field(..., description="Similarity score")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")
    document_title: Optional[str] = Field(None, description="Source document title")
    source_url: Optional[str] = Field(None, description="Source document URL")


class QueryMetadata(BaseModel):
    """Metadata for query processing."""

    query_id: str = Field(..., description="Unique query identifier")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    llm_tokens_used: Optional[int] = Field(None, description="Number of LLM tokens used")
    embedding_time_ms: Optional[float] = Field(None, description="Embedding generation time")
    retrieval_time_ms: Optional[float] = Field(None, description="Vector search time")
    generation_time_ms: Optional[float] = Field(None, description="LLM generation time")
    timestamp: datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), description="Query timestamp")


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Response status")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceChunk] = Field(default_factory=list, description="Source chunks used")
    metadata: QueryMetadata = Field(..., description="Query processing metadata")
    confidence_score: Optional[float] = Field(None, description="Answer confidence score")
    message: Optional[str] = Field(None, description="Additional response message")


class FeedbackResponse(BaseModel):
    """Response model for feedback collection endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Response status")
    feedback_id: str = Field(..., description="Unique feedback identifier")
    timestamp: datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), description="Feedback timestamp")
    message: str = Field(default="Feedback recorded successfully", description="Response message")


class HealthStatus(BaseModel):
    """Health status for individual service components."""

    name: str = Field(..., description="Service component name")
    status: str = Field(..., description="Service status (healthy/unhealthy/degraded)")
    last_check: datetime = Field(..., description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: StatusEnum = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(default="1.0.0", description="Service version")
    services: Optional[List[HealthStatus]] = Field(None, description="Individual service health")
    message: str = Field(default="Service is healthy", description="Health status message")


class IngestMetadata(BaseModel):
    """Metadata for document ingestion."""

    total_documents: int = Field(..., description="Total documents processed")
    total_chunks: int = Field(..., description="Total chunks created")
    processing_time_ms: float = Field(..., description="Total processing time")
    successful_documents: int = Field(..., description="Successfully processed documents")
    failed_documents: int = Field(default=0, description="Failed document count")
    errors: List[str] = Field(default_factory=list, description="Processing errors")


class IngestResponse(BaseModel):
    """Response model for document ingestion endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Ingestion status")
    metadata: IngestMetadata = Field(..., description="Ingestion metadata")
    message: str = Field(default="Documents ingested successfully", description="Response message")
    timestamp: datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), description="Ingestion timestamp")


class MetricData(BaseModel):
    """Individual metric data point."""

    name: str = Field(..., description="Metric name")
    value: Union[int, float, str] = Field(..., description="Metric value")
    unit: Optional[str] = Field(None, description="Metric unit")
    timestamp: datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), description="Metric timestamp")
    tags: Optional[Dict[str, str]] = Field(None, description="Metric tags")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Response status")
    metrics: List[MetricData] = Field(..., description="List of metrics")
    time_range: Dict[str, datetime] = Field(..., description="Metrics time range")
    summary: Optional[Dict[str, Any]] = Field(None, description="Metrics summary")
    message: str = Field(default="Metrics retrieved successfully", description="Response message")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    status: StatusEnum = Field(default=StatusEnum.ERROR, description="Error status")
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class BulkQueryResult(BaseModel):
    """Individual result in bulk query response."""

    query: str = Field(..., description="Original query")
    answer: Optional[str] = Field(None, description="Generated answer")
    sources: List[SourceChunk] = Field(default_factory=list, description="Source chunks")
    metadata: Optional[QueryMetadata] = Field(None, description="Query metadata")
    error: Optional[str] = Field(None, description="Error message if query failed")


class BulkQueryResponse(BaseModel):
    """Response model for bulk query processing."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Overall status")
    results: List[BulkQueryResult] = Field(..., description="Query results")
    total_queries: int = Field(..., description="Total number of queries")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(default=0, description="Number of failed queries")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    message: str = Field(default="Bulk queries processed", description="Response message")


class ServiceInfo(BaseModel):
    """Basic service information."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment")
    status: str = Field(..., description="Service status")


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information."""

    field: str = Field(..., description="Field name that failed validation")
    message: str = Field(..., description="Validation error message")
    value: Optional[Any] = Field(None, description="Invalid value provided")


class ValidationErrorResponse(BaseModel):
    """Response model for validation errors."""

    status: StatusEnum = Field(default=StatusEnum.ERROR, description="Error status")
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(default="Request validation failed", description="Error message")
    details: List[ValidationErrorDetail] = Field(..., description="Validation error details")
    timestamp: datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), description="Error timestamp")