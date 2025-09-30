from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    user_id: str = Field(..., description="Discord user ID")
    channel_id: Optional[str] = Field(None, description="Discord channel ID")
    max_chunks: Optional[int] = Field(
        default=5, ge=1, le=20, description="Maximum chunks to retrieve"
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature"
    )
    include_sources: Optional[bool] = Field(
        default=True, description="Include source information in response"
    )

    @field_validator("query")
    def validate_query(cls, v):
        """Validate query is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        return v.strip()


class FeedbackRequest(BaseModel):
    """Request model for feedback collection endpoint."""

    query_id: str = Field(..., description="ID of the query being rated")
    user_id: str = Field(..., description="Discord user ID")
    feedback_type: str = Field(
        ..., pattern="^(thumbs_up|thumbs_down)$", description="Type of feedback"
    )
    rating: Optional[int] = Field(
        None, ge=1, le=5, description="Rating from 1-5 (optional)"
    )
    comment: Optional[str] = Field(
        None, max_length=1000, description="Optional feedback comment"
    )

    @field_validator("comment")
    def validate_comment(cls, v):
        """Validate comment if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class DocumentRequest(BaseModel):
    """Single document for ingestion."""

    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=1, description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional document metadata"
    )
    source_url: Optional[str] = Field(None, description="Source URL if available")

    @field_validator("title", "content")
    def validate_text_fields(cls, v):
        """Validate text fields are not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Field cannot be empty or just whitespace")
        return v.strip()


class IngestRequest(BaseModel):
    """Request model for document ingestion endpoint."""

    documents: List[DocumentRequest] = Field(
        ..., min_items=1, max_items=100, description="Documents to ingest"
    )
    chunk_size: Optional[int] = Field(
        default=500, ge=100, le=2000, description="Text chunk size"
    )
    chunk_overlap: Optional[int] = Field(
        default=50, ge=0, le=500, description="Overlap between chunks"
    )
    replace_existing: Optional[bool] = Field(
        default=False, description="Replace existing documents with same title"
    )

    @field_validator("chunk_overlap")
    def validate_overlap(cls, v, info):
        """Validate chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 500) if hasattr(info, 'data') and info.data else 500
        if v >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class HealthCheckRequest(BaseModel):
    """Request model for health check (usually empty)."""

    include_detailed: Optional[bool] = Field(
        default=False, description="Include detailed health information"
    )


class MetricsRequest(BaseModel):
    """Request model for metrics endpoint."""

    start_time: Optional[datetime] = Field(
        None, description="Start time for metrics range"
    )
    end_time: Optional[datetime] = Field(
        None, description="End time for metrics range"
    )
    metric_types: Optional[List[str]] = Field(
        default=["requests", "errors", "latency", "uptime"],
        description="Types of metrics to retrieve"
    )

    @field_validator("end_time")
    def validate_time_range(cls, v, info):
        """Validate end time is after start time."""
        start_time = info.data.get("start_time") if hasattr(info, 'data') and info.data else None
        if start_time and v and v <= start_time:
            raise ValueError("End time must be after start time")
        return v


class BulkQueryRequest(BaseModel):
    """Request model for bulk query processing."""

    queries: List[str] = Field(
        ..., min_items=1, max_items=10, description="List of queries to process"
    )
    user_id: str = Field(..., description="Discord user ID")
    channel_id: Optional[str] = Field(None, description="Discord channel ID")
    max_chunks: Optional[int] = Field(
        default=5, ge=1, le=20, description="Maximum chunks to retrieve per query"
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature"
    )

    @field_validator("queries")
    def validate_queries(cls, v):
        """Validate all queries are not empty."""
        validated_queries = []
        for query in v:
            if not query.strip():
                raise ValueError("All queries must be non-empty")
            validated_queries.append(query.strip())
        return validated_queries