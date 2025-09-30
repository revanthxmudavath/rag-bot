import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
import datetime as dt

from app.models.requests import QueryRequest, IngestRequest
from app.models.responses import QueryResponse, IngestResponse, QueryMetadata, IngestMetadata, SourceChunk, StatusEnum
from app.services.logger_service import get_logger_service
from app.services.metrics_service import get_metrics_service
from data_science.chunker import get_chunker_service, DocumentChunker
from data_science.embedder import get_embedding_service, EmbeddingService
from app.integrations.vector_db import get_vector_db_service, VectorDatabase
from app.integrations.llm_client import get_llm_client_service, LLMClient

logger_service = get_logger_service()
metrics_service = get_metrics_service()


class RAGService:
    """
    Service that orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.
    Coordinates chunking, embedding, vector search, and LLM generation.
    """

    def __init__(self,
                 chunker: Optional[DocumentChunker] = None,
                 embedder: Optional[EmbeddingService] = None,
                 vector_db: Optional[VectorDatabase] = None,
                 llm_client: Optional[LLMClient] = None):
        """
        Initialize the RAG service with component dependencies.

        Args:
            chunker: Document chunking service
            embedder: Embedding generation service
            vector_db: Vector database service
            llm_client: LLM client service
        """
        self.chunker = chunker or get_chunker_service()
        self.embedder = embedder or get_embedding_service()
        self.vector_db = vector_db or get_vector_db_service()
        self.llm_client = llm_client or get_llm_client_service()

        logger_service.get_logger("rag_service").info("RAG service initialized")

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a user query through the complete RAG pipeline.

        Args:
            request: Query request with user question and parameters

        Returns:
            Query response with answer, sources, and metadata
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())

        try:
            logger_service.set_request_context(
                request_id=query_id,
                user_id=request.user_id,
                endpoint="rag_query"
            )

            logger_service.get_logger("rag_service").info(
                f"Processing query: '{request.query[:100]}...' for user {request.user_id}"
            )

            # Step 1: Generate query embedding
            embedding_start = time.time()
            query_embedding = await self.embedder.embed_text(request.query)
            embedding_time_ms = (time.time() - embedding_start) * 1000

            logger_service.log_performance(
                operation="query_embedding",
                duration_ms=embedding_time_ms,
                query_length=len(request.query)
            )

            # Step 2: Search for similar chunks
            retrieval_start = time.time()
            similar_chunks = await self.vector_db.search_similar(
                query_embedding=query_embedding,
                top_k=request.max_chunks,
                filter_criteria=None  
            )
            retrieval_time_ms = (time.time() - retrieval_start) * 1000

            logger_service.log_performance(
                operation="vector_search",
                duration_ms=retrieval_time_ms,
                chunks_retrieved=len(similar_chunks),
                top_k=request.max_chunks
            )

            # Step 3: Generate LLM response with context
            generation_start = time.time()
            llm_result = await self.llm_client.generate_with_chunks(
                query=request.query,
                chunks=similar_chunks,
                temperature=request.temperature
            )
            generation_time_ms = (time.time() - generation_start) * 1000

            logger_service.log_performance(
                operation="llm_generation",
                duration_ms=generation_time_ms,
                tokens_used=llm_result.get("metadata", {}).get("total_tokens", 0)
            )

            # Step 4: Format response
            total_time_ms = (time.time() - start_time) * 1000

            # Convert chunks to SourceChunk objects
            source_chunks = []
            if request.include_sources:
                for chunk in similar_chunks:
                    source_chunk = SourceChunk(
                        content=chunk.get("content", ""),
                        metadata=chunk.get("metadata", {}),
                        similarity_score=chunk.get("similarity_score", 0.0),
                        chunk_id=chunk.get("chunk_id"),
                        document_title=chunk.get("metadata", {}).get("title"),
                        source_url=chunk.get("metadata", {}).get("source_url")
                    )
                    source_chunks.append(source_chunk)

            # Create query metadata
            query_metadata = QueryMetadata(
                query_id=query_id,
                processing_time_ms=total_time_ms,
                chunks_retrieved=len(similar_chunks),
                llm_tokens_used=llm_result.get("metadata", {}).get("total_tokens"),
                embedding_time_ms=embedding_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                timestamp=dt.datetime.now(dt.timezone.utc)
            )

            # Log comprehensive query processing
            logger_service.log_rag_query(
                query=request.query,
                chunks_retrieved=len(similar_chunks),
                processing_time_ms=total_time_ms,
                user_id=request.user_id,
                query_id=query_id,
                tokens_used=llm_result.get("metadata", {}).get("total_tokens", 0)
            )

            return QueryResponse(
                status=StatusEnum.SUCCESS,
                answer=llm_result.get("answer", ""),
                sources=source_chunks,
                metadata=query_metadata,
                confidence_score=self._calculate_confidence_score(similar_chunks, llm_result),
                message="Query processed successfully"
            )

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000

            logger_service.log_error(e, {
                "operation": "rag_process_query",
                "query_id": query_id,
                "user_id": request.user_id,
                "query_length": len(request.query),
                "processing_time_ms": total_time_ms
            })

            # Return error response
            return QueryResponse(
                status=StatusEnum.ERROR,
                answer="I apologize, but I encountered an error while processing your question. Please try again later.",
                sources=[],
                metadata=QueryMetadata(
                    query_id=query_id,
                    processing_time_ms=total_time_ms,
                    chunks_retrieved=0,
                    timestamp=dt.datetime.now(dt.timezone.utc)
                ),
                message=f"Error processing query: {str(e)}"
            )

        finally:
            logger_service.clear_request_context()

    async def ingest_documents(self, request: IngestRequest) -> IngestResponse:
        """
        Ingest documents into the vector database.

        Args:
            request: Ingest request with documents and parameters

        Returns:
            Ingest response with processing results
        """
        start_time = time.time()
        ingestion_id = str(uuid.uuid4())

        try:
            logger_service.set_request_context(
                request_id=ingestion_id,
                endpoint="ingest_documents"
            )

            logger_service.get_logger("rag_service").info(
                f"Starting document ingestion: {len(request.documents)} documents"
            )

            # Step 1: Chunk documents
            chunking_start = time.time()
            documents_for_chunking = []
            for doc in request.documents:
                documents_for_chunking.append({
                    "title": doc.title,
                    "content": doc.content,
                    "metadata": {
                        **doc.metadata,
                        "source_url": doc.source_url,
                        "ingestion_id": ingestion_id,
                        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat()
                    }
                })

            chunks = await self.chunker.chunk_documents(
                documents=documents_for_chunking,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap
            )
            chunking_time_ms = (time.time() - chunking_start) * 1000

            logger_service.log_performance(
                operation="document_chunking",
                duration_ms=chunking_time_ms,
                documents_processed=len(request.documents),
                chunks_created=len(chunks)
            )

            # Step 2: Generate embeddings
            embedding_start = time.time()
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedder.embed_batch(chunk_texts)
            embedding_time_ms = (time.time() - embedding_start) * 1000

            logger_service.log_performance(
                operation="batch_embedding",
                duration_ms=embedding_time_ms,
                texts_processed=len(chunk_texts)
            )

            # Step 3: Prepare documents for vector storage
            documents_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                doc_data = {
                    "document_id": chunk.source_document_id or str(uuid.uuid4()),
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "embedding": embedding,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index
                }
                documents_with_embeddings.append(doc_data)

            # Step 4: Store in vector database
            storage_start = time.time()
            storage_result = await self.vector_db.store_documents(documents_with_embeddings)
            storage_time_ms = (time.time() - storage_start) * 1000

            logger_service.log_performance(
                operation="vector_storage",
                duration_ms=storage_time_ms,
                documents_stored=storage_result.get("inserted_count", 0) + storage_result.get("updated_count", 0)
            )

            # Calculate totals
            total_time_ms = (time.time() - start_time) * 1000
            successful_documents = len(request.documents) - len(storage_result.get("errors", []))

            # Create response metadata
            ingest_metadata = IngestMetadata(
                total_documents=len(request.documents),
                total_chunks=len(chunks),
                processing_time_ms=total_time_ms,
                successful_documents=successful_documents,
                failed_documents=len(storage_result.get("errors", [])),
                errors=storage_result.get("errors", [])
            )

            logger_service.get_logger("rag_service").info(
                f"Document ingestion completed: {successful_documents}/{len(request.documents)} documents, "
                f"{len(chunks)} chunks, {total_time_ms:.2f}ms"
            )

            return IngestResponse(
                status=StatusEnum.SUCCESS if successful_documents == len(request.documents) else StatusEnum.WARNING,
                metadata=ingest_metadata,
                message=f"Successfully ingested {successful_documents}/{len(request.documents)} documents",
                timestamp=dt.datetime.now(dt.timezone.utc)
            )

        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000

            logger_service.log_error(e, {
                "operation": "rag_ingest_documents",
                "ingestion_id": ingestion_id,
                "document_count": len(request.documents),
                "processing_time_ms": total_time_ms
            })

            return IngestResponse(
                status=StatusEnum.ERROR,
                metadata=IngestMetadata(
                    total_documents=len(request.documents),
                    total_chunks=0,
                    processing_time_ms=total_time_ms,
                    successful_documents=0,
                    failed_documents=len(request.documents),
                    errors=[f"Ingestion failed: {str(e)}"]
                ),
                message=f"Document ingestion failed: {str(e)}",
                timestamp=dt.datetime.now(dt.timezone.utc)
            )

        finally:
            logger_service.clear_request_context()

    def _calculate_confidence_score(self, chunks: List[Dict[str, Any]], llm_result: Dict[str, Any]) -> Optional[float]:
        """
        Calculate a confidence score for the response based on chunk similarities and other factors.

        Args:
            chunks: Retrieved chunks with similarity scores
            llm_result: LLM generation result

        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not chunks:
                return 0.0

            # Base confidence on average similarity score of top chunks
            similarity_scores = [chunk.get("similarity_score", 0) for chunk in chunks]
            avg_similarity = sum(similarity_scores) / len(similarity_scores)

            # Factor in the number of chunks (more context = higher confidence, up to a point)
            chunk_factor = min(len(chunks) / 3, 1.0)  # Normalize to max of 1.0

            # Factor in response length (very short responses might be less confident)
            response_length = len(llm_result.get("answer", ""))
            length_factor = min(response_length / 100, 1.0)  # Normalize to max of 1.0

            # Combine factors
            confidence = (avg_similarity * 0.6) + (chunk_factor * 0.2) + (length_factor * 0.2)

            return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1

        except Exception as e:
            logger_service.log_error(e, {"operation": "calculate_confidence_score"})
            return None

    async def health_check(self) -> Dict[str, bool]:
        """
        Comprehensive health check for all RAG components.

        Returns:
            Dictionary with health status of each component
        """
        health_results = {}

        try:
            # Check each component concurrently
            health_checks = await asyncio.gather(
                self.chunker.health_check(),
                self.embedder.health_check(),
                self.vector_db.health_check(),
                self.llm_client.health_check(),
                return_exceptions=True
            )

            component_names = ["chunker", "embedder", "vector_db", "llm_client"]

            for name, result in zip(component_names, health_checks):
                if isinstance(result, Exception):
                    health_results[name] = False
                    logger_service.log_error(result, {"operation": f"{name}_health_check"})
                else:
                    health_results[name] = bool(result)

            # Overall health
            health_results["overall"] = all(health_results.values())

            logger_service.get_logger("rag_service").info(
                f"RAG service health check completed: {health_results}"
            )

            return health_results

        except Exception as e:
            logger_service.log_error(e, {"operation": "rag_service_health_check"})
            return {
                "chunker": False,
                "embedder": False,
                "vector_db": False,
                "llm_client": False,
                "overall": False
            }

    async def get_service_stats(self) -> Dict[str, Any]:
        """
        Get statistics and information about the RAG service.

        Returns:
            Service statistics
        """
        try:
            return {
                "chunker": self.chunker.get_chunk_stats([]),
                "embedder": self.embedder.get_model_info(),
                "vector_db": await self.vector_db.get_collection_stats(),
                "llm_client": self.llm_client.get_usage_stats(),
                "service_info": {
                    "name": "RAG Service",
                    "version": "1.0.0",
                    "components": ["chunker", "embedder", "vector_db", "llm_client"]
                }
            }
        except Exception as e:
            logger_service.log_error(e, {"operation": "get_rag_service_stats"})
            return {}


# Global instance
rag_service = RAGService()


def get_rag_service() -> RAGService:
    """Get the global RAG service instance."""
    return rag_service