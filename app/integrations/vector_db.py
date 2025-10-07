import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING
from pymongo.errors import DuplicateKeyError, OperationFailure
import datetime as dt

from app.services.logger_service import get_logger_service
from app.config import get_settings
from data_science.chunker import DocumentChunk

logger_service = get_logger_service()
settings = get_settings()


class VectorDatabase:
    """
    MongoDB Atlas Vector Search integration for storing and retrieving document embeddings.
    Supports similarity search and document management operations.
    """

    def __init__(self, mongodb_uri: Optional[str] = None, database_name: Optional[str] = None):
        """
        Initialize the vector database connection.

        Args:
            mongodb_uri: MongoDB connection string
            database_name: Name of the database to use
        """
        self.mongodb_uri = mongodb_uri or settings.mongodb_uri
        self.database_name = database_name or settings.database_name
        self.collection_name = settings.collection_name
        self.vector_index_name = settings.vector_index_name

        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.collection: Optional[AsyncIOMotorCollection] = None
        self._connected = False

    async def connect(self):
        """Establish connection to MongoDB Atlas."""
        try:
            if self._connected:
                return

            if not self.mongodb_uri:
                raise RuntimeError("MongoDB URI is not configured. Set MONGODB_URI to enable vector storage.")

            logger_service.get_logger("vector_db").info("Connecting to MongoDB Atlas")
            start_time = time.time()

            self.client = AsyncIOMotorClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                maxPoolSize=10,
                minPoolSize=1
            )

            # Get database and collection references
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]

            # Test the connection using the target database (not admin)
            # This avoids authentication errors when user doesn't have admin access
            await self.database.command('ping')

            connection_time = time.time() - start_time
            self._connected = True

            logger_service.get_logger("vector_db").info(
                f"Successfully connected to MongoDB Atlas "
                f"(database: {self.database_name}, collection: {self.collection_name}, "
                f"connection_time: {connection_time:.2f}s)"
            )

            # Ensure indexes exist
            await self._ensure_indexes()

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "mongodb_connect",
                "database": self.database_name,
                "collection": self.collection_name
            })
            raise

    async def disconnect(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger_service.get_logger("vector_db").info("Disconnected from MongoDB Atlas")

    async def _ensure_indexes(self):
        """Create necessary indexes for the collection."""
        try:
            indexes = [
                IndexModel([("document_id", ASCENDING)]),
                IndexModel([("chunk_index", ASCENDING)]),
                IndexModel([("metadata.title", ASCENDING)]),
                IndexModel([("created_at", ASCENDING)]),
                # Vector search index is created separately in Atlas UI or via Atlas CLI
            ]

            await self.collection.create_indexes(indexes)
            logger_service.get_logger("vector_db").info("Ensured database indexes exist")

        except Exception as e:
            logger_service.log_error(e, {"operation": "create_indexes"})
            # Don't raise - indexes might already exist

    async def store_documents(self, documents_with_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store documents with their embeddings in the vector database.

        Args:
            documents_with_embeddings: List of documents with embeddings

        Returns:
            Dictionary with storage results
        """
        try:
            if not self._connected:
                await self.connect()

            start_time = time.time()
            inserted_count = 0
            updated_count = 0
            errors = []

            for doc in documents_with_embeddings:
                try:
                    # Prepare document for storage
                    doc_data = {
                        "document_id": doc["document_id"],
                        "chunk_id": doc["chunk_id"],
                        "content": doc["content"],
                        "embedding": doc["embedding"],
                        "metadata": doc.get("metadata", {}),
                        "chunk_index": doc.get("chunk_index", 0),
                        "created_at": dt.datetime.now(dt.timezone.utc),
                        "updated_at": dt.datetime.now(dt.timezone.utc)
                    }

                    # Use upsert to handle duplicates
                    result = await self.collection.replace_one(
                        {"chunk_id": doc["chunk_id"]},
                        doc_data,
                        upsert=True
                    )

                    if result.upserted_id:
                        inserted_count += 1
                    else:
                        updated_count += 1

                except Exception as e:
                    error_msg = f"Failed to store document {doc.get('chunk_id', 'unknown')}: {str(e)}"
                    errors.append(error_msg)
                    logger_service.get_logger("vector_db").error(error_msg)

            duration_ms = (time.time() - start_time) * 1000

            result = {
                "inserted_count": inserted_count,
                "updated_count": updated_count,
                "total_processed": len(documents_with_embeddings),
                "errors": errors,
                "duration_ms": duration_ms
            }

            logger_service.log_performance(
                operation="store_documents",
                duration_ms=duration_ms,
                documents_processed=len(documents_with_embeddings),
                inserted_count=inserted_count,
                updated_count=updated_count,
                error_count=len(errors)
            )

            return result

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "store_documents",
                "document_count": len(documents_with_embeddings)
            })
            raise

    async def search_similar(self,
                           query_embedding: List[float],
                           top_k: int = 5,
                           filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_criteria: Optional filters to apply

        Returns:
            List of similar documents with similarity scores
        """
        try:
            if not self._connected:
                await self.connect()

            start_time = time.time()

            # MongoDB Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": min(top_k * 10, 1000),  # More candidates for better recall
                        "limit": top_k
                    }
                }
            ]

            # Add filters if provided
            if filter_criteria:
                pipeline.append({"$match": filter_criteria})

            # Add score and format output
            pipeline.extend([
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "document_id": 1,
                        "chunk_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "chunk_index": 1,
                        "similarity_score": 1,
                        "created_at": 1
                    }
                }
            ])

            # Execute the search
            cursor = self.collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)

            duration_ms = (time.time() - start_time) * 1000

            logger_service.log_performance(
                operation="vector_search",
                duration_ms=duration_ms,
                query_vector_dimensions=len(query_embedding),
                top_k=top_k,
                results_found=len(results)
            )

            return results

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "search_similar",
                "top_k": top_k,
                "has_filters": filter_criteria is not None
            })
            # Return empty results on error
            return []

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its ID.

        Args:
            document_id: The document ID to retrieve

        Returns:
            Document data or None if not found
        """
        try:
            if not self._connected:
                await self.connect()

            result = await self.collection.find_one({"document_id": document_id})
            return result

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "get_document_by_id",
                "document_id": document_id
            })
            return None

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks of a document.

        Args:
            document_id: The document ID to delete

        Returns:
            Deletion results
        """
        try:
            if not self._connected:
                await self.connect()

            result = await self.collection.delete_many({"document_id": document_id})

            logger_service.get_logger("vector_db").info(
                f"Deleted {result.deleted_count} chunks for document {document_id}"
            )

            return {
                "deleted_count": result.deleted_count,
                "document_id": document_id
            }

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "delete_document",
                "document_id": document_id
            })
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.

        Returns:
            Collection statistics
        """
        try:
            if not self._connected:
                await self.connect()

            # Get basic collection stats
            stats = await self.database.command("collStats", self.collection_name)

            # Get document count by type
            pipeline = [
                {
                    "$group": {
                        "_id": "$metadata.title",
                        "chunk_count": {"$sum": 1},
                        "avg_content_length": {"$avg": {"$strLenCP": "$content"}}
                    }
                },
                {"$sort": {"chunk_count": -1}}
            ]

            doc_stats = await self.collection.aggregate(pipeline).to_list(length=None)

            return {
                "total_documents": stats.get("count", 0),
                "storage_size_bytes": stats.get("storageSize", 0),
                "index_sizes": stats.get("indexSizes", {}),
                "document_types": doc_stats
            }

        except Exception as e:
            logger_service.log_error(e, {"operation": "get_collection_stats"})
            return {}

    async def health_check(self) -> bool:
        """
        Check if the vector database is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._connected:
                await self.connect()

            # Simple ping to test connection using target database (not admin)
            await self.database.command('ping')

            # Test basic collection access
            await self.collection.count_documents({}, limit=1)

            return True

        except Exception as e:
            logger_service.log_error(e, {"operation": "vector_db_health_check"})
            return False

    async def create_vector_index(self, index_definition: Optional[Dict[str, Any]] = None):
        """
        Create vector search index. Note: This requires Atlas CLI or Atlas UI in practice.
        This method provides the index definition for reference.

        Args:
            index_definition: Custom index definition
        """
        default_index = {
            "name": self.vector_index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 384,  # all-MiniLM-L6-v2 dimensions
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "metadata.title"
                    },
                    {
                        "type": "filter",
                        "path": "document_id"
                    }
                ]
            }
        }

        index_def = index_definition or default_index

        logger_service.get_logger("vector_db").info(
            f"Vector index definition for '{self.vector_index_name}': {index_def}"
        )

        # Note: Actual index creation must be done via Atlas CLI or UI
        logger_service.get_logger("vector_db").warning(
            "Vector search indexes must be created via MongoDB Atlas UI or CLI. "
            "Use the definition logged above."
        )

        return index_def


# Global instance
vector_db_service = VectorDatabase()


def get_vector_db_service() -> VectorDatabase:
    """Get the global vector database service instance."""
    return vector_db_service