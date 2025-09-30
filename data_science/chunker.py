import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

from app.services.logger_service import get_logger_service
from app.config import get_settings

logger_service = get_logger_service()
settings = get_settings()


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    source_document_id: Optional[str] = None

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())


class DocumentChunker:
    """
    Service for chunking documents into smaller pieces suitable for embedding and retrieval.
    Uses LangChain's RecursiveCharacterTextSplitter for intelligent text splitting.
    """

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 separators: Optional[List[str]] = None):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Default separators optimized for various content types
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            " ",     # Word breaks
            ".",     # Sentence breaks
            ",",     # Clause breaks
            "\t",    # Tab characters
            ""       # Character-level fallback
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators,
            keep_separator=True
        )

        logger_service.get_logger("chunker").info(
            f"DocumentChunker initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    async def chunk_documents(self,
                            documents: List[Dict[str, Any]],
                            chunk_size: Optional[int] = None,
                            chunk_overlap: Optional[int] = None) -> List[DocumentChunk]:
        """
        Chunk multiple documents into smaller pieces.

        Args:
            documents: List of documents with 'content', 'title', and optional 'metadata'
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap

        Returns:
            List of DocumentChunk objects
        """
        try:
            # Update splitter if different parameters provided
            if chunk_size or chunk_overlap:
                self._update_splitter(chunk_size, chunk_overlap)

            all_chunks = []

            for doc_idx, document in enumerate(documents):
                doc_id = document.get('id', str(uuid.uuid4()))
                title = document.get('title', f'Document_{doc_idx}')
                content = document.get('content', '')
                doc_metadata = document.get('metadata', {})

                if not content.strip():
                    logger_service.get_logger("chunker").warning(
                        f"Empty content for document: {title}"
                    )
                    continue

                # Chunk this document
                doc_chunks = await self.chunk_text(
                    text=content,
                    metadata={
                        'title': title,
                        'source_document_id': doc_id,
                        'document_index': doc_idx,
                        **doc_metadata
                    }
                )

                all_chunks.extend(doc_chunks)

            logger_service.get_logger("chunker").info(
                f"Successfully chunked {len(documents)} documents into {len(all_chunks)} chunks"
            )

            return all_chunks

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "chunk_documents",
                "document_count": len(documents)
            })
            raise

    async def chunk_text(self,
                        text: str,
                        metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Chunk a single text into smaller pieces.

        Args:
            text: The text content to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of DocumentChunk objects
        """
        try:
            if not text.strip():
                return []

            metadata = metadata or {}

            # Run text splitting in thread pool to avoid blocking
            chunks = await asyncio.get_event_loop().run_in_executor(
                None,
                self.text_splitter.split_text,
                text
            )

            # Create DocumentChunk objects with preserved metadata
            document_chunks = []
            for idx, chunk_content in enumerate(chunks):
                if chunk_content.strip():  # Skip empty chunks
                    chunk = DocumentChunk(
                        id=str(uuid.uuid4()),
                        content=chunk_content.strip(),
                        metadata={
                            **metadata,
                            'chunk_index': idx,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk_content),
                        },
                        chunk_index=idx,
                        source_document_id=metadata.get('source_document_id')
                    )
                    document_chunks.append(chunk)

            logger_service.get_logger("chunker").debug(
                f"Chunked text into {len(document_chunks)} pieces",
                text_length=len(text),
                chunk_count=len(document_chunks)
            )

            return document_chunks

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "chunk_text",
                "text_length": len(text) if text else 0
            })
            raise

    def _update_splitter(self, chunk_size: Optional[int], chunk_overlap: Optional[int]):
        """Update the text splitter with new parameters."""
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators,
            keep_separator=True
        )

    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunked documents.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_content_length": 0
            }

        chunk_sizes = [len(chunk.content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_content_length": sum(chunk_sizes),
            "unique_documents": len(set(chunk.source_document_id for chunk in chunks if chunk.source_document_id))
        }

    async def health_check(self) -> bool:
        """
        Check if the chunker service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test with a simple text
            test_text = "This is a test text for health check."
            test_chunks = await self.chunk_text(test_text)
            return len(test_chunks) > 0
        except Exception as e:
            logger_service.log_error(e, {"operation": "chunker_health_check"})
            return False


# Global instance
chunker_service = DocumentChunker()


def get_chunker_service() -> DocumentChunker:
    """Get the global chunker service instance."""
    return chunker_service