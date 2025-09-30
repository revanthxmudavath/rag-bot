import asyncio
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import time
from functools import lru_cache

from app.services.logger_service import get_logger_service
from app.config import get_settings

logger_service = get_logger_service()
settings = get_settings()


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.
    Optimized for batch processing and includes caching for performance.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.model_dimensions = None
        self._model_lock = None

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            start_time = time.time()
            logger_service.get_logger("embedder").info(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)
            self.model_dimensions = self.model.get_sentence_embedding_dimension()

            load_time = time.time() - start_time
            logger_service.get_logger("embedder").info(
                f"Successfully loaded model {self.model_name} "
                f"(dimensions: {self.model_dimensions}, load_time: {load_time:.2f}s)"
            )

        except Exception as e:
            logger_service.log_error(e, {"operation": "load_embedding_model", "model_name": self.model_name})
            raise

    async def _ensure_model(self):
        """Load the embedding model lazily when first used."""
        if self.model is not None:
            return

        if self._model_lock is None:
            self._model_lock = asyncio.Lock()

        async with self._model_lock:
            if self.model is None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._load_model)

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding vector
        """
        try:
            await self._ensure_model()

            if not text.strip():
                # Return zero vector for empty text
                return [0.0] * self.model_dimensions

            start_time = time.time()

            # Run embedding generation in thread pool to avoid blocking
            embedding = await asyncio.get_running_loop().run_in_executor(
                None,
                self._generate_embedding,
                text
            )

            duration_ms = (time.time() - start_time) * 1000

            logger_service.log_performance(
                operation="embed_single_text",
                duration_ms=duration_ms,
                text_length=len(text)
            )

            return embedding.tolist()

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "embed_text",
                "text_length": len(text) if text else 0
            })
            raise

    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors (one per input text)
        """
        try:
            if not texts:
                return []

            await self._ensure_model()

            start_time = time.time()
            all_embeddings = []

            # Filter out empty texts and keep track of indices
            valid_texts = []
            text_indices = []
            for i, text in enumerate(texts):
                if text.strip():
                    valid_texts.append(text)
                    text_indices.append(i)

            if not valid_texts:
                # Return zero vectors for all empty texts
                return [[0.0] * self.model_dimensions for _ in texts]

            # Process in batches
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]

                # Run batch embedding generation in thread pool
                batch_embeddings = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._generate_batch_embeddings,
                    batch_texts
                )

                all_embeddings.extend(batch_embeddings)

            # Create final result with zero vectors for empty texts
            result = [[0.0] * self.model_dimensions for _ in texts]
            for i, embedding in enumerate(all_embeddings):
                original_index = text_indices[i]
                result[original_index] = embedding.tolist()

            duration_ms = (time.time() - start_time) * 1000

            logger_service.log_performance(
                operation="embed_batch",
                duration_ms=duration_ms,
                batch_size=len(texts),
                valid_texts=len(valid_texts)
            )

            return result

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "embed_batch",
                "text_count": len(texts) if texts else 0
            })
            raise

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text (synchronous)."""
        return self.model.encode(text, normalize_embeddings=True)

    def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts (synchronous)."""
        return self.model.encode(texts, normalize_embeddings=True)

    @staticmethod
    def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger_service.log_error(e, {"operation": "calculate_similarity"})
            return 0.0

    @staticmethod
    def calculate_batch_similarities(query_embedding: List[float],
                                   candidate_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate similarities between a query embedding and multiple candidate embeddings.

        Args:
            query_embedding: The query embedding vector
            candidate_embeddings: List of candidate embedding vectors

        Returns:
            List of similarity scores
        """
        try:
            if not candidate_embeddings:
                return []

            query_vec = np.array(query_embedding)
            candidate_matrix = np.array(candidate_embeddings)

            # Calculate cosine similarities using vectorized operations
            dot_products = np.dot(candidate_matrix, query_vec)
            query_norm = np.linalg.norm(query_vec)
            candidate_norms = np.linalg.norm(candidate_matrix, axis=1)

            # Avoid division by zero
            valid_mask = (candidate_norms != 0) & (query_norm != 0)
            similarities = np.zeros(len(candidate_embeddings))

            if query_norm != 0:
                similarities[valid_mask] = dot_products[valid_mask] / (
                    candidate_norms[valid_mask] * query_norm
                )

            return similarities.tolist()

        except Exception as e:
            logger_service.log_error(e, {"operation": "calculate_batch_similarities"})
            return [0.0] * len(candidate_embeddings)

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.model_dimensions,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "model_loaded": self.model is not None
        }

    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test with a simple text
            test_text = "This is a health check test."
            embedding = await self.embed_text(test_text)
            return len(embedding) == self.model_dimensions and all(isinstance(x, float) for x in embedding)
        except Exception as e:
            logger_service.log_error(e, {"operation": "embedder_health_check"})
            return False

    async def warmup(self, sample_texts: Optional[List[str]] = None):
        """
        Warm up the model with sample texts to reduce cold start latency.

        Args:
            sample_texts: Optional list of sample texts for warmup
        """
        try:
            if sample_texts is None:
                sample_texts = [
                    "This is a sample text for model warmup.",
                    "Another example text to ensure the model is ready.",
                    "A third text to complete the warmup process."
                ]

            logger_service.get_logger("embedder").info("Starting model warmup")
            start_time = time.time()

            # Generate embeddings for warmup texts
            await self.embed_batch(sample_texts)

            warmup_time = time.time() - start_time
            logger_service.get_logger("embedder").info(
                f"Model warmup completed in {warmup_time:.2f}s"
            )

        except Exception as e:
            logger_service.log_error(e, {"operation": "model_warmup"})


# Global instance
embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    return embedding_service


# Utility function for backwards compatibility
async def embed_text(text: str) -> List[float]:
    """Convenience function to embed a single text."""
    return await embedding_service.embed_text(text)


async def embed_batch(texts: List[str]) -> List[List[float]]:
    """Convenience function to embed multiple texts."""
    return await embedding_service.embed_batch(texts)