import asyncio
import time
from typing import List, Dict, Any, Optional
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

from app.services.logger_service import get_logger_service
from app.config import get_settings

logger_service = get_logger_service()
settings = get_settings()


class LLMClient:
    """
    Azure OpenAI client for text generation with RAG context.
    Includes retry logic, token tracking, and prompt optimization.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 deployment_name: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Deployment name for the model
        """
        self.api_key = api_key or settings.azure_openai_api_key
        self.endpoint = endpoint or settings.azure_openai_endpoint
        self.deployment_name = deployment_name or settings.azure_openai_deployment_name

        self.client: Optional[AsyncAzureOpenAI] = None

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            # Fallback to a general tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.total_tokens_used = 0
        self.total_requests = 0

        if self.api_key and self.endpoint and self.deployment_name:
            logger_service.get_logger("llm_client").info(
                f"LLMClient initialised with deployment: {self.deployment_name}"
            )
        else:
            logger_service.get_logger("llm_client").warning(
                "Azure OpenAI credentials are not fully configured; LLM client will remain inactive until provided."
            )

    def _ensure_client(self) -> AsyncAzureOpenAI:
        """Create the Azure OpenAI client if credentials are available."""
        if self.client is not None:
            return self.client

        missing = [name for name, value in {"api_key": self.api_key, "endpoint": self.endpoint, "deployment_name": self.deployment_name}.items() if not value]
        if missing:
            raise RuntimeError(f"Azure OpenAI configuration missing values: {', '.join(missing)}")

        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version="2024-02-01"
        )

        logger_service.get_logger("llm_client").info(
            f"Azure OpenAI client initialised for deployment: {self.deployment_name}"
        )
        return self.client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_response(self,
                              prompt: str,
                              context: str = "",
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response using the LLM with optional context.

        Args:
            prompt: The user prompt/question
            context: Retrieved context from vector search
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()

            # Create the full prompt with context
            full_prompt = self.create_augmented_prompt(prompt, context)

            # Count input tokens
            input_tokens = len(self.tokenizer.encode(full_prompt))

            # Set max tokens if not provided
            if max_tokens is None:
                context_budget = 4096 - input_tokens - 100  # Leave buffer for safety
                if context_budget <= 0:
                    raise ValueError("Prompt and retrieved context exceed the model's maximum context window.")
                max_tokens = min(settings.llm_max_tokens, context_budget)

            max_tokens = max(1, max_tokens)

            # Prepare the messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for an AI Engineering Bootcamp. "
                              "Provide accurate, helpful answers based on the provided context. "
                              "If the context doesn't contain enough information to answer the question, "
                              "say so clearly and suggest what additional information might be needed."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]

            # Make the API call
            client = self._ensure_client()
            response = await client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Extract response data
            answer = response.choices[0].message.content
            usage = response.usage

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            # Update tracking
            self.total_tokens_used += total_tokens
            self.total_requests += 1

            # Log performance
            logger_service.log_llm_request(
                model=self.deployment_name,
                tokens_used=total_tokens,
                duration_ms=duration_ms,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                temperature=temperature
            )

            return {
                "answer": answer,
                "metadata": {
                    "model": self.deployment_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "duration_ms": duration_ms,
                    "temperature": temperature,
                    "finish_reason": response.choices[0].finish_reason
                }
            }

        except Exception as e:
            logger_service.log_llm_request(
                model=self.deployment_name,
                tokens_used=0,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )
            logger_service.log_error(e, {
                "operation": "llm_generate_response",
                "prompt_length": len(prompt),
                "context_length": len(context),
                "temperature": temperature
            })
            raise

    def create_augmented_prompt(self, query: str, context: str) -> str:
        """
        Create an augmented prompt combining the query with retrieved context.

        Args:
            query: User's question
            context: Retrieved context from vector search

        Returns:
            Formatted prompt with context
        """
        if not context.strip():
            return f"""Question: {query}

I don't have specific context to answer this question. Please provide more details or rephrase your question."""

        return f"""Context Information:
{context}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information to fully answer the question, please say so and explain what additional information would be helpful.

Answer:"""

    def create_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved document chunks.

        Args:
            chunks: List of retrieved chunks from vector search

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("metadata", {}).get("title", "Unknown Document")
            content = chunk.get("content", "")
            score = chunk.get("similarity_score", 0)

            context_parts.append(
                f"Source {i} - {title} (Relevance: {score:.3f}):\n{content}\n"
            )

        return "\n".join(context_parts)

    async def generate_with_chunks(self,
                                 query: str,
                                 chunks: List[Dict[str, Any]],
                                 temperature: float = 0.7,
                                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate response using query and retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response with sources and metadata
        """
        try:
            # Build context from chunks
            context = self.create_context_from_chunks(chunks)

            # Generate response
            result = await self.generate_response(
                prompt=query,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Add source information
            result["sources"] = [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "title": chunk.get("metadata", {}).get("title", "Unknown"),
                    "similarity_score": chunk.get("similarity_score", 0),
                    "content_preview": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", "")
                }
                for chunk in chunks
            ]

            return result

        except Exception as e:
            logger_service.log_error(e, {
                "operation": "generate_with_chunks",
                "query_length": len(query),
                "chunks_count": len(chunks)
            })
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger_service.log_error(e, {"operation": "count_tokens"})
            # Fallback: rough estimation
            return len(text) // 4

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the LLM client.

        Returns:
            Usage statistics
        """
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_request": self.total_tokens_used / max(self.total_requests, 1),
            "deployment_name": self.deployment_name
        }

    async def health_check(self) -> bool:
        """
        Check if the LLM service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test with minimal tokens
            test_result = await self.generate_response(
                prompt="Hello",
                context="",
                temperature=0.1,
                max_tokens=10
            )

            return (
                test_result.get("answer") is not None and
                len(test_result.get("answer", "")) > 0
            )

        except Exception as e:
            logger_service.log_error(e, {"operation": "llm_health_check"})
            return False

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Azure OpenAI and return detailed info.

        Returns:
            Connection test results
        """
        try:
            start_time = time.time()

            # Make a minimal API call
            client = self._ensure_client()
            response = await client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                temperature=0
            )

            duration_ms = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "model": self.deployment_name,
                "response_time_ms": duration_ms,
                "api_version": "2024-02-01",
                "endpoint": self.endpoint.split('.')[0] + "..." if self.endpoint else "unknown"
            }

        except Exception as e:
            logger_service.log_error(e, {"operation": "test_llm_connection"})
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.deployment_name,
                "endpoint": self.endpoint.split('.')[0] + "..." if self.endpoint else "unknown"
            }


# Global instance
llm_client_service = LLMClient()


def get_llm_client_service() -> LLMClient:
    """Get the global LLM client service instance."""
    return llm_client_service