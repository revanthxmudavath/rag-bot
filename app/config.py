from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Discord Bot Configuration
    discord_bot_token: Optional[str] = Field(default=None, description="Discord bot token")

    # MongoDB Atlas Configuration
    mongodb_uri: Optional[str] = Field(default=None, description="MongoDB connection URI")

    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint URL")
    azure_openai_deployment_name: str = Field(
        default="gpt-4o",
        description="Azure OpenAI deployment name"
    )

    # Application Configuration
    app_name: str = Field(default="Discord RAG Bot", description="Application name")
    environment: str = Field(default="development", description="Environment (development/production)")
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    log_level: str = Field(default="INFO", description="Logging level")
    log_to_stdout: bool = Field(default=False, description="Force logging to stdout/stderr only")

    # RAG Configuration
    chunk_size: int = Field(default=500, description="Text chunk size for document processing")
    chunk_overlap: int = Field(default=50, description="Overlap between text chunks")
    max_chunks_retrieval: int = Field(default=5, description="Maximum chunks to retrieve for RAG")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )

    # Optional Configuration
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins"
    )

    # Database Configuration
    database_name: str = Field(default="rag-backend-db", description="MongoDB database name")
    collection_name: str = Field(default="documents", description="MongoDB collection name")
    vector_index_name: str = Field(default="vector_index", description="Vector search index name")

    # API Configuration
    api_prefix: str = Field(default="/api", description="API route prefix")
    docs_url: Optional[str] = Field(default="/docs", description="FastAPI docs URL")
    redoc_url: Optional[str] = Field(default="/redoc", description="FastAPI ReDoc URL")

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute per user")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")

    # LLM Configuration
    llm_temperature: float = Field(default=0.7, description="LLM temperature for generation")
    llm_max_tokens: int = Field(default=1000, description="Maximum tokens for LLM response")
    llm_timeout: int = Field(default=30, description="LLM request timeout in seconds")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
