"""
Core configuration module for AI Memory API.
Loads settings from environment variables using Pydantic Settings.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import secrets


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "AI Memory API"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    API_V1_PREFIX: str = "/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = True

    # CORS
    ALLOWED_ORIGINS: str = "*"

    @validator("ALLOWED_ORIGINS")
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]

    # Database - PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "memory_ai"
    POSTGRES_PASSWORD: str = "memory_ai_password"
    POSTGRES_DB: str = "memory_ai_db"

    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Milvus Vector Database
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: str = "root"
    MILVUS_PASSWORD: str = "Milvus"
    MILVUS_COLLECTION_PREFIX: str = "memory_ai"

    # Neo4j Knowledge Graph
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_password"
    NEO4J_DATABASE: str = "neo4j"

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_TRAJECTORIES: str = "memory_ai.trajectories"
    KAFKA_TOPIC_MEMORIES: str = "memory_ai.memories"
    KAFKA_CONSUMER_GROUP: str = "memory_ai_processors"

    # S3 Object Storage
    S3_ENDPOINT: str = "https://s3.amazonaws.com"
    S3_ACCESS_KEY: str = "test_access_key"
    S3_SECRET_KEY: str = "test_secret_key"
    S3_BUCKET_TRAJECTORIES: str = "memory-ai-trajectories"
    S3_BUCKET_MODELS: str = "memory-ai-models"
    S3_REGION: str = "us-east-1"

    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"

    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    GEMINI_THINKING_BUDGET: str = "dynamic"

    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"

    DEFAULT_LLM_PROVIDER: str = "gemini"

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DIMENSION: int = 1024
    USE_API_EMBEDDINGS: bool = False
    EMBEDDING_API_PROVIDER: str = "openai"

    # Reasoning Engine
    REASONING_TIMEOUT_SECONDS: int = 30
    REASONING_MAX_RETRIES: int = 3
    REASONING_CACHE_TTL: int = 3600
    ENABLE_THOUGHT_SUMMARIES: bool = True
    ENABLE_SYMBOLIC_VALIDATION: bool = True

    # Memory Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    CHUNKING_STRATEGY: str = "recursive"

    DEFAULT_SEARCH_LIMIT: int = 10
    MAX_SEARCH_LIMIT: int = 100
    HYBRID_SEARCH_ALPHA: float = 0.5
    ENABLE_CROSS_ENCODER_RERANKING: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    ENABLE_MEMORY_DECAY: bool = True
    DEFAULT_DECAY_RATE: float = 0.001
    IMPORTANT_MEMORY_DECAY_RATE: float = 0.0001

    # Reinforcement Learning
    ENABLE_RL_LOGGING: bool = True
    RL_TRAINING_SCHEDULE: str = "weekly"
    MIN_TRAJECTORIES_FOR_TRAINING: int = 1000
    RL_BATCH_SIZE: int = 512
    RL_LEARNING_RATE: float = 0.0001
    RL_PPO_EPSILON: float = 0.2
    RL_KL_PENALTY: float = 0.1

    # Authentication
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200  # 30 days
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    API_KEY_PREFIX: str = "mem_sk"

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_FREE_TIER: str = "100/minute"
    RATE_LIMIT_STARTER_TIER: str = "1000/minute"
    RATE_LIMIT_PRO_TIER: str = "10000/minute"

    # Observability
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    ENABLE_TRACING: bool = True
    JAEGER_AGENT_HOST: str = "localhost"
    JAEGER_AGENT_PORT: int = 6831
    SENTRY_DSN: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1

    # Feature Flags
    ENABLE_WEBHOOKS: bool = True
    ENABLE_STREAMING_RESPONSES: bool = True
    ENABLE_BATCH_OPERATIONS: bool = True
    ENABLE_MULTI_TENANCY: bool = True

    # Security
    ENABLE_HTTPS_REDIRECT: bool = False
    ALLOWED_HOSTS: str = "*"
    ENABLE_CSRF_PROTECTION: bool = False

    # Performance
    ENABLE_RESPONSE_CACHING: bool = True
    CACHE_TTL_SECONDS: int = 300
    CONNECTION_POOL_SIZE: int = 20
    MAX_OVERFLOW: int = 10

    # Development
    ENABLE_API_DOCS: bool = True
    ENABLE_PROFILING: bool = False
    MOCK_LLM_RESPONSES: bool = False

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env file


# Global settings instance
settings = Settings()
