from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Telegram
    telegram_bot_token: str

    # PostgreSQL
    postgres_user: str = "arxiv_bot"
    postgres_password: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "arxiv_bot"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "arxiv_chunks"

    # Groq
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # Embedding model
    embedding_repo_id: str = "enacimie/Qwen3-Embedding-0.6B-Q4_K_M-GGUF"
    embedding_filename: str = "qwen3-embedding-0.6b-q4_k_m.gguf"
    embedding_dim: int = 1024
    models_cache_dir: str = "/app/data/models"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_chunks: int = 5
    max_history_tokens: int = 120000

    # Session
    session_ttl_hours: int = 24
    pdf_download_dir: str = "/app/data/pdfs"

    # Langfuse (optional — bot works without these)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
