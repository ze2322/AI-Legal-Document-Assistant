"""Application settings, loaded once from the environment.

Every tunable value lives here so the rest of the code never reads
``os.environ`` directly. Values come from environment variables (or a local
``.env`` file) and fall back to sensible defaults, which keeps the app runnable
out of the box while still being configurable in Docker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the application configuration."""

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2"
    ollama_host: str | None = None  # None -> the ollama client default

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 3

    # Generation
    llm_temperature: float = 0.3
    llm_max_tokens: int = 512
    llm_timeout_seconds: float = 120.0

    # Storage / limits
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss_index")
    max_upload_mb: int = 25

    # Logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        """Build a ``Settings`` instance from environment variables."""
        index_dir = os.getenv("INDEX_DIR", "").strip()
        return cls(
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            ollama_model=_env_str("OLLAMA_MODEL", cls.ollama_model),
            ollama_host=os.getenv("OLLAMA_HOST", "").strip() or None,
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_env_int("TOP_K", cls.top_k),
            llm_temperature=_env_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", cls.llm_max_tokens),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", cls.llm_timeout_seconds),
            index_dir=Path(index_dir) if index_dir else PROJECT_ROOT / "data" / "faiss_index",
            max_upload_mb=_env_int("MAX_UPLOAD_MB", cls.max_upload_mb),
            log_level=_env_str("LOG_LEVEL", cls.log_level).upper(),
        )


settings = Settings.from_env()
