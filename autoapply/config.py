"""all the env var / config stuff lives here, everything else just imports `settings`"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("PROVIDER", "openai").lower())
    fallback_provider: str | None = field(
        default_factory=lambda: (os.getenv("FALLBACK_PROVIDER", "anthropic") or None)
    )

    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or None)

    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    )
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./chroma_store"))
    cache_dir: str = field(default_factory=lambda: os.getenv("CACHE_DIR", "./cache_store"))

    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    )
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    batch_concurrency: int = field(default_factory=lambda: int(os.getenv("BATCH_CONCURRENCY", "5")))

    cache_enabled: bool = field(default_factory=lambda: _get_bool("CACHE_ENABLED", True))

    def model_for(self, provider: str) -> str:
        if provider == "openai":
            return self.openai_model
        if provider == "anthropic":
            return self.anthropic_model
        if provider == "gemini":
            return self.gemini_model
        raise ValueError(f"Unknown provider: {provider}")

    def api_key_for(self, provider: str) -> str | None:
        if provider == "openai":
            return self.openai_api_key
        if provider == "anthropic":
            return self.anthropic_api_key
        if provider == "gemini":
            return self.gemini_api_key
        raise ValueError(f"Unknown provider: {provider}")


settings = Settings()
