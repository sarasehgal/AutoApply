"""Bridges AutoApply's provider-agnostic LLMClient to ChromaDB's
EmbeddingFunction interface, so Chroma can embed documents/queries
using whichever embedding provider is configured (OpenAI or a local
sentence-transformers model) without knowing anything about either.
"""

from __future__ import annotations

from chromadb import Documents, EmbeddingFunction, Embeddings

from autoapply.llm.provider import LLMClient


class AutoApplyEmbeddingFunction(EmbeddingFunction):
    """Chroma calls this like a function; it delegates to LLMClient.embed."""

    def __init__(self, client: LLMClient | None = None):
        self._client = client or LLMClient()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's required signature
        return self._client.embed(list(input))


def embed_texts(texts: list[str]) -> list[list[float]]:
    return LLMClient().embed(texts)
