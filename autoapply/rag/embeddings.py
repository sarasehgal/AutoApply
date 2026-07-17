"""glue between our LLMClient.embed() and chroma's EmbeddingFunction interface"""

from __future__ import annotations

from chromadb import Documents, EmbeddingFunction, Embeddings

from autoapply.llm.provider import LLMClient


class AutoApplyEmbeddingFunction(EmbeddingFunction):
    """chroma calls this like a fn, just forwards to LLMClient.embed"""

    def __init__(self, client: LLMClient | None = None):
        self._client = client or LLMClient()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chroma wants this exact param name
        return self._client.embed(list(input))


def embed_texts(texts: list[str]) -> list[list[float]]:
    return LLMClient().embed(texts)
