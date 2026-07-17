"""ChromaDB-backed vector store.

Two collections:

- ``resume_chunks`` — the active resume, chunked bullet-by-bullet.
  The Matcher and Tailoring agents retrieve from this to ground their
  output in the user's real experience.
- ``postings_corpus`` — an optional history of past job postings, so
  the app can semantically search "postings like this one" across
  everything the user has ever analyzed.
"""

from __future__ import annotations

import uuid

import chromadb

from autoapply.config import settings
from autoapply.rag.chunking import chunk_resume, chunk_text
from autoapply.rag.embeddings import AutoApplyEmbeddingFunction

RESUME_COLLECTION = "resume_chunks"
POSTINGS_COLLECTION = "postings_corpus"


class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        self._client = chromadb.PersistentClient(path=persist_dir or settings.chroma_dir)
        self._embedding_fn = AutoApplyEmbeddingFunction()

    def _collection(self, name: str):
        return self._client.get_or_create_collection(name=name, embedding_function=self._embedding_fn)

    # ------------------------------------------------------------- resume
    def add_resume(self, resume_text: str, resume_id: str = "default") -> int:
        """Chunk + embed a resume, replacing any prior chunks for this id."""
        collection = self._collection(RESUME_COLLECTION)

        existing = collection.get(where={"resume_id": resume_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        chunks = chunk_resume(resume_text)
        if not chunks:
            return 0

        ids = [f"{resume_id}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"resume_id": resume_id, "chunk_index": i} for i in range(len(chunks))]
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def semantic_search(self, query: str, n_results: int = 5, resume_id: str = "default") -> list[dict]:
        """Return the resume chunks most relevant to ``query``, most relevant first."""
        collection = self._collection(RESUME_COLLECTION)
        count = collection.count()
        if count == 0:
            return []

        result = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
            where={"resume_id": resume_id},
        )
        return _flatten_query_result(result)

    # ----------------------------------------------------------- postings
    def add_postings(self, postings: list[dict]) -> int:
        """Embed a corpus of past postings for semantic search over history.

        Each posting needs at least ``{"id": ..., "text": ...}``; any
        other keys (title, company, url, ...) are stored as metadata.
        """
        collection = self._collection(POSTINGS_COLLECTION)
        added = 0
        for posting in postings:
            chunks = chunk_text(posting["text"])
            if not chunks:
                continue
            extra_meta = {k: v for k, v in posting.items() if k != "text"}
            ids = [f"{posting['id']}-{i}" for i in range(len(chunks))]
            metadatas = [{**extra_meta, "chunk_index": i} for i in range(len(chunks))]
            collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
            added += len(chunks)
        return added

    def search_postings(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search over every posting previously stored with add_postings()."""
        collection = self._collection(POSTINGS_COLLECTION)
        count = collection.count()
        if count == 0:
            return []
        result = collection.query(query_texts=[query], n_results=min(n_results, count))
        return _flatten_query_result(result)

    # --------------------------------------------------------------- admin
    def reset(self) -> None:
        """Drop both collections. Mainly for tests."""
        for name in (RESUME_COLLECTION, POSTINGS_COLLECTION):
            try:
                self._client.delete_collection(name)
            except Exception:  # noqa: BLE001 - collection may not exist yet
                pass


def _flatten_query_result(result: dict) -> list[dict]:
    docs = result["documents"][0] if result.get("documents") else []
    metas = result["metadatas"][0] if result.get("metadatas") else []
    dists = result["distances"][0] if result.get("distances") else [None] * len(docs)
    ids = result["ids"][0] if result.get("ids") else []
    return [
        {"id": cid, "text": doc, "metadata": meta, "distance": dist}
        for cid, doc, meta, dist in zip(ids, docs, metas, dists)
    ]
