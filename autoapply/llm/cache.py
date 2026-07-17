"""Hash-keyed, on-disk response cache.

Every LLM call is cached by a hash of (provider, system prompt, user
prompt, temperature, expected schema). Identical calls — e.g. re-running
the same posting twice while iterating on the UI — hit disk instead of
paying for another API call.
"""

from __future__ import annotations

import hashlib

import diskcache

from autoapply.config import settings


def hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class ResponseCache:
    """Thin wrapper around diskcache so callers never touch the library directly."""

    def __init__(self, directory: str | None = None):
        self._cache = diskcache.Cache(directory or settings.cache_dir)

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def set(self, key: str, value: str, expire: float | None = None) -> None:
        self._cache.set(key, value, expire=expire)

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()
