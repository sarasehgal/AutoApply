"""disk cache for llm responses, keyed by a hash of the call params so repeat calls don't cost $$"""

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
    """wrapper so nobody has to import diskcache directly elsewhere"""

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
