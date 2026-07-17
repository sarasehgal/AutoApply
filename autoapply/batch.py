"""Async batch scoring of multiple postings.

A single posting can run through :func:`autoapply.graph.orchestrator.run_pipeline`
directly. Scoring many postings at once (e.g. "score these 20 postings
against my resume") runs them concurrently via ``asyncio``, capped by
``settings.batch_concurrency`` so we don't blow past provider rate
limits. One posting failing (bad URL, malformed output after retries)
does not abort the rest of the batch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from autoapply.config import settings
from autoapply.graph.orchestrator import AutoApplyState, run_pipeline
from autoapply.llm.provider import LLMClient
from autoapply.rag.store import VectorStore

logger = logging.getLogger("autoapply.batch")


@dataclass
class BatchResult:
    posting_input: str
    state: AutoApplyState | None
    error: str | None
    latency_seconds: float


async def _score_one(
    posting_input: str,
    resume_text: str,
    *,
    resume_id: str,
    store: VectorStore,
    client: LLMClient,
    semaphore: asyncio.Semaphore,
) -> BatchResult:
    async with semaphore:
        start = time.monotonic()
        try:
            state = await run_pipeline(
                posting_input, resume_text, resume_id=resume_id, store=store, client=client
            )
            latency = time.monotonic() - start
            logger.info("batch posting=%r latency=%.2fs status=ok", posting_input[:60], latency)
            return BatchResult(posting_input=posting_input, state=state, error=None, latency_seconds=latency)
        except Exception as exc:  # noqa: BLE001 - one bad posting must not abort the batch
            latency = time.monotonic() - start
            logger.warning(
                "batch posting=%r latency=%.2fs status=error error=%s", posting_input[:60], latency, exc
            )
            return BatchResult(posting_input=posting_input, state=None, error=str(exc), latency_seconds=latency)


async def score_postings(
    posting_inputs: list[str],
    resume_text: str,
    *,
    resume_id: str = "default",
    concurrency: int | None = None,
    store: VectorStore | None = None,
    client: LLMClient | None = None,
) -> list[BatchResult]:
    """Score many postings concurrently, at most ``concurrency`` in flight at once."""
    store = store or VectorStore()
    client = client or LLMClient()
    semaphore = asyncio.Semaphore(concurrency or settings.batch_concurrency)

    tasks = [
        _score_one(posting, resume_text, resume_id=resume_id, store=store, client=client, semaphore=semaphore)
        for posting in posting_inputs
    ]
    return await asyncio.gather(*tasks)
