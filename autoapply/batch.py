"""
scores a bunch of postings at once, concurrently, capped so we don't nuke rate limits.
one posting failing (bad url, retries exhausted etc) doesn't take the rest down with it
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
        except Exception as exc:  # noqa: BLE001 - don't let one bad posting kill the whole batch
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
    """runs postings concurrently, capped at `concurrency` in flight"""
    store = store or VectorStore()
    client = client or LLMClient()
    semaphore = asyncio.Semaphore(concurrency or settings.batch_concurrency)

    tasks = [
        _score_one(posting, resume_text, resume_id=resume_id, store=store, client=client, semaphore=semaphore)
        for posting in posting_inputs
    ]
    return await asyncio.gather(*tasks)


def _demo() -> None:
    """scores everything in data/sample_postings/ against data/sample_resume.md
    run: python -m autoapply.batch (needs an api key in .env)"""
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    resume_text = (data_dir / "sample_resume.md").read_text()
    posting_paths = sorted((data_dir / "sample_postings").glob("*.txt"))
    postings = [p.read_text() for p in posting_paths]

    store = VectorStore()
    store.add_resume(resume_text, resume_id="demo")

    results = asyncio.run(score_postings(postings, resume_text, resume_id="demo", store=store))

    for path, result in zip(posting_paths, results):
        if result.error:
            print(f"{path.name}: FAILED in {result.latency_seconds:.2f}s - {result.error}")
        else:
            score = result.state["match_result"].score
            print(f"{path.name}: score={score}/100 in {result.latency_seconds:.2f}s")


if __name__ == "__main__":
    _demo()
