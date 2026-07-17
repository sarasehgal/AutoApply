"""
matcher agent - grabs the resume chunks relevant to a posting, asks the llm to score the match
and back up every covered/partial call with the actual chunk ids. no vibes-based scoring
"""

from __future__ import annotations

import asyncio
import logging

from autoapply.agents.schemas import MatchResult, ParsedPosting
from autoapply.llm.provider import LLMClient
from autoapply.prompts import matcher as prompts
from autoapply.rag.store import VectorStore

logger = logging.getLogger("autoapply.agents.matcher")

AGENT_NAME = "matcher"
DEFAULT_CHUNKS_PER_QUERY = 4


def gather_grounding_chunks(
    posting: ParsedPosting,
    store: VectorStore,
    resume_id: str,
    chunks_per_query: int = DEFAULT_CHUNKS_PER_QUERY,
) -> list[dict]:
    """fires off a bunch of targeted queries (title, each skill, each responsibility) and dedupes the hits"""
    queries = [
        f"{posting.title} at {posting.company}",
        *posting.required_skills,
        *posting.preferred_skills,
        *posting.responsibilities,
    ]
    seen_ids: set[str] = set()
    chunks: list[dict] = []
    for query in queries:
        if not query or not query.strip():
            continue
        for chunk in store.semantic_search(query, n_results=chunks_per_query, resume_id=resume_id):
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                chunks.append(chunk)
    return chunks


def format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(no resume chunks found - the resume may not be indexed yet)"
    return "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


def match_posting(
    posting: ParsedPosting,
    *,
    store: VectorStore | None = None,
    resume_id: str = "default",
    client: LLMClient | None = None,
) -> MatchResult:
    store = store or VectorStore()
    chunks = gather_grounding_chunks(posting, store, resume_id)

    client = client or LLMClient()
    return client.complete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(posting, format_chunks(chunks)),
        response_model=MatchResult,
        temperature=0.1,
        agent_name=AGENT_NAME,
    )


async def amatch_posting(
    posting: ParsedPosting,
    *,
    store: VectorStore | None = None,
    resume_id: str = "default",
    client: LLMClient | None = None,
) -> MatchResult:
    store = store or VectorStore()
    # chroma's client is sync only, so shove it in the executor or it'll block the loop during batch runs
    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(None, gather_grounding_chunks, posting, store, resume_id)

    client = client or LLMClient()
    return await client.acomplete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(posting, format_chunks(chunks)),
        response_model=MatchResult,
        temperature=0.1,
        agent_name=AGENT_NAME,
    )
