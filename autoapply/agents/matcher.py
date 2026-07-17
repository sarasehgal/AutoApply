"""Matcher Agent: RAG-grounded match scoring.

Retrieves the resume chunks most relevant to a posting's title,
required/preferred skills, and responsibilities, then asks the LLM to
score the match and justify every "covered"/"partial" judgment by
citing exact chunk IDs — so the score is traceable back to real resume
text rather than the model's guess about the candidate.
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
    """Retrieve resume chunks relevant to this posting via multiple targeted queries."""
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
    # Chroma's client is sync-only; offload to the default executor so it
    # doesn't block the event loop during concurrent batch scoring.
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
