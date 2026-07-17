"""Cover-Letter Agent: drafts a concise, specific cover letter grounded
in the match analysis and the candidate's strongest matched experience.
"""

from __future__ import annotations

import asyncio
import logging

from autoapply.agents.matcher import format_chunks, gather_grounding_chunks
from autoapply.agents.schemas import CoverLetter, MatchResult, ParsedPosting
from autoapply.llm.provider import LLMClient
from autoapply.prompts import cover_letter as prompts
from autoapply.rag.store import VectorStore

logger = logging.getLogger("autoapply.agents.cover_letter")

AGENT_NAME = "cover_letter"


def write_cover_letter(
    posting: ParsedPosting,
    match: MatchResult,
    *,
    store: VectorStore | None = None,
    resume_id: str = "default",
    client: LLMClient | None = None,
) -> CoverLetter:
    store = store or VectorStore()
    chunks = gather_grounding_chunks(posting, store, resume_id)

    client = client or LLMClient()
    return client.complete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(posting, match, format_chunks(chunks)),
        response_model=CoverLetter,
        temperature=0.4,
        agent_name=AGENT_NAME,
    )


async def awrite_cover_letter(
    posting: ParsedPosting,
    match: MatchResult,
    *,
    store: VectorStore | None = None,
    resume_id: str = "default",
    client: LLMClient | None = None,
) -> CoverLetter:
    store = store or VectorStore()
    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(None, gather_grounding_chunks, posting, store, resume_id)

    client = client or LLMClient()
    return await client.acomplete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(posting, match, format_chunks(chunks)),
        response_model=CoverLetter,
        temperature=0.4,
        agent_name=AGENT_NAME,
    )
