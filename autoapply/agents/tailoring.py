"""
rewrites resume bullets for a posting. prompt says "don't make stuff up" but prompts aren't
enough on their own, so there's also a dumb-but-reliable check afterward that:

1. rejects bullets citing a chunk id that was never actually retrieved (can't ground a claim
   in a chunk that doesn't exist)
2. flags bullets with numbers/%s/tech-looking words that don't show up anywhere in the real
   resume - i.e. stuff it can't prove

both are just string comparisons against the real resume text, no extra llm call needed, and
that also makes them easy to unit test
"""

from __future__ import annotations

import asyncio
import logging
import re

from autoapply.agents.matcher import format_chunks, gather_grounding_chunks
from autoapply.agents.schemas import ParsedPosting, TailoredResume
from autoapply.llm.provider import LLMClient
from autoapply.prompts import tailoring as prompts
from autoapply.rag.store import VectorStore

logger = logging.getLogger("autoapply.agents.tailoring")

AGENT_NAME = "tailoring"

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_PROPER_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9+#.]{1,}\b")
_IGNORED_TOKENS = {
    "I", "The", "A", "An", "In", "On", "For", "With", "And", "Or", "To", "Of",
    "Built", "Led", "Designed", "Using", "Shipped", "Drove", "Improved", "Reduced",
    "Increased", "Managed", "Created", "Developed", "Implemented", "Deployed",
    "Optimized", "Architected", "Automated", "Maintained", "Established",
    "Delivered", "Achieved", "Enhanced", "Streamlined", "Owned", "Scaled",
    "Wrote", "Authored", "Launched", "Migrated", "Refactored", "Resolved",
    "Supported", "Upgraded", "Utilized", "Collaborated", "Coordinated",
}


def _extract_claim_tokens(text: str) -> set[str]:
    """grabs the claim-y bits of a sentence: numbers and capitalized/proper-noun-ish words"""
    tokens = set(_NUMBER_RE.findall(text))
    tokens |= {tok for tok in _PROPER_TOKEN_RE.findall(text) if tok not in _IGNORED_TOKENS}
    return tokens


def _unsupported_claims(tailored_text: str, source_resume_text: str) -> list[str]:
    source_lower = source_resume_text.lower()
    return sorted(tok for tok in _extract_claim_tokens(tailored_text) if tok.lower() not in source_lower)


def validate_no_fabrication(
    tailored: TailoredResume, source_resume_text: str, known_chunk_ids: set[str]
) -> TailoredResume:
    """returns a copy of tailored with flagged_unsupported_claims filled in (adds to whatever was already there)"""
    flags: list[str] = list(tailored.flagged_unsupported_claims)

    for bullet in tailored.bullets:
        unknown_ids = [cid for cid in bullet.source_chunk_ids if cid not in known_chunk_ids]
        if unknown_ids:
            flags.append(f"bullet '{bullet.tailored[:60]}' cites unknown chunk id(s): {unknown_ids}")

        unsupported = _unsupported_claims(bullet.tailored, source_resume_text)
        if unsupported:
            flags.append(f"bullet '{bullet.tailored[:60]}' has claim(s) not found in source resume: {unsupported}")

    return tailored.model_copy(update={"flagged_unsupported_claims": flags})


def tailor_resume(
    posting: ParsedPosting,
    resume_text: str,
    *,
    store: VectorStore | None = None,
    resume_id: str = "default",
    client: LLMClient | None = None,
) -> TailoredResume:
    store = store or VectorStore()
    chunks = gather_grounding_chunks(posting, store, resume_id)
    known_ids = {c["id"] for c in chunks}

    client = client or LLMClient()
    result = client.complete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(posting, format_chunks(chunks)),
        response_model=TailoredResume,
        temperature=0.3,
        agent_name=AGENT_NAME,
    )
    return validate_no_fabrication(result, resume_text, known_ids)


async def atailor_resume(
    posting: ParsedPosting,
    resume_text: str,
    *,
    store: VectorStore | None = None,
    resume_id: str = "default",
    client: LLMClient | None = None,
) -> TailoredResume:
    store = store or VectorStore()
    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(None, gather_grounding_chunks, posting, store, resume_id)
    known_ids = {c["id"] for c in chunks}

    client = client or LLMClient()
    result = await client.acomplete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(posting, format_chunks(chunks)),
        response_model=TailoredResume,
        temperature=0.3,
        agent_name=AGENT_NAME,
    )
    return validate_no_fabrication(result, resume_text, known_ids)
