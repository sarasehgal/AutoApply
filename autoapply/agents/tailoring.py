"""Tailoring Agent: rewrites resume bullets for a specific posting.

The system prompt forbids fabrication, but prompts alone are not a
safety mechanism — this module also runs a deterministic validation
pass over the model's output that:

1. Rejects any bullet that cites a resume chunk ID that doesn't
   actually exist (the model can't ground a claim in a chunk it made up).
2. Flags any bullet whose claim-bearing tokens (numbers, percentages,
   capitalized technology/proper-noun-like words) don't appear anywhere
   in the source resume text, i.e. it isn't traceable back to what the
   candidate actually wrote.

Both checks are pure string/token comparisons against the real resume,
so they run without another LLM call and are deterministic to test.
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
    """Pull out the parts of a sentence that carry a factual claim: numbers and proper-noun-ish words."""
    tokens = set(_NUMBER_RE.findall(text))
    tokens |= {tok for tok in _PROPER_TOKEN_RE.findall(text) if tok not in _IGNORED_TOKENS}
    return tokens


def _unsupported_claims(tailored_text: str, source_resume_text: str) -> list[str]:
    source_lower = source_resume_text.lower()
    return sorted(tok for tok in _extract_claim_tokens(tailored_text) if tok.lower() not in source_lower)


def validate_no_fabrication(
    tailored: TailoredResume, source_resume_text: str, known_chunk_ids: set[str]
) -> TailoredResume:
    """Return a copy of ``tailored`` with flagged_unsupported_claims populated.

    Any pre-existing flags from the model itself are preserved; this
    only adds to them.
    """
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
