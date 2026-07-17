"""Pydantic schemas for every agent's structured output.

Centralizing these means the LLM provider layer, the orchestrator, and
the Streamlit UI all agree on one shape per agent, and every agent
response is validated (not just hopefully-JSON-shaped) before the rest
of the pipeline touches it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Seniority(str, Enum):
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    STAFF_PLUS = "staff_plus"
    UNKNOWN = "unknown"


class ParsedPosting(BaseModel):
    """Structured output of the Parser Agent."""

    title: str
    company: str
    seniority: Seniority = Seniority.UNKNOWN
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class CoverageStatus(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"


class RequirementBreakdown(BaseModel):
    requirement: str
    status: CoverageStatus
    supporting_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of resume chunks that support this judgment"
    )
    explanation: str


class MatchResult(BaseModel):
    """Structured output of the Matcher Agent."""

    score: int = Field(ge=0, le=100)
    summary: str
    breakdown: list[RequirementBreakdown] = Field(default_factory=list)
    top_gaps: list[str] = Field(default_factory=list)


class TailoredBullet(BaseModel):
    original: str
    tailored: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class TailoredResume(BaseModel):
    """Structured output of the Tailoring Agent."""

    summary: str
    bullets: list[TailoredBullet] = Field(default_factory=list)
    flagged_unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the tailored output the validation pass could not trace to the source resume",
    )


class CoverLetter(BaseModel):
    """Structured output of the Cover-Letter Agent."""

    greeting: str
    body_paragraphs: list[str] = Field(default_factory=list)
    closing: str

    def full_text(self) -> str:
        parts = [self.greeting, *self.body_paragraphs, self.closing]
        return "\n\n".join(p for p in parts if p.strip())
