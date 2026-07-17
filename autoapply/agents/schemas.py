"""pydantic models for what each agent spits out - keeps everything (llm layer, graph, ui) on the same page"""

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
    """parser output"""

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
        default_factory=list, description="chunk ids backing this up"
    )
    explanation: str


class MatchResult(BaseModel):
    """matcher output"""

    score: int = Field(ge=0, le=100)
    summary: str
    breakdown: list[RequirementBreakdown] = Field(default_factory=list)
    top_gaps: list[str] = Field(default_factory=list)


class TailoredBullet(BaseModel):
    original: str
    tailored: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class TailoredResume(BaseModel):
    """tailoring output"""

    summary: str
    bullets: list[TailoredBullet] = Field(default_factory=list)
    flagged_unsupported_claims: list[str] = Field(
        default_factory=list,
        description="stuff the validation pass couldn't trace back to the real resume",
    )


class CoverLetter(BaseModel):
    """cover letter output"""

    greeting: str
    body_paragraphs: list[str] = Field(default_factory=list)
    closing: str

    def full_text(self) -> str:
        parts = [self.greeting, *self.body_paragraphs, self.closing]
        return "\n\n".join(p for p in parts if p.strip())
