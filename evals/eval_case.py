"""one eval case: a resume + posting pair with hand-picked expectations about what the
matcher should find covered vs missing. not pydantic - this never touches the llm, just
plain data the eval runner checks against."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    name: str
    resume_text: str
    posting_text: str
    expect_covered: list[str] = field(default_factory=list)  # must NOT come back "missing"
    expect_missing: list[str] = field(default_factory=list)  # must come back "missing"
