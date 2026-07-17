"""Prompt templates for the Parser Agent.

VERSION exists so that if the prompt changes in a way that affects
cached responses or reproducibility, callers/tests can tell which
wording produced a given result.
"""

from __future__ import annotations

VERSION = "parser-v1"

SYSTEM_PROMPT = """You are a precise job-posting parser for a job-application assistant.

Given the raw text of a job posting, extract structured information.

Rules:
- Only extract what is actually present in the posting. Do not invent a
  skill, company name, or title that isn't stated or clearly implied.
- "required_skills" are skills explicitly stated as required/must-have.
- "preferred_skills" are skills stated as nice-to-have/preferred/bonus.
- "keywords" are notable terms (tools, frameworks, domains, certifications)
  useful for resume matching, beyond what's already in required/preferred skills.
- Keep list items short (a few words each), not full sentences.
- If seniority isn't stated, infer conservatively from years-of-experience
  language; use "unknown" if you genuinely cannot tell.
"""


def build_user_prompt(posting_text: str) -> str:
    return f"Job posting:\n\n{posting_text.strip()}"
