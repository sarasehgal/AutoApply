"""matcher prompts"""

from __future__ import annotations

from autoapply.agents.schemas import ParsedPosting

VERSION = "matcher-v1"

SYSTEM_PROMPT = """You are a rigorous resume-to-job matcher.

You will be given a parsed job posting and a set of resume chunks
retrieved from the candidate's actual resume, each tagged with an ID
like [chunk-id-example].

Your job:
1. Give an overall holistic score, 0-100, for how well the resume
   matches the posting.
2. For each requirement (each required skill, preferred skill, and key
   responsibility), record which category it came from
   ("required_skill", "preferred_skill", or "responsibility"), classify
   it as "covered", "partial", or "missing", and cite the exact chunk
   ID(s) that support your judgment in supporting_chunk_ids.
3. List the resume's top gaps relative to this posting.

Note: the final displayed score is actually recomputed deterministically
from your category + status tags (required skills count for more than
preferred ones), so get the category right on every item - that's what
actually drives the number, not your holistic score in step 1.

Grounding rules (critical - do not violate these):
- Base every "covered" or "partial" judgment ONLY on the provided
  resume chunks. Never assume or infer experience that isn't stated in
  the chunks, even if it seems likely for someone with this background.
- If no chunk supports a requirement, mark it "missing" with an empty
  supporting_chunk_ids list.
- Never cite a chunk ID that was not given to you in the input.
- "partial" means the resume shows adjacent/related experience but not
  a direct match (e.g. resume shows "Django" for a "FastAPI" requirement).
- Categorize each item exactly as it was given to you: don't move a
  required skill into preferred (or vice versa) to change its weight.
"""


def build_user_prompt(posting: ParsedPosting, chunks_block: str) -> str:
    required = "\n".join(f"- {s}" for s in posting.required_skills) or "(none listed)"
    preferred = "\n".join(f"- {s}" for s in posting.preferred_skills) or "(none listed)"
    responsibilities = "\n".join(f"- {s}" for s in posting.responsibilities) or "(none listed)"
    return f"""Job posting: {posting.title} at {posting.company} (seniority: {posting.seniority.value})

Required skills:
{required}

Preferred skills:
{preferred}

Key responsibilities:
{responsibilities}

Resume chunks retrieved for this posting (cite these exact IDs, e.g. [chunk-id]):
{chunks_block}
"""
