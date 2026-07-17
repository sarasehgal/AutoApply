"""
small eval harness for the matcher - catches prompt regressions unit tests can't, since it
needs a real llm call. NOT part of the pytest suite (costs real tokens, needs a real api
key) - run by hand after touching prompts/matcher.py:

    python -m evals.run_matcher_eval
"""

from __future__ import annotations

import sys

from autoapply.agents.matcher import match_posting
from autoapply.agents.parser import parse_posting
from autoapply.agents.schemas import CoverageStatus, RequirementBreakdown
from autoapply.rag.store import VectorStore
from evals.cases import CASES


def _find_status(breakdown: list[RequirementBreakdown], needle: str) -> CoverageStatus | None:
    needle_lower = needle.lower()
    for item in breakdown:
        if needle_lower in item.requirement.lower():
            return item.status
    return None


def run() -> bool:
    store = VectorStore()
    all_passed = True

    for case in CASES:
        resume_id = f"eval-{case.name}"
        store.add_resume(case.resume_text, resume_id=resume_id)

        try:
            posting = parse_posting(case.posting_text)
            match = match_posting(posting, store=store, resume_id=resume_id)
        except Exception as exc:  # noqa: BLE001 - one case erroring (rate limit, etc) shouldn't kill the run
            print(f"\n=== {case.name} ===\n  ERROR - {exc}")
            all_passed = False
            continue

        print(f"\n=== {case.name} (score={match.score}/100) ===")
        case_passed = True

        for needle in case.expect_covered:
            status = _find_status(match.breakdown, needle)
            ok = status is not None and status != CoverageStatus.MISSING
            print(f"  {'PASS' if ok else 'FAIL'} expected '{needle}' covered/partial, got {status}")
            case_passed = case_passed and ok

        for needle in case.expect_missing:
            status = _find_status(match.breakdown, needle)
            ok = status == CoverageStatus.MISSING
            print(f"  {'PASS' if ok else 'FAIL'} expected '{needle}' missing, got {status}")
            case_passed = case_passed and ok

        all_passed = all_passed and case_passed

    print(f"\n{'ALL CASES PASSED' if all_passed else 'SOME CASES FAILED'}")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
