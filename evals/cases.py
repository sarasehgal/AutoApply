"""hand-picked eval cases built from the same sample data used for demos, so the expectations
are grounded in resume text a human can actually go read"""

from __future__ import annotations

from pathlib import Path

from evals.eval_case import EvalCase

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SAMPLE_RESUME = (_DATA_DIR / "sample_resume.md").read_text()


def _posting(name: str) -> str:
    return (_DATA_DIR / "sample_postings" / name).read_text()


CASES = [
    EvalCase(
        name="ml_engineer_strong_match",
        resume_text=_SAMPLE_RESUME,
        posting_text=_posting("senior_ml_engineer.txt"),
        expect_covered=["Python", "PyTorch", "Kubernetes", "Docker"],
        expect_missing=["Ranking", "Personalization"],
    ),
    EvalCase(
        name="frontend_engineer_weak_match",
        resume_text=_SAMPLE_RESUME,
        posting_text=_posting("frontend_engineer.txt"),
        expect_covered=[],
        expect_missing=["React", "TypeScript"],
    ),
]
