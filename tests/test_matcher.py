"""tests for matcher retrieval/scoring. fake store stands in for chroma so these are
fast and don't need real embeddings - just testing our dedup/prompt logic"""

from __future__ import annotations

import pytest

from autoapply.agents.matcher import amatch_posting, compute_weighted_score, format_chunks, gather_grounding_chunks, match_posting
from autoapply.agents.schemas import (
    CoverageStatus,
    MatchResult,
    ParsedPosting,
    RequirementBreakdown,
    RequirementCategory,
)


class FakeStore:
    """same chunk back for every query, makes dedup easy to check"""

    def __init__(self, chunks_by_query: dict[str, list[dict]] | None = None):
        self.queries: list[str] = []
        self._chunks_by_query = chunks_by_query

    def semantic_search(self, query, n_results=4, resume_id="default"):
        self.queries.append(query)
        if self._chunks_by_query is not None:
            return self._chunks_by_query.get(query, [])
        return [{"id": "chunk-1", "text": "Built things with Python.", "metadata": {}, "distance": 0.1}]


def _posting(**overrides) -> ParsedPosting:
    defaults = dict(
        title="ML Engineer",
        company="Acme",
        required_skills=["PyTorch", "Kubernetes"],
        preferred_skills=["AWS"],
        responsibilities=["Ship models"],
    )
    defaults.update(overrides)
    return ParsedPosting(**defaults)


def test_gather_grounding_chunks_dedupes_across_queries():
    store = FakeStore()
    posting = _posting()

    chunks = gather_grounding_chunks(posting, store, "default")

    # 5 queries fired (title, 2 required, 1 preferred, 1 responsibility) but
    # every query returns the same chunk id, so the result should have 1 entry.
    assert len(store.queries) == 5
    assert len(chunks) == 1
    assert chunks[0]["id"] == "chunk-1"


def test_gather_grounding_chunks_collects_distinct_chunks():
    store = FakeStore(
        chunks_by_query={
            "PyTorch": [{"id": "chunk-a", "text": "PyTorch experience", "metadata": {}, "distance": 0.1}],
            "Kubernetes": [{"id": "chunk-b", "text": "Kubernetes experience", "metadata": {}, "distance": 0.1}],
        }
    )
    posting = _posting(preferred_skills=[], responsibilities=[])

    chunks = gather_grounding_chunks(posting, store, "default")
    chunk_ids = {c["id"] for c in chunks}

    assert chunk_ids == {"chunk-a", "chunk-b"}


def test_format_chunks_empty():
    assert "no resume chunks" in format_chunks([])


def test_format_chunks_includes_ids():
    chunks = [{"id": "chunk-1", "text": "hello", "metadata": {}, "distance": 0.1}]
    formatted = format_chunks(chunks)
    assert "[chunk-1]" in formatted
    assert "hello" in formatted


def test_match_posting_returns_validated_result_and_grounds_prompt():
    store = FakeStore()
    posting = _posting()
    captured_user_prompt = {}

    class FakeClient:
        def complete(self, *, system, user, response_model, temperature, agent_name):
            captured_user_prompt["value"] = user
            assert response_model is MatchResult
            assert agent_name == "matcher"
            return MatchResult(
                score=75,  # raw llm score - gets overridden by compute_weighted_score() below
                summary="Solid match",
                breakdown=[
                    RequirementBreakdown(
                        requirement="PyTorch",
                        category=RequirementCategory.REQUIRED_SKILL,
                        status=CoverageStatus.COVERED,
                        supporting_chunk_ids=["chunk-1"],
                        explanation="resume shows Python/ML work",
                    )
                ],
                top_gaps=["Kubernetes"],
            )

    result = match_posting(posting, store=store, client=FakeClient())

    assert isinstance(result, MatchResult)
    assert result.score == 100  # one required skill, fully covered -> deterministic 100
    assert "chunk-1" in captured_user_prompt["value"]
    # every cited chunk id must have actually come from retrieval
    cited_ids = {cid for b in result.breakdown for cid in b.supporting_chunk_ids}
    assert cited_ids <= {c["id"] for c in gather_grounding_chunks(posting, store, "default")}


@pytest.mark.asyncio
async def test_amatch_posting_matches_sync_behavior():
    store = FakeStore()
    posting = _posting()

    class FakeClient:
        async def acomplete(self, *, system, user, response_model, temperature, agent_name):
            assert response_model is MatchResult
            return MatchResult(score=50, summary="ok", breakdown=[], top_gaps=[])

    result = await amatch_posting(posting, store=store, client=FakeClient())

    assert result.score == 0  # empty breakdown -> nothing to weight -> 0, not the raw llm score


def _item(category, status):
    return RequirementBreakdown(requirement="x", category=category, status=status, explanation="")


def test_weighted_score_required_skills_count_more_than_preferred():
    # one required skill missing, one preferred skill covered - a naive average would look ok,
    # but missing a required skill should hurt more than nailing a preferred one helps
    breakdown = [
        _item(RequirementCategory.REQUIRED_SKILL, CoverageStatus.MISSING),
        _item(RequirementCategory.PREFERRED_SKILL, CoverageStatus.COVERED),
    ]
    score = compute_weighted_score(breakdown)
    assert score < 50  # required weight (1.0) > preferred weight (0.5), so this should skew low


def test_weighted_score_all_covered_is_100():
    breakdown = [
        _item(RequirementCategory.REQUIRED_SKILL, CoverageStatus.COVERED),
        _item(RequirementCategory.PREFERRED_SKILL, CoverageStatus.COVERED),
        _item(RequirementCategory.RESPONSIBILITY, CoverageStatus.COVERED),
    ]
    assert compute_weighted_score(breakdown) == 100


def test_weighted_score_all_missing_is_0():
    breakdown = [
        _item(RequirementCategory.REQUIRED_SKILL, CoverageStatus.MISSING),
        _item(RequirementCategory.PREFERRED_SKILL, CoverageStatus.MISSING),
    ]
    assert compute_weighted_score(breakdown) == 0


def test_weighted_score_empty_breakdown_is_0():
    assert compute_weighted_score([]) == 0


def test_weighted_score_partial_counts_as_half_credit():
    breakdown = [_item(RequirementCategory.REQUIRED_SKILL, CoverageStatus.PARTIAL)]
    assert compute_weighted_score(breakdown) == 50
