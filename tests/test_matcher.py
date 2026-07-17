"""Tests for the Matcher Agent's retrieval and scoring wiring.

No real Chroma/embedding calls are made: a fake store stands in for
VectorStore so these tests are fast and deterministic, and focus on
AutoApply's own logic (dedup, prompt construction) rather than
ChromaDB's.
"""

from __future__ import annotations

import pytest

from autoapply.agents.matcher import amatch_posting, format_chunks, gather_grounding_chunks, match_posting
from autoapply.agents.schemas import CoverageStatus, MatchResult, ParsedPosting, RequirementBreakdown


class FakeStore:
    """Returns the same chunk for every query, to make dedup easy to assert on."""

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
                score=75,
                summary="Solid match",
                breakdown=[
                    RequirementBreakdown(
                        requirement="PyTorch",
                        status=CoverageStatus.COVERED,
                        supporting_chunk_ids=["chunk-1"],
                        explanation="resume shows Python/ML work",
                    )
                ],
                top_gaps=["Kubernetes"],
            )

    result = match_posting(posting, store=store, client=FakeClient())

    assert isinstance(result, MatchResult)
    assert result.score == 75
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

    assert result.score == 50
