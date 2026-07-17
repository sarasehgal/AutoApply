"""No-fabrication guarantee for the Tailoring Agent.

The core promise of this agent is: it may rephrase or reprioritize
real resume content, but it must never invent a skill the candidate
doesn't have. These tests feed a resume that deliberately lacks a
skill the posting wants, simulate a model that (incorrectly) claims
that skill anyway, and assert the deterministic validation pass in
agents/tailoring.py catches it every time - independent of what the
LLM actually produced.
"""

from __future__ import annotations

import pytest

from autoapply.agents.schemas import ParsedPosting, TailoredBullet, TailoredResume
from autoapply.agents.tailoring import atailor_resume, tailor_resume, validate_no_fabrication

RESUME_WITHOUT_KUBERNETES = (
    "- Built a real-time recommendation engine serving 2M users using Python and PyTorch, "
    "improving CTR by 18%.\n"
    "- Led a team of 4 engineers to migrate a monolith to a service-oriented architecture."
)


class FakeStore:
    def semantic_search(self, query, n_results=4, resume_id="default"):
        return [
            {
                "id": "chunk-1",
                "text": "Built a real-time recommendation engine serving 2M users using Python and PyTorch, improving CTR by 18%.",
                "metadata": {},
                "distance": 0.1,
            }
        ]


def _posting() -> ParsedPosting:
    return ParsedPosting(title="ML Engineer", company="Acme", required_skills=["Kubernetes"])


class FabricatingClient:
    """Simulates a model that claims Kubernetes experience the resume never mentions."""

    def complete(self, *, system, user, response_model, temperature, agent_name):
        return TailoredResume(
            summary="Engineer experienced in Kubernetes-based deployments.",
            bullets=[
                TailoredBullet(
                    original="Built a real-time recommendation engine...",
                    tailored="Deployed and scaled Kubernetes infrastructure for a recommendation engine serving 2M users.",
                    source_chunk_ids=["chunk-1"],
                )
            ],
        )

    async def acomplete(self, *, system, user, response_model, temperature, agent_name):
        return self.complete(
            system=system, user=user, response_model=response_model, temperature=temperature, agent_name=agent_name
        )


class HonestClient:
    """Simulates a model that only rephrases what's actually in the resume."""

    def complete(self, *, system, user, response_model, temperature, agent_name):
        return TailoredResume(
            summary="Engineer with a track record shipping ML-powered recommendation systems.",
            bullets=[
                TailoredBullet(
                    original="Built a real-time recommendation engine...",
                    tailored="Built and shipped a real-time PyTorch recommendation engine serving 2M users, +18% CTR.",
                    source_chunk_ids=["chunk-1"],
                )
            ],
        )


def test_fabricated_skill_is_flagged():
    result = tailor_resume(_posting(), RESUME_WITHOUT_KUBERNETES, store=FakeStore(), client=FabricatingClient())

    assert result.flagged_unsupported_claims, "expected the Kubernetes claim to be flagged"
    assert any("Kubernetes" in flag for flag in result.flagged_unsupported_claims)


def test_fabricated_skill_never_silently_passes_through():
    """The tailored bullet text may still contain the fabricated claim, but it must always
    be accompanied by a flag - the caller (UI) must never present it as unvalidated fact."""
    result = tailor_resume(_posting(), RESUME_WITHOUT_KUBERNETES, store=FakeStore(), client=FabricatingClient())

    bullet_mentions_kubernetes = any("Kubernetes" in b.tailored for b in result.bullets)
    if bullet_mentions_kubernetes:
        assert result.flagged_unsupported_claims


def test_honest_tailoring_produces_no_flags():
    result = tailor_resume(_posting(), RESUME_WITHOUT_KUBERNETES, store=FakeStore(), client=HonestClient())

    assert result.flagged_unsupported_claims == []


def test_validate_no_fabrication_catches_unknown_chunk_id():
    tailored = TailoredResume(
        summary="s",
        bullets=[TailoredBullet(original="x", tailored="Built stuff.", source_chunk_ids=["chunk-does-not-exist"])],
    )

    validated = validate_no_fabrication(tailored, RESUME_WITHOUT_KUBERNETES, known_chunk_ids={"chunk-1"})

    assert any("unknown chunk id" in flag for flag in validated.flagged_unsupported_claims)


def test_validate_no_fabrication_ignores_common_action_verbs():
    """Common resume verbs (Built, Led, Deployed, ...) shouldn't trip the heuristic on their own."""
    tailored = TailoredResume(
        summary="s",
        bullets=[
            TailoredBullet(
                original="x",
                tailored="Built and shipped a real-time PyTorch recommendation engine serving 2M users, +18% CTR.",
                source_chunk_ids=["chunk-1"],
            )
        ],
    )

    validated = validate_no_fabrication(tailored, RESUME_WITHOUT_KUBERNETES, known_chunk_ids={"chunk-1"})

    assert validated.flagged_unsupported_claims == []


@pytest.mark.asyncio
async def test_async_tailoring_also_flags_fabrication():
    result = await atailor_resume(
        _posting(), RESUME_WITHOUT_KUBERNETES, store=FakeStore(), client=FabricatingClient()
    )

    assert any("Kubernetes" in flag for flag in result.flagged_unsupported_claims)
