"""LangGraph orchestrator wiring the four agents into a pipeline:

    Parser -> Matcher -> (Tailoring ‖ Cover-Letter)

Tailoring and Cover-Letter both depend only on the parsed posting and
match result, not on each other, so they're wired as two edges out of
"matcher" rather than a chain — LangGraph schedules nodes with
satisfied dependencies in the same superstep and awaits them
concurrently when the graph is run with ``ainvoke``.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from autoapply.agents.cover_letter import awrite_cover_letter
from autoapply.agents.matcher import amatch_posting
from autoapply.agents.parser import aparse_posting
from autoapply.agents.schemas import CoverLetter, MatchResult, ParsedPosting, TailoredResume
from autoapply.agents.tailoring import atailor_resume
from autoapply.llm.provider import LLMClient
from autoapply.rag.store import VectorStore

logger = logging.getLogger("autoapply.graph")


class AutoApplyState(TypedDict, total=False):
    """State object threaded through every node in the graph."""

    posting_input: str
    resume_text: str
    resume_id: str
    parsed_posting: ParsedPosting
    match_result: MatchResult
    tailored_resume: TailoredResume
    cover_letter: CoverLetter


def build_graph(*, store: VectorStore | None = None, client: LLMClient | None = None):
    """Compile the AutoApply graph, injecting a shared VectorStore/LLMClient into every node."""
    store = store or VectorStore()
    client = client or LLMClient()

    async def parser_node(state: AutoApplyState) -> dict:
        parsed = await aparse_posting(state["posting_input"], client=client)
        return {"parsed_posting": parsed}

    async def matcher_node(state: AutoApplyState) -> dict:
        match = await amatch_posting(
            state["parsed_posting"], store=store, resume_id=state.get("resume_id", "default"), client=client
        )
        return {"match_result": match}

    async def tailoring_node(state: AutoApplyState) -> dict:
        tailored = await atailor_resume(
            state["parsed_posting"],
            state["resume_text"],
            store=store,
            resume_id=state.get("resume_id", "default"),
            client=client,
        )
        return {"tailored_resume": tailored}

    async def cover_letter_node(state: AutoApplyState) -> dict:
        letter = await awrite_cover_letter(
            state["parsed_posting"],
            state["match_result"],
            store=store,
            resume_id=state.get("resume_id", "default"),
            client=client,
        )
        return {"cover_letter": letter}

    graph = StateGraph(AutoApplyState)
    graph.add_node("parser", parser_node)
    graph.add_node("matcher", matcher_node)
    graph.add_node("tailoring", tailoring_node)
    graph.add_node("cover_letter", cover_letter_node)

    graph.set_entry_point("parser")
    graph.add_edge("parser", "matcher")
    graph.add_edge("matcher", "tailoring")
    graph.add_edge("matcher", "cover_letter")
    graph.add_edge("tailoring", END)
    graph.add_edge("cover_letter", END)

    return graph.compile()


async def run_pipeline(
    posting_input: str,
    resume_text: str,
    *,
    resume_id: str = "default",
    store: VectorStore | None = None,
    client: LLMClient | None = None,
) -> AutoApplyState:
    """Run the full pipeline for one posting and return the final state."""
    app = build_graph(store=store, client=client)
    initial_state: AutoApplyState = {
        "posting_input": posting_input,
        "resume_text": resume_text,
        "resume_id": resume_id,
    }
    return await app.ainvoke(initial_state)
