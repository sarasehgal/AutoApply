"""streamlit app - paste your resume, index it, paste a posting, hit analyze, get results"""

from __future__ import annotations

import asyncio
import difflib
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from autoapply.graph.orchestrator import run_pipeline
from autoapply.llm.provider import LLMClient, clear_cache
from autoapply.rag.store import VectorStore
from autoapply.resume_export import render_tailored_resume_pdf
from autoapply.resume_parsing import UnsupportedResumeFormatError, extract_resume_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="AutoApply", page_icon="📄", layout="wide")

SAMPLE_RESUME_PATH = Path(__file__).parent / "data" / "sample_resume.md"
RESUME_ID = "default"


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource
def get_client() -> LLMClient:
    return LLMClient()


def run_pipeline_sync(posting_input: str, resume_text: str):
    return asyncio.run(
        run_pipeline(posting_input, resume_text, resume_id=RESUME_ID, store=get_store(), client=get_client())
    )


def word_diff_markdown(original: str, tailored: str) -> str:
    """word-level diff, strikethrough for removed bits and bold for added bits"""
    original_words = original.split()
    tailored_words = tailored.split()
    matcher = difflib.SequenceMatcher(None, original_words, tailored_words)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(tailored_words[j1:j2]))
        elif tag == "insert":
            parts.append(f"**{' '.join(tailored_words[j1:j2])}**")
        elif tag == "delete":
            parts.append(f"~~{' '.join(original_words[i1:i2])}~~")
        elif tag == "replace":
            parts.append(f"~~{' '.join(original_words[i1:i2])}~~ **{' '.join(tailored_words[j1:j2])}**")
    return " ".join(parts)


STATUS_ICON = {"covered": "✅", "partial": "\U0001f7e1", "missing": "❌"}
CATEGORY_LABEL = {"required_skill": "required", "preferred_skill": "preferred", "responsibility": "responsibility"}


st.title("AutoApply")
st.caption(
    "Paste a job posting (or its URL) and your resume to get a grounded match score, "
    "a tailored resume, and a draft cover letter."
)

with st.sidebar:
    st.header("Your resume")

    if SAMPLE_RESUME_PATH.exists() and st.button("Load sample resume"):
        st.session_state["resume_draft"] = SAMPLE_RESUME_PATH.read_text()

    uploaded = st.file_uploader("Or upload a resume", type=["txt", "md", "pdf", "docx"])
    if uploaded is not None:
        try:
            st.session_state["resume_draft"] = extract_resume_text(uploaded.name, uploaded.read())
        except UnsupportedResumeFormatError as exc:
            st.error(str(exc))

    resume_text = st.text_area(
        "Paste your resume (plain text or markdown)",
        height=320,
        key="resume_draft",
    )

    if st.button("Index resume", type="primary"):
        if not resume_text.strip():
            st.error("Paste or upload a resume first.")
        else:
            n_chunks = get_store().add_resume(resume_text, resume_id=RESUME_ID)
            st.session_state["indexed_resume_text"] = resume_text
            st.success(f"Indexed {n_chunks} resume chunks.")

    if st.session_state.get("indexed_resume_text"):
        st.caption("Resume is indexed and ready.")

    with st.expander("Advanced"):
        st.caption("Every LLM call is cached on disk. Clear it to force fresh calls (e.g. after tweaking a prompt).")
        if st.button("Clear cache"):
            clear_cache()
            st.success("Cache cleared.")

st.subheader("Job posting")
input_mode = st.radio("Input type", ["Paste text", "URL"], horizontal=True)
if input_mode == "Paste text":
    posting_input = st.text_area("Paste the job posting text", height=200)
else:
    posting_input = st.text_input("Job posting URL", placeholder="https://...")

analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:
    indexed_resume = st.session_state.get("indexed_resume_text")
    if not indexed_resume:
        st.error("Index your resume in the sidebar first.")
    elif not posting_input or not posting_input.strip():
        st.error("Paste a job posting or URL first.")
    else:
        with st.spinner("Running Parser → Matcher → Tailoring + Cover Letter..."):
            start = time.monotonic()
            try:
                state = run_pipeline_sync(posting_input.strip(), indexed_resume)
                st.session_state["result_state"] = state
                st.session_state["result_elapsed"] = time.monotonic() - start
                get_store().add_postings(
                    [
                        {
                            "id": f"posting-{uuid.uuid4().hex[:12]}",
                            "text": posting_input.strip(),
                            "title": state["parsed_posting"].title,
                            "company": state["parsed_posting"].company,
                            "score": state["match_result"].score,
                            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        }
                    ]
                )
            except Exception as exc:  # noqa: BLE001 - show it in the ui instead of crashing
                st.error(f"Pipeline failed: {exc}")

with st.expander("🔍 Search past postings"):
    history_query = st.text_input("Search postings you've already analyzed", key="history_query")
    if history_query.strip():
        hits = get_store().search_postings(history_query.strip(), n_results=5)
        if not hits:
            st.caption("No past postings match that yet.")
        for hit in hits:
            meta = hit["metadata"]
            title = meta.get("title", "Untitled posting")
            company = meta.get("company", "Unknown company")
            score = meta.get("score")
            when = meta.get("analyzed_at", "")
            score_str = f"{score}/100" if score is not None else "n/a"
            st.markdown(f"**{title}** at **{company}** — score {score_str}")
            st.caption(when)

state = st.session_state.get("result_state")
if state:
    posting = state["parsed_posting"]
    match = state["match_result"]
    tailored = state.get("tailored_resume")
    letter = state.get("cover_letter")
    elapsed = st.session_state.get("result_elapsed")

    st.divider()
    header = f"{posting.title} at {posting.company}"
    st.header(header)
    if elapsed:
        st.caption(f"Completed in {elapsed:.1f}s")

    col_score, col_breakdown = st.columns([1, 2])
    with col_score:
        st.metric("Match score", f"{match.score}/100")
        st.write(match.summary)
        if match.top_gaps:
            st.write("**Top gaps**")
            for gap in match.top_gaps:
                st.write(f"- {gap}")
    with col_breakdown:
        st.write("**Requirement breakdown**")
        for item in match.breakdown:
            icon = STATUS_ICON.get(item.status.value, "•")
            category_label = CATEGORY_LABEL.get(item.category.value, item.category.value)
            st.markdown(f"{icon} **{item.requirement}** _({category_label})_ — {item.explanation}")
            if item.supporting_chunk_ids:
                st.caption("Sources: " + ", ".join(item.supporting_chunk_ids))

    tab_resume, tab_letter = st.tabs(["Tailored resume", "Cover letter"])

    with tab_resume:
        if tailored:
            if tailored.flagged_unsupported_claims:
                st.warning(
                    "Validation flagged possible unsupported claims (not shown as fact until reviewed):\n\n"
                    + "\n".join(f"- {f}" for f in tailored.flagged_unsupported_claims)
                )
            st.write(tailored.summary)
            st.write("**Bullet-by-bullet diff**")
            for bullet in tailored.bullets:
                st.markdown(word_diff_markdown(bullet.original, bullet.tailored))
                st.caption("Sources: " + ", ".join(bullet.source_chunk_ids))

            tailored_text = tailored.summary + "\n\n" + "\n".join(f"- {b.tailored}" for b in tailored.bullets)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "Download tailored resume (.txt)", tailored_text, file_name="tailored_resume.txt"
                )
            with dl_col2:
                pdf_bytes = render_tailored_resume_pdf(tailored, title=f"{posting.title} - Tailored Resume")
                st.download_button(
                    "Download tailored resume (.pdf)",
                    pdf_bytes,
                    file_name="tailored_resume.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("No tailored resume in this result.")

    with tab_letter:
        if letter:
            st.write(letter.full_text())
            st.download_button(
                "Download cover letter (.txt)", letter.full_text(), file_name="cover_letter.txt"
            )
        else:
            st.info("No cover letter in this result.")
