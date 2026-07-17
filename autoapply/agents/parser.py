"""turns a posting (text or url) into a ParsedPosting"""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

from autoapply.agents.schemas import ParsedPosting
from autoapply.llm.provider import LLMClient
from autoapply.prompts import parser as prompts

logger = logging.getLogger("autoapply.agents.parser")

AGENT_NAME = "parser"
_USER_AGENT = "AutoApply/1.0 (job-posting fetcher)"


def _is_url(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith("http://") or value.startswith("https://")


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def fetch_posting_text(url: str, timeout: float = 20.0) -> str:
    """grabs a posting url, strips scripts/styles/nav junk, returns plain text"""
    resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    return _clean_html(resp.text)


async def afetch_posting_text(url: str, timeout: float = 20.0) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as http:
        resp = await http.get(url, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
    return _clean_html(resp.text)


def parse_posting(posting_input: str, *, client: LLMClient | None = None) -> ParsedPosting:
    """text or url in, structured posting out"""
    text = fetch_posting_text(posting_input) if _is_url(posting_input) else posting_input
    if not text.strip():
        raise ValueError("posting text is empty after fetching/cleaning")

    client = client or LLMClient()
    return client.complete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(text),
        response_model=ParsedPosting,
        temperature=0.0,
        agent_name=AGENT_NAME,
    )


async def aparse_posting(posting_input: str, *, client: LLMClient | None = None) -> ParsedPosting:
    """same thing but async, so the batch scorer isn't blocked on url fetch + llm call"""
    text = await afetch_posting_text(posting_input) if _is_url(posting_input) else posting_input
    if not text.strip():
        raise ValueError("posting text is empty after fetching/cleaning")

    client = client or LLMClient()
    return await client.acomplete(
        system=prompts.SYSTEM_PROMPT,
        user=prompts.build_user_prompt(text),
        response_model=ParsedPosting,
        temperature=0.0,
        agent_name=AGENT_NAME,
    )
