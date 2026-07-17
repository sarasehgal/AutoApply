"""
the one place that knows about openai/anthropic/gemini. agents just call complete()/acomplete()/embed()
and don't care which provider is behind it. handles retries, timeouts, fallback, structured
output validation, caching, latency logs - basically all the annoying reliability stuff
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

from anthropic import Anthropic, AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from autoapply.config import settings
from autoapply.llm.cache import ResponseCache, hash_key

logger = logging.getLogger("autoapply.llm")

T = TypeVar("T", bound=BaseModel)

_cache = ResponseCache()


def clear_cache() -> None:
    """wipes every cached llm response - the only way to force a fresh call for something already cached"""
    _cache.clear()


class ProviderError(Exception):
    """one call to one provider blew up - bad request, timeout, bad json, whatever"""


class AllProvidersFailedError(Exception):
    """primary AND fallback both died, nothing left to try"""


def _retryer() -> Retrying:
    return Retrying(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )


def _async_retryer() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )


def _schema_instructions(response_model: type[BaseModel]) -> str:
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    return (
        "\n\nYou must respond with ONLY valid JSON matching this JSON Schema. "
        "No markdown code fences, no commentary before or after the JSON.\n\n"
        f"Schema:\n{schema}"
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    return text.strip()


def _validate(raw: str, response_model: type[T]) -> T:
    cleaned = _strip_code_fence(raw)
    try:
        return response_model.model_validate_json(cleaned)
    except ValidationError as exc:
        raise ProviderError(f"structured output failed schema validation: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"model did not return valid JSON: {exc}") from exc


class LLMClient:
    """chat + embedding client, provider swap is just a string, retry/timeout/fallback/cache all built in"""

    def __init__(self, provider: str | None = None):
        self.primary = (provider or settings.provider).lower()
        self.fallback = (settings.fallback_provider or "").lower() or None
        if self.fallback == self.primary:
            self.fallback = None

    # ---------------------------------------------------------------- sync
    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T] | None = None,
        temperature: float = 0.2,
        use_cache: bool = True,
        agent_name: str = "unknown",
    ) -> T | str:
        cache_key = hash_key(
            self.primary, self.fallback or "", system, user, str(temperature),
            response_model.__name__ if response_model else "raw",
        )
        raw = _cache.get(cache_key) if (use_cache and settings.cache_enabled) else None

        if raw is None:
            start = time.monotonic()
            raw, provider_used = self._raw_with_fallback(system, user, response_model, temperature)
            latency = time.monotonic() - start
            logger.info(
                "agent=%s provider=%s latency=%.2fs cache=miss", agent_name, provider_used, latency
            )
            if use_cache and settings.cache_enabled:
                _cache.set(cache_key, raw)
        else:
            logger.info("agent=%s provider=cache latency=0.00s cache=hit", agent_name)

        if response_model is None:
            return raw
        return _validate(raw, response_model)

    def _raw_with_fallback(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> tuple[str, str]:
        try:
            raw = _retryer()(self._call_raw, self.primary, system, user, response_model, temperature)
            return raw, self.primary
        except ProviderError as primary_exc:
            if not self.fallback:
                raise AllProvidersFailedError(str(primary_exc)) from primary_exc
            logger.warning("provider=%s failed (%s); falling back to provider=%s",
                            self.primary, primary_exc, self.fallback)
            try:
                raw = _retryer()(self._call_raw, self.fallback, system, user, response_model, temperature)
                return raw, self.fallback
            except ProviderError as fallback_exc:
                raise AllProvidersFailedError(
                    f"primary({self.primary})={primary_exc}; fallback({self.fallback})={fallback_exc}"
                ) from fallback_exc

    def _call_raw(
        self, provider: str, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        try:
            if provider == "openai":
                raw = self._call_openai(system, user, response_model, temperature)
            elif provider == "anthropic":
                raw = self._call_anthropic(system, user, response_model, temperature)
            elif provider == "gemini":
                raw = self._call_gemini(system, user, response_model, temperature)
            else:
                raise ProviderError(f"unknown provider: {provider}")
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - catch-all so retry/fallback still kicks in
            raise ProviderError(f"{provider} call failed: {exc}") from exc
        if response_model is not None:
            _validate(raw, response_model)  # raises -> retry picks it up
        return raw

    # --------------------------------------------------------------- async
    async def acomplete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T] | None = None,
        temperature: float = 0.2,
        use_cache: bool = True,
        agent_name: str = "unknown",
    ) -> T | str:
        cache_key = hash_key(
            self.primary, self.fallback or "", system, user, str(temperature),
            response_model.__name__ if response_model else "raw",
        )
        raw = _cache.get(cache_key) if (use_cache and settings.cache_enabled) else None

        if raw is None:
            start = time.monotonic()
            raw, provider_used = await self._araw_with_fallback(system, user, response_model, temperature)
            latency = time.monotonic() - start
            logger.info(
                "agent=%s provider=%s latency=%.2fs cache=miss", agent_name, provider_used, latency
            )
            if use_cache and settings.cache_enabled:
                _cache.set(cache_key, raw)
        else:
            logger.info("agent=%s provider=cache latency=0.00s cache=hit", agent_name)

        if response_model is None:
            return raw
        return _validate(raw, response_model)

    async def _araw_with_fallback(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> tuple[str, str]:
        try:
            raw = await self._aretry_call(self.primary, system, user, response_model, temperature)
            return raw, self.primary
        except ProviderError as primary_exc:
            if not self.fallback:
                raise AllProvidersFailedError(str(primary_exc)) from primary_exc
            logger.warning("provider=%s failed (%s); falling back to provider=%s",
                            self.primary, primary_exc, self.fallback)
            try:
                raw = await self._aretry_call(self.fallback, system, user, response_model, temperature)
                return raw, self.fallback
            except ProviderError as fallback_exc:
                raise AllProvidersFailedError(
                    f"primary({self.primary})={primary_exc}; fallback({self.fallback})={fallback_exc}"
                ) from fallback_exc

    async def _aretry_call(
        self, provider: str, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        async for attempt in _async_retryer():
            with attempt:
                return await self._acall_raw(provider, system, user, response_model, temperature)
        raise ProviderError("retry loop exhausted without result")  # pragma: no cover

    async def _acall_raw(
        self, provider: str, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        try:
            if provider == "openai":
                raw = await self._acall_openai(system, user, response_model, temperature)
            elif provider == "anthropic":
                raw = await self._acall_anthropic(system, user, response_model, temperature)
            elif provider == "gemini":
                raw = await self._acall_gemini(system, user, response_model, temperature)
            else:
                raise ProviderError(f"unknown provider: {provider}")
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - catch-all so retry/fallback still kicks in
            raise ProviderError(f"{provider} call failed: {exc}") from exc
        if response_model is not None:
            _validate(raw, response_model)
        return raw

    # ------------------------------------------------------------- openai
    def _openai_client(self) -> OpenAI:
        api_key = settings.api_key_for("openai")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        return OpenAI(api_key=api_key, timeout=settings.request_timeout_seconds)

    def _aopenai_client(self) -> AsyncOpenAI:
        api_key = settings.api_key_for("openai")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        return AsyncOpenAI(api_key=api_key, timeout=settings.request_timeout_seconds)

    def _call_openai(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        client = self._openai_client()
        kwargs: dict[str, Any] = {}
        if response_model is not None:
            system = system + _schema_instructions(response_model)
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - just wrap it as ProviderError
            raise ProviderError(f"openai call failed: {exc}") from exc
        return resp.choices[0].message.content or ""

    async def _acall_openai(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        client = self._aopenai_client()
        kwargs: dict[str, Any] = {}
        if response_model is not None:
            system = system + _schema_instructions(response_model)
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"openai call failed: {exc}") from exc
        return resp.choices[0].message.content or ""

    # ----------------------------------------------------------- anthropic
    def _anthropic_client(self) -> Anthropic:
        api_key = settings.api_key_for("anthropic")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        return Anthropic(api_key=api_key, timeout=settings.request_timeout_seconds)

    def _aanthropic_client(self) -> AsyncAnthropic:
        api_key = settings.api_key_for("anthropic")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        return AsyncAnthropic(api_key=api_key, timeout=settings.request_timeout_seconds)

    def _call_anthropic(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        client = self._anthropic_client()
        messages, prefill, system = self._anthropic_messages(system, user, response_model)
        try:
            resp = client.messages.create(
                model=settings.anthropic_model,
                system=system,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"anthropic call failed: {exc}") from exc
        text = "".join(block.text for block in resp.content if block.type == "text")
        return (prefill + text) if prefill else text

    async def _acall_anthropic(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        client = self._aanthropic_client()
        messages, prefill, system = self._anthropic_messages(system, user, response_model)
        try:
            resp = await client.messages.create(
                model=settings.anthropic_model,
                system=system,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"anthropic call failed: {exc}") from exc
        text = "".join(block.text for block in resp.content if block.type == "text")
        return (prefill + text) if prefill else text

    @staticmethod
    def _anthropic_messages(
        system: str, user: str, response_model: type[BaseModel] | None
    ) -> tuple[list[dict[str, str]], str | None, str]:
        messages = [{"role": "user", "content": user}]
        prefill = None
        if response_model is not None:
            system = system + _schema_instructions(response_model)
            prefill = "{"
            messages.append({"role": "assistant", "content": prefill})
        return messages, prefill, system

    # -------------------------------------------------------------- gemini
    def _gemini_client(self) -> genai.Client:
        api_key = settings.api_key_for("gemini")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        timeout_ms = settings.request_timeout_seconds * 1000
        return genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(timeout=timeout_ms))

    def _gemini_config(
        self, system: str, response_model: type[BaseModel] | None, temperature: float
    ) -> genai_types.GenerateContentConfig:
        if response_model is not None:
            system = system + _schema_instructions(response_model)
        kwargs: dict[str, Any] = {"system_instruction": system, "temperature": temperature}
        if response_model is not None:
            kwargs["response_mime_type"] = "application/json"
        return genai_types.GenerateContentConfig(**kwargs)

    def _call_gemini(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        client = self._gemini_client()
        config = self._gemini_config(system, response_model, temperature)
        try:
            resp = client.models.generate_content(model=settings.gemini_model, contents=user, config=config)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"gemini call failed: {exc}") from exc
        return resp.text or ""

    async def _acall_gemini(
        self, system: str, user: str, response_model: type[BaseModel] | None, temperature: float
    ) -> str:
        client = self._gemini_client()
        config = self._gemini_config(system, response_model, temperature)
        try:
            resp = await client.aio.models.generate_content(model=settings.gemini_model, contents=user, config=config)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"gemini call failed: {exc}") from exc
        return resp.text or ""

    # ----------------------------------------------------------- embedding
    def embed(self, texts: list[str]) -> list[list[float]]:
        if settings.embedding_provider == "openai":
            client = self._openai_client()
            try:
                resp = client.embeddings.create(model=settings.openai_embedding_model, input=texts)
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"openai embedding call failed: {exc}") from exc
            return [d.embedding for d in resp.data]
        if settings.embedding_provider == "local":
            return _local_embed(texts)
        raise ProviderError(f"unknown embedding provider: {settings.embedding_provider}")


_local_model_cache: dict[str, Any] = {}


def _local_embed(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    name = settings.local_embedding_model
    if name not in _local_model_cache:
        _local_model_cache[name] = SentenceTransformer(name)
    model = _local_model_cache[name]
    return model.encode(texts, convert_to_numpy=True).tolist()
