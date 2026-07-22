"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import httpx


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        # refusal: source text tripped safety — no answer coming, drop this row.
        # content may be [] or the refusal message; either way, no explanation.
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out


def _chat_completions_endpoint(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL or full endpoint."""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _chat_completion_text(payload: dict) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        raise AssertionError(f"chat completion response has no choices: {payload!r}")
    first = choices[0]
    finish_reason = first.get("finish_reason")
    if finish_reason == "content_filter":
        return None
    message = first.get("message") or {}
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
            elif block is not None:
                parts.append(str(block))
        content = "".join(parts)
    if (
        finish_reason != "length"
        and (content is None or content == "")
        and isinstance(reasoning_content, str)
        and reasoning_content.strip()
    ):
        # NVIDIA-hosted reasoning models such as Kimi can return the full
        # answer in reasoning_content while leaving message.content null.
        content = reasoning_content
    if not isinstance(content, str):
        if content is None and finish_reason in {"length", "stop"}:
            return None
        raise AssertionError(f"chat completion response has no text content: {payload!r}")
    text = content.strip()
    assert text, "empty completion — refusing to emit blank explanation"
    return text


class OpenAIChatCompletionsProvider(CompletionProvider):
    """OpenAI-compatible chat completions provider.

    This supports endpoints such as NVIDIA's ``/v1/chat/completions`` API while
    preserving Stage 2's `CompletionProvider` interface. Credentials are read
    from explicit constructor args first, then OPENAI/NLA variables, then the
    ANTHROPIC variables commonly used by the reference pipeline.
    """

    _TOLERATED_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        model: str,
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        extra_body: dict[str, Any] | None = None,
    ):
        raw_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("NLA_STAGE2_BASE_URL")
            or os.getenv("ANTHROPIC_BASE_URL")
        )
        raw_api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("NLA_STAGE2_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if not raw_base_url:
            raise ValueError("OpenAIChatCompletionsProvider requires base_url, OPENAI_BASE_URL, NLA_STAGE2_BASE_URL, or ANTHROPIC_BASE_URL")
        if not raw_api_key:
            raise ValueError("OpenAIChatCompletionsProvider requires api_key, OPENAI_API_KEY, NLA_STAGE2_API_KEY, or ANTHROPIC_API_KEY")
        self.endpoint = _chat_completions_endpoint(raw_base_url)
        self.api_key = raw_api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})

    async def _one(self, client: httpx.AsyncClient, sem: asyncio.Semaphore, prompt: str) -> str | None:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(self.extra_body)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
                except (httpx.TimeoutException, httpx.TransportError):
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    return None

                if resp.status_code in self._TOLERATED_STATUS:
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 30))
                        continue
                    return None
                if resp.status_code >= 400:
                    snippet = resp.text[:500]
                    raise httpx.HTTPStatusError(
                        f"chat completions request failed with {resp.status_code}: {snippet}",
                        request=resp.request,
                        response=resp,
                    )
                return _chat_completion_text(resp.json())
        raise AssertionError("unreachable")

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient() as client:
                return await asyncio.gather(
                    *(self._one(client, sem, p) for p in prompts),
                    return_exceptions=True,
                )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, BaseException):
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [OpenAIChatCompletionsProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out
