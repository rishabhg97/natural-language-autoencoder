#!/usr/bin/env python3
"""Reusable AV text-generation helpers for Nano NLA evals."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


@dataclass(frozen=True)
class GenerationJob:
    row_index: int
    source_row_index: int
    split: str
    doc_id: str | None
    control_name: str
    control_vector: Any
    row: dict[str, Any]
    target_explanation: str

    @property
    def job_key(self) -> str:
        return f"{self.split}:{self.row_index}:{self.control_name}"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    cache_used: bool
    fallback_reason: str | None
    steps: int


@dataclass
class _GenerationCache:
    kind: str | None = None
    value: Any = None
    processed_tokens: int = 0
    fallback_reason: str | None = None


_TOKENIZER_EOS = object()


def plan_generation_jobs(
    *,
    rows_by_index: dict[int, dict[str, Any]],
    row_indices: Iterable[int],
    controls_by_row: dict[int, dict[str, Any]],
    controls_requested: Iterable[str],
    target_explanations: dict[int, str],
) -> list[GenerationJob]:
    jobs: list[GenerationJob] = []
    for raw_index in row_indices:
        row_index = int(raw_index)
        row = rows_by_index[row_index]
        controls = controls_by_row[row_index]
        for control_name in controls_requested:
            jobs.append(
                GenerationJob(
                    row_index=row_index,
                    source_row_index=int(row.get("source_row_index", row_index)),
                    split=str(row.get("split")),
                    doc_id=None if row.get("doc_id") is None else str(row.get("doc_id")),
                    control_name=str(control_name),
                    control_vector=controls[str(control_name)],
                    row=row,
                    target_explanation=str(target_explanations[row_index]),
                )
            )
    return jobs


def append_generation_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def iter_generation_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_completed_job_keys(path: Path) -> set[str]:
    completed: set[str] = set()
    for record in iter_generation_records(path):
        if record.get("status", "complete") == "complete" and record.get("job_key"):
            completed.add(str(record["job_key"]))
    return completed


def _token_id_from_argmax(torch: Any, logits: Any) -> tuple[int, Any]:
    next_id = torch.argmax(logits[:, -1, :], dim=-1)
    try:
        token_id = int(next_id[0].detach().cpu())
    except TypeError:
        token_id = int(next_id[0].detach().cpu().item())
    return token_id, next_id


def _embed_next_token(model: Any, next_id: Any, *, dtype: Any) -> Any:
    return model.get_input_embeddings()(next_id[:, None]).to(dtype=dtype)


def _token_ids_from_argmax_tensor(next_ids: Any) -> list[int]:
    values = next_ids.detach().cpu().tolist()
    if isinstance(values, int):
        return [int(values)]
    return [int(value) for value in values]


def _initial_generation_cache(model: Any, initial_embeds: Any) -> _GenerationCache:
    """Initialize remote-code cache objects that cannot be created lazily."""

    if (
        model.__class__.__name__ == "NemotronHForCausalLM"
        and os.environ.get("NLA_ENABLE_EXPERIMENTAL_NEMOTRON_CACHE") != "1"
    ):
        return _GenerationCache(
            kind="cache_params",
            fallback_reason="nemotron_cache_equivalence_unverified",
        )
    forward = getattr(model, "forward", None)
    if forward is None:
        return _GenerationCache()
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return _GenerationCache()
    if "cache_params" not in parameters:
        return _GenerationCache()

    module = sys.modules.get(model.__class__.__module__)
    if module is None:
        return _GenerationCache()
    cache_cls = getattr(module, "HybridMambaAttentionDynamicCache", None)
    if cache_cls is None:
        cache_cls = getattr(module, "NemotronHHybridDynamicCache", None)
    if cache_cls is None:
        return _GenerationCache()

    cache = cache_cls(
        model.config,
        int(initial_embeds.shape[0]),
        initial_embeds.dtype,
        device=initial_embeds.device,
    )
    return _GenerationCache(kind="cache_params", value=cache)


def _cache_call_kwargs(torch: Any, cache: _GenerationCache, cur_embeds: Any) -> dict[str, Any]:
    if cache.value is None:
        return {}
    if cache.kind == "cache_params":
        start = cache.processed_tokens
        stop = start + int(cur_embeds.shape[1])
        return {
            "cache_params": cache.value,
            "cache_position": torch.arange(start, stop, device=cur_embeds.device),
        }
    return {"past_key_values": cache.value}


def _next_generation_cache(out: Any, current: _GenerationCache, input_length: int) -> _GenerationCache:
    candidates = []
    if current.kind is not None:
        candidates.append(current.kind)
    candidates.extend(name for name in ("past_key_values", "cache_params") if name not in candidates)
    for kind in candidates:
        value = getattr(out, kind, None)
        if value is not None:
            return _GenerationCache(
                kind=kind,
                value=value,
                processed_tokens=current.processed_tokens + int(input_length),
                fallback_reason=current.fallback_reason,
            )
    return _GenerationCache(kind=current.kind, fallback_reason=current.fallback_reason)


def _missing_cache_reason(cache: _GenerationCache) -> str:
    if cache.fallback_reason:
        return cache.fallback_reason
    return "missing_cache_params" if cache.kind == "cache_params" else "missing_past_key_values"


def greedy_generate_with_cache(
    model: Any,
    tokenizer: Any,
    *,
    initial_embeds: Any,
    attention_mask: Any,
    max_new_tokens: int,
    generated_ids: list[int] | None = None,
    stop_text: str | None = None,
    eos_token_id: int | list[int] | None | object = _TOKENIZER_EOS,
) -> GenerationResult:
    """Greedy-decode from injected embeddings, using KV cache when available."""

    import torch

    eos = (
        getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is _TOKENIZER_EOS
        else eos_token_id
    )
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    token_ids = list(generated_ids or [])
    cache = _initial_generation_cache(model, initial_embeds)
    cur_embeds = initial_embeds
    cur_attention_mask = attention_mask
    cache_used = False
    fallback_reason = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            kwargs = {
                "inputs_embeds": cur_embeds,
                "attention_mask": cur_attention_mask,
                "use_cache": True,
            }
            if cache.value is not None and cache.processed_tokens > 0:
                cache_used = True
            kwargs.update(_cache_call_kwargs(torch, cache, cur_embeds))
            input_length = int(cur_embeds.shape[1])
            out = model(**kwargs)
            token_id, next_id = _token_id_from_argmax(torch, out.logits)
            token_ids.append(token_id)
            if token_id in eos_ids:
                break
            if stop_text and stop_text in tokenizer.decode(token_ids, skip_special_tokens=True):
                break

            next_embed = _embed_next_token(model, next_id, dtype=cur_embeds.dtype)
            cur_attention_mask = torch.cat(
                [
                    cur_attention_mask,
                    torch.ones(
                        (cur_attention_mask.shape[0], 1),
                        device=cur_attention_mask.device,
                        dtype=cur_attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            next_cache = _next_generation_cache(out, cache, input_length)
            if next_cache.value is None:
                fallback_reason = fallback_reason or _missing_cache_reason(cache)
                cache = next_cache
                cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            else:
                cache = next_cache
                cur_embeds = next_embed

    return GenerationResult(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        token_ids=token_ids,
        cache_used=cache_used,
        fallback_reason=fallback_reason,
        steps=len(token_ids) - len(generated_ids or []),
    )


def _greedy_generate_batch(
    model: Any,
    tokenizer: Any,
    *,
    initial_embeds: Any,
    attention_mask: Any,
    max_new_tokens: int,
    generated_ids: list[list[int]] | None = None,
    stop_text: str | None = None,
    use_cache: bool,
) -> list[GenerationResult]:
    import torch

    batch_size = int(initial_embeds.shape[0])
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    initial_ids = generated_ids or [[] for _ in range(batch_size)]
    token_ids = [list(ids) for ids in initial_ids]
    initial_lengths = [len(ids) for ids in token_ids]
    done = [False] * batch_size
    cache = _initial_generation_cache(model, initial_embeds) if use_cache else _GenerationCache()
    cur_embeds = initial_embeds
    cur_attention_mask = attention_mask
    cache_used = [False] * batch_size
    fallback_reason = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            kwargs = {
                "inputs_embeds": cur_embeds,
                "attention_mask": cur_attention_mask,
                "use_cache": use_cache,
            }
            if use_cache and cache.value is not None and cache.processed_tokens > 0:
                for index, is_done in enumerate(done):
                    if not is_done:
                        cache_used[index] = True
            if use_cache:
                kwargs.update(_cache_call_kwargs(torch, cache, cur_embeds))
            input_length = int(cur_embeds.shape[1])
            out = model(**kwargs)
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1)
            next_id_values = _token_ids_from_argmax_tensor(next_ids)
            for index, token_id in enumerate(next_id_values):
                if done[index]:
                    continue
                token_ids[index].append(token_id)
                if token_id in eos_ids:
                    done[index] = True
                elif stop_text and stop_text in tokenizer.decode(token_ids[index], skip_special_tokens=True):
                    done[index] = True
            if all(done):
                break

            next_embeds = _embed_next_token(model, next_ids, dtype=cur_embeds.dtype)
            cur_attention_mask = torch.cat(
                [
                    cur_attention_mask,
                    torch.ones(
                        (cur_attention_mask.shape[0], 1),
                        device=cur_attention_mask.device,
                        dtype=cur_attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            if use_cache:
                next_cache = _next_generation_cache(out, cache, input_length)
                if next_cache.value is None:
                    fallback_reason = fallback_reason or _missing_cache_reason(cache)
                    cache = next_cache
                    cur_embeds = torch.cat([cur_embeds, next_embeds], dim=1)
                else:
                    cache = next_cache
                    cur_embeds = next_embeds
            else:
                cur_embeds = torch.cat([cur_embeds, next_embeds], dim=1)

    return [
        GenerationResult(
            text=tokenizer.decode(ids, skip_special_tokens=True),
            token_ids=ids,
            cache_used=cache_used[index],
            fallback_reason=fallback_reason,
            steps=len(ids) - initial_lengths[index],
        )
        for index, ids in enumerate(token_ids)
    ]


def greedy_generate_batch_with_cache(
    model: Any,
    tokenizer: Any,
    *,
    initial_embeds: Any,
    attention_mask: Any,
    max_new_tokens: int,
    generated_ids: list[list[int]] | None = None,
    stop_text: str | None = None,
) -> list[GenerationResult]:
    """Greedy-decode a same-length batch, falling back when cache is unsafe."""

    return _greedy_generate_batch(
        model,
        tokenizer,
        initial_embeds=initial_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        generated_ids=generated_ids,
        stop_text=stop_text,
        use_cache=True,
    )


def greedy_generate_batch_full_prefix(
    model: Any,
    tokenizer: Any,
    *,
    initial_embeds: Any,
    attention_mask: Any,
    max_new_tokens: int,
    generated_ids: list[list[int]] | None = None,
    stop_text: str | None = None,
) -> list[GenerationResult]:
    """Greedy-decode a same-length batch by recomputing each full prefix."""

    return _greedy_generate_batch(
        model,
        tokenizer,
        initial_embeds=initial_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        generated_ids=generated_ids,
        stop_text=stop_text,
        use_cache=False,
    )
