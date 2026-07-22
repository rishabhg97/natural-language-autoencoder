#!/usr/bin/env python3
"""Async local Stage 2 teacher labeling with Nemotron Super reasoning.

This is a local/Mac-friendly companion to ``nla.datagen.stage2_api_explain``:

- reads a base parquet with Nano activations and source prefixes;
- sends source prefixes to an OpenAI-compatible NVIDIA endpoint;
- enables Super reasoning with a configurable reasoning budget;
- extracts only the final tagged answer, never the hidden reasoning trace;
- writes crash-resumable chunk parquet files and merges them into one explained
  parquet with an NLA sidecar.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "natural_language_autoencoders"
if str(EXTERNAL) not in sys.path:
    sys.path.insert(0, str(EXTERNAL))

from nla.datagen.sidecar import NLAApiSummaryMeta, read_sidecar_local, write_sidecar_local  # noqa: E402


DEFAULT_ENDPOINT = "https://inference-api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nvidia/nemotron-3-super-v3"
DEFAULT_PROMPT = ROOT / "prompts" / "super_teacher_predictive_features.txt"
TOLERATED_STATUS = {408, 409, 429, 500, 502, 503, 504}
TAG_RE = re.compile(r"<\s*(analysis|explanation)\s*>(.*?)<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+•]|\d+[\.)])\s+")
FEATURE_PREFIX_RE = re.compile(r"^\s*(?:feature\s*)?\d+\s*[:.)-]\s*", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAnalysis:
    text: str
    source: str
    feature_count: int


@dataclass(frozen=True)
class RowResult:
    parsed: ParsedAnalysis | None
    error: str | None = None
    status_code: int | None = None


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
            elif block is not None:
                parts.append(str(block))
        return "".join(parts)
    return "" if content is None else str(content)


def _clean_feature_lines(block: str) -> list[str]:
    lines: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = LIST_PREFIX_RE.sub("", line)
        line = FEATURE_PREFIX_RE.sub("", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = line.strip().strip("*_").strip()
        if line:
            lines.append(line)
    return lines


def extract_teacher_analysis(raw: str, *, source: str = "analysis", min_features: int = 3) -> ParsedAnalysis | None:
    """Extract the last complete analysis/explanation block from a teacher response."""
    matches = list(TAG_RE.finditer(raw or ""))
    if not matches:
        return None
    block = matches[-1].group(2)
    lines = _clean_feature_lines(block)
    if len(lines) < min_features:
        return None
    return ParsedAnalysis(text="\n\n".join(lines), source=source, feature_count=len(lines))


def parse_chat_completion(payload: dict[str, Any], *, min_features: int = 3) -> ParsedAnalysis | None:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"chat completion response has no choices: {payload!r}")
    first = choices[0]
    if first.get("finish_reason") == "content_filter":
        return None
    message = first.get("message") or {}

    # Prefer final answer content. Only fall back to reasoning_content if the
    # endpoint fails to put the final tagged answer in content.
    content = _message_text(message.get("content"))
    parsed = extract_teacher_analysis(content, source="content", min_features=min_features)
    if parsed is not None:
        return parsed

    reasoning_content = _message_text(message.get("reasoning_content"))
    if reasoning_content:
        return extract_teacher_analysis(reasoning_content, source="reasoning_content", min_features=min_features)
    return None


def build_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    reasoning_budget: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if enable_thinking:
        payload["reasoning_effort"] = reasoning_effort
        payload["reasoning_budget"] = reasoning_budget
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "thinking": True,
            "reasoning_budget": reasoning_budget,
            "force_nonempty_content": True,
        }
    else:
        payload["reasoning_effort"] = "none"
        payload["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
    return payload


async def _post_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    max_retries: int,
    min_features: int,
) -> RowResult:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(endpoint, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < max_retries:
                    await asyncio.sleep(min(2**attempt, 30))
                    continue
                return RowResult(None, error=f"{type(exc).__name__}: {exc}")

            if resp.status_code in TOLERATED_STATUS:
                if attempt < max_retries:
                    await asyncio.sleep(min(2**attempt, 30))
                    continue
                return RowResult(None, error=f"retry_exhausted_status_{resp.status_code}", status_code=resp.status_code)
            if resp.status_code >= 400:
                snippet = resp.text[:500]
                raise httpx.HTTPStatusError(
                    f"chat completion failed with {resp.status_code}: {snippet}",
                    request=resp.request,
                    response=resp,
                )

            parsed = parse_chat_completion(resp.json(), min_features=min_features)
            if parsed is None:
                return RowResult(None, error="parse_failed", status_code=resp.status_code)
            return RowResult(parsed, status_code=resp.status_code)
    raise AssertionError("unreachable")


async def complete_prompts(
    prompts: list[str],
    *,
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    reasoning_budget: int,
    enable_thinking: bool,
    concurrency: int,
    timeout: float,
    max_retries: int,
    min_features: int,
) -> list[RowResult]:
    sem = asyncio.Semaphore(concurrency)
    timeout_cfg = httpx.Timeout(timeout, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        tasks = []
        for prompt in prompts:
            payload = build_payload(
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_budget=reasoning_budget,
                enable_thinking=enable_thinking,
            )
            tasks.append(
                _post_one(
                    client,
                    sem,
                    endpoint=endpoint,
                    api_key=api_key,
                    payload=payload,
                    max_retries=max_retries,
                    min_features=min_features,
                )
            )
        return await asyncio.gather(*tasks)


def _slice_table(path: Path, row_offset: int, row_limit: int | None) -> pa.Table:
    table = pq.read_table(path)
    if row_offset:
        table = table.slice(row_offset)
    if row_limit is not None:
        table = table.slice(0, row_limit)
    return table


def _append_explanations(chunk: pa.Table, results: list[RowResult]) -> tuple[pa.Table, dict[str, int]]:
    keep_mask: list[bool] = []
    explanations: list[str] = []
    counts = {"kept": 0, "dropped": 0, "content": 0, "reasoning_content": 0, "parse_failed": 0, "request_failed": 0}
    for result in results:
        if result.parsed is None:
            keep_mask.append(False)
            counts["dropped"] += 1
            if result.error == "parse_failed":
                counts["parse_failed"] += 1
            else:
                counts["request_failed"] += 1
            continue
        keep_mask.append(True)
        explanations.append(result.parsed.text)
        counts["kept"] += 1
        counts[result.parsed.source] = counts.get(result.parsed.source, 0) + 1
    if not all(keep_mask):
        chunk = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
    return chunk.append_column("api_explanation", pa.array(explanations, type=pa.string())), counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.getenv(args.api_key_env) or os.getenv("NLA_STAGE2_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(f"missing API key: set {args.api_key_env}, NLA_STAGE2_API_KEY, or OPENAI_API_KEY")
    prompt_template = args.prompt_file.read_text()
    if "{text}" not in prompt_template:
        raise SystemExit(f"prompt file must contain {{text}} placeholder: {args.prompt_file}")

    table = _slice_table(args.input, args.row_offset, args.row_limit)
    if "detokenized_text_truncated" not in table.column_names:
        raise SystemExit("input parquet must contain detokenized_text_truncated")
    if table.num_rows == 0:
        raise SystemExit("selected input slice is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = Path(f"{args.output}.chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_output or args.output.with_suffix(args.output.suffix + ".report.json")

    chunk_paths: list[Path] = []
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "row_offset": args.row_offset,
        "row_limit": args.row_limit,
        "selected_rows": table.num_rows,
        "chunk_size": args.chunk_size,
        "model": args.model,
        "endpoint": args.endpoint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort if args.enable_thinking else "none",
        "reasoning_budget": args.reasoning_budget if args.enable_thinking else 0,
        "enable_thinking": args.enable_thinking,
        "concurrency": args.concurrency,
        "min_features": args.min_features,
        "chunks_total": 0,
        "chunks_skipped": 0,
        "kept": 0,
        "dropped": 0,
        "content": 0,
        "reasoning_content": 0,
        "parse_failed": 0,
        "request_failed": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for chunk_start in range(0, table.num_rows, args.chunk_size):
        summary["chunks_total"] += 1
        chunk_path = chunks_dir / f"chunk_{chunk_start:08d}.parquet"
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            summary["chunks_skipped"] += 1
            existing = pq.read_table(chunk_path)
            summary["kept"] += existing.num_rows
            continue

        chunk = table.slice(chunk_start, args.chunk_size)
        texts = chunk.column("detokenized_text_truncated").to_pylist()
        prompts = [prompt_template.format(text=text) for text in texts]
        results = asyncio.run(
            complete_prompts(
                prompts,
                endpoint=args.endpoint,
                api_key=api_key,
                model=args.model,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                reasoning_budget=args.reasoning_budget,
                enable_thinking=args.enable_thinking,
                concurrency=args.concurrency,
                timeout=args.timeout,
                max_retries=args.max_retries,
                min_features=args.min_features,
            )
        )
        chunk_out, counts = _append_explanations(chunk, results)
        tmp = chunk_path.with_suffix(".tmp")
        pq.write_table(chunk_out, tmp)
        tmp.rename(chunk_path)
        for key, value in counts.items():
            summary[key] = int(summary.get(key, 0)) + int(value)
        summary["last_completed_chunk_start"] = chunk_start
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(
            f"chunk {chunk_start:08d}: kept={counts['kept']} dropped={counts['dropped']} "
            f"content={counts.get('content', 0)} reasoning={counts.get('reasoning_content', 0)}",
            flush=True,
        )

    row_count = 0
    out_schema: pa.Schema | None = None
    with pq.ParquetWriter(args.output, pq.read_table(chunk_paths[0]).schema) as writer:
        for path in chunk_paths:
            chunk = pq.read_table(path)
            if out_schema is None:
                out_schema = chunk.schema
            writer.write_table(chunk)
            row_count += chunk.num_rows

    in_meta = read_sidecar_local(args.input)
    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__super_thinking_explained",
        row_count=row_count,
        api_summaries=NLAApiSummaryMeta(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            instruction_prompt=prompt_template,
        ),
        parent_datasets=[in_meta.dataset_id],
        created_by="scripts.nano_stage2_super_teacher",
        created_at="",
        git_commit="",
    )
    write_sidecar_local(args.output, out_meta)

    summary["row_count"] = row_count
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--system-prompt", default="You are a careful teacher labeling predictive features for model activations.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", default="high", choices=["none", "low", "high"])
    parser.add_argument("--reasoning-budget", type=int, default=4096)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--min-features", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
