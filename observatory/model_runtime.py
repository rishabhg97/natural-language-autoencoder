#!/usr/bin/env python3
"""Shared one-GPU model runtime for the offline NLA Observatory."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .common import ObservatoryConfigError


def hf_checkpoint_complete(path: Path) -> bool:
    import json

    if not (path / "config.json").is_file():
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            files = {str(value) for value in (index.get("weight_map") or {}).values()}
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return bool(files) and all(
            (path / filename).is_file() and (path / filename).stat().st_size > 0
            for filename in files
        )
    files = list(path.glob("*.safetensors"))
    return bool(files) and all(file.stat().st_size > 0 for file in files)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def selected_rows(
    rows: list[dict[str, Any]], row_ids: list[str], *, exact: bool = True
) -> list[dict[str, Any]]:
    by_id = {f"validation-{int(row['row_index'])}": row for row in rows}
    missing = [row_id for row_id in row_ids if row_id not in by_id]
    if missing and exact:
        raise ObservatoryConfigError(f"selected source rows are missing: {missing}")
    return [by_id[row_id] for row_id in row_ids if row_id in by_id]


def load_train_mean(prediction_cache: Path) -> np.ndarray:
    with np.load(prediction_cache, allow_pickle=False) as cache:
        mean = np.asarray(cache["train_mean"], dtype=np.float32)
    if mean.shape != (2688,) or not np.isfinite(mean).all():
        raise ObservatoryConfigError(
            f"invalid train mean in {prediction_cache}: shape={mean.shape}"
        )
    return mean


def load_av_model(checkpoint: Path, *, torch_dtype: str, device_map: str) -> tuple[Any, Any]:
    from transformers import PreTrainedTokenizerFast

    from eval_nano_av_miles_checkpoint import _check_hf_checkpoint
    from eval_nano_ar_miles_checkpoint import _patch_remote_code_for_eval
    from nano_introspection import load_model_from_args

    _check_hf_checkpoint(checkpoint)
    _patch_remote_code_for_eval(checkpoint)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args = SimpleNamespace(
        model_id=str(checkpoint),
        model_revision=None,
        tokenizer_revision=None,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=None,
        load_mode="full",
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model = load_model_from_args(args).eval()
    return model, tokenizer


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def causal_token_logprobs(
    logits: Any,
    input_ids: Any,
    *,
    label_starts: list[int],
    lengths: list[int],
) -> list[dict[str, Any]]:
    """Extract causal token log-probabilities over each response span."""

    import torch

    shifted_logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    shifted_targets = input_ids[:, 1:].to(device=shifted_logprobs.device)
    gathered = shifted_logprobs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
    output: list[dict[str, Any]] = []
    for row_index, (label_start, length) in enumerate(
        zip(label_starts, lengths, strict=True)
    ):
        first_target = max(1, int(label_start))
        positions = list(range(first_target, int(length)))
        values = [
            float(gathered[row_index, position - 1].detach().cpu())
            for position in positions
        ]
        token_ids = [int(input_ids[row_index, position].detach().cpu()) for position in positions]
        if not values:
            raise ObservatoryConfigError(
                f"response span is empty: label_start={label_start} length={length}"
            )
        output.append(
            {
                "positions": positions,
                "token_ids": token_ids,
                "logprobs": values,
                "loss": float(-np.mean(values)),
                "target_tokens": len(values),
            }
        )
    return output


def teacher_forced_scores(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    rows: list[dict[str, Any]],
    vectors: list[Any | None],
    *,
    injection_scale: float | None,
    max_target_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score AV targets in right-padded batches and retain per-token evidence."""

    import torch

    from nano_av_warmstart_smoke import (
        _with_injection_char,
        build_loss_sequence,
        injected_embeddings,
    )

    if len(rows) != len(vectors):
        raise ObservatoryConfigError("rows and vectors must have equal lengths")
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start : start + batch_size]
        chunk_vectors = vectors[start : start + batch_size]
        examples: list[tuple[Any, Any, int, str]] = []
        for row, vector in zip(chunk_rows, chunk_vectors, strict=True):
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or not all(
                isinstance(message, dict)
                and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str)
                for message in prompt
            ):
                raise ObservatoryConfigError(
                    "AV source prompt must be a native list of role/content messages"
                )
            messages = _with_injection_char(prompt, cfg.injection_char)
            input_ids, label_start, mode = build_loss_sequence(
                tokenizer,
                messages,
                row["response"],
                max_target_tokens=max_target_tokens,
            )
            ids, embeds = injected_embeddings(
                model,
                input_ids,
                vector,
                cfg=cfg,
                injection_scale=injection_scale,
            )
            examples.append((ids, embeds, label_start, mode))
        max_length = max(int(ids.shape[1]) for ids, _, _, _ in examples)
        device = examples[0][0].device
        embedding_dtype = examples[0][1].dtype
        hidden_size = int(examples[0][1].shape[-1])
        input_ids = torch.full(
            (len(examples), max_length),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        embeddings = torch.zeros(
            (len(examples), max_length, hidden_size),
            dtype=embedding_dtype,
            device=device,
        )
        lengths: list[int] = []
        label_starts: list[int] = []
        modes: list[str] = []
        for index, (ids, embeds, label_start, mode) in enumerate(examples):
            length = int(ids.shape[1])
            input_ids[index, :length] = ids[0]
            attention_mask[index, :length] = 1
            embeddings[index, :length] = embeds[0]
            lengths.append(length)
            label_starts.append(int(label_start))
            modes.append(mode)
        with torch.no_grad():
            output = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                use_cache=False,
            )
        scores = causal_token_logprobs(
            output.logits,
            input_ids,
            label_starts=label_starts,
            lengths=lengths,
        )
        for row, score, mode in zip(chunk_rows, scores, modes, strict=True):
            score.update(
                {
                    "row_index": int(row["row_index"]),
                    "doc_id": str(row["doc_id"]),
                    "tokenization_mode": mode,
                    "tokens": [tokenizer.decode([token]) for token in score["token_ids"]],
                }
            )
            records.append(score)
    return records


def control_vectors(
    rows: list[dict[str, Any]], train_mean: np.ndarray
) -> list[dict[str, Any | None]]:
    import torch

    real = [torch.tensor(row["activation_vector"], dtype=torch.float32) for row in rows]
    if len(real) < 2:
        raise ObservatoryConfigError("at least two rows are required for shuffled controls")
    mean = torch.tensor(train_mean, dtype=torch.float32)
    zero = torch.zeros_like(mean)
    output: list[dict[str, Any | None]] = []
    for index, vector in enumerate(real):
        output.append(
            {
                "real": vector,
                "shuffled": real[(index + 1) % len(real)],
                "zero": zero,
                "mean": mean,
                "none": None,
            }
        )
    return output


def compare_score_batches(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ObservatoryConfigError("batch-equivalence row counts differ")
    loss_deltas: list[float] = []
    token_deltas: list[float] = []
    token_details: list[dict[str, Any]] = []
    for first, second in zip(left, right, strict=True):
        if first["row_index"] != second["row_index"]:
            raise ObservatoryConfigError("batch-equivalence row order differs")
        if first["token_ids"] != second["token_ids"]:
            raise ObservatoryConfigError("batch-equivalence target tokens differ")
        loss_deltas.append(abs(float(first["loss"]) - float(second["loss"])))
        for offset, (a, b) in enumerate(
            zip(first["logprobs"], second["logprobs"], strict=True)
        ):
            delta = abs(float(a) - float(b))
            token_deltas.append(delta)
            token_details.append(
                {
                    "row_index": int(first["row_index"]),
                    "response_offset": offset,
                    "position": int(first["positions"][offset]),
                    "token_id": int(first["token_ids"][offset]),
                    "batch1_logprob": float(a),
                    "batched_logprob": float(b),
                    "abs_delta": delta,
                }
            )
    max_detail = max(token_details, key=lambda item: item["abs_delta"], default=None)
    return {
        "rows": len(left),
        "tokens": len(token_deltas),
        "max_abs_loss_delta": max(loss_deltas, default=0.0),
        "mean_abs_loss_delta": float(np.mean(loss_deltas)) if loss_deltas else 0.0,
        "max_abs_token_logprob_delta": max(token_deltas, default=0.0),
        "mean_abs_token_logprob_delta": float(np.mean(token_deltas)) if token_deltas else 0.0,
        "p95_abs_token_logprob_delta": float(np.quantile(token_deltas, 0.95))
        if token_deltas
        else 0.0,
        "p99_abs_token_logprob_delta": float(np.quantile(token_deltas, 0.99))
        if token_deltas
        else 0.0,
        "max_token": max_detail,
    }


def rowwise_reconstruction_metrics(
    predictions: np.ndarray, targets: np.ndarray, *, eps: float = 1e-12
) -> dict[str, np.ndarray]:
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ObservatoryConfigError(
            f"prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    prediction_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    prediction_unit = prediction / np.maximum(prediction_norm[:, None], eps)
    target_unit = target / np.maximum(target_norm[:, None], eps)
    return {
        "directional_mse": np.square(prediction_unit - target_unit).sum(axis=1),
        "raw_mse": np.square(prediction - target).mean(axis=1),
        "cosine": np.sum(prediction_unit * target_unit, axis=1),
        "norm_ratio": prediction_norm / np.maximum(target_norm, eps),
    }


def write_prediction_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("cell_id", pa.string()),
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("content_family_id", pa.string()),
            ("family", pa.string()),
            ("variant", pa.string()),
            ("depth", pa.string()),
            ("critic", pa.string()),
            ("directional_mse", pa.float32()),
            ("raw_mse", pa.float32()),
            ("cosine", pa.float32()),
            ("norm_ratio", pa.float32()),
            ("prediction_vector", pa.list_(pa.float16(), 2688)),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
    )
    temporary.replace(path)


def _sample_next_tokens(
    logits: Any,
    *,
    temperature: float,
    top_p: float,
    generators: list[Any],
) -> tuple[Any, list[float]]:
    import torch

    if temperature <= 0.0 or not 0.0 < top_p <= 1.0:
        raise ObservatoryConfigError("sampling requires temperature > 0 and 0 < top_p <= 1")
    scaled = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    probabilities = torch.softmax(sorted_logits, dim=-1)
    chosen_sorted: list[Any] = []
    chosen_logprobs: list[float] = []
    for row_index, generator in enumerate(generators):
        sampled = torch.multinomial(
            probabilities[row_index], 1, generator=generator
        )
        chosen_sorted.append(sampled)
        chosen_logprobs.append(
            float(torch.log(probabilities[row_index, sampled]).detach().cpu())
        )
    sampled_sorted = torch.stack(chosen_sorted).view(-1, 1)
    next_ids = sorted_indices.gather(1, sampled_sorted).squeeze(1)
    return next_ids, chosen_logprobs


def sample_generate_batch_full_prefix(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    row: dict[str, Any],
    vector: Any,
    *,
    seeds: list[int],
    injection_scale: float | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_text: str | None,
) -> list[dict[str, Any]]:
    """Sample same-prompt AV tellings with the verified full-prefix backend."""

    import torch

    from nano_av_warmstart_smoke import (
        _with_injection_char,
        build_prompt_ids,
        injected_embeddings,
    )

    messages = _with_injection_char(row["prompt"], cfg.injection_char)
    prompt_ids = build_prompt_ids(tokenizer, messages)
    _, embeds = injected_embeddings(
        model,
        prompt_ids,
        vector,
        cfg=cfg,
        injection_scale=injection_scale,
    )
    batch_size = len(seeds)
    current_embeds = embeds.repeat(batch_size, 1, 1)
    attention_mask = torch.ones(
        current_embeds.shape[:2], dtype=torch.long, device=current_embeds.device
    )
    generators = [
        torch.Generator(device=current_embeds.device).manual_seed(int(seed))
        for seed in seeds
    ]
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    token_ids: list[list[int]] = [[] for _ in seeds]
    token_logprobs: list[list[float]] = [[] for _ in seeds]
    done = [False for _ in seeds]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            next_ids, logprobs = _sample_next_tokens(
                output.logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                generators=generators,
            )
            for index, token_id in enumerate(next_ids.detach().cpu().tolist()):
                if done[index]:
                    continue
                token_ids[index].append(int(token_id))
                token_logprobs[index].append(float(logprobs[index]))
                if int(token_id) in eos_ids:
                    done[index] = True
                elif stop_text and stop_text in tokenizer.decode(
                    token_ids[index], skip_special_tokens=True
                ):
                    done[index] = True
            if all(done):
                break
            next_embeds = model.get_input_embeddings()(next_ids[:, None]).to(
                dtype=current_embeds.dtype
            )
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
    return [
        {
            "text": tokenizer.decode(ids, skip_special_tokens=True),
            "token_ids": ids,
            "token_logprobs": logprobs,
            "steps": len(ids),
            "seed": int(seed),
        }
        for ids, logprobs, seed in zip(token_ids, token_logprobs, seeds, strict=True)
    ]


def topk_distribution(logits: Any, *, k: int) -> dict[str, list[float] | list[int]]:
    import torch

    probabilities = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probabilities, min(k, probabilities.shape[-1]))
    return {
        "token_ids": [int(value) for value in indices.detach().cpu().tolist()],
        "probabilities": [float(value) for value in values.detach().cpu().tolist()],
    }


def run_functional_pass_detailed(
    *,
    model: Any,
    boundary_module: Any,
    entries: list[dict[str, Any]],
    original_logits: dict[tuple[Any, ...], Any],
    batch_size: int,
    pad_token_id: int,
    top_k: int,
) -> list[dict[str, Any]]:
    import torch

    from nano_eval_core import functional_logit_metrics
    from nano_r33_functional_runtime import _module_device, _pad_prefixes, _patched_forward

    start_device = _module_device(model.get_input_embeddings())
    output: list[dict[str, Any]] = []
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        input_ids, attention_mask, positions = _pad_prefixes(
            [entry["prefix"] for entry in chunk],
            pad_token_id=pad_token_id,
            device=start_device,
        )
        replacements = torch.stack([entry["replacement"] for entry in chunk])
        patched = _patched_forward(
            model,
            boundary_module,
            input_ids,
            attention_mask,
            positions,
            replacements,
        ).detach().float().cpu()
        for offset, entry in enumerate(chunk):
            key = tuple(entry["provenance_key"])
            baseline = original_logits[key].float().cpu()
            output.append(
                {
                    "split": str(entry["split"]),
                    "row_index": int(entry["row_index"]),
                    "provenance_key": list(key),
                    "content_family_id": str(entry.get("content_family_id") or ""),
                    "variant": str(entry["variant"]),
                    "cell_id": entry.get("cell_id"),
                    "metrics": functional_logit_metrics(
                        baseline.numpy(), patched[offset].numpy()
                    ),
                    "original_topk": topk_distribution(baseline, k=top_k),
                    "patched_topk": topk_distribution(patched[offset], k=top_k),
                }
            )
    return output


def greedy_generate_unpatched(
    *,
    model: Any,
    tokenizer: Any,
    prefix: list[int],
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int | list[int] | None,
    backend: str = "full_prefix",
) -> list[int]:
    """Greedy continuation with a selectable, auditable generation backend."""

    import torch

    from nano_av_generation import greedy_generate_with_cache
    from nano_r33_functional_runtime import _model_logits, _module_device

    if not prefix or max_new_tokens < 1:
        raise ObservatoryConfigError("greedy generation requires a prefix and budget")
    device = _module_device(model.get_input_embeddings())
    if backend == "full_prefix":
        eos_ids = (
            set()
            if eos_token_id is None
            else {int(eos_token_id)}
            if isinstance(eos_token_id, int)
            else {int(value) for value in eos_token_id}
        )
        tokens = list(prefix)
        generated: list[int] = []
        for _ in range(int(max_new_tokens)):
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            next_id = int(
                _model_logits(output)[0, -1].argmax(dim=-1).detach().cpu().item()
            )
            generated.append(next_id)
            tokens.append(next_id)
            if next_id in eos_ids:
                break
        return generated
    if backend != "explicit_cache":
        raise ObservatoryConfigError(
            f"unsupported functional generation backend: {backend}"
        )
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        initial_embeds = model.get_input_embeddings()(input_ids)
        result = greedy_generate_with_cache(
            model,
            tokenizer,
            initial_embeds=initial_embeds,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=eos_token_id,
        )
    if max_new_tokens > 1 and not result.cache_used:
        reason = result.fallback_reason or "cache_not_used"
        raise ObservatoryConfigError(
            f"explicit cache unavailable for baseline generation: {reason}"
        )
    return result.token_ids


def greedy_generate_patched_cached(
    *,
    model: Any,
    boundary_module: Any,
    tokenizer: Any,
    prefix: list[int],
    replacement: Any,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int | list[int] | None,
) -> list[int]:
    """Greedy continuation after one boundary replacement in the cached prefill."""

    import torch

    from nano_av_generation import greedy_generate_with_cache
    from nano_r33_functional_core import make_boundary_replacement_hook
    from nano_r33_functional_runtime import _module_device

    if not prefix or max_new_tokens < 1:
        raise ObservatoryConfigError("patched generation requires a prefix and budget")
    device = _module_device(model.get_input_embeddings())
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        initial_embeds = model.get_input_embeddings()(input_ids)
    positions = torch.tensor([len(prefix) - 1], dtype=torch.long, device=device)
    replacement_batch = replacement.reshape(1, -1)
    replacement_hook = make_boundary_replacement_hook(
        replacement_batch, positions=positions
    )
    applied = False

    def patch_prefill(module: Any, inputs: Any, output: Any) -> Any:
        nonlocal applied
        if applied:
            return output
        applied = True
        return replacement_hook(module, inputs, output)

    handle = boundary_module.register_forward_hook(patch_prefill)
    try:
        with torch.no_grad():
            result = greedy_generate_with_cache(
                model,
                tokenizer,
                initial_embeds=initial_embeds,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                eos_token_id=eos_token_id,
            )
    finally:
        handle.remove()
    if not applied:
        raise ObservatoryConfigError("boundary hook did not fire during cached generation")
    if max_new_tokens > 1 and not result.cache_used:
        reason = result.fallback_reason or "cache_not_used"
        raise ObservatoryConfigError(
            f"explicit cache unavailable for patched generation: {reason}"
        )
    return result.token_ids


def greedy_generate_patched_full_prefix(
    *,
    model: Any,
    boundary_module: Any,
    prefix: list[int],
    replacement: Any,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
) -> list[int]:
    """Reference greedy generation that replays and patches the full prefix."""

    import torch

    from nano_r33_functional_core import make_boundary_replacement_hook
    from nano_r33_functional_runtime import _model_logits, _module_device

    if not prefix or max_new_tokens < 1:
        raise ObservatoryConfigError("reference generation requires a prefix and budget")
    eos_ids = (
        set()
        if eos_token_id is None
        else {int(eos_token_id)}
        if isinstance(eos_token_id, int)
        else {int(value) for value in eos_token_id}
    )
    device = _module_device(model.get_input_embeddings())
    tokens = list(prefix)
    generated: list[int] = []
    original_position = len(prefix) - 1
    for _ in range(int(max_new_tokens)):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        positions = torch.tensor([original_position], dtype=torch.long, device=device)
        hook = boundary_module.register_forward_hook(
            make_boundary_replacement_hook(replacement.reshape(1, -1), positions=positions)
        )
        try:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            hook.remove()
        token = int(torch.argmax(_model_logits(output)[0, -1]).detach().cpu())
        generated.append(token)
        tokens.append(token)
        if token in eos_ids:
            break
    return generated


def functional_wake_metrics(
    *,
    model: Any,
    boundary_module: Any,
    prefix: list[int],
    baseline_continuation: list[int],
    replacement: Any,
    wake_positions: int,
    baseline_position_logits: Any | None = None,
) -> list[dict[str, float | int]]:
    """Compare baseline and patched logits along a fixed baseline continuation."""

    import torch

    from nano_eval_core import functional_logit_metrics
    from nano_r33_functional_core import make_boundary_replacement_hook
    from nano_r33_functional_runtime import _model_logits, _module_device

    count = min(int(wake_positions), len(baseline_continuation))
    if not prefix or count < 1:
        raise ObservatoryConfigError("wake evaluation requires continuation tokens")
    sequence = [*prefix, *baseline_continuation[:count]]
    device = _module_device(model.get_input_embeddings())
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    original_position = len(prefix) - 1
    positions = torch.tensor([original_position], dtype=torch.long, device=device)
    if baseline_position_logits is None:
        with torch.no_grad():
            baseline_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        baseline_logits = (
            _model_logits(baseline_output)[0, original_position : original_position + count]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    else:
        baseline_logits = np.asarray(baseline_position_logits, dtype=np.float32)
        if baseline_logits.ndim != 2 or baseline_logits.shape[0] < count:
            raise ObservatoryConfigError("cached baseline wake logits have invalid shape")
        baseline_logits = baseline_logits[:count]
    hook = boundary_module.register_forward_hook(
        make_boundary_replacement_hook(replacement.reshape(1, -1), positions=positions)
    )
    try:
        with torch.no_grad():
            patched_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        hook.remove()
    patched_logits = (
        _model_logits(patched_output)[0, original_position : original_position + count]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    output: list[dict[str, float | int]] = []
    vocabulary_size = int(baseline_logits.shape[-1])
    top_ks = tuple(sorted({min(10, vocabulary_size), min(50, vocabulary_size)}))
    for offset in range(count):
        output.append(
            {
                "offset": offset + 1,
                "token_id": int(baseline_continuation[offset]),
                **functional_logit_metrics(
                    baseline_logits[offset],
                    patched_logits[offset],
                    top_ks=top_ks,
                ),
            }
        )
    return output


def baseline_wake_logits(
    *,
    model: Any,
    prefix: list[int],
    baseline_continuation: list[int],
    wake_positions: int,
) -> Any:
    """Compute reusable baseline logits for a fixed wake continuation."""

    import torch

    from nano_r33_functional_runtime import _model_logits, _module_device

    count = min(int(wake_positions), len(baseline_continuation))
    if not prefix or count < 1:
        raise ObservatoryConfigError("wake baseline requires continuation tokens")
    sequence = [*prefix, *baseline_continuation[:count]]
    device = _module_device(model.get_input_embeddings())
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    original_position = len(prefix) - 1
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return (
        _model_logits(output)[0, original_position : original_position + count]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def select_trajectory_positions(
    length: int, *, minimum_context: int, count: int
) -> list[int]:
    if length < minimum_context or minimum_context < 1 or count < 1:
        raise ObservatoryConfigError(
            f"invalid trajectory selection: length={length} minimum={minimum_context} count={count}"
        )
    positions = np.linspace(minimum_context - 1, length - 1, min(count, length - minimum_context + 1))
    return sorted({int(round(value)) for value in positions.tolist()})


def write_trajectory_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("doc_id", pa.string()),
            ("content_family_id", pa.string()),
            ("position", pa.int64()),
            ("n_context_tokens", pa.int64()),
            ("token_id", pa.int64()),
            ("token_text", pa.string()),
            ("activation_vector", pa.list_(pa.float16(), 2688)),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
    )
    temporary.replace(path)
