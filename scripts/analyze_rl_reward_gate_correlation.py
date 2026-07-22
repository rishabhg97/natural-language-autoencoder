#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
NLA_ROOT = REPO_ROOT / "external" / "natural_language_autoencoders"
MILES_ROOT = Path(os.environ.get("MILES_ROOT", "/workspace/interp/code/miles-051cd15"))
INTERPRETATION = "fixed_policy_correlation_not_policy_gradient_proof"


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = rank
        i = j
    return ranks


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(x_var * y_var)
    if denom == 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / denom


def spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson_corr(_rank(xs), _rank(ys))


def pair_rewards_with_gate_losses(
    split: dict[str, Any],
    rewards_by_row_index: dict[int, float],
    *,
    variant: str = "av_real",
) -> dict[str, Any]:
    row_indices = [int(value) for value in split.get("row_indices", [])]
    losses = split.get("rowwise_normalized_mse", {}).get(variant, [])
    paired_rows: list[int] = []
    paired_rewards: list[float] = []
    paired_losses: list[float] = []
    for row_index, loss in zip(row_indices, losses):
        if row_index not in rewards_by_row_index:
            continue
        reward = rewards_by_row_index[row_index]
        loss_value = float(loss)
        if math.isfinite(reward) and math.isfinite(loss_value):
            paired_rows.append(row_index)
            paired_rewards.append(float(reward))
            paired_losses.append(loss_value)
    return {
        "row_indices": paired_rows,
        "rewards": paired_rewards,
        "gate_losses": paired_losses,
    }


def summarize_pairs(pairs: dict[str, Any]) -> dict[str, Any]:
    rewards = pairs["rewards"]
    gate_losses = pairs["gate_losses"]
    return {
        "paired_row_count": len(rewards),
        "pearson_reward_vs_gate_loss": pearson_corr(rewards, gate_losses),
        "spearman_reward_vs_gate_loss": spearman_corr(rewards, gate_losses),
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "gate_loss_mean": sum(gate_losses) / len(gate_losses) if gate_losses else None,
        "row_indices": pairs["row_indices"],
    }


def analyze_report_with_rewards(
    report: dict[str, Any],
    rewards_by_split: dict[str, dict[int, float]],
    *,
    variant: str = "av_real",
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    for split_name, split in report.get("splits", {}).items():
        pairs = pair_rewards_with_gate_losses(split, rewards_by_split.get(split_name, {}), variant=variant)
        summary = summarize_pairs(pairs)
        summary["variant"] = variant
        summary["negative_correlation_is_aligned"] = (
            summary["spearman_reward_vs_gate_loss"] is not None
            and summary["spearman_reward_vs_gate_loss"] < 0.0
        )
        split_summaries[split_name] = summary
    return {
        "schema_version": "nano_rl_reward_gate_correlation.v1",
        "interpretation": INTERPRETATION,
        "variant": variant,
        "splits": split_summaries,
    }


def _ensure_pad_token(tokenizer: Any) -> None:
    tokenizer.padding_side = "right"
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    for token_attr, id_attr in (("eos_token", "eos_token_id"), ("unk_token", "unk_token_id")):
        token = getattr(tokenizer, token_attr, None)
        token_id = getattr(tokenizer, id_attr, None)
        if token is not None:
            tokenizer.pad_token = token
            return
        if token_id is not None:
            tokenizer.pad_token_id = token_id
            return
    raise ValueError("critic tokenizer needs a pad/eos/unk token for batched padding")


def load_generated_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _real_explanation(record: dict[str, Any]) -> str | None:
    if str(NLA_ROOT) not in sys.path:
        sys.path.insert(0, str(NLA_ROOT))
    from nla.schema import extract_explanation

    real = (record.get("controls") or {}).get("real") or {}
    generated = str(real.get("generated") or "")
    explanation = extract_explanation(generated)
    if explanation is None:
        parsed = real.get("parsed") or {}
        explanation = parsed.get("explanation")
    if explanation is None:
        return None
    explanation = str(explanation).strip()
    return explanation or None


def _load_split_rows(validation_parquet: Path, test_parquet: Path) -> dict[str, list[dict[str, Any]]]:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from nano_av_warmstart_smoke import load_av_rows

        return {
            "validation": load_av_rows(validation_parquet),
            "test": load_av_rows(test_parquet),
        }
    finally:
        try:
            sys.path.remove(str(REPO_ROOT / "scripts"))
        except ValueError:
            pass


def recompute_rewards_for_generated_rows(
    *,
    generated_jsonl: Path,
    critic_checkpoint_dir: Path,
    validation_parquet: Path,
    test_parquet: Path,
    batch_size: int,
    device: str,
) -> dict[str, dict[int, float]]:
    if str(NLA_ROOT) not in sys.path:
        sys.path.insert(0, str(NLA_ROOT))
    if MILES_ROOT.exists() and str(MILES_ROOT) not in sys.path:
        sys.path.insert(0, str(MILES_ROOT))

    import torch
    from miles.utils.processing_utils import load_tokenizer
    from nla.config import load_nla_config
    from nla.models import NLACriticModel
    from nla.reward import _mse_to_reward
    from nla.train_actor import _normalize_mamba_time_step_limits, _temporarily_disable_mamba_fast_path

    os.environ.setdefault("NLA_CRITIC_FWD_DISABLE_MAMBA_FAST_PATH", "1")
    value_head = critic_checkpoint_dir / "value_head.safetensors"
    if not value_head.exists():
        raise FileNotFoundError(f"missing {value_head}; cannot recompute reward safely")

    records = load_generated_records(generated_jsonl)
    rows_by_split = _load_split_rows(validation_parquet, test_parquet)
    tokenizer = load_tokenizer(str(critic_checkpoint_dir), trust_remote_code=True)
    _ensure_pad_token(tokenizer)
    cfg = load_nla_config(str(critic_checkpoint_dir), tokenizer)
    if cfg.critic_prompt_template is None:
        raise ValueError(f"critic sidecar at {critic_checkpoint_dir} has no critic_prompt_template")

    model = NLACriticModel.from_pretrained(
        str(critic_checkpoint_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if getattr(model.value_head.weight, "is_meta", False):
        raise RuntimeError("critic value_head is still meta after load; reward recompute is unsafe")
    _normalize_mamba_time_step_limits(model)
    model.to(device)
    model.eval()

    rewards_by_split: dict[str, dict[int, float]] = {"validation": {}, "test": {}}
    examples: list[tuple[str, int, int, str]] = []
    for record in records:
        split = str(record.get("split"))
        if split not in rewards_by_split:
            continue
        row_index = int(record["row_index"])
        source_row_index = int(record.get("source_row_index", row_index))
        explanation = _real_explanation(record)
        if explanation is None:
            continue
        examples.append((split, row_index, source_row_index, cfg.critic_prompt_template.format(explanation=explanation)))

    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [item[3] for item in batch]
            tok = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
            input_ids = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)
            value_indices = attention_mask.sum(dim=1).to(torch.long) - 1
            with _temporarily_disable_mamba_fast_path(model):
                output = model(input_ids=input_ids, attention_mask=attention_mask, nla_value_indices=value_indices)
            pred = output.values.detach().float().cpu()
            gold_rows = [rows_by_split[split][source_row_index] for split, _row_index, source_row_index, _prompt in batch]
            gold = torch.tensor([row["activation_vector"] for row in gold_rows], dtype=torch.float32)
            rewards = _mse_to_reward(pred, gold, cfg.mse_scale)
            for (split, row_index, _source_row_index, _prompt), reward in zip(batch, rewards):
                rewards_by_split[split][row_index] = float(reward)
    return rewards_by_split


def run_analysis(
    *,
    roundtrip_report_json: Path,
    reward_loader: Callable[[], dict[str, dict[int, float]]],
    output_json: Path | None = None,
) -> dict[str, Any]:
    report = json.loads(roundtrip_report_json.read_text())
    rewards_by_split = reward_loader()
    summary = analyze_report_with_rewards(report, rewards_by_split)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roundtrip-report-json", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--critic-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    summary = run_analysis(
        roundtrip_report_json=args.roundtrip_report_json,
        reward_loader=lambda: recompute_rewards_for_generated_rows(
            generated_jsonl=args.generated_jsonl,
            critic_checkpoint_dir=args.critic_checkpoint_dir,
            validation_parquet=args.validation_parquet,
            test_parquet=args.test_parquet,
            batch_size=args.batch_size,
            device=args.device,
        ),
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
