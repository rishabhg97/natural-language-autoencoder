"""NLAFSDPActor: FSDP training actor with model-type-dispatched NLA support.

Two orthogonal dimensions:
  - model type: LM (.logits) vs NLACriticModel (.values) — gated by _is_critic_model
  - role: "actor" (rollout_data as-is) vs "critic" (swap actor tokens → critic tokens)

Key simplification: override _compute_log_prob to no-op for critic models (they
have no .logits). Stock _train_core then works — compute_advantages_and_returns
early-returns when log_probs/values are None, _train_step override handles .values.
Override _train_core (not train) so parent handles get_rollout_data + timers + perf log.
"""

import os
import json
import threading
import shutil
import subprocess
import sys
import contextlib

import ray

from miles.utils.ray_utils import Box
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor, apply_fsdp2
from miles.backends.training_utils.data import get_batch
from miles.backends.training_utils.log_utils import aggregate_forward_results
from miles.backends.training_utils.loss import get_log_probs_and_entropy
from miles.utils.timer import timer
from tqdm import tqdm
from miles.backends.training_utils.loss import loss_function

from nla.audit_runtime import (
    keep_router_fp32,
    mse_ratio_agreement,
    padded_critic_inputs_from_tokens,
    should_synchronize_microbatch,
)
from nla.arch_adapters import resolve_text_config, resolve_text_model
from nla.config import NLAConfig, load_nla_config_from_args, write_model_sidecar
from nla.critic_repartition import (
    REPORT_KEY as CRITIC_REPARTITION_REPORT_KEY,
    balance_critic_partition,
    require_minimum_retained_fraction,
)
from nla.injection import inject_at_marked_positions
from nla.mamba_runtime import (
    resolve_mamba_train_kernel_mode,
    temporarily_disable_mamba_fast_path,
    temporarily_select_mamba_training_kernel,
)
from nla.model_dtype import normalize_floating_module_dtype
from nla.models import NLACriticModel, embed_dump_path
from nla.packed_equivalence import (
    build_bshd_attention_mask,
    build_bshd_max_seq_lens,
    build_packed_padded_inputs,
    packed_equivalence_metrics,
    response_mean_nlls,
)
from nla.schema import (
    MM_ACTIVATION_KEY, MM_CRITIC_TOKENS_KEY, MM_MSE_SCALE_KEY,
    load_predict_mean_baselines, normalize_activation,
)
from nla.runtime_flags import env_flag, env_float
from nla.storage import _load_storage, is_remote
from nla.system_metrics import (
    RouterEntropyTracker,
    SystemMetricsLogger,
    append_metrics_to_miles_loss_dict,
)


CRITIC_ONLY_MM_KEYS = {MM_CRITIC_TOKENS_KEY}


def _swap_rollout_to_critic_tokens(
    rollout_data: dict,
    device: torch.device,
    *,
    bshd_pad_size: int | None = None,
) -> dict:
    """Rewire rollout_data: actor tokens → critic tokens, filter failed extractions.

    Pure data transform — unit-testable. The RL rollout fn stashed
    `nla_critic_tokens` (tokenized <text>{payload}</text> <summary>{pm}) in
    multimodal_train_inputs for samples where <explanation> extraction succeeded.
    Missing key → sample filtered.

    Returns a NEW dict. Caller must handle cross-rank divergence — len(kept) may
    differ per rank; see _truncate_to_cross_rank_min.
    """
    kept: list[int] = []
    critic_tokens: list[torch.Tensor] = []
    mm_list = rollout_data["multimodal_train_inputs"]
    for i, mm in enumerate(mm_list):
        if mm is None or MM_CRITIC_TOKENS_KEY not in mm:
            continue
        critic_tokens.append(mm[MM_CRITIC_TOKENS_KEY])
        kept.append(i)
    # No assert on len(kept) here — it's a PER-RANK check. If one rank has zero
    # and asserts, the others enter _truncate_to_cross_rank_min's all_reduce and
    # hang forever. Let the collective assert n_min > 0 fire on ALL ranks together.
    empty_mask = torch.empty(0, dtype=torch.int, device=device)
    critic_view = {
        "tokens": critic_tokens,
        "total_lengths": [t.shape[0] for t in critic_tokens],
        "response_lengths": [0] * len(critic_tokens),
        "loss_masks": [empty_mask] * len(critic_tokens),
        "multimodal_train_inputs": [
            {MM_ACTIVATION_KEY: mm_list[i][MM_ACTIVATION_KEY]} for i in kept
        ],
    }
    if bshd_pad_size is not None:
        critic_view["max_seq_lens"] = build_bshd_max_seq_lens(
            critic_tokens,
            pad_size=bshd_pad_size,
        )
    return critic_view


def _critic_token_rows_and_golds(rollout_data: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
    mm_list = rollout_data.get("multimodal_train_inputs") or []
    rollout_tokens = rollout_data.get("tokens") or []
    toks: list[torch.Tensor] = []
    golds: list[torch.Tensor] = []
    for i, mm in enumerate(mm_list):
        if not mm or MM_ACTIVATION_KEY not in mm:
            continue
        if MM_CRITIC_TOKENS_KEY in mm:
            tok = mm[MM_CRITIC_TOKENS_KEY]
        elif i < len(rollout_tokens):
            tok = rollout_tokens[i]
        else:
            continue
        toks.append(tok)
        golds.append(mm[MM_ACTIVATION_KEY])
    if not golds:
        return [], torch.empty(0)
    return toks, torch.cat(golds, dim=0)


def _env_assert_packed_equiv(default: bool) -> bool:
    value = os.environ.get("NLA_ASSERT_PACKED_EQUIV", "auto").strip().lower()
    if value == "auto":
        return default
    return value in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _temporarily_disable_mamba_fast_path(model: torch.nn.Module):
    """Disable Nemotron-H Mamba Triton fast path only for critic reward scoring."""

    if not env_flag("NLA_CRITIC_FWD_DISABLE_MAMBA_FAST_PATH", False):
        yield
        return

    with temporarily_disable_mamba_fast_path(model):
        yield


def _decode_json_float_sentinel(value):
    """Decode checkpoint config sentinels such as {"__float__": "Infinity"}."""

    if isinstance(value, dict) and set(value) == {"__float__"}:
        tag = str(value["__float__"]).strip().lower()
        if tag in {"infinity", "inf", "+infinity", "+inf"}:
            return float("inf")
        if tag in {"-infinity", "-inf"}:
            return -float("inf")
        if tag == "nan":
            return float("nan")
    if isinstance(value, (list, tuple)):
        return tuple(_decode_json_float_sentinel(item) for item in value)
    return value


def _normalize_mamba_time_step_limits(model: torch.nn.Module) -> int:
    """Normalize remote-code Mamba module time_step_limit values after HF load."""

    changed = 0
    for module in model.modules():
        if not hasattr(module, "time_step_limit"):
            continue
        old_value = getattr(module, "time_step_limit")
        new_value = _decode_json_float_sentinel(old_value)
        if repr(new_value) == repr(old_value):
            continue
        setattr(module, "time_step_limit", new_value)
        changed += 1
    return changed


def _emit_nla_advantage_stats(rollout_id: int, rollout_data: dict) -> None:
    """Print true rollout advantage stats from the actor training path."""

    try:
        from nla.rollout.rl_metrics import advantage_stats_from_rollout_data

        metrics = advantage_stats_from_rollout_data(rollout_data)
        if not metrics:
            keys = ",".join(sorted(str(key) for key in rollout_data.keys()))
            print(
                f"[NLA ADVANTAGE] rollout_id={rollout_id} scope=actor_shard status=missing keys={keys}",
                flush=True,
            )
            return
        payload = " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))
        print(f"[NLA ADVANTAGE] rollout_id={rollout_id} scope=actor_shard {payload}", flush=True)
    except Exception as exc:
        print(f"[NLA ADVANTAGE] failed to collect advantage stats: {exc}", flush=True)


def _critic_forward_padded_last_values_from_padded(
    model: torch.nn.Module,
    ids: torch.Tensor,
    mask: torch.Tensor,
    last_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward critic on padded inputs and return selected last-token rows."""

    out = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        nla_value_indices=last_idx,
    )
    return out.values.float(), out.backbone_last_hidden


def _critic_forward_padded_last_values(
    model: torch.nn.Module,
    unconcat_tokens: list[torch.Tensor],
    *,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward critic on a padded microbatch and return selected last-token rows."""

    ids, mask, last_idx = padded_critic_inputs_from_tokens(
        unconcat_tokens,
        torch.cuda.current_device(),
        pad_id=pad_id,
    )
    return _critic_forward_padded_last_values_from_padded(model, ids, mask, last_idx)


def _mse_ratio_tolerance(name: str, default: float) -> float:
    """Read an explicit, bounded tolerance for the live critic path check."""

    tolerance = env_float(name, default)
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {tolerance}")
    return tolerance


def _assert_reward_train_paths_agree(
    critic_fwd_fn,
    model: torch.nn.Module,
    rollout_data: dict,
    mse_scale: float,
    tol: float = 0.10,
    training_kernel_mode: str = "auto",
) -> None:
    """Live step-0 check: reward critic_fwd MSE == padded training MSE.

    Preflight (rl_preflight.py) validates this on dummy data with the HF-loaded
    critic. This runs it once at step 0 on REAL rollout data with the REAL
    post-DCP-overlay critic — catches anything the dummy batch misses (e.g. DCP
    load corrupted weights, or a tokenizer edge case the dummies don't hit).

    The old left-pad bug (left-pad + mask.sum-1) produced per-sample ratios
    around 1.5-2.0.  The reward scorer deliberately runs the critic in eval
    mode while optimization runs it in train mode, so validate those two
    invariants separately: the reward/eval-layout comparison is strict, while
    the train-mode comparison allows bounded BF16 kernel drift.
    """
    toks, golds = _critic_token_rows_and_golds(rollout_data)
    # Two paths must agree on ANY subset — a handful of varied-length samples
    # exercises padding edge cases; 32 from the rank-partition adds ~1s.
    # critic_fwd returns .cpu() but rollout_data's golds are on the rank's
    # CUDA device (miles moved them during data prep). Unify on CPU.
    toks, golds = toks[:32], golds[:32].float().cpu()
    n = len(toks)
    if n < 4:
        print(f"[NLA STEP0 CHECK] skipped: n={n} < 4 (smoke-test batch too small for varied-length padding)", flush=True)
        return

    # Reward path: pad to max, attention_mask, critic_fwd picks last_idx.
    ids, mask, _last_idx = padded_critic_inputs_from_tokens(toks, "cpu", pad_id=0)
    pred_reward = critic_fwd_fn(ids, mask)  # [n, d] CPU

    def _mse(p: torch.Tensor) -> torch.Tensor:
        pn = normalize_activation(p, mse_scale)
        gn = normalize_activation(golds.float(), mse_scale)
        return ((pn - gn) ** 2).mean(dim=1)

    reward_mse = _mse(pred_reward)

    # Compare the reward scorer to the exact padded training layout in eval
    # mode first. This isolates token positions, padding, and kernel selection
    # from train/eval implementation differences.
    was_training = model.training
    model.eval()
    try:
        with temporarily_select_mamba_training_kernel(
            model, training_kernel_mode
        ), torch.no_grad():
            values, _backbone_h = _critic_forward_padded_last_values(model, toks)
            pred_eval_layout = values.float().cpu()
    finally:
        model.train(was_training)

    # Then measure the actual optimization-mode forward. Nano's checkpoint
    # has zero dropout, but BF16 train/eval kernel selection can still make a
    # small sequence-dependent difference. Keep this as a separate fail-closed
    # guard with a config-controlled, conservative tolerance.
    with temporarily_select_mamba_training_kernel(
        model, training_kernel_mode
    ), torch.no_grad():
        values, _backbone_h = _critic_forward_padded_last_values(model, toks)
        pred_train = values.float().cpu()

    reward_layout = mse_ratio_agreement(reward_mse, _mse(pred_eval_layout))
    train_layout = mse_ratio_agreement(_mse(pred_train), _mse(pred_eval_layout))
    layout_tol = _mse_ratio_tolerance(
        "NLA_CRITIC_REWARD_LAYOUT_MSE_RATIO_TOL", tol
    )
    train_mode_tol = _mse_ratio_tolerance(
        "NLA_CRITIC_TRAIN_MODE_MSE_RATIO_TOL", 0.05
    )
    print(
        "[NLA STEP0 CHECK] "
        f"reward/eval-layout MSE ratio: mean={reward_layout.mean_ratio:.4f} "
        f"p95|r-1|={reward_layout.p95_abs_deviation:.4f} "
        f"max|r-1|={reward_layout.max_abs_deviation:.4f} n={n}; "
        f"train/eval-layout MSE ratio: mean={train_layout.mean_ratio:.4f} "
        f"p95|r-1|={train_layout.p95_abs_deviation:.4f} "
        f"max|r-1|={train_layout.max_abs_deviation:.4f}",
        flush=True,
    )
    assert reward_layout.max_abs_deviation < layout_tol, (
        "step-0 reward-path and eval-layout MSE diverge by "
        f"{reward_layout.max_abs_deviation:.1%} (tol {layout_tol:.0%}) on real rollout data. "
        "The critic reward layout must match the padded training layout. "
        f"Per-sample ratios: {reward_layout.ratios}"
    )
    assert train_layout.max_abs_deviation < train_mode_tol, (
        "step-0 train-mode and eval-layout MSE diverge by "
        f"{train_layout.max_abs_deviation:.1%} (tol {train_mode_tol:.0%}) on real rollout data. "
        "The critic train-mode kernel drift exceeds the registered bound. "
        f"Per-sample ratios: {train_layout.ratios}"
    )

    # NOT checking raw pred_norm/gold_norm: normalize_activation(v,s) does
    # v/‖v‖·s — MSE loss is scale-invariant, head output norm is unconstrained.
    # Gemma's head naturally outputs at backbone scale (~3× gold). Preflight's
    # normalize(pred).norm/√d > 0.1 is the right check for random-direction
    # (Mar 13 bug); this step-0 check covers path-divergence only.


def _truncate_to_cross_rank_min(
    rollout_data: dict, dp_group, micro_batch_size: int | None
) -> dict:
    """All-reduce len(tokens) to the cross-rank MIN and truncate all lists.

    After _swap_rollout_to_critic_tokens, each rank may have a different
    len(kept). get_data_iterator computes num_steps = len(tokens) // (gbs/dp);
    different lengths → different num_steps → FSDP grad-allreduce desync → hang.
    Or with dynamic batching, mismatched tensor shapes in the allreduce → hang.

    Also sets dynamic_global_batch_size so num_steps == 1 regardless of original gbs.
    """
    n = torch.tensor([len(rollout_data["tokens"])], device=torch.cuda.current_device())
    dist.all_reduce(n, op=dist.ReduceOp.MIN, group=dp_group)
    n_min = n.item()
    if micro_batch_size is not None:
        n_min = (n_min // micro_batch_size) * micro_batch_size
    assert n_min > 0, (
        f"cross-rank min(len(kept)) rounded to {n_min} — at least one rank has "
        f"no valid <explanation> extractions. Actor is not emitting tags "
        f"reliably. Raise rollout_batch_size or check actor SFT checkpoint."
    )
    out = {k: v[:n_min] for k, v in rollout_data.items()}
    out["dynamic_global_batch_size"] = n_min * dist.get_world_size(dp_group)
    return out


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set")
    return parsed


def _cap_actor_sft_rollout_tokens(rollout_data: dict) -> dict[str, int | None] | None:
    max_sequence_tokens = _positive_int_env("NLA_SFT_MAX_SEQUENCE_TOKENS")
    max_response_tokens = _positive_int_env("NLA_SFT_MAX_RESPONSE_TOKENS")
    if max_sequence_tokens is None and max_response_tokens is None:
        return None

    tokens = rollout_data.get("tokens") or []
    response_lengths = rollout_data.get("response_lengths") or []
    loss_masks = rollout_data.get("loss_masks") or []
    if not tokens or len(response_lengths) != len(tokens) or len(loss_masks) != len(tokens):
        return None

    original_max = max(len(t) for t in tokens)
    capped_tokens = []
    capped_response_lengths = []
    capped_loss_masks = []
    for token_seq, response_length, loss_mask in zip(tokens, response_lengths, loss_masks, strict=True):
        response_length = int(response_length)
        prompt_length = len(token_seq) - response_length
        keep = len(token_seq)
        if max_response_tokens is not None and response_length > max_response_tokens:
            keep = min(keep, prompt_length + max_response_tokens)
        if max_sequence_tokens is not None:
            keep = min(keep, max_sequence_tokens)
        new_response_length = max(0, min(response_length, keep - prompt_length))
        if new_response_length <= 0:
            raise ValueError(
                "NLA SFT actor token cap removed all response tokens; "
                "increase max_sequence_tokens or max_response_tokens"
            )
        capped_tokens.append(token_seq[:keep])
        capped_response_lengths.append(new_response_length)
        capped_loss_masks.append(loss_mask[:new_response_length])

    rollout_data["tokens"] = capped_tokens
    rollout_data["response_lengths"] = capped_response_lengths
    rollout_data["loss_masks"] = capped_loss_masks
    rollout_data["total_lengths"] = [len(t) for t in capped_tokens]
    return {
        "max_sequence_tokens": max_sequence_tokens,
        "max_response_tokens": max_response_tokens,
        "original_max_tokens": original_max,
        "capped_max_tokens": max(len(t) for t in capped_tokens),
    }


def _repartition_for_critic(
    rollout_data_ref,
    actor_dp,
    critic_rank,
    critic_dp,
    *,
    micro_batch_size=1,
):
    """Build one parse-valid, row-balanced critic shard from actor shards."""
    assert len(rollout_data_ref) == actor_dp, (
        f"expected {actor_dp} actor partitions, got {len(rollout_data_ref)}"
    )
    fetched = [ray.get(ref.inner) for ref in rollout_data_ref]
    merged, _ = balance_critic_partition(
        fetched,
        critic_rank=critic_rank,
        critic_dp=critic_dp,
        alignment=micro_batch_size,
        required_multimodal_key=MM_CRITIC_TOKENS_KEY,
    )

    # Re-wrap: critic_dp Boxes. process_rollout_data does refs[dp_rank].inner,
    # so only our rank's Box needs real data. Others are None-inner placeholders
    # (never accessed). Box class: miles.utils.ray_utils.Box.
    new_refs = [Box(None)] * critic_dp
    new_refs[critic_rank] = Box(ray.put(merged))
    return new_refs


class NLATextOnlyCausalLM:
    """Auto-class shim: load + unwrap multimodal → text-only CausalLM.

    miles' FSDPTrainRayActor.get_model_cls() returns AutoModelForImageTextToText
    when hf_config has vision_config (Gemma-3 triggers this) → actor has
    vision_tower params → RL weight-sync to text-only sglang 400s on the first
    vision_tower key. resolve_text_model unwraps to a CausalLM wrapper around
    the text side only. No-op for Qwen/Llama/Mistral (no .language_model attr).

    The shim interface is the minimum miles needs: `.from_pretrained(...)` is
    the only callsite (fsdp_utils/actor.py: model_cls.from_pretrained(...)).
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return resolve_text_model(model)


class _SGLangKeyRemap:
    """Wrap a model so state_dict() keys match sglang's multimodal naming.

    Our actor (via NLATextOnlyCausalLM) is Gemma3ForCausalLM — keys `model.*`.
    sglang loads `google/gemma-3-12b-it` as Gemma3ForConditionalGeneration
    (architectures=['Gemma3ForConditionalGeneration'] in HF config) — keys
    `language_model.model.*`. Weight-sync iterates actor state_dict and sends
    names verbatim; sglang's load_weights does params_dict[name] → KeyError →
    HTTP 400 on the very first param.

    This wrapper prepends the prefix for weight-sync only (doesn't touch
    FSDP/training — weight_updater captures its own model ref at init).
    """

    def __init__(self, model: torch.nn.Module, prefix: str):
        self._model = model
        self._prefix = prefix

    def state_dict(self):
        return {self._prefix + k: v for k, v in self._model.state_dict().items()}


class NLAFSDPActor(FSDPTrainRayActor):
    def _nla_is_dp_leader(self) -> bool:
        parallel_state = getattr(self, "parallel_state", None)
        if parallel_state is not None and hasattr(parallel_state, "dp_rank"):
            return int(parallel_state.dp_rank) == 0
        return (not dist.is_initialized()) or dist.get_rank() == 0

    def init(self, args, role, with_ref=False):
        self._nla_mamba_training_kernel_mode = resolve_mamba_train_kernel_mode(
            role
        )
        if role == "critic":
            assert args.critic_save is not None, (
                "NLA RL requires --critic-save (reward fn reads from there)"
            )
            args.hf_checkpoint = args.critic_load
            if args.critic_load_dcp:
                tracker = Path(args.critic_load_dcp) / "latest_checkpointed_iteration.txt"
                assert tracker.is_file(), (
                    f"--critic-load-dcp={args.critic_load_dcp!r} has no tracker file. "
                    f"checkpoint.load() would silently return None → critic keeps "
                    f"HF weights from --critic-load, ignoring the DCP overlay you asked for."
                )
            args.load = args.critic_load_dcp or args.critic_load
            args.save = args.critic_save
            args.lr = args.critic_lr or args.lr
            # Megatron wires this at megatron_utils/actor.py:93; FSDP doesn't.
            if args.critic_lr_warmup_iters:
                args.lr_warmup_iters = args.critic_lr_warmup_iters
            args.loss_type = "custom_loss"
            args.custom_loss_function_path = "nla.loss.nla_critic_loss"
            args.nla_model_is_critic = True
            # Critic's sidecar lives at critic_load — it has critic_num_layers,
            # mse_scale, suffix_ids. --nla-sidecar-source on CLI is the ACTOR's
            # override (for injection_scale from its model sidecar). Swap to the
            # critic's source: --nla-critic-sidecar-source if set, else None →
            # resolve_sidecar_source falls through to hf_checkpoint = critic_load.
            # (Megatron REQUIRES nla_critic_sidecar_source since its critic_load
            # is torch_dist with no sidecar; FSDP's fall-through to critic_load
            # HF dir is why None works here.)
            args.nla_sidecar_source = args.nla_critic_sidecar_source

        self._is_critic_model = getattr(args, "nla_model_is_critic", False)
        self._nla_actor_packed_equiv_enabled = (
            not self._is_critic_model
            and env_flag("NLA_ASSERT_ACTOR_PACKED_EQUIV", False)
        )
        self._nla_actor_packed_equiv_checked = False
        self._nla_actor_packed_equiv_attempts = 0
        self._nla_padded_layout_logged = False
        self._nla_actor_packed_equiv_rtol = float(
            os.environ.get("NLA_ACTOR_PACKED_EQUIV_RTOL", "0.02")
        )
        self._nla_actor_packed_equiv_atol = float(
            os.environ.get("NLA_ACTOR_PACKED_EQUIV_ATOL", "0.05")
        )
        if self._nla_actor_packed_equiv_rtol < 0 or self._nla_actor_packed_equiv_atol < 0:
            raise ValueError("actor packed-equivalence tolerances must be non-negative")
        self._nla_freeze_critic_train = (
            self._is_critic_model
            and role == "critic"
            and env_flag("NLA_FREEZE_CRITIC_TRAIN", False)
        )

        rollout_id = super().init(args, role, with_ref)
        normalized_time_step_limits = _normalize_mamba_time_step_limits(self.model)
        if getattr(self, "ref_model", None) is not None:
            normalized_time_step_limits += _normalize_mamba_time_step_limits(self.ref_model)
        if normalized_time_step_limits and self._nla_is_dp_leader():
            print(
                f"[NLA] normalized {normalized_time_step_limits} Mamba time_step_limit "
                "JSON float sentinel values",
                flush=True,
            )
        if self._nla_freeze_critic_train and self._nla_is_dp_leader():
            print(
                "[NLA] NLA_FREEZE_CRITIC_TRAIN=1: critic participates in reward "
                "forward/checks but skips critic SFT optimizer updates.",
                flush=True,
            )
        if self._nla_mamba_training_kernel_mode != "auto" and self._nla_is_dp_leader():
            print(
                "[NLA] Mamba training kernel mode: "
                f"{self._nla_mamba_training_kernel_mode}",
                flush=True,
            )
        if self._nla_actor_packed_equiv_enabled and self._nla_is_dp_leader():
            print(
                "[NLA] actor packed-equivalence gate enabled: "
                f"rtol={self._nla_actor_packed_equiv_rtol} "
                f"atol={self._nla_actor_packed_equiv_atol}",
                flush=True,
            )
        if os.environ.get("NLA_REPIN_ROUTER_FP32_AFTER_FSDP", "0").strip().lower() in {"1", "true", "yes", "on"}:
            router_fp32_count = keep_router_fp32(self.model)
            if router_fp32_count and dist.get_rank() == 0:
                print(f"[NLA] restored {router_fp32_count} router tensors/buffers to fp32 storage", flush=True)

        # Parent keeps the full wrapper config (needs .vision_config for its own
        # checks); NLA only cares about text-side hidden_size/num_hidden_layers.
        self._text_config = resolve_text_config(self.hf_config)

        # sglang loads the HF checkpoint's architecture (multimodal for Gemma),
        # weight-sync sends OUR text-only keys. If we unwrapped, bridge with a
        # prefix remapper. weight_updater only exists for the actor role (sglang
        # sync path) — critic role doesn't create one. _text_config != hf_config
        # is the exact signal: we stripped a multimodal wrapper.
        if (
            not self._is_critic_model
            and self._text_config is not self.hf_config
            and hasattr(self, "weight_updater")
        ):
            arch = (getattr(self.hf_config, "architectures", None) or [""])[0]
            prefix = "language_model." if "ConditionalGeneration" in arch else ""
            if prefix:
                self.weight_updater.model = _SGLangKeyRemap(self.model, prefix)

        assert self.parallel_state.cp_size == 1, (
            "NLA requires cp_size=1. With cp>1, slice_with_cp splits each sample "
            "into non-contiguous chunks; injection token + neighbors can land on "
            "different CP ranks, breaking the in-hook scan."
        )

        # get_grpo_returns (ppo_utils.py) takes kl but only uses it for .ones_like
        # (shape) — the value is discarded. So --kl-coef with grpo/gspo computes
        # ref_log_probs (slow!), builds the kl tensor, then throws it away. The
        # actual GRPO KL path is --use-kl-loss, which adds KL to the policy loss
        # instead (logs as train/kl_loss). This silently ate early RL runs.
        if role == "actor" and args.advantage_estimator in ("grpo", "gspo"):
            assert args.kl_coef == 0, (
                f"--kl-coef={args.kl_coef} is a NO-OP under "
                f"--advantage-estimator={args.advantage_estimator}: "
                f"get_grpo_returns discards the kl tensor. Use --use-kl-loss "
                f"--kl-loss-coef {args.kl_coef} instead (adds KL to policy loss, "
                f"logs as train/kl_loss). Or set --kl-coef 0 explicitly if you "
                f"don't want KL."
            )

        if role == "critic" and args.force_use_critic:
            actor_dp = args.actor_num_nodes * args.actor_num_gpus_per_node
            critic_dp = args.critic_num_nodes * args.critic_num_gpus_per_node
            # RolloutManager partitions by actor DP. The critic always rebuilds
            # a globally balanced, parse-valid view, including when the two DP
            # sizes happen to match; otherwise uneven parser failures can still
            # force cross-rank truncation.
            self._nla_actor_dp = actor_dp
            print(
                f"[NLA] critic row rebalance: actor_dp={actor_dp} "
                f"critic_dp={critic_dp}",
                flush=True,
            )

        cfg, sidecar_source = load_nla_config_from_args(args, self.tokenizer)
        assert cfg.d_model == self._text_config.hidden_size, (
            f"sidecar d_model={cfg.d_model} != model hidden_size="
            f"{self._text_config.hidden_size}. Wrong checkpoint for this dataset."
        )
        if self._is_critic_model:
            # arguments.py:1796 defaults critic_load=load, so a missing
            # --critic-load silently loads the full-depth actor checkpoint.
            # Positive arch check catches that.
            assert cfg.critic_num_layers is not None, (
                f"critic model loaded from {args.hf_checkpoint!r} but sidecar "
                f"has no critic_num_layers. Did --critic-load default to the "
                f"actor checkpoint? Point it at the prepared K+1-layer critic."
            )
            assert self._text_config.num_hidden_layers == cfg.critic_num_layers + 1, (
                f"critic checkpoint has {self._text_config.num_hidden_layers} "
                f"layers, sidecar says extraction layer_index K="
                f"{cfg.critic_num_layers} → expect K+1="
                f"{cfg.critic_num_layers + 1} layers. Wrong checkpoint."
            )

        # injection_scale is a TRAINING HYPERPARAMETER — REQUIRED for actor
        # training. load_nla_config_from_args already applied any CLI override;
        # here we assert the value was resolved (via CLI, model sidecar, or
        # --nla-sidecar-source). Dataset sidecars deliberately don't carry
        # injection_scale — pick explicitly.
        #
        # INFERENCE MUST MATCH: nla_generate.py also calls load_nla_config_from_args
        # (same helper, same resolution), so train/infer scale cannot diverge.
        injects = not self._is_critic_model and args.loss_type in ("sft_loss", "policy_loss")
        if injects:
            assert cfg.injection_scale is not None, (
                "Actor training requires injection_scale. Set --nla-injection-scale "
                "(e.g. '150', 'raw', 'sqrt_d_model'), or point --nla-sidecar-source "
                "at a model sidecar that has it. Dataset sidecars don't carry it — "
                "it's a training hyperparameter, pick explicitly. "
                f"(Resolved sidecar: {sidecar_source!r}, injection_scale: None.)"
            )
        self._nla_cfg: NLAConfig = cfg
        self._nla_vectors: torch.Tensor | None = None
        # Expose mse_scale on args so nla_critic_loss can read it backend-agnostically.
        # (Megatron's forward_step closure can't mutate batch; args is the shared channel.)
        self.args.nla_mse_scale = cfg.mse_scale

        # Predict-the-mean baselines for FVE. If passed via CLI (--nla-baseline-*,
        # precomputed from schema.compute_predict_mean_baselines), use those
        # directly — skips the init-time parquet read. Otherwise rank 0 reads
        # + broadcasts. Megatron uses CLI-only (no fallback compute).
        if self._is_critic_model and args.prompt_data is not None and args.nla_baseline_rawvar is None:
            baselines = [0.0]
            if dist.get_rank() == 0:
                t0 = time.perf_counter()
                source = args.prompt_data.split("@[")[0]
                if is_remote(source):
                    assert args.nla_storage_cls is not None
                    source = _load_storage(args.nla_storage_cls).open_read(source)
                _, b_rv = load_predict_mean_baselines(source, cfg.mse_scale)
                baselines[0] = b_rv
                dt = time.perf_counter() - t0
                print(f"[NLA] FVE baseline rawvar={b_rv:.4f} "
                      f"(mse_scale={cfg.mse_scale}, took {dt:.1f}s)")
            dist.broadcast_object_list(baselines, src=0)
            self.args.nla_baseline_rawvar = baselines[0]

        # miles calls gradient_checkpointing_enable() with no kwargs
        # (fsdp_utils/actor.py:123) — HF defaults to use_reentrant=True.
        # Reentrant checkpoint's backward re-runs forward via a custom
        # autograd.Function whose recompute does NOT trigger FSDP2's
        # post-forward reshard hook. All-gather buffers from the recompute
        # stay alive through the rest of backward. At 62 layers × 826MB
        # (27b) = 51GB pileup. 74GB OOM at rollout 1 once adam state lands.
        # Memory snapshot 2026-03-13: 54 × 826MB foreach_all_gather at OOM,
        # post-forward only 17GB (FWDMEM hook) → backward-only pileup.
        # Standalone FSDP test WITHOUT grad-ckpt → 10.64GB → confirms.
        #
        # use_reentrant=False (non-reentrant) runs recompute via a normal
        # forward call, module hooks fire, FSDP reshards correctly. PyTorch
        # FSDP docs explicitly recommend this. miles should fix upstream.
        # NLACriticModel.gradient_checkpointing_enable/_disable delegate
        # to backbone so this works for both roles.
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Hook registration AFTER grad-ckpt re-enable — defensive ordering.
        # HF's gradient_checkpointing_disable() clears forward hooks on
        # submodules (it removes checkpoint wrappers by clearing hooks), so
        # registering before the disable/re-enable cycle risks losing them.
        # Earlier 12b configs had no grad-ckpt re-enable so ordering was moot.
        if not self._is_critic_model and args.loss_type in ("sft_loss", "policy_loss"):
            self._register_injection_hook(self.model)
            if self.ref_model is not None:
                self._register_injection_hook(self.ref_model)

        self._nla_system_metrics = self._build_system_metrics_logger(role)
        self._nla_router_entropy = (
            RouterEntropyTracker.attach(self.model)
            if os.environ.get("NLA_ROUTER_METRICS", "0").strip().lower() in {"1", "true", "yes", "on"}
            else None
        )
        return rollout_id

    def _build_system_metrics_logger(self, role: str) -> SystemMetricsLogger:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            try:
                local_rank = int(torch.cuda.current_device())
            except Exception:
                pass
        train_env_vars = getattr(self.args, "train_env_vars", None)
        logger = SystemMetricsLogger.from_env_mapping(
            train_env_vars,
            rank=rank,
            local_rank=local_rank,
            role=role,
        )
        if rank == 0:
            keys = sorted(train_env_vars) if isinstance(train_env_vars, dict) else []
            print(
                f"[NLA] system_metrics role={role} enabled={logger.enabled} "
                f"interval={logger.interval_steps} nvsm_interval={logger.nvidia_smi_interval_steps} "
                f"phase_metrics={logger.phase_metrics_enabled} "
                f"phase_all_gpus={logger.phase_metrics_all_gpus} "
                f"train_env_keys={keys}",
                flush=True,
            )
        return logger

    def _attach_system_metrics(self, log_dict: dict, step_id: int) -> dict:
        metrics: dict[str, float | int] = {}
        logger = getattr(self, "_nla_system_metrics", None)
        if logger is not None:
            metrics.update(logger.collect(step_id=step_id))
        router_entropy = getattr(self, "_nla_router_entropy", None)
        if router_entropy is not None:
            metrics.update(router_entropy.collect())
        repartition = getattr(self, "_nla_critic_repartition_report", None)
        if repartition:
            for key, value in repartition.items():
                if isinstance(value, (int, float)):
                    metrics[f"nla/critic_batch/{key}"] = value
        return append_metrics_to_miles_loss_dict(log_dict, metrics)

    def _emit_phase_snapshot(self, phase: str, event: str, *, rollout_id=None, extra: dict | None = None) -> None:
        logger = getattr(self, "_nla_system_metrics", None)
        if logger is None:
            return
        snapshot_extra = dict(extra or {})
        if rollout_id is not None:
            snapshot_extra["rollout_id"] = int(rollout_id)
        try:
            step_id = 0 if rollout_id is None else int(rollout_id)
        except Exception:
            step_id = 0
        logger.emit_phase_snapshot(
            step_id=step_id,
            phase=phase,
            event=event,
            extra=snapshot_extra,
        )

    def get_model_cls(self):
        if self._is_critic_model:
            return NLACriticModel
        # NLA is text-only. Parent returns AutoModelForImageTextToText when
        # hf_config has vision_config (Gemma-3) → actor has vision_tower params
        # → RL weight-sync to text-only sglang 400s. resolve_text_model unwraps
        # to Gemma3ForCausalLM (no-op for Qwen/Llama). See arch_adapters.py.
        return NLATextOnlyCausalLM

    def connect_actor_critic(self, critic_group):
        # Miles' PPO critic creates an actor↔critic NCCL group for syncing
        # per-token values into the actor's GAE computation (megatron_utils/actor.py:552).
        # NLA's critic is independent — GRPO advantages come from group-normed
        # rewards, not critic values. Both groups consume the same rollout_data_ref;
        # nothing to sync.
        pass

    def update_weights(self):
        """Sync actor weights to SGLang, then dump embedding for nla_generate.

        The rollout worker's cached embedding goes stale after each train step.
        Since the trainer is idle during rollout, and update_weights fires right
        before rollout starts, this is the moment to dump a fresh copy.
        nla_generate._maybe_reload_embed reads it.
        """
        super().update_weights()
        # debug_train_only (SFT mode): no SGLang rollout worker, so nla_generate
        # never runs → no consumer for the dump. Skip — saves ~2.2s/step
        # (FSDP all-gather of 1.1GB embedding + torch.save to disk).
        if (self._is_critic_model or self.args.save is None
                or self.args.debug_rollout_only or self.args.debug_train_only):
            return
        # --offload-train moves model to CPU before this (train.py:92 → sleep()
        # → model.cpu()). Mirror the parent updater's .cuda() (update_weight_utils.py:58)
        # so .full_tensor() runs its NCCL all-gather on GPU.
        weight = self.model.get_input_embeddings().weight.detach().cuda()
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        if dist.get_rank() == 0:
            out_path = embed_dump_path(self.args.save)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            torch.save(weight.detach().cpu(), tmp)
            tmp.rename(out_path)  # atomic — no mid-write read
        dist.barrier()

    def _register_injection_hook(self, model):
        embed = model.get_input_embeddings()
        inj = self._nla_cfg.injection_token_id
        left = self._nla_cfg.injection_left_neighbor_id
        right = self._nla_cfg.injection_right_neighbor_id

        def hook(_module, inputs, output):
            if self._nla_vectors is None or os.environ.get("NLA_SKIP_INJECTION") == "1":
                return output
            assert len(inputs) == 1 and inputs[0].dtype == torch.long
            return inject_at_marked_positions(
                input_ids=inputs[0],
                embeddings=output,
                vectors=self._nla_vectors,
                inj_id=inj, left_id=left, right_id=right,
            )

        embed.register_forward_hook(hook)

    def _prepare_nla_multimodal_inputs(self, batch):
        mm = batch.get("multimodal_train_inputs")
        if mm is not None and MM_ACTIVATION_KEY in mm:
            popped = mm.pop(MM_ACTIVATION_KEY)  # [B, d_model], raw from dataset
            if self._is_critic_model:
                batch[MM_ACTIVATION_KEY] = popped
                batch[MM_MSE_SCALE_KEY] = self._nla_cfg.mse_scale
            else:
                self._nla_vectors = normalize_activation(popped, self._nla_cfg.injection_scale)

    def _get_model_inputs_args(self, batch):
        self._prepare_nla_multimodal_inputs(batch)
        model_args = super()._get_model_inputs_args(batch)
        if str(getattr(self.args, "qkv_format", "thd")) == "bshd":
            if int(getattr(self.parallel_state, "cp_size", 1)) != 1:
                raise RuntimeError("NLA FSDP bshd batches currently require cp_size=1")
            token_rows = batch.get("unconcat_tokens") or []
            input_ids = model_args["input_ids"]
            attention_mask = build_bshd_attention_mask(token_rows, input_ids)
            if tuple(model_args["position_ids"].shape) != tuple(input_ids.shape):
                raise RuntimeError(
                    "bshd position_ids must match input_ids: "
                    f"positions={tuple(model_args['position_ids'].shape)} "
                    f"inputs={tuple(input_ids.shape)}"
                )
            model_args["attention_mask"] = attention_mask
            if not self._nla_padded_layout_logged and self._nla_is_dp_leader():
                self._nla_padded_layout_logged = True
                print(
                    "[NLA PADDED BATCH] "
                    f"samples={input_ids.shape[0]} width={input_ids.shape[1]} "
                    f"valid_tokens={int(attention_mask.sum().item())}",
                    flush=True,
                )
        # use_cache=False kills TWO bugs, both via the same DynamicCache:
        #
        # (1) v22 ref_lp: Gemma3TextModel.forward:518 creates DynamicCache when
        #     `use_cache and past_key_values is None and not self.training`.
        #     ref.eval() → cache; actor (train mode) → none. Gemma3 sliding-window
        #     attn picks different mask based on cache presence → ref_lp=-3.39 vs
        #     actor_lp=-1.32, identical weights. Qwen has no sliding window.
        #
        # (2) thd-packed cross-sequence contamination (miles passes
        #     attention_mask=None at fsdp_utils/actor.py:645). transformers HAS
        #     packed detection — masking_utils.py:735 infers block-diag from
        #     position_id resets — but it's gated on `past_key_values is None`.
        #     DynamicCache → detection bypassed → SDPA/eager fall through to full
        #     causal mask over the pack → seq N attends to seq 1..N-1. Verified
        #     2026-03-19: eval+default=4.2%
        #     L2 drift, eval+use_cache=False=0.6% (→0.0% in fp32, bf16 GEMM-tiling
        #     noise from batch-shape diff). Qwen FA2: varlen from position_ids
        #     directly, never touches this path.
        #
        # DO NOT remove — without this, every thd microbatch has contaminated
        # gradients on seqs 2..N. All training callers funnel through here
        # (_train_step, _compute_log_prob, _ref_log_probs_no_swap).
        model_args["use_cache"] = False
        return model_args

    def _create_ref_model(self, ref_load_path):
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")
        if not Path(ref_load_path).is_dir():
            return super()._create_ref_model(ref_load_path)

        # --nla-ref-on-gpu: UNTESTED since the hook + DynamicCache fixes (both
        # landed after this was dropped at v12 for the kl=7.14 symptom that
        # turned out to be those bugs). Gives back the ~20s/step CPU swap at
        # ~7.5GB VRAM. Re-validate KL init ~=0 before using in production.
        ref_on_gpu = getattr(self.args, "nla_ref_on_gpu", False)
        ref_cpu_offload = not ref_on_gpu
        if ref_on_gpu:
            # Miles hardcodes cpu_offload=True for ref → ~20s/step actor↔ref CPU swap.
            # At m16+resp150: actor ~31GB + ref ~7.5GB (FSDP-sharded) = ~38GB, fits on 80GB.
            print(f"[NLA] --nla-ref-on-gpu: ref from {ref_load_path} stays on GPU "
                  f"(skips ~20s/step swap, costs ~7.5GB VRAM)")
        elif dist.get_rank() == 0:
            print(f"[NLA] creating bf16-normalized CPU-offloaded ref from {ref_load_path}", flush=True)

        with self._get_init_weight_context_manager()():
            ref = self.get_model_cls().from_pretrained(
                ref_load_path, trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
                # convert_fsdp_to_hf saves fp32 (DCP is fp32). Without this
                # cast, ref is 2× on GPU: 18GB sharded at DP=6 vs 9GB bf16.
                # 45GB pre-train baseline vs 36 expected → OOM at step 0.
                # (Same fix as actor.py:611 for the CPUOffload path.)
                torch_dtype=torch.bfloat16,
            )
        changed = normalize_floating_module_dtype(ref, torch.bfloat16)
        if changed and dist.get_rank() == 0:
            print(f"[NLA] normalized {changed} ref floating tensors/buffers to bf16 before FSDP", flush=True)
        full_state = ref.state_dict()
        ref = apply_fsdp2(ref, mesh=self.parallel_state.dp_mesh, cpu_offload=ref_cpu_offload, args=self.args)
        ref = self._fsdp2_load_full_state_dict(
            ref,
            full_state,
            self.parallel_state.dp_mesh,
            cpu_offload=ref_cpu_offload,
        )
        if ref_on_gpu:
            ref.cuda()  # from_pretrained→CPU, FSDP cpu_offload=False won't move it — pin to GPU now
        ref.eval()
        return ref

    def _compute_log_prob(self, model_tag, data_iterator, num_microbatches, store_prefix=""):
        # Critic model has no .logits. compute_advantages_and_returns early-returns
        # when log_probs/values are None (loss.py:315); get_batch returns None for
        # absent keys (data.py:300). Stock _train_core handles the rest.
        # sft_loss (loss.py:785-835) recomputes from logits — this pass is wasted
        # (full model forward, injection hook, clone — ~2× step time).
        if self._is_critic_model or self.args.loss_type == "sft_loss":
            return {}
        if (model_tag == "ref" and self.ref_model is not None
                and getattr(self.args, "nla_ref_on_gpu", False)):
            return self._ref_log_probs_no_swap(data_iterator, num_microbatches, store_prefix)
        return super()._compute_log_prob(model_tag, data_iterator, num_microbatches, store_prefix)

    def _ref_log_probs_no_swap(self, data_iterator, num_microbatches, store_prefix):
        # Same forward loop as parent's _compute_log_prob ref branch, but without
        # the model.cpu()/model.cuda() swap (ref is on-GPU already). Parent's version
        # is at fsdp_utils/actor.py:310-392 — this is that minus lines 318-321, 386-392.
        forward_data_store = []
        data_iterator.reset()
        with timer(f"{store_prefix}log_probs"), torch.no_grad():
            for step_id in range(len(num_microbatches)):
                for _ in self.prof.iterate_train_log_probs(
                    tqdm(range(num_microbatches[step_id]),
                         desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    batch = get_batch(
                        data_iterator,
                        ["tokens", "loss_masks", "multimodal_train_inputs",
                         "total_lengths", "response_lengths", "max_seq_lens"],
                        self.parallel_state,
                        self.args.data_pad_size_multiplier,
                        self.args.qkv_format,
                        get_position_ids=True,
                    )
                    model_args = self._get_model_inputs_args(batch)
                    logits = self.ref_model(**model_args).logits.float()
                    result = get_log_probs_and_entropy(
                        logits=logits, args=self.args, parallel_state=self.parallel_state,
                        unconcat_tokens=batch["unconcat_tokens"],
                        total_lengths=batch["total_lengths"],
                        response_lengths=batch["response_lengths"],
                        with_entropy=False,
                        max_seq_lens=batch.get("max_seq_lens", None),
                    )
                    forward_data_store.append({f"{store_prefix}log_probs": result["log_probs"]})
        return aggregate_forward_results(forward_data_store, data_iterator, self.args, store_prefix)

    def _maybe_assert_actor_packed_equivalence(self, batch, num_microbatches):
        if not self._nla_actor_packed_equiv_enabled or self._nla_actor_packed_equiv_checked:
            return
        if str(getattr(self.args, "qkv_format", "thd")) != "thd":
            self._nla_actor_packed_equiv_checked = True
            return

        self._nla_actor_packed_equiv_attempts += 1
        token_rows = batch.get("unconcat_tokens") or []
        local_count = torch.tensor(len(token_rows), device=torch.cuda.current_device())
        dist.all_reduce(local_count, op=dist.ReduceOp.MIN)
        if int(local_count.item()) < 2:
            if self._nla_actor_packed_equiv_attempts >= int(num_microbatches):
                raise RuntimeError(
                    "actor packed-equivalence gate found no multi-sample microbatch "
                    "before the first optimizer step"
                )
            return

        multimodal = batch.get("multimodal_train_inputs") or {}
        raw_vectors = multimodal.get(MM_ACTIVATION_KEY)
        response_lengths = batch.get("response_lengths") or []
        local_ready = torch.tensor(
            int(
                isinstance(raw_vectors, torch.Tensor)
                and raw_vectors.shape[0] >= 2
                and len(response_lengths) >= 2
            ),
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(local_ready, op=dist.ReduceOp.MIN)
        if not int(local_ready.item()):
            raise RuntimeError(
                "actor packed-equivalence gate requires two activation-backed samples "
                "on every FSDP rank"
            )

        inputs = build_packed_padded_inputs(token_rows, sample_limit=2)
        selected_response_lengths = [int(value) for value in response_lengths[:2]]
        previous_vectors = self._nla_vectors
        try:
            self._nla_vectors = normalize_activation(
                raw_vectors[:2], self._nla_cfg.injection_scale
            )
            with torch.no_grad():
                packed_logits = self.model(
                    input_ids=inputs.packed_input_ids,
                    position_ids=inputs.packed_position_ids,
                    attention_mask=None,
                    use_cache=False,
                ).logits
                packed_nlls = response_mean_nlls(
                    packed_logits,
                    inputs.tokens,
                    selected_response_lengths,
                    packed=True,
                )
                del packed_logits
                padded_logits = self.model(
                    input_ids=inputs.padded_input_ids,
                    position_ids=inputs.padded_position_ids,
                    attention_mask=inputs.padded_attention_mask,
                    use_cache=False,
                ).logits
                padded_nlls = response_mean_nlls(
                    padded_logits,
                    inputs.tokens,
                    selected_response_lengths,
                    packed=False,
                )
                del padded_logits
        finally:
            self._nla_vectors = previous_vectors

        metrics = packed_equivalence_metrics(
            packed_nlls,
            padded_nlls,
            rtol=self._nla_actor_packed_equiv_rtol,
            atol=self._nla_actor_packed_equiv_atol,
        )
        global_metrics = torch.tensor(
            [
                float(metrics["max_abs_diff"]),
                float(metrics["max_rel_diff"]),
                0.0 if metrics["passed"] else 1.0,
            ],
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(global_metrics, op=dist.ReduceOp.MAX)
        self._nla_actor_packed_equiv_checked = True
        if self._nla_is_dp_leader():
            print(
                "[NLA PACKED EQUIV] "
                f"samples={int(metrics['sample_count'])} "
                f"packed_mean_nll={metrics['packed_mean_nll']:.8f} "
                f"padded_mean_nll={metrics['padded_mean_nll']:.8f} "
                f"global_max_abs={global_metrics[0].item():.8f} "
                f"global_max_rel={global_metrics[1].item():.8f} "
                f"passed={global_metrics[2].item() == 0.0}",
                flush=True,
            )
        if global_metrics[2].item() != 0.0:
            raise RuntimeError(
                "actor packed-equivalence gate failed: "
                f"global_max_abs={global_metrics[0].item():.8f}, "
                f"global_max_rel={global_metrics[1].item():.8f}"
            )

    def _train_step(self, batch, step_id, num_microbatches):
        with temporarily_select_mamba_training_kernel(
            self.model, self._nla_mamba_training_kernel_mode
        ):
            if self._is_critic_model:
                self._prepare_nla_multimodal_inputs(batch)
                values, backbone_h = _critic_forward_padded_last_values(
                    self.model, batch["unconcat_tokens"]
                )
                batch["_nla_backbone_last_hidden"] = backbone_h.detach()
                batch["_nla_value_head_weight_norm"] = (
                    self.model.value_head.weight.detach().float().norm()
                )
                loss, _, log_dict = loss_function(
                    self.args, self.parallel_state, batch, num_microbatches, values,
                )
                loss.backward()
                return self._attach_system_metrics(log_dict, step_id)
            self._maybe_assert_actor_packed_equivalence(batch, num_microbatches)
            log_dict = super()._train_step(batch, step_id, num_microbatches)
        # FSDP2 overlaps this microbatch's reduce-scatter with the next
        # microbatch's all-gather prefetch. Gemma's 1.41B tied embedding
        # (262k vocab × 5376 d, root FSDP group per fsdp_utils/actor.py:687)
        # lands 5 embedding-sized tensors alive at the boundary = 16.9GB peak
        # on top of model + optimizer. Step 0 survives (no Adam state); once
        # Adam exists (+20GB), any seq-len variance tips it. Diagnosed via
        # torch.cuda.memory._dump_snapshot — _fsdp_collectives.py:508
        # foreach_reduce (5.64GB fp32) + :262 foreach_all_gather (2×2.82GB).
        # Synchronize forces reduce-scatter completion before next all-gather.
        # Keep it for huge embedding tables that need the memory bound; Nano's
        # smaller untied embedding can skip by default unless NLA_SYNC_MICROBATCH
        # forces the legacy behavior.
        if should_synchronize_microbatch(self.model):
            torch.cuda.synchronize()
        return self._attach_system_metrics(log_dict, step_id)

    def critic_fwd(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Inference-only forward, returns values at each sample's last real token.

        Ray-callable from RolloutManager during generation (when trainer is idle).
        FSDP collective — ALL ranks must call this (RayTrainGroup.critic_fwd dispatches
        to every rank). Each rank computes identical output; caller takes rank 0's.

        Returns [B, d] CPU tensor (small → cheap to ship back over Ray object store).
        """
        assert self._is_critic_model, "critic_fwd called on non-critic actor"
        self._emit_phase_snapshot(
            "critic_reward_fwd",
            "start",
            extra={
                "batch_size": int(input_ids.shape[0]),
                "seq_len": int(input_ids.shape[1]),
            },
        )
        ids = input_ids.cuda(non_blocking=True)
        mask = attention_mask.cuda(non_blocking=True)
        # Rightmost True in mask, robust to either padding side.
        # GemmaTokenizerFast defaults to padding_side='left' (mask
        # [0,0,1,1,1]) where the old mask.sum-1 gave n_real-1 instead
        # of T-1. Qwen defaults right, so it worked by accident. At
        # v19 Gemma RL this picked the wrong pos for 31/32 samples —
        # actor chased an artificial length gradient (longer → less
        # padding → less-wrong idx) instead of explanation quality.
        last_idx = mask.cumsum(dim=1).argmax(dim=1)
        # no_grad, NOT inference_mode: the latter marks FSDP's gathered param
        # tensors as inference-only → next training forward crashes at F.linear
        # with "Inference tensors cannot be saved for backward". no_grad is
        # autograd-compatible.
        was_training = self.model.training
        self.model.eval()
        try:
            with _temporarily_disable_mamba_fast_path(self.model), torch.no_grad():
                values, _backbone_h = _critic_forward_padded_last_values_from_padded(
                    self.model,
                    ids,
                    mask,
                    last_idx,
                )
        finally:
            self.model.train(was_training)
            self._emit_phase_snapshot(
                "critic_reward_fwd",
                "end",
                extra={
                    "batch_size": int(input_ids.shape[0]),
                    "seq_len": int(input_ids.shape[1]),
                },
            )
        return values.float().cpu()

    def train(self, rollout_id, rollout_data_ref):
        # Rebuild the critic view from all actor partitions before Miles selects
        # its rank-local ref. This prevents whole-shard assignment followed by
        # cross-rank-min truncation from silently discarding valid rows.
        if self._is_critic_model and self.role == "critic":
            actor_dp = getattr(self, "_nla_actor_dp", self.parallel_state.dp_size)
            alignment = (
                1
                if self.args.use_dynamic_batch_size
                else int(self.args.micro_batch_size)
            )
            rollout_data_ref = _repartition_for_critic(
                rollout_data_ref, actor_dp,
                self.parallel_state.dp_rank, self.parallel_state.dp_size,
                micro_batch_size=alignment,
            )
        return super().train(rollout_id, rollout_data_ref)

    def _train_core(self, rollout_id, rollout_data):
        # All data prep happens here — parent's train() already did
        # get_rollout_data + timers + perf logging.
        if self._is_critic_model:
            repartition_report = rollout_data.pop(
                CRITIC_REPARTITION_REPORT_KEY, None
            )
            if self.role == "critic":
                assert repartition_report is not None, (
                    "online critic rollout is missing its balanced repartition "
                    "report"
                )
                minimum = float(
                    os.environ.get("NLA_MIN_CRITIC_RETAINED_FRACTION", "0.95")
                )
                require_minimum_retained_fraction(repartition_report, minimum)
                self._nla_critic_repartition_report = dict(repartition_report)
                if self._nla_is_dp_leader():
                    print(
                        "[NLA CRITIC BATCH] "
                        + json.dumps(repartition_report, sort_keys=True),
                        flush=True,
                    )
            packed_check_default = self.role == "critic" or int(getattr(self.args, "micro_batch_size", 1)) > 1
            if rollout_id == 0 and _env_assert_packed_equiv(packed_check_default):
                # All ranks run (FSDP forward is collective); each sees its own
                # slice of rollout_data but per-sample ratios must all be ~1.0.
                # Rank-0-gated assert would leave other ranks hanging in FSDP
                # allgather when rank 0 dies — let the exception fire everywhere.
                _assert_reward_train_paths_agree(
                    self.critic_fwd,
                    self.model,
                    rollout_data,
                    self._nla_cfg.mse_scale,
                    tol=0.02,
                    training_kernel_mode=self._nla_mamba_training_kernel_mode,
                )
            if self.role != "critic":
                return super()._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
            bshd_pad_size = None
            if self.args.qkv_format == "bshd":
                bshd_pad_size = (
                    self.parallel_state.tp_size
                    * self.args.data_pad_size_multiplier
                )
            rollout_data = _swap_rollout_to_critic_tokens(
                rollout_data,
                torch.cuda.current_device(),
                bshd_pad_size=bshd_pad_size,
            )
            if repartition_report is not None:
                expected_local = int(repartition_report["local_samples"])
                assert len(rollout_data["tokens"]) == expected_local, (
                    "critic rows changed after parse-valid repartition: "
                    f"expected={expected_local} actual={len(rollout_data['tokens'])}"
                )
            rollout_data = _truncate_to_cross_rank_min(
                rollout_data,
                self.parallel_state.dp_group,
                None if self.args.use_dynamic_batch_size else self.args.micro_batch_size,
            )
            if repartition_report is not None:
                post_filter_retained = (
                    len(rollout_data["tokens"]) * self.parallel_state.dp_size
                )
                assert post_filter_retained == int(
                    repartition_report["retained_samples"]
                ), (
                    "critic safety truncation dropped rows after balanced "
                    f"repartition: expected={repartition_report['retained_samples']} "
                    f"actual={post_filter_retained}"
                )
                self._nla_critic_repartition_report[
                    "post_filter_retained_samples"
                ] = post_filter_retained
            if self._nla_freeze_critic_train:
                self._emit_phase_snapshot(
                    "critic_frozen_train_skip",
                    "skip",
                    rollout_id=rollout_id,
                    extra={
                        "local_samples": len(rollout_data.get("tokens", [])),
                    },
                )
                if self._nla_is_dp_leader():
                    print(
                        "[NLA] frozen critic train: validated critic rollout data; "
                        "skipping backward/optimizer step.",
                        flush=True,
                    )
                self._nla_vectors = None
                return
        elif not self._is_critic_model:
            # LM-actor: strip variable-length critic tokens (would flow to
            # model(**kwargs) as unknown kwarg after multimodal concat).
            for mm in rollout_data.get("multimodal_train_inputs") or []:
                if mm is not None:
                    for k in CRITIC_ONLY_MM_KEYS:
                        mm.pop(k, None)
            cap_report = _cap_actor_sft_rollout_tokens(rollout_data)
            if cap_report is not None and not getattr(self, "_nla_sft_cap_debug_printed", False):
                print(f"[NLA SFT ACTOR TRAIN CAP] {cap_report}", flush=True)
                self._nla_sft_cap_debug_printed = True
            # _compute_log_prob truncates to microbatch boundary (n // micro_bsz * micro_bsz)
            # but rollout_data's per-sample lists stay at the original length. With
            # indivisible counts (e.g. 512 rollouts / 3 DP = 170.67 → 171/170/170, then
            # 171 // 8 = 21 batches = 168), downstream gets len(rewards)=171 vs
            # len(log_probs)=168 → IndexError. Qwen's batch sizes were divisible.
            #
            # Cross-rank sync: each rank's n may differ (171/170/170) → different
            # n_aligned (168/168/168 here, but not guaranteed). get_data_iterator
            # reads dynamic_global_batch_size to compute num_microbatches — if ranks
            # disagree, one gets [] (all_reduce MIN hangs or asserts fail). Same
            # pattern as the critic's _truncate_to_cross_rank_min above.
            n_local = torch.tensor(
                [len(rollout_data.get("tokens", []))],
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(n_local, op=dist.ReduceOp.MIN, group=self.parallel_state.dp_group)
            micro = self.args.micro_batch_size
            n_aligned = (n_local.item() // micro) * micro
            assert n_aligned > 0, (
                f"actor has {n_local.item()} samples after cross-rank MIN, "
                f"fewer than micro_batch_size={micro}. Raise rollout_batch_size."
            )
            n_orig = len(rollout_data.get("tokens", []))
            for k, v in list(rollout_data.items()):
                if isinstance(v, list) and len(v) == n_orig:
                    rollout_data[k] = v[:n_aligned]
            rollout_data["dynamic_global_batch_size"] = (
                n_aligned * dist.get_world_size(self.parallel_state.dp_group)
            )
        self._emit_phase_snapshot(
            "actor_train_core",
            "start",
            rollout_id=rollout_id,
            extra={
                "local_samples": len(rollout_data.get("tokens", [])),
                "dynamic_global_batch_size": int(rollout_data.get("dynamic_global_batch_size", 0) or 0),
            },
        )
        try:
            super()._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
            _emit_nla_advantage_stats(rollout_id, rollout_data)
        finally:
            self._emit_phase_snapshot(
                "actor_train_core",
                "end",
                rollout_id=rollout_id,
                extra={
                    "local_samples": len(rollout_data.get("tokens", [])),
                    "dynamic_global_batch_size": int(rollout_data.get("dynamic_global_batch_size", 0) or 0),
                },
            )
        self._nla_vectors = None

    def save_model(self, rollout_id, force_sync=False):
        if self._is_critic_model and self._nla_freeze_critic_train:
            iter_dir = Path(f"{self.args.save}/iter_{rollout_id + 1:07d}")
            if self._nla_is_dp_leader():
                iter_dir.mkdir(parents=True, exist_ok=True)
                marker = {
                    "schema_version": "nano_nla_frozen_critic_checkpoint.v1",
                    "rollout_id": rollout_id,
                    "source_critic_load": self.args.critic_load,
                    "reason": "NLA_FREEZE_CRITIC_TRAIN=1",
                }
                (iter_dir / "frozen_critic_train_skipped.json").write_text(
                    json.dumps(marker, indent=2, sort_keys=True) + "\n"
                )
                self._write_sidecar(str(iter_dir), rollout_id)
            dist.barrier()
            return
        super().save_model(rollout_id, force_sync)
        if self.args.debug_rollout_only or self.args.save is None:
            return

        # get_model_state_dict with full_state_dict=True is a COLLECTIVE.
        # All ranks must call it or rank 0 deadlocks in the all-gather.
        # actor.py:96 doesn't pass torch_dtype → model stored fp32;
        # MixedPrecision is compute-only. Cast here for 2× smaller saves.
        full_sd = None
        if self._is_critic_model:
            full_sd = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            full_sd = {
                k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v)
                for k, v in full_sd.items()
            }

        # Match fsdp_utils/checkpoint.py:199's iter_{rollout_id+1} convention.
        iter_dir = f"{self.args.save}/iter_{rollout_id + 1:07d}"

        if dist.get_rank() == 0:
            if self._is_critic_model:
                hf_dir = f"{iter_dir}/hf"
                self.model.save_pretrained(hf_dir, state_dict=full_sd)
                self.tokenizer.save_pretrained(hf_dir)
                # Write sidecar at BOTH hf/ and iter_N/ — the hf/ one is what
                # --critic-load needs (alongside config.json), the iter_N/ one
                # is a footgun defuser (if someone points at the wrong level).
                self._write_sidecar(hf_dir, rollout_id)
                self._write_sidecar(iter_dir, rollout_id)
            else:
                # Actor saves DCP only (model/, optimizer/). To load for RL:
                #   --hf-checkpoint = base model (from_pretrained needs safetensors)
                #   --load = this iter_dir (DCP overwrites weights)
                #   --nla-sidecar-source = this iter_dir (injection_scale from sidecar)
                # SGLang also loads base model; update_weights syncs SFT weights via NCCL.
                self._write_sidecar(iter_dir, rollout_id)
            keep_n = max(1, int(os.environ.get("NLA_KEEP_LOCAL", "1")))
            save_dir = self.args.save

            def _bg():
                self._maybe_background_push()
                prune = (f"ls -1d {save_dir}/iter_* 2>/dev/null | "
                         f"head -n -{keep_n} | xargs -r rm -rf")
                subprocess.run(["bash", "-c", prune], check=False)

            threading.Thread(target=_bg, daemon=True).start()
        dist.barrier()

    def _maybe_background_push(self):
        """Fire-and-forget GCS push after each checkpoint, if NLA_BACKUP_REMOTE is set.

        For gs:// remotes, uses gsutil -m cp -r directly (no extra dep).
        For other schemes, goes through push_checkpoint + storage_cls.
        start_new_session detaches — upload survives later pkill of trainer.
        """
        remote = os.environ.get("NLA_BACKUP_REMOTE")
        if not remote:
            return
        # Serialize with the previous push: save_model's _bg thread prunes old
        # iter_* right after this returns. With keep_n=2 and one-behind push,
        # save N+1's prune deletes iter_{N-1} — which save N's gsutil may still
        # be uploading → truncated/corrupt remote. Block until that prior upload
        # finishes; the next prune is then safe.
        prev = getattr(self, "_push_proc", None)
        if prev is not None:
            prev.wait()
        role = "critic" if self._is_critic_model else "actor"
        remote_dir = f"{remote}/{role}"
        log = f"/tmp/push_{role}_iter.log"
        tracker = Path(self.args.save) / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            # First save with --async-save: super() started the write, no prior
            # to finalize → no tracker yet. Push is one-behind by design; nothing
            # to push on the first save.
            return
        latest = tracker.read_text().strip()
        iter_dir = f"{self.args.save}/iter_{int(latest):07d}"
        if remote.startswith("gs://"):
            if role == "actor":
                train_log = os.environ.get("NLA_TRAIN_LOG")
                if train_log and Path(train_log).exists():
                    shutil.copy(train_log, f"{iter_dir}/train.log")
                # RolloutManager writes sample_offset/epoch_id to a SIBLING
                # rollout/ dir — not inside iter_dir, so gsutil cp -r iter_dir
                # misses it. Snapshot into iter_dir so resume from GCS restores
                # data offset (else fresh pod → multi-epoch on first ~128k rows).
                rollout_state_dir = Path(self.args.save) / "rollout"
                if rollout_state_dir.exists():
                    for f in rollout_state_dir.glob("global_dataset_state_dict_*.pt"):
                        shutil.copy(f, f"{iter_dir}/{f.name}")
            # env -u PYTHONPATH: training's PYTHONPATH (Megatron-LM checkout)
            # leaks into nix gsutil's subprocess → boto's
            # platform.python_version() chokes on conda-forge sys.version string.
            # Push only — caller handles prune (both backends: daemon thread
            # in save_model, push-then-prune). Chaining prune here previously
            # meant the `ls` readdir hung under async-save bg-write saturation.
            cmd = ["bash", "-c",
                   f"env -u PYTHONPATH gsutil -m cp -r {iter_dir} {remote_dir}/"]
        else:
            storage_cls = os.environ.get("NLA_BACKUP_STORAGE_CLS")
            assert storage_cls, "NLA_BACKUP_STORAGE_CLS required for non-gs:// remote"
            cmd = [sys.executable, "-m", "nla.scripts.push_checkpoint",
                   "--local", self.args.save, "--remote", remote_dir,
                   "--storage-cls", storage_cls, "--only-latest"]
        self._push_proc = subprocess.Popen(
            cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"[NLA] background push fired: {iter_dir} → {remote_dir} (log: {log})")

    def _write_sidecar(self, checkpoint_dir: str, rollout_id: int):
        cfg = self._nla_cfg
        if self._is_critic_model:
            # num_hidden_layers is K+1 (blocks 0..K inclusive — we need
            # the output OF block K). Sidecar stores K (the extraction layer_index,
            # matching datagen's extraction.layer_index convention).
            cfg = replace(cfg, critic_num_layers=self._text_config.num_hidden_layers - 1)
        write_model_sidecar(
            checkpoint_dir, cfg,
            role="critic" if self._is_critic_model else "actor",
            stage="rl" if self.args.loss_type == "policy_loss" else "sl",
            base_checkpoint=self.args.hf_checkpoint,
            trained_on=[self.args.prompt_data] if self.args.prompt_data else [],
            parent_checkpoints=[self.args.hf_checkpoint],
            created_by="nla.train_actor.NLAFSDPActor",
            training_args={
                "rollout_id": rollout_id,
                "lr": self.args.lr,
                "loss_type": self.args.loss_type,
                "global_batch_size": self.args.global_batch_size,
            },
        )
