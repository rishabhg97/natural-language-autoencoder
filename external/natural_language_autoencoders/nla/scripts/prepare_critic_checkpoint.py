"""One-time preprocessing: base model → truncated critic-init checkpoint.

Loads a base HF model, keeps blocks 0..K (K+1 layers), strips lm_head + final
LN, saves as a standalone HF checkpoint with config.num_hidden_layers = K+1.
The critic's last_hidden_state is then the output OF block K — exactly what
datagen captured at extraction layer_index K.

After this, Critic-SL's --hf-checkpoint points at the output dir and
NLACriticModel.from_pretrained just works — no layer-count arg needed.

Also writes an nla_meta.yaml sidecar so load_nla_config succeeds. Token IDs
and prompt templates are copied from the DATASET sidecar (the one next to
the parquet you'll train on) since those are dataset-pinned, not model-pinned.

Usage:
    python -m nla.scripts.prepare_critic_checkpoint \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --num-layers 20 \
        --dataset-sidecar path/to/ar_sft.parquet \
        --output /path/to/critic_init
"""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from nla.config import load_nla_config, write_model_sidecar
from nla.models import NLACriticModel


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _parameter_group_sha256(parameters: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = parameter.detach().contiguous().cpu().view(torch.uint8)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _initialize_value_head(
    weight: torch.Tensor,
    *,
    mode: str,
    seed: int,
    rotation_radians: float,
) -> dict[str, object]:
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError(f"value head must be square, got {tuple(weight.shape)}")
    if mode not in {"identity", "seeded_givens"}:
        raise ValueError(f"unsupported value-head initialization: {mode}")
    if mode == "seeded_givens" and not 0.0 < rotation_radians < math.pi:
        raise ValueError("rotation_radians must be in (0, pi) for seeded_givens")

    before_sha256 = _tensor_sha256(weight)
    dimension = weight.shape[0]
    with torch.no_grad():
        weight.zero_()
        diagonal = torch.arange(dimension, device=weight.device)
        weight[diagonal, diagonal] = 1
        if mode == "seeded_givens":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            permutation = torch.randperm(dimension, generator=generator)
            left = permutation[0 : 2 * (dimension // 2) : 2]
            right = permutation[1 : 2 * (dimension // 2) : 2]
            signs = torch.randint(0, 2, (left.numel(),), generator=generator, dtype=torch.int64)
            signs = signs.mul(2).sub(1).to(dtype=torch.float32)
            left = left.to(weight.device)
            right = right.to(weight.device)
            signs = signs.to(device=weight.device, dtype=weight.dtype)
            cosine = math.cos(rotation_radians)
            sine = math.sin(rotation_radians)
            weight[left, left] = cosine
            weight[right, right] = cosine
            weight[left, right] = -signs * sine
            weight[right, left] = signs * sine

    return {
        "mode": mode,
        "seed": seed if mode == "seeded_givens" else None,
        "dimension": dimension,
        "rotation_radians": rotation_radians if mode == "seeded_givens" else 0.0,
        "before_sha256": before_sha256,
        "after_sha256": _tensor_sha256(weight),
    }


def _is_router_parameter(name: str) -> bool:
    return name.endswith(".mixer.gate.weight") or name.endswith(
        ".mixer.gate.e_score_correction_bias"
    )


def _perturb_router_parameters(
    model: torch.nn.Module,
    *,
    mode: str,
    seed: int,
    relative_std: float,
) -> dict[str, object]:
    if mode not in {"pretrained", "seeded_relative_noise"}:
        raise ValueError(f"unsupported router initialization: {mode}")
    if mode == "seeded_relative_noise" and relative_std <= 0.0:
        raise ValueError("relative_std must be positive for seeded_relative_noise")

    parameters = sorted(
        ((name, parameter) for name, parameter in model.named_parameters() if _is_router_parameter(name)),
        key=lambda item: item[0],
    )
    before_sha256 = _parameter_group_sha256(parameters)
    parameter_reports: list[dict[str, object]] = []
    if mode == "seeded_relative_noise":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        with torch.no_grad():
            for name, parameter in parameters:
                parameter_rms = float(parameter.detach().float().square().mean().sqrt().cpu())
                noise_std = max(parameter_rms, 1e-6) * relative_std
                noise = torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                parameter.add_(noise.to(device=parameter.device, dtype=parameter.dtype), alpha=noise_std)
                parameter_reports.append(
                    {
                        "name": name,
                        "parameter_rms_before": parameter_rms,
                        "noise_std": noise_std,
                        "after_sha256": _tensor_sha256(parameter),
                    }
                )

    return {
        "mode": mode,
        "seed": seed if mode == "seeded_relative_noise" else None,
        "relative_std": relative_std if mode == "seeded_relative_noise" else 0.0,
        "parameter_count": len(parameters),
        "before_sha256": before_sha256,
        "after_sha256": _parameter_group_sha256(parameters),
        "parameters": parameter_reports,
    }


def _auto_map_module_names(auto_map: object) -> set[str]:
    names: set[str] = set()
    if not isinstance(auto_map, dict):
        return names
    values: list[object] = list(auto_map.values())
    for value in values:
        refs = value if isinstance(value, (list, tuple)) else [value]
        for ref in refs:
            if not isinstance(ref, str) or "." not in ref:
                continue
            module_ref = ref.rsplit(".", 1)[0].split("--")[-1]
            names.add(module_ref)
    return names


def _copy_remote_code_files(base_model: str, output_dir: str, auto_map: object) -> None:
    """Preserve HF remote-code modules referenced by config.auto_map.

    Transformers' save_pretrained writes the config, but local remote-code
    checkpoints may not copy every `auto_map` module into the destination. A
    truncated Nano critic must remain loadable with trust_remote_code=True.
    """
    module_names = _auto_map_module_names(auto_map)
    if not module_names:
        return

    source = Path(base_model)
    if not source.exists():
        source = Path(snapshot_download(base_model))
    out = Path(output_dir)
    for module_name in module_names:
        rel = Path(*module_name.split(".")).with_suffix(".py")
        src = source / rel
        if not src.exists():
            raise FileNotFoundError(f"config.auto_map references {rel}, but it is missing from {source}")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        if dst.name == "modeling_nemotron_h.py":
            from nla.remote_code_patches import patch_nemotron_h_file_if_needed

            patch_nemotron_h_file_if_needed(dst)


def _add_megatron_compat_keys(
    output_dir: str | Path,
    value_head_weight: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """mbridge's Qwen2Bridge hard-requires model.norm.weight + lm_head.weight.
    NLACriticModel drops both (Identity norm, d×d value_head separately saved).
    Convert builds output_layer as d×d (critic_output_size=hidden_size), so
    lm_head must be the exact d×d initialized value head for mbridge scatter to
    preserve the critic contract. norm.weight=ones is a no-op under Identity."""
    out = Path(output_dir)
    hidden_size = value_head_weight.shape[0]
    if value_head_weight.shape != (hidden_size, hidden_size):
        raise ValueError(f"value head must be square, got {tuple(value_head_weight.shape)}")
    compat_file = "model-megatron-compat.safetensors"
    save_file(
        {
            "model.norm.weight": torch.ones(hidden_size, dtype=dtype),
            "lm_head.weight": value_head_weight.detach().to(device="cpu", dtype=dtype).contiguous(),
        },
        out / compat_file,
    )
    idx_path = out / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    else:
        single = out / "model.safetensors"
        assert single.exists(), f"neither {idx_path.name} nor {single.name} found in {out}"
        with safe_open(single, framework="pt") as f:
            idx = {"metadata": {}, "weight_map": {k: single.name for k in f.keys()}}
    idx["weight_map"]["model.norm.weight"] = compat_file
    idx["weight_map"]["lm_head.weight"] = compat_file
    idx_path.write_text(json.dumps(idx, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True,
                   help="HF checkpoint to truncate (local dir or hub name)")
    p.add_argument("--num-layers", type=int, required=True,
                   help="The datagen extraction layer_index (K). Critic keeps "
                        "blocks 0..K inclusive (num_hidden_layers = K+1 in config.json) "
                        "so last_hidden_state = output of block K = what datagen captured.")
    p.add_argument("--dataset-sidecar", required=True,
                   help="Path to the dataset parquet whose sidecar has token IDs + templates "
                        "(reads {path}.nla_meta.yaml)")
    p.add_argument("--output", required=True, help="Output directory for truncated checkpoint")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument(
        "--value-head-init",
        choices=("identity", "seeded_givens"),
        default="identity",
    )
    p.add_argument("--initialization-seed", type=int, default=0)
    p.add_argument("--value-head-rotation-radians", type=float, default=0.2)
    p.add_argument(
        "--router-init",
        choices=("pretrained", "seeded_relative_noise"),
        default="pretrained",
    )
    p.add_argument("--router-relative-std", type=float, default=0.01)
    p.add_argument("--megatron-compat", action="store_true",
                   help="Write dummy model.norm.weight + lm_head.weight so mbridge "
                        "convert_hf_to_torch_dist can handle the non-standard structure. "
                        "NLAMegatronActor replaces both post-load.")
    args = p.parse_args()

    dtype = getattr(torch, args.torch_dtype)

    print(f"Loading {args.base_model} (truncating to {args.num_layers} layers)...")
    model = NLACriticModel.from_pretrained(
        args.base_model,
        nla_num_layers=args.num_layers,
        torch_dtype=dtype,
    )
    value_head_initialization = _initialize_value_head(
        model.value_head.weight,
        mode=args.value_head_init,
        seed=args.initialization_seed,
        rotation_radians=args.value_head_rotation_radians,
    )
    router_initialization = _perturb_router_parameters(
        model,
        mode=args.router_init,
        seed=args.initialization_seed,
        relative_std=args.router_relative_std,
    )
    print(f"Saving to {args.output}...")
    model.save_pretrained(args.output)
    _copy_remote_code_files(args.base_model, args.output, getattr(model.backbone.config, "auto_map", None))
    if args.megatron_compat:
        _add_megatron_compat_keys(
            args.output,
            model.value_head.weight,
            dtype,
        )
    initialization_manifest = {
        "schema_version": "nano_critic_initialization.v1",
        "base_model": args.base_model,
        "dataset_sidecar": args.dataset_sidecar,
        "extraction_layer_index": args.num_layers,
        "torch_dtype": args.torch_dtype,
        "value_head": value_head_initialization,
        "router": router_initialization,
    }
    (Path(args.output) / "critic_initialization.json").write_text(
        json.dumps(initialization_manifest, indent=2, sort_keys=True) + "\n"
    )

    print(f"Loading dataset sidecar from {args.dataset_sidecar}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    cfg = load_nla_config(args.dataset_sidecar, tokenizer)
    assert cfg.d_model == model.config.hidden_size, (
        f"dataset d_model={cfg.d_model} != model hidden_size={model.config.hidden_size}"
    )

    # Bake critic_num_layers into the model sidecar so downstream loads know
    # the truncation without re-reading the dataset sidecar.
    cfg_with_k = replace(cfg, critic_num_layers=args.num_layers)

    print(f"Writing nla_meta.yaml...")
    write_model_sidecar(
        args.output, cfg_with_k,
        role="critic", stage="init",
        base_checkpoint=args.base_model,
        trained_on=[], parent_checkpoints=[args.base_model],
        created_by="nla.scripts.prepare_critic_checkpoint",
    )

    # LlamaTokenizerFast/GemmaTokenizerFast default padding_side='left' (for
    # generation). Downstream reward.py tokenizes with this tokenizer; Megatron's
    # critic_fwd passes attention_mask=None (causal-only) so left-pad tokens would
    # be attended by the last real token. Force right-pad at save so the causal-only
    # assumption holds and the the left-pad fix cumsum fix isn't load-bearing.
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(args.output)
    print(f"Done. Use --hf-checkpoint {args.output} for Critic-SL.")


if __name__ == "__main__":
    main()
