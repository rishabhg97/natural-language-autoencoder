import argparse
import json
import os
import pickle
import re
import shutil
import time

import torch
import torch.distributed.checkpoint as dist_cp
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM
from typing_extensions import override


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _detect_model_dir(input_dir: str) -> str:
    model_dir = os.path.join(input_dir, "model")
    return model_dir if os.path.isdir(model_dir) else input_dir


def _resolve_torch_dtype(name: str | None) -> torch.dtype | None:
    if name in {None, "", "auto", "preserve"}:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return mapping[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype {name!r}; choose preserve, float32, bfloat16, or float16") from exc


def _dtype_config_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    mapping = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }
    return mapping.get(dtype, str(dtype).removeprefix("torch."))


def _cast_floating_tensors(
    tensor_items: dict[str, torch.Tensor],
    *,
    torch_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    if torch_dtype is None:
        return tensor_items
    return {
        key: tensor.to(dtype=torch_dtype) if tensor.is_floating_point() else tensor
        for key, tensor in tensor_items.items()
    }


def _load_fsdp_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state_dict


def _get_candidate_prefixes(keys: list[str]) -> list[str]:
    predefined = [
        "model_state.model.",
        "model_state.",
        "model.",
        "module.",
        "",
    ]

    detected: set[str] = set()
    for key in keys:
        for prefix in predefined:
            if prefix and key.startswith(prefix):
                detected.add(prefix)

    # Always keep empty string as a fall back option for exact match.
    detected.add("")
    # Preserve predefined order while keeping only detected prefixes.
    return [p for p in predefined if p in detected]


def _strip_best_prefix_with_normalizer(
    keys: list[str],
    target_keys: set[str],
    normalizer,
) -> tuple[str, int]:
    best_prefix = ""
    best_match = -1

    for prefix in _get_candidate_prefixes(keys):
        mapped_keys = {normalizer(k.removeprefix(prefix)) for k in keys}
        match_count = len(mapped_keys & target_keys)
        if match_count > best_match:
            best_match = match_count
            best_prefix = prefix

    return best_prefix, best_match


def _strip_best_prefix(keys: list[str], target_keys: set[str]) -> tuple[str, int]:
    return _strip_best_prefix_with_normalizer(keys, target_keys, _normalize_model_key)


def _strip_best_raw_prefix(keys: list[str], target_keys: set[str]) -> tuple[str, int]:
    return _strip_best_prefix_with_normalizer(keys, target_keys, lambda key: key)


_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<base>model\.layers\.\d+\.mixer\.experts)\."
    r"(?P<expert>\d+)\."
    r"(?P<proj>up_proj|down_proj)\.weight$"
)


def _normalize_model_key(key: str) -> str:
    if key.startswith("backbone."):
        return "model." + key.removeprefix("backbone.")
    return key


def _build_hf_model_state(
    tensor_items: dict[str, torch.Tensor],
    best_prefix: str,
    target_keys: set[str],
) -> dict[str, torch.Tensor]:
    model_state: dict[str, torch.Tensor] = {}
    expert_parts: dict[str, dict[int, torch.Tensor]] = {}

    for key, tensor in tensor_items.items():
        normalized_key = _normalize_model_key(key.removeprefix(best_prefix))
        expert_match = _EXPERT_WEIGHT_RE.match(normalized_key)
        if expert_match:
            packed_key = f"{expert_match.group('base')}.{expert_match.group('proj')}"
            expert_parts.setdefault(packed_key, {})[int(expert_match.group("expert"))] = tensor
            continue
        model_state[normalized_key] = tensor

    for packed_key, parts in expert_parts.items():
        if packed_key not in target_keys:
            raise KeyError(f"Packed expert key {packed_key!r} is not present in the HF skeleton")
        ordered_indices = sorted(parts)
        expected_indices = list(range(ordered_indices[-1] + 1))
        if ordered_indices != expected_indices:
            raise ValueError(
                f"Expert shards for {packed_key!r} are not contiguous: "
                f"got {ordered_indices[:3]}...{ordered_indices[-3:]}"
            )
        model_state[packed_key] = torch.stack([parts[index] for index in ordered_indices], dim=0)

    return model_state


def _resolve_skeleton_config(origin_config, dcp_keys: list[str]):
    """Return the config to build the load-target skeleton from.

    DCP matches by key name, so the skeleton's state_dict keys must match what
    was saved. If origin is a multimodal wrapper (Gemma-3: has .text_config) but
    the DCP was saved from the unwrapped text-only CausalLM (NLATextOnlyCausalLM
    in nla/train_actor.py), the wrapper skeleton's keys are language_model.model.*
    while the DCP has model.* -- zero overlap, load_state_dict(strict=False)
    silently keeps random init. See PORTING_NEW_ARCHITECTURES.md.

    Detection: wrapper config + no DCP key contains the wrapper module prefix
    -> DCP is text-only -> build skeleton from text_config.
    AutoModelForCausalLM._model_mapping[Gemma3TextConfig] -> Gemma3ForCausalLM,
    whose state_dict keys are model.* -- matches the DCP.

    Qwen/Llama/Mistral configs have no .text_config -> pass through unchanged.
    """
    text_config = getattr(origin_config, "text_config", None)
    if text_config is None:
        return origin_config
    if any("language_model." in k for k in dcp_keys):
        return origin_config
    print(
        f"Origin config {type(origin_config).__name__} is a multimodal wrapper but "
        f"DCP has no language_model.* keys -- building text-only skeleton from "
        f"{type(text_config).__name__} (model_type={text_config.model_type})."
    )
    return text_config


def _patch_config_for_skeleton_init(config):
    """Fill config defaults required by newer HF model constructors.

    Some Nemotron-H checkpoints were saved with configs that omit
    ``moe_latent_size``. Transformers 5.9 reads that attribute directly during
    ``AutoModelForCausalLM.from_config``; the intended default is ``None``.
    """
    if getattr(config, "model_type", None) == "nemotron_h" and not hasattr(config, "moe_latent_size"):
        config.moe_latent_size = None
    return config


def _write_origin_config_with_compat(
    origin_hf_dir: str,
    output_dir: str,
    *,
    torch_dtype: torch.dtype | None = None,
) -> None:
    origin_config_path = os.path.join(origin_hf_dir, "config.json")
    if not os.path.isfile(origin_config_path):
        return
    with open(origin_config_path) as f:
        data = json.load(f)
    if data.get("model_type") == "nemotron_h":
        data.setdefault("moe_latent_size", None)
    dtype_name = _dtype_config_name(torch_dtype)
    if dtype_name is not None:
        data["torch_dtype"] = dtype_name
        if isinstance(data.get("text_config"), dict):
            data["text_config"]["torch_dtype"] = dtype_name
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _convert_with_origin_safetensors_layout(
    origin_hf_dir: str,
    tensor_items: dict[str, torch.Tensor],
    output_dir: str,
    *,
    torch_dtype: torch.dtype | None = None,
) -> bool:
    """Write DCP tensors in the origin checkpoint's HF safetensors key layout.

    Nano/Nemotron-H's remote-code checkpoint stores expert tensors split as
    ``backbone.layers.*.experts.N.*.weight``. The built-in Transformers module
    exposes packed ``model.layers.*.experts.*`` parameters, but the local Nano
    checkpoint should stay faithful to the origin remote-code layout.
    """
    index_path = os.path.join(origin_hf_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return False
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return False

    target_keys = set(weight_map)
    best_prefix, best_match = _strip_best_raw_prefix(list(tensor_items.keys()), target_keys)
    tensor_items = _cast_floating_tensors(tensor_items, torch_dtype=torch_dtype)
    model_state = {key.removeprefix(best_prefix): tensor for key, tensor in tensor_items.items()}
    missing = sorted(target_keys - set(model_state))
    unexpected = sorted(set(model_state) - target_keys)
    print(
        f"Origin safetensors layout match using prefix '{best_prefix}': "
        f"{best_match}/{len(tensor_items)} DCP keys."
    )
    if missing or unexpected:
        print(
            "Origin safetensors layout did not exactly match; falling back to "
            f"skeleton conversion. Missing={missing[:3]} Unexpected={unexpected[:3]}"
        )
        return False

    os.makedirs(output_dir, exist_ok=True)
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    for shard, keys in sorted(shard_to_keys.items()):
        shard_state = {key: model_state[key] for key in keys}
        save_file(shard_state, os.path.join(output_dir, shard), metadata={"format": "pt"})

    total_size = sum(int(tensor.numel() * tensor.element_size()) for tensor in model_state.values())
    out_index = {
        "metadata": {**index.get("metadata", {}), "total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(out_index, f, indent=2)
        f.write("\n")
    _write_origin_config_with_compat(origin_hf_dir, output_dir, torch_dtype=torch_dtype)
    dtype_note = f", torch_dtype={_dtype_config_name(torch_dtype)}" if torch_dtype is not None else ""
    print(f"Model weights saved to {output_dir} with origin safetensors layout{dtype_note}")
    return True


def _convert_fsdp_to_hf(
    origin_hf_dir: str,
    input_dir: str,
    output_dir: str,
    *,
    torch_dtype: torch.dtype | None = None,
) -> None:
    print(f"loading FSDP model from {input_dir}")
    t = time.time()
    state_dict = _load_fsdp_state_dict(input_dir)
    print(f"FSDP model loaded in {time.time()-t:.2f} sec.")

    tensor_items = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if not tensor_items:
        raise ValueError(
            "No model weights found in checkpoint. "
            "Please pass the checkpoint directory (e.g. iter_xxx or iter_xxx/model)."
        )
    tensor_items = _cast_floating_tensors(tensor_items, torch_dtype=torch_dtype)
    if _convert_with_origin_safetensors_layout(origin_hf_dir, tensor_items, output_dir, torch_dtype=torch_dtype):
        return

    origin_config = AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)
    config = _resolve_skeleton_config(origin_config, list(tensor_items.keys()))
    config = _patch_config_for_skeleton_init(config)
    hf_model = AutoModelForCausalLM.from_config(config)
    if torch_dtype is not None:
        hf_model.to(dtype=torch_dtype)
    target_keys = set(hf_model.state_dict().keys())

    best_prefix, best_match = _strip_best_prefix(list(tensor_items.keys()), target_keys)
    print(
        f"Skeleton: {type(hf_model).__name__} ({len(target_keys)} params). "
        f"Using prefix '{best_prefix}', matched {best_match}/{len(tensor_items)} DCP keys."
    )

    model_state = _build_hf_model_state(tensor_items, best_prefix, target_keys)
    missing, unexpected = hf_model.load_state_dict(model_state, strict=False)
    print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    assert not missing, (
        f"{len(missing)} skeleton params received no DCP weights -- saved model would "
        f"have random-init garbage. DCP key shape does not match "
        f"{type(hf_model).__name__}. First missing: {missing[:3]}"
    )
    assert not unexpected, (
        f"{len(unexpected)} DCP weights did not map into the HF skeleton. "
        f"First unexpected: {unexpected[:3]}"
    )

    # save_pretrained writes config.torch_dtype from the skeleton (init'd on meta
    # device, never had weights → dtype unset → None). sglang reads None → dtype
    # mismatch: fp32 embeddings hit bf16 weights → RuntimeError at qkv_proj.
    # DCP tensors carry their own dtype; take it from the first one.
    tensor_dtype = torch_dtype or next(iter(model_state.values())).dtype
    hf_model.config.torch_dtype = tensor_dtype
    text_cfg = getattr(hf_model.config, "text_config", None)
    if text_cfg is not None:
        text_cfg.torch_dtype = tensor_dtype
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model weights saved to {output_dir} (torch_dtype={tensor_dtype})")


def copy_assets(origin_hf_dir: str, output_dir: str) -> None:
    if not os.path.isdir(origin_hf_dir):
        print(f"Skip copy_assets: {origin_hf_dir} is not a local directory (hub ID?). "
              f"config.json already written by save_pretrained; fetch tokenizer separately if needed.")
        return
    # config.json was written by save_pretrained with the correct (possibly text-only)
    # architectures -- copying origin's multimodal config.json would clobber it.
    skip = {"model.safetensors.index.json", "config.json"}
    for filename in os.listdir(origin_hf_dir):
        if filename in skip or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="The original Hugging Face model directory to load config/tokenizer assets.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "--torch-dtype",
        default="preserve",
        choices=("preserve", "auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"),
        help="Optional dtype for floating model tensors in the emitted HF checkpoint. Default preserves DCP dtype.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    model_dir = _detect_model_dir(args.input_dir)
    _convert_fsdp_to_hf(
        args.origin_hf_dir,
        model_dir,
        args.output_dir,
        torch_dtype=_resolve_torch_dtype(args.torch_dtype),
    )
    copy_assets(args.origin_hf_dir, args.output_dir)
