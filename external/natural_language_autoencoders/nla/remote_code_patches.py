"""Patch helpers for Nano/Nemotron-H Hugging Face remote-code files.

The base model and prepared critic checkpoints carry their own
``modeling_nemotron_h.py``. These helpers make the audit remediations explicit
and repeatable instead of editing checkpoint directories by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


PATCH_MARKER = "# NLA_AUDIT_PATCHED_NEMOTRON_H_20260610"
PACKED_ATTENTION_MARKER = "# NLA_PACKED_ATTENTION_BOUNDARY_MASK"
RMSNORM_FALLBACK_MARKER = "# NLA_MAMBA_RMSNORM_TORCH_FALLBACK"

RMSNORM_FALLBACK = '''try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    # NLA_MAMBA_RMSNORM_TORCH_FALLBACK
    def rmsnorm_fn(
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
    ):
        """PyTorch equivalent of mamba_ssm's gated RMSNorm reference path."""
        output_dtype = x.dtype
        x_float = x.float()
        z_float = z.float() if z is not None else None
        weight_float = weight.float()
        bias_float = bias.float() if bias is not None else None

        if z_float is not None and not norm_before_gate:
            x_float = x_float * torch.nn.functional.silu(z_float)

        width = x_float.shape[-1]
        normalized_group_size = width if group_size is None else int(group_size)
        if normalized_group_size <= 0 or width % normalized_group_size != 0:
            raise ValueError(
                f"group_size must divide the hidden width: "
                f"group_size={normalized_group_size}, width={width}"
            )
        grouped = x_float.reshape(*x_float.shape[:-1], -1, normalized_group_size)
        reciprocal_rms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
        output = (grouped * reciprocal_rms).reshape_as(x_float) * weight_float
        if bias_float is not None:
            output = output + bias_float
        if z_float is not None and norm_before_gate:
            output = output * torch.nn.functional.silu(z_float)
        return output.to(output_dtype)
'''

NEMOTRON_H_SOURCE_MARKERS = (
    "class NemotronH",
    "mamba_split_conv1d_scan_combined",
    "mamba_chunk_scan_combined",
)

SEQ_IDX_HELPER = '''def _nla_seq_idx_from_position_ids(position_ids, batch_size=None):
    if position_ids is None:
        return None
    pos = position_ids
    if getattr(pos, "ndim", 0) == 1:
        pos = pos.unsqueeze(0)
    if pos.numel() == 0:
        return None
    if batch_size is not None and getattr(pos, "ndim", 0) == 2 and pos.shape[0] == 1 and batch_size != 1:
        pos = pos.expand(batch_size, -1)
    boundaries = pos == 0
    return boundaries.to(dtype=torch.int32).cumsum(dim=-1, dtype=torch.int32) - 1
'''


@dataclass
class PatchReport:
    changed: bool = False
    already_patched: bool = False
    seq_idx_replacements: int = 0
    kernel_seq_idx_replacements: int = 0
    mamba_signature_replacements: int = 0
    mamba_forward_call_replacements: int = 0
    block_signature_replacements: int = 0
    block_mixer_call_replacements: int = 0
    block_attention_call_replacements: int = 0
    model_seq_idx_replacements: int = 0
    model_block_call_replacements: int = 0
    attention_mask_replacements: int = 0
    causal_mask_signature_replacements: int = 0
    causal_mask_call_replacements: int = 0
    packed_attention_boundary_replacements: int = 0
    causal_lm_position_ids_replacements: int = 0
    moe_replaced: bool = False
    moe_signature_replacements: int = 0
    router_post_init_replacements: int = 0
    generation_cache_conv_kernel_replacements: int = 0
    generation_cache_device_replacements: int = 0
    generation_cache_reset_replacements: int = 0
    rmsnorm_fallback_replacements: int = 0
    helper_injected: bool = False
    validation_errors: list[str] | None = None

    def add_error(self, message: str) -> None:
        if self.validation_errors is None:
            self.validation_errors = []
        self.validation_errors.append(message)


def _patch_rmsnorm_fallback(source: str, report: PatchReport) -> str:
    if RMSNORM_FALLBACK_MARKER in source:
        return source
    pattern = re.compile(
        r"^try:\n"
        r"(?:[ \t]+[^\n]*\n)*?"
        r"[ \t]+from mamba_ssm\.ops\.triton\.layernorm_gated import rmsnorm_fn\n"
        r"except ImportError:\n"
        r"[ \t]+raise ImportError\([^\n]+\)\n",
        flags=re.MULTILINE,
    )
    source, count = pattern.subn(RMSNORM_FALLBACK, source, count=1)
    report.rmsnorm_fallback_replacements += count
    return source


def _inject_helper(source: str, report: PatchReport) -> str:
    if PATCH_MARKER in source:
        report.already_patched = True
        source, count = re.subn(
            r"def _nla_seq_idx_from_position_ids\(.*?\n\n(?=(?:def|class|import|from)\s)",
            SEQ_IDX_HELPER + "\n\n",
            source,
            count=1,
            flags=re.DOTALL,
        )
        report.helper_injected = bool(count)
        return source
    helper = f'''

{PATCH_MARKER}
{SEQ_IDX_HELPER}

def _nla_keep_router_buffers_fp32(model):
    for module in model.modules():
        name = type(module).__name__.lower()
        if "router" not in name and "gate" not in name:
            continue
        weight = getattr(module, "weight", None)
        if weight is not None and getattr(weight, "dtype", None) is not None:
            try:
                module.weight.data = module.weight.data.float()
            except Exception:
                pass
        bias = getattr(module, "e_score_correction_bias", None)
        if bias is not None and getattr(bias, "dtype", None) is not None:
            try:
                module.e_score_correction_bias = bias.float()
            except Exception:
                pass
'''
    match = re.search(r"^(import .*\n|from .* import .*\n)+", source, flags=re.MULTILINE)
    if match:
        insert_at = match.end()
        report.helper_injected = True
        return source[:insert_at] + helper + source[insert_at:]
    report.helper_injected = True
    return helper + "\n" + source


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _class_span(lines: list[str], class_name: str) -> tuple[int, int] | None:
    pattern = re.compile(rf"^class\s+{re.escape(class_name)}\b")
    for start, line in enumerate(lines):
        if not pattern.match(line):
            continue
        indent = _line_indent(line)
        end = len(lines)
        for idx in range(start + 1, len(lines)):
            if lines[idx].strip() and _line_indent(lines[idx]) <= indent and not lines[idx].startswith((" ", "\t")):
                end = idx
                break
        return start, end
    return None


def _function_signature_span(lines: list[str], class_name: str, function_name: str) -> tuple[int, int] | None:
    span = _class_span(lines, class_name)
    if span is None:
        return None
    start, end = span
    pattern = re.compile(rf"^\s+def\s+{re.escape(function_name)}\(")
    for idx in range(start + 1, end):
        if not pattern.match(lines[idx]):
            continue
        sig_end = idx
        while sig_end < end and not re.search(r"\)\s*(?:->.*)?\s*:\s*$", lines[sig_end]):
            sig_end += 1
        if sig_end >= end:
            return None
        return idx, sig_end
    return None


def _add_param_to_function(source: str, class_name: str, function_name: str, param: str) -> tuple[str, int]:
    lines = source.splitlines(keepends=True)
    span = _function_signature_span(lines, class_name, function_name)
    if span is None:
        return source, 0
    start, end = span
    signature = "".join(lines[start : end + 1])
    param_name = param.split("=", 1)[0].split(":", 1)[0].strip()
    if re.search(rf"\b{re.escape(param_name)}\b", signature):
        return source, 0
    if start == end:
        lines[start] = re.sub(r"\)\s*:", f", {param}):", lines[start], count=1)
        return "".join(lines), 1
    indent = re.match(r"^(\s*)", lines[end]).group(1)  # type: ignore[union-attr]
    param_indent = ""
    for idx in range(end - 1, start, -1):
        if lines[idx].strip():
            param_indent = re.match(r"^(\s*)", lines[idx]).group(1)  # type: ignore[union-attr]
            break
    if not param_indent:
        param_indent = indent + "    "
    newline = "\n" if lines[end].endswith("\n") else ""
    lines.insert(end, f"{param_indent}{param},{newline}")
    return "".join(lines), 1


def _insert_after_once(source: str, needle: str, insertion: str) -> tuple[str, int]:
    if insertion.strip() in source:
        return source, 0
    index = source.find(needle)
    if index < 0:
        return source, 0
    index += len(needle)
    return source[:index] + insertion + source[index:], 1


def _replace_seq_idx_none_in_mamba_calls(source: str) -> tuple[str, int]:
    call_names = ("mamba_split_conv1d_scan_combined", "mamba_chunk_scan_combined")
    output: list[str] = []
    cursor = 0
    replacements = 0
    while cursor < len(source):
        matches = [
            (idx, name)
            for name in call_names
            if (idx := source.find(f"{name}(", cursor)) >= 0
        ]
        if not matches:
            output.append(source[cursor:])
            break
        start, _name = min(matches, key=lambda item: item[0])
        output.append(source[cursor:start])
        depth = 0
        end = start
        while end < len(source):
            char = source[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        call_text = source[start:end]
        patched_call, count = re.subn(r"seq_idx\s*=\s*None", "seq_idx=seq_idx", call_text)
        output.append(patched_call)
        replacements += count
        cursor = end
    return "".join(output), replacements


def _patch_seq_idx(source: str, report: PatchReport) -> str:
    source, kernel_count = _replace_seq_idx_none_in_mamba_calls(source)
    report.kernel_seq_idx_replacements += kernel_count
    report.seq_idx_replacements += kernel_count

    for class_name in ("NemotronHMamba2Mixer", "NemotronHMixer"):
        for function_name in ("cuda_kernels_forward", "torch_forward", "forward"):
            source, count = _add_param_to_function(source, class_name, function_name, "seq_idx=None")
            report.mamba_signature_replacements += count
            report.seq_idx_replacements += count

    for old, new in (
        (
            "return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)",
            "return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask, seq_idx)",
        ),
        (
            "return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)",
            "return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask, seq_idx)",
        ),
    ):
        source, count = re.subn(re.escape(old), new, source)
        report.mamba_forward_call_replacements += count
        report.seq_idx_replacements += count

    source, count = _add_param_to_function(source, "NemotronHBlock", "forward", "seq_idx=None")
    report.block_signature_replacements += count
    report.seq_idx_replacements += count

    seq_idx_assignment = "        seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])\n"
    source, count = re.subn(
        r"^        seq_idx = _nla_seq_idx_from_position_ids\(position_ids\)\n",
        seq_idx_assignment,
        source,
        flags=re.MULTILINE,
    )
    if count == 0 and seq_idx_assignment not in source:
        causal_mask_call = "        causal_mask = self._update_causal_mask("
        call_offset = source.find(causal_mask_call)
        if call_offset >= 0:
            source = source[:call_offset] + seq_idx_assignment + source[call_offset:]
            count = 1
        else:
            source, count = _insert_after_once(
                source,
                "mamba_mask = self._update_mamba_mask(attention_mask, cache_position)\n",
                seq_idx_assignment,
            )

    # Earlier patch revisions derived seq_idx after constructing the causal
    # mask. Packed attention needs the boundary ids while the mask is built.
    causal_mask_offset = source.find("        causal_mask = self._update_causal_mask(")
    seq_idx_offset = source.find(seq_idx_assignment)
    if causal_mask_offset >= 0 and seq_idx_offset > causal_mask_offset:
        source = source[:seq_idx_offset] + source[seq_idx_offset + len(seq_idx_assignment) :]
        causal_mask_offset = source.find("        causal_mask = self._update_causal_mask(")
        source = source[:causal_mask_offset] + seq_idx_assignment + source[causal_mask_offset:]
        count += 1
    source, dedupe_count = re.subn(
        rf"(?:{re.escape(seq_idx_assignment)}){{2,}}",
        seq_idx_assignment,
        source,
    )
    count += dedupe_count
    report.model_seq_idx_replacements += count
    report.seq_idx_replacements += count

    source, count = re.subn(
        r"(mixer_block\.__call__, hidden_states, cache_params, cache_position, layer_mask)(\s*\))",
        r"\1, seq_idx\2",
        source,
    )
    report.model_block_call_replacements += count
    report.seq_idx_replacements += count

    source, count = re.subn(
        r"(mixer_block\(\n\s*hidden_states,\n\s*cache_params=cache_params,\n\s*cache_position=cache_position,\n\s*attention_mask=layer_mask,\n)(\s*\))",
        r"\1                    seq_idx=seq_idx,\n\2",
        source,
    )
    report.model_block_call_replacements += count
    report.seq_idx_replacements += count
    return source


def _patch_generation_cache(source: str, report: PatchReport) -> str:
    """Repair list-backed cache state handling in Nano's bundled remote code."""

    if "class HybridMambaAttentionDynamicCache" not in source or "self.conv_states = []" not in source:
        return source

    source, count = re.subn(
        r"^(?P<indent>[ \t]*)conv_kernel_size = config\.conv_kernel\n"
        r"(?![ \t]*self\.conv_kernel_size = conv_kernel_size\n)",
        lambda match: match.group(0) + f"{match.group('indent')}self.conv_kernel_size = conv_kernel_size\n",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    report.generation_cache_conv_kernel_replacements += count

    device_replacements = (
        (r"self\.conv_states\.device", "self.conv_states[layer_idx].device"),
        (r"self\.ssm_states\.device", "self.ssm_states[layer_idx].device"),
        (r"cache_params\.conv_states\.device", "cache_params.conv_states[self.layer_idx].device"),
        (r"cache_params\.ssm_states\.device", "cache_params.ssm_states[self.layer_idx].device"),
    )
    for pattern, replacement in device_replacements:
        source, count = re.subn(pattern, replacement, source)
        report.generation_cache_device_replacements += count

    source, count = re.subn(
        r"(?P<indent>^[ \t]+)def reset\(self\):\n"
        r"(?P<body>[ \t]+)self\.conv_states\.zero_\(\)\n"
        r"(?P=body)self\.ssm_states\.zero_\(\)",
        lambda match: (
            f"{match.group('indent')}def reset(self):\n"
            f"{match.group('body')}for state in self.conv_states:\n"
            f"{match.group('body')}    state.zero_()\n"
            f"{match.group('body')}for state in self.ssm_states:\n"
            f"{match.group('body')}    state.zero_()"
        ),
        source,
        count=1,
        flags=re.MULTILINE,
    )
    report.generation_cache_reset_replacements += count
    return source


def _patch_attention_mask(source: str, report: PatchReport) -> str:
    patterns = [
        (
            r"self\.mixer\(hidden_states, cache_position=cache_position\)",
            "self.mixer(hidden_states, cache_position=cache_position, attention_mask=attention_mask, seq_idx=seq_idx)",
        ),
        (
            r"self\.attention\(hidden_states, cache_position=cache_position\)",
            "self.attention(hidden_states, cache_position=cache_position, attention_mask=attention_mask)",
        ),
    ]
    for pattern, repl in patterns:
        source, count = re.subn(pattern, repl, source)
        report.attention_mask_replacements += count
        if "seq_idx=seq_idx" in repl:
            report.block_mixer_call_replacements += count
        else:
            report.block_attention_call_replacements += count
    source, count = re.subn(
        r"(elif self\.block_type == \"attention\":\n\s+hidden_states = self\.mixer\(\n\s+hidden_states, cache_position=cache_position\n\s+\))",
        "elif self.block_type == \"attention\":\n"
        "                hidden_states = self.mixer(\n"
        "                    hidden_states, cache_position=cache_position, attention_mask=attention_mask\n"
        "                )",
        source,
    )
    report.attention_mask_replacements += count
    report.block_attention_call_replacements += count
    source, count = re.subn(
        r"(hidden_states,\s*cache_params=cache_params,\s*cache_position=cache_position\n\s*\))",
        "hidden_states, cache_params=cache_params, cache_position=cache_position,\n"
        "                    attention_mask=attention_mask, seq_idx=seq_idx\n                )",
        source,
    )
    report.attention_mask_replacements += count
    report.block_mixer_call_replacements += count
    return source


def _patch_packed_attention_boundaries(source: str, report: PatchReport) -> str:
    source, count = _add_param_to_function(
        source,
        "NemotronHModel",
        "_update_causal_mask",
        "seq_idx=None",
    )
    report.causal_mask_signature_replacements += count

    source, count = re.subn(
        r"self\._update_causal_mask\(\s*attention_mask,\s*inputs_embeds,\s*cache_position\s*\)",
        "self._update_causal_mask(attention_mask, inputs_embeds, cache_position, seq_idx)",
        source,
    )
    report.causal_mask_call_replacements += count

    if PACKED_ATTENTION_MARKER in source:
        return source
    needle = (
        "        causal_mask = causal_mask[None, None, :, :].expand("
        "input_tensor.shape[0], 1, -1, -1)\n"
    )
    insertion = f'''        {PACKED_ATTENTION_MARKER}
        if (
            seq_idx is not None
            and seq_idx.ndim == 2
            and seq_idx.shape[-1] == sequence_length
            and int(target_length) == sequence_length
        ):
            sequence_ids = seq_idx.to(device=device)
            same_sequence = (
                sequence_ids[:, None, :, None]
                == sequence_ids[:, None, None, :]
            )
            causal_mask = causal_mask.masked_fill(~same_sequence, min_dtype)
'''
    source, count = _insert_after_once(source, needle, insertion)
    report.packed_attention_boundary_replacements += count
    return source


def _patch_causal_lm_position_ids(source: str, report: PatchReport) -> str:
    """Forward packed position resets through the causal-LM wrapper."""

    source, count = re.subn(
        r"(?P<prefix>nemotron_h_outputs = self\.backbone\(\n"
        r"\s+input_ids,\n"
        r"\s+cache_params=cache_params,\n"
        r"(?P<indent>\s+)inputs_embeds=inputs_embeds,\n)"
        r"(?!\s+position_ids=position_ids,\n)",
        lambda match: (
            match.group("prefix")
            + f"{match.group('indent')}position_ids=position_ids,\n"
        ),
        source,
        count=1,
    )
    report.causal_lm_position_ids_replacements += count
    return source


def _patch_moe(source: str, report: PatchReport) -> str:
    if "segmented_moe(" in source:
        report.moe_replaced = True
        return source
    pattern = re.compile(
        r"(\n\s+def moe\(self,\s*hidden_states(?:\s*:\s*[^,\n)]+)?\s*,\s*topk_indices(?:\s*:\s*[^,\n)]+)?\s*,\s*topk_weights(?:\s*:\s*[^,\n)]+)?\s*\):\n)(?P<body>.*?)(?=\n\s+def |\nclass |\Z)",
        flags=re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        indent = re.match(r"\n(\s+)def", match.group(1)).group(1)  # type: ignore[union-attr]
        body_indent = indent + "    "
        report.moe_replaced = True
        report.moe_signature_replacements += 1
        return (
            match.group(1)
            + f"{body_indent}from nla.nemotron_moe import segmented_moe\n"
            + f"{body_indent}return segmented_moe(hidden_states, topk_indices, topk_weights, self.experts)\n"
        )

    return pattern.sub(repl, source, count=1)


def _patch_router_post_init(source: str, report: PatchReport) -> str:
    if "_nla_keep_router_buffers_fp32(self)" in source:
        return source
    source, count = re.subn(
        r"(class NemotronHPreTrainedModel\(PreTrainedModel\):\n(?:\s+\"\"\".*?\"\"\"\n)?)",
        r"\1\n    def post_init(self):\n        super().post_init()\n        _nla_keep_router_buffers_fp32(self)\n\n",
        source,
        count=1,
        flags=re.DOTALL,
    )
    report.router_post_init_replacements += count
    if count:
        return source
    source, count = re.subn(
        r"(def post_init\(self\):\n(?P<indent>\s+).*?)(?=\n\s+def |\nclass |\Z)",
        lambda m: m.group(1) + f"{m.group('indent')}_nla_keep_router_buffers_fp32(self)\n",
        source,
        count=1,
        flags=re.DOTALL,
    )
    report.router_post_init_replacements += count
    return source


def _validate_patch(source: str, report: PatchReport) -> None:
    has_list_backed_hybrid_cache = (
        "class HybridMambaAttentionDynamicCache" in source and "self.conv_states = []" in source
    )
    if has_list_backed_hybrid_cache and "conv_kernel_size = config.conv_kernel" in source:
        if "self.conv_kernel_size = conv_kernel_size" not in source:
            report.add_error("hybrid generation cache does not retain conv_kernel_size")
    if has_list_backed_hybrid_cache:
        stale_device_accesses = (
            "self.conv_states.device",
            "self.ssm_states.device",
            "cache_params.conv_states.device",
            "cache_params.ssm_states.device",
        )
        if any(access in source for access in stale_device_accesses):
            report.add_error("hybrid generation cache still treats list-backed state as a tensor")
    if report.kernel_seq_idx_replacements and report.mamba_signature_replacements == 0:
        report.add_error("rewrote Mamba kernel seq_idx kwargs without adding any Mamba seq_idx signatures")
    if report.kernel_seq_idx_replacements and report.block_signature_replacements == 0:
        report.add_error("rewrote Mamba kernel seq_idx kwargs without adding NemotronHBlock.forward seq_idx")
    has_model_layer_loop = "for layer_idx, mixer_block in enumerate(self.layers)" in source
    if report.kernel_seq_idx_replacements and has_model_layer_loop and report.model_block_call_replacements == 0:
        report.add_error("rewrote Mamba kernel seq_idx kwargs without passing seq_idx from model to blocks")
    if (
        report.kernel_seq_idx_replacements
        and has_model_layer_loop
        and "seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])" not in source
    ):
        report.add_error("rewrote Mamba kernel seq_idx kwargs without deriving seq_idx from position_ids")
    has_causal_mask = "def _update_causal_mask" in source
    if has_model_layer_loop and has_causal_mask:
        if PACKED_ATTENTION_MARKER not in source:
            report.add_error("packed attention mask does not isolate seq_idx boundaries")
        if "self._update_causal_mask(attention_mask, inputs_embeds, cache_position, seq_idx)" not in source:
            report.add_error("model forward does not pass seq_idx into the causal mask")
        seq_idx_assignment = (
            "seq_idx = _nla_seq_idx_from_position_ids(position_ids, hidden_states.shape[0])"
        )
        causal_mask_call = (
            "self._update_causal_mask(attention_mask, inputs_embeds, cache_position, seq_idx)"
        )
        if (
            seq_idx_assignment in source
            and causal_mask_call in source
            and source.index(seq_idx_assignment) > source.index(causal_mask_call)
        ):
            report.add_error("model forward derives seq_idx after constructing the causal mask")
    if "class NemotronHForCausalLM" in source and "nemotron_h_outputs = self.backbone(" in source:
        causal_lm_call = source.split("nemotron_h_outputs = self.backbone(", 1)[1].split(
            ")", 1
        )[0]
        if "position_ids=position_ids" not in causal_lm_call:
            report.add_error("causal-LM forward drops packed position_ids before the backbone")
    try:
        compile(source, "modeling_nemotron_h.py", "exec")
    except SyntaxError as exc:
        report.add_error(f"patched source does not compile: {exc}")
    if report.validation_errors:
        raise RuntimeError("; ".join(report.validation_errors))


def patch_nemotron_h_source(source: str) -> tuple[str, PatchReport]:
    report = PatchReport()
    before = source
    source = _patch_rmsnorm_fallback(source, report)
    source = _inject_helper(source, report)
    source = _patch_generation_cache(source, report)
    source = _patch_seq_idx(source, report)
    source = _patch_attention_mask(source, report)
    source = _patch_packed_attention_boundaries(source, report)
    source = _patch_causal_lm_position_ids(source, report)
    source = _patch_moe(source, report)
    source = _patch_router_post_init(source, report)
    report.changed = source != before
    _validate_patch(source, report)
    return source, report


def patch_nemotron_h_file(path: str | Path) -> PatchReport:
    path = Path(path)
    source = path.read_text()
    patched, report = patch_nemotron_h_source(source)
    if report.changed:
        path.write_text(patched)
        (path.with_suffix(path.suffix + ".nla_patch_report.json")).write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
        )
    return report


def looks_like_nemotron_h_source(source: str) -> bool:
    return all(marker in source for marker in NEMOTRON_H_SOURCE_MARKERS)


def patch_nemotron_h_file_if_needed(path: str | Path) -> PatchReport | None:
    """Patch real Nemotron-H remote-code files while leaving simple test stubs untouched."""
    path = Path(path)
    if path.name != "modeling_nemotron_h.py" or not path.exists():
        return None
    source = path.read_text()
    if not looks_like_nemotron_h_source(source):
        return None
    return patch_nemotron_h_file(path)


def patch_nemotron_h_checkpoint_dir(path: str | Path) -> PatchReport | None:
    root = Path(path)
    modeling = root / "modeling_nemotron_h.py"
    return patch_nemotron_h_file_if_needed(modeling)


def clear_transformers_dynamic_module_cache(path: str | Path) -> list[str]:
    """Remove cached remote code for a local checkpoint after source patching."""

    checkpoint_name = Path(path).resolve().name
    sanitized_name = checkpoint_name.replace(".", "_dot_").replace("-", "_hyphen_")
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "modules"
        / "transformers_modules"
    )
    removed: list[str] = []
    for name in sorted({checkpoint_name, sanitized_name}):
        candidate = cache_root / name
        if candidate.is_dir():
            shutil.rmtree(candidate)
            removed.append(str(candidate))
    return removed


def prepare_nemotron_h_checkpoint_for_load(path: str | Path) -> PatchReport | None:
    """Patch local Nemotron-H remote code and invalidate stale HF module caches."""

    report = patch_nemotron_h_checkpoint_dir(path)
    if report is not None and report.changed:
        clear_transformers_dynamic_module_cache(path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Checkpoint directory or modeling_nemotron_h.py path")
    args = parser.parse_args()
    path = args.path
    report = patch_nemotron_h_file(path) if path.is_file() else patch_nemotron_h_checkpoint_dir(path)
    print(json.dumps(None if report is None else asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
