"""Actor-SFT rollout: no generation — tokenize prompt+response, stash activation.

Pattern follows miles/rollout/sft_rollout.py. The data_buffer yields Samples
whose .prompt is a list[dict] (from NLADataSource, <INJECT>→㊗ already substituted)
and whose .metadata["response"] is the <explanation>...</explanation> string.
"""

import os

import torch

from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer

from nla.schema import MM_ACTIVATION_KEY


_TOKENIZER = None
_MASK_GEN = None
_CAP_DEBUG_PRINTED = False


def _positive_int_from_env(name):
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set")
    return parsed


def _masked_response_length(loss_mask):
    return sum(1 for value in loss_mask if int(value) != 0)


def _truncate_sft_tokens(
    token_ids,
    loss_mask,
    *,
    response_length,
    max_sequence_tokens=None,
    max_response_tokens=None,
):
    if max_response_tokens is not None and response_length > max_response_tokens:
        prompt_length = len(token_ids) - response_length
        keep_tokens = prompt_length + max_response_tokens
        token_ids = token_ids[:keep_tokens]
        loss_mask = loss_mask[:keep_tokens]
        response_length = _masked_response_length(loss_mask)

    if max_sequence_tokens is not None and len(token_ids) > max_sequence_tokens:
        token_ids = token_ids[:max_sequence_tokens]
        loss_mask = loss_mask[:max_sequence_tokens]
        response_length = _masked_response_length(loss_mask)

    if response_length <= 0:
        raise ValueError(
            "AV SFT token caps removed all response tokens; increase "
            "NLA_SFT_MAX_SEQUENCE_TOKENS or NLA_SFT_MAX_RESPONSE_TOKENS"
        )
    return token_ids, loss_mask, response_length


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global _TOKENIZER, _MASK_GEN
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if _MASK_GEN is None:
        _MASK_GEN = MultiTurnLossMaskGenerator(_TOKENIZER, tokenizer_type=args.loss_mask_type)

    samples = data_buffer.get_samples(args.rollout_batch_size)
    max_sequence_tokens = _positive_int_from_env("NLA_SFT_MAX_SEQUENCE_TOKENS")
    max_response_tokens = _positive_int_from_env("NLA_SFT_MAX_RESPONSE_TOKENS")

    for group in samples:
        (sample,) = group
        messages = sample.prompt
        assert isinstance(messages, list), (
            f"actor SFT requires list[dict] prompt (got {type(messages).__name__}). "
            f"NLADataSource must use apply_chat_template=False."
        )
        response = sample.metadata["response"]
        messages = messages + [{"role": "assistant", "content": response}]

        token_ids, loss_mask = _MASK_GEN.get_loss_mask(messages)
        original_token_length = len(token_ids)
        response_length = _MASK_GEN.get_response_lengths([loss_mask])[0]
        token_ids, loss_mask, response_length = _truncate_sft_tokens(
            token_ids,
            loss_mask,
            response_length=response_length,
            max_sequence_tokens=max_sequence_tokens,
            max_response_tokens=max_response_tokens,
        )
        global _CAP_DEBUG_PRINTED
        if not _CAP_DEBUG_PRINTED:
            print(
                "[NLA SFT ACTOR CAP] "
                f"max_sequence_tokens={max_sequence_tokens} "
                f"max_response_tokens={max_response_tokens} "
                f"original_tokens={original_token_length} "
                f"capped_tokens={len(token_ids)} "
                f"response_length={response_length}",
                flush=True,
            )
            _CAP_DEBUG_PRINTED = True

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0.0
        sample.loss_mask = loss_mask[-response_length:]

        activation = torch.tensor(
            sample.metadata["activation_vector"], dtype=torch.float32
        ).view(1, -1)
        sample.multimodal_train_inputs = {MM_ACTIVATION_KEY: activation}

    return samples
