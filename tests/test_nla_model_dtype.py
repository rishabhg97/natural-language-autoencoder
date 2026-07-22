import pytest

torch = pytest.importorskip("torch")

from external.natural_language_autoencoders.nla.model_dtype import normalize_floating_module_dtype


def test_normalize_floating_module_dtype_casts_float_parameters_and_buffers_only():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
    module.bias = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    module.register_buffer("float_buffer", torch.ones(2, dtype=torch.float32))
    module.register_buffer("int_buffer", torch.ones(2, dtype=torch.int64))

    changed = normalize_floating_module_dtype(module, torch.bfloat16)

    assert changed == 3
    assert module.weight.dtype == torch.bfloat16
    assert module.bias.dtype == torch.bfloat16
    assert module.float_buffer.dtype == torch.bfloat16
    assert module.int_buffer.dtype == torch.int64
