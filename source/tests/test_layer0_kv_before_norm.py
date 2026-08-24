from unittest.mock import patch

import pytest

from prime_rl.parity_router_trace import (
    _capture_layer0_kv_before_norm,
    _install_layer0_token_diagnostics,
)


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def detach(self):
        return self


class HookModule:
    def __init__(self):
        self.forward_hooks = []
        self.forward_pre_hooks = []

    def register_forward_hook(self, hook):
        self.forward_hooks.append(hook)
        return hook

    def register_forward_pre_hook(self, hook):
        self.forward_pre_hooks.append(hook)
        return hook


def test_layer0_kv_before_norm_is_one_bounded_512_value():
    value = FakeTensor((4, 512))
    with patch("prime_rl.parity_router_trace.write_hidden_diagnostic") as write:
        _capture_layer0_kv_before_norm(None, (value,))
    write.assert_called_once_with(
        "layer0_kv_before_norm", value, expected_width=512, token_limit=1
    )


@pytest.mark.parametrize("inputs", [None, (), (object(),)])
def test_layer0_kv_before_norm_rejects_invalid_inputs(inputs):
    with pytest.raises(ValueError, match="layer 0 KV input"):
        _capture_layer0_kv_before_norm(None, inputs)


def test_layer0_token_hooks_capture_before_and_after_kv_norm():
    input_norm = HookModule()
    kv_norm = HookModule()
    output_projection = HookModule()
    attention_modules = type(
        "AttentionModules",
        (),
        {"kv_a_layernorm": kv_norm, "o_proj": output_projection},
    )()
    attention = type("Attention", (), {"mla_attn": attention_modules})()
    layer = type(
        "Layer",
        (),
        {"input_layernorm": input_norm, "self_attn": attention},
    )()

    with patch("prime_rl.parity_router_trace._diagnostic_hooks", []):
        _install_layer0_token_diagnostics(layer)

    assert kv_norm.forward_pre_hooks == [_capture_layer0_kv_before_norm]
    assert len(kv_norm.forward_hooks) == 1
