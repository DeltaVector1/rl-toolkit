from prime_rl.trainer.models.layers.hf_ring_attn import (
    _transformers_window_for_upstream,
)


def test_current_transformers_causal_window_uses_inclusive_boundary():
    assert _transformers_window_for_upstream(None) is None
    assert _transformers_window_for_upstream(2048) == 2047
    assert _transformers_window_for_upstream(1) == 0
