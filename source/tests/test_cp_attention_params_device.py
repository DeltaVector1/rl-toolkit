import torch

from prime_rl.utils import cp


def test_cp_attention_params_preserves_seq_lens_device(monkeypatch) -> None:
    seen = {}

    def fake_cu_seqlens(seq_lens, total_tokens):
        seen["device"] = seq_lens.device
        seen["total_tokens"] = total_tokens
        return torch.empty(2, dtype=torch.int32, device=seq_lens.device), total_tokens

    monkeypatch.setattr(cp, "get_cu_seqlens_from_seq_lens", fake_cu_seqlens)
    monkeypatch.setattr(cp, "update_ring_flash_attn_params", lambda *_: None)

    cp.setup_cp_attention_params(
        torch.empty(1, 32, dtype=torch.long),
        cp_group=object(),
        seq_lens=torch.empty(1, dtype=torch.long, device="meta"),
        cp_style="ring",
    )

    assert seen == {"device": torch.device("meta"), "total_tokens": 32}
