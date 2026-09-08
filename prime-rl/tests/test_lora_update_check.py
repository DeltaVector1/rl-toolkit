import torch
from torch import nn

import prime_rl.trainer.rl.train as train_module


class TinyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(4, 4))
        self.lora_B = nn.Parameter(torch.zeros(4, 4))


def test_lora_update_stats_are_no_grad_and_detect_updates(monkeypatch):
    monkeypatch.setattr(train_module.dist, "all_reduce", lambda tensor, op=None: tensor)
    model = TinyLoRA()

    initial, finite = train_module._lora_update_check_stats(
        model, device=torch.device("cpu")
    )
    assert not initial.requires_grad
    assert not finite.requires_grad
    assert initial[3].item() == 1
    assert initial[4].item() == 1
    assert initial[1].item() == 0

    model.lora_A.grad = torch.ones_like(model.lora_A)
    model.lora_B.data.fill_(1)
    updated, finite = train_module._lora_update_check_stats(
        model, device=torch.device("cpu")
    )
    assert not updated.requires_grad
    assert updated[0].item() > 0
    assert updated[1].item() > 0
    assert finite.item() == 1
