"""Expose a valid Muse tensor-parallel plan to vLLM's Transformers backend.

The outer multimodal model does not publish the language model's TP plan.
Muse has two KV heads. TP2 shards them; larger TP sizes must replicate them.
Keep Q, the attention control projection, the output projection, and the MLP tensor-parallel.
"""

import os

import torch
from transformers.models.muse_glimmer.configuration_muse_glimmer import (
    MuseGlimmerTextConfig,
)
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerModel,
    MuseGlimmerTextAttention,
)
from vllm.model_executor.models.transformers import TransformersMultiModalForCausalLM
try:
    from vllm.model_executor.models.transformers.fusers.qkv import QKVFuser
except ModuleNotFoundError:
    QKVFuser = None


tp_plan = dict(MuseGlimmerTextConfig.base_model_tp_plan)
tp_size = int(os.environ.get("MUSE_TP_SIZE", "8"))
if tp_size > 2:
    for projection in ("k_proj", "v_proj"):
        tp_plan[f"layers.*.self_attn.{projection}"] = "replicate"
MuseGlimmerTextConfig.base_model_tp_plan = tp_plan
MuseGlimmerModel._tp_plan = {
    f"language_model.{name}": style for name, style in tp_plan.items()
}


if QKVFuser is not None:
    _original_qkv_match = QKVFuser.match.__func__


    def _match_non_muse_qkv(cls, graph, module):
        if isinstance(module, MuseGlimmerTextAttention):
            return None
        return _original_qkv_match(cls, graph, module)


    QKVFuser.match = classmethod(_match_non_muse_qkv)


_original_compute_logits = TransformersMultiModalForCausalLM.compute_logits


def _compute_muse_logits(self, hidden_states):
    if getattr(self.config, "model_type", "") != "muse_glimmer":
        return _original_compute_logits(self, hidden_states)
    logits = self.logits_processor._get_logits(
        hidden_states,
        self.lm_head,
        self.lm_head.bias,
    )
    if logits is None:
        return None
    multiplier = float(self.text_config.output_multiplier)
    softcap = float(self.text_config.final_logit_softcapping)
    return torch.tanh(logits * multiplier / softcap) * softcap


TransformersMultiModalForCausalLM.compute_logits = _compute_muse_logits
