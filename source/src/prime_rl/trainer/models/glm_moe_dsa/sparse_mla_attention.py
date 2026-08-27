from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer
from prime_rl.trainer.models.layers.lora.multi_linear import (
    MultiLoRALinear,
    fused_frozen_linear_pair,
)
from prime_rl.trainer.models.layers.checkpointing import current_reentrant_block_phase
from prime_rl.trainer.models.layers.norms import LayerNorm, RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import rotate_half
from prime_rl.trainer.phase_profiler import get_phase_profiler, phase_span
from prime_rl.utils.cp import gather_for_cp

_flex_attention = torch.compile(flex_attention, dynamic=True)
_ADA_FLEX_OPTIONS = {
    "BLOCK_M": 16,
    "BLOCK_N": 16,
    "BLOCK_M1": 16,
    "BLOCK_N1": 16,
    "BLOCK_M2": 16,
    "BLOCK_N2": 16,
    "num_warps": 4,
    "num_stages": 1,
    "PRESCALE_QK": True,
}

try:
    from prime_rl.trainer.models.kernels.sparse_mla_bwd import sparse_mla_bwd
    from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface
except ImportError:
    sparse_mla_fwd_interface = None  # type: ignore
    sparse_mla_bwd = None  # type: ignore


@dataclass(frozen=True)
class SparseMlaAttentionArgs:
    hidden_size: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    attention_bias: bool
    rms_norm_eps: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    use_index_cache: bool = False
    skip_topk: bool = False


class _SparseMLA(torch.autograd.Function):
    """Autograd wrapper for tilelang sparse MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale):
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, out, indices, lse)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    @phase_span("sparse_mla_backward")
    def backward(ctx, do):
        q, kv, out, indices, lse = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(
            q, kv, out, do.contiguous(), indices, lse, sm_scale=ctx.sm_scale
        )
        return dq, dkv, None, None


def apply_rope_interleave_single(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    b, h, s, d = t.shape
    t = t.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    # Under `quantization.dequantize_checkpoint = false` FSDP wraps the blocks
    # with a bare MixedPrecisionPolicy() (model.py:813), so nothing normalizes
    # activation dtypes and float8 tensors reach this multiply. Torch has no fp8
    # promotion, so ANY fp8 operand raises. Either side can be the fp8 one: t on
    # the main path, cos/sin via compute_sparse_indices. Widening fp8 -> bf16 is
    # exact, and rope must not run in fp8 regardless -- e4m3 has 3 mantissa bits,
    # which would wreck the positional phase.
    _REAL = (torch.float32, torch.float16, torch.bfloat16)
    if t.dtype not in _REAL or cos.dtype not in _REAL or sin.dtype not in _REAL:
        common = next((x.dtype for x in (t, cos, sin) if x.dtype in _REAL), torch.bfloat16)
        if t.dtype not in _REAL:
            t = t.to(common)
        if cos.dtype not in _REAL:
            cos = cos.to(common)
        if sin.dtype not in _REAL:
            sin = sin.to(common)
    return (t * cos) + (rotate_half(t) * sin)


class Indexer(nn.Module):
    def __init__(self, args: SparseMlaAttentionArgs):
        super().__init__()
        self.n_head = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=args.attention_bias)
        self.k_norm = LayerNorm(dim=self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_head, bias=False)
        self.weight_scale = (self.head_dim**-0.5) * (self.n_head**-0.5)

    @torch.no_grad()
    def compute_sparse_indices(
        self,
        hidden_states_local: torch.Tensor,
        q_latent_local: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        index_topk: int,
        position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
        cp_group: dist.ProcessGroup | None,
        cp_world_size: int,
        cp_rank: int,
    ) -> torch.Tensor:
        assert index_topk % 64 == 0, f"index_topk must be divisible by 64 (block_I), got {index_topk}"

        s_local = hidden_states_local.shape[1]

        q_idx = self.wq_b(q_latent_local[0]).view(s_local, self.n_head, self.head_dim)
        k_idx_local = self.k_norm(self.wk(hidden_states_local[0]))
        w = self.weights_proj(hidden_states_local[0])

        if cp_world_size > 1:
            k_idx = gather_for_cp(k_idx_local.unsqueeze(0), cp_group).squeeze(0)
        else:
            k_idx = k_idx_local

        cos_full, sin_full = position_embeddings_full
        if cp_world_size > 1:
            cos_local = cos_full[:, cp_rank * s_local : (cp_rank + 1) * s_local, :]
            sin_local = sin_full[:, cp_rank * s_local : (cp_rank + 1) * s_local, :]
        else:
            cos_local, sin_local = cos_full, sin_full

        q_pe = q_idx[..., : self.rope_dim]
        q_nope = q_idx[..., self.rope_dim :]
        k_pe = k_idx[..., : self.rope_dim]
        k_nope = k_idx[..., self.rope_dim :]

        q_pe = q_pe.unsqueeze(0).transpose(1, 2)  # [1, H, S_local, rope_dim]
        q_pe = apply_rope_interleave_single(q_pe, cos_local, sin_local)
        q_pe = q_pe.transpose(1, 2).squeeze(0)

        k_pe = k_pe.unsqueeze(0).unsqueeze(1)  # [1, 1, S_full, rope_dim]
        k_pe = apply_rope_interleave_single(k_pe, cos_full, sin_full)
        k_pe = k_pe.squeeze(1).squeeze(0)

        q_idx = torch.cat([q_pe, q_nope], dim=-1)
        k_idx = torch.cat([k_pe, k_nope], dim=-1)

        indices = fp8_indexer(q_idx, k_idx, w, ks, ke, index_topk, self.weight_scale)
        # indices shape: [S_local, topk] in K's coordinate space (sentinel = s_full)
        # KV passed to sparse MLA has length s_full + 1 (sentinel zeros at index s_full).
        return indices.view(1, s_local, 1, index_topk)


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, args: SparseMlaAttentionArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim

        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=args.attention_bias)
        self.q_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=args.q_lora_rank, eps=args.rms_norm_eps))
        self.q_b_proj = nn.Linear(args.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(RMSNormConfig(hidden_size=self.kv_lora_rank, eps=args.rms_norm_eps))
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, args.hidden_size, bias=args.attention_bias)
        # IndexShare (GLM-5.2): layers that reuse cached indices carry no indexer weights.
        self.indexer = Indexer(args) if not args.skip_topk else None
        self.use_index_cache = args.use_index_cache
        self.skip_topk = args.skip_topk
        self.scaling = self.qk_head_dim ** (-0.5)

        self._cp_group: dist.ProcessGroup | None = None
        self._cp_rank: int = 0
        self._cp_world_size: int = 1

        self._phase_backward_timer = None
        if get_phase_profiler().enabled:
            self.register_full_backward_pre_hook(self._phase_backward_start)
            self.register_full_backward_hook(self._phase_backward_end)

    def _phase_backward_start(self, _module, _grad_output) -> None:
        self._phase_backward_timer = get_phase_profiler().start_span(
            "attention_backward"
        )

    def _phase_backward_end(self, _module, _grad_input, _grad_output) -> None:
        get_phase_profiler().end_span(self._phase_backward_timer)
        self._phase_backward_timer = None

    def set_context_parallel_attributes(self, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
        self._cp_group = cp_group
        self._cp_rank = cp_rank
        self._cp_world_size = cp_world_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_world_size > 1

    def attn_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_projection = self.q_a_proj
        kv_projection = self.kv_a_proj_with_mqa
        if isinstance(q_projection, MultiLoRALinear) and isinstance(
            kv_projection, MultiLoRALinear
        ):
            q_latent, compressed_kv = fused_frozen_linear_pair(
                hidden_states,
                q_projection.base_layer,
                kv_projection.base_layer,
            )
            q_latent = q_latent + q_projection.adapter_output(hidden_states)
            compressed_kv = compressed_kv + kv_projection.adapter_output(hidden_states)
        elif isinstance(q_projection, MultiLoRALinear) or isinstance(
            kv_projection, MultiLoRALinear
        ):
            raise ValueError("Q-A and KV-A must use LoRA together")
        else:
            q_latent = q_projection(hidden_states)
            compressed_kv = kv_projection(hidden_states)
        q_latent = self.q_a_layernorm(q_latent)
        k_compressed, k_rope = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        return q_latent, self.kv_a_layernorm(k_compressed), k_rope

    def mla_up_proj(
        self,
        q_latent_local: torch.Tensor,
        k_compressed_normed_full: torch.Tensor,
        k_rope_full: torch.Tensor,
        position_embeddings_full: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, s_local, _ = q_latent_local.shape
        s_full = k_compressed_normed_full.shape[1]

        q_full = self.q_b_proj(q_latent_local).view(batch_size, s_local, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q_full.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        cos_full, sin_full = position_embeddings_full
        if self.cp_enabled:
            cos_local = cos_full[:, self._cp_rank * s_local : (self._cp_rank + 1) * s_local, :]
            sin_local = sin_full[:, self._cp_rank * s_local : (self._cp_rank + 1) * s_local, :]
        else:
            cos_local, sin_local = cos_full, sin_full

        q_rope_r = q_rope.transpose(1, 2)  # [B, H, S_local, rope_dim]
        q_rope_r = apply_rope_interleave_single(q_rope_r, cos_local, sin_local)
        q_rope = q_rope_r.transpose(1, 2)

        k_rope_r = k_rope_full.unsqueeze(1)  # [B, 1, S_full, rope_dim]
        k_rope_r = apply_rope_interleave_single(k_rope_r, cos_full, sin_full)
        k_rope_full = k_rope_r.squeeze(1)

        kv_b_w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_k_nope = kv_b_w[:, : self.qk_nope_head_dim, :]
        w_v = kv_b_w[:, self.qk_nope_head_dim :, :]
        q_absorbed = torch.einsum("bshd,hdk->bshk", q_nope, w_k_nope)

        active_lora_factors = getattr(self.kv_b_proj, "active_lora_factors", None)
        if active_lora_factors is not None:
            lora_a, lora_b, lora_scaling = active_lora_factors()
            lora_b = lora_b.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                lora_b.shape[-1],
            )
            key_lora_b = lora_b[:, : self.qk_nope_head_dim, :]
            value_lora_b = lora_b[:, self.qk_nope_head_dim :, :]
            q_low_rank = torch.einsum("bshd,hdr->bshr", q_nope, key_lora_b)
            q_update = torch.einsum("bshr,rk->bshk", q_low_rank, lora_a)
            q_update.mul_(lora_scaling)
            q_absorbed.add_(q_update)
            w_v = (w_v, lora_a, value_lora_b, lora_scaling)

        sparse_q = torch.cat([q_absorbed, q_rope], dim=-1)
        sparse_kv = torch.cat([k_compressed_normed_full, k_rope_full], dim=-1).unsqueeze(2)

        sentinel = torch.zeros(batch_size, 1, 1, sparse_kv.shape[-1], dtype=sparse_kv.dtype, device=sparse_kv.device)
        sparse_kv = torch.cat([sparse_kv, sentinel], dim=1)
        assert sparse_kv.shape[1] == s_full + 1
        return sparse_q, sparse_kv, w_v

    def _mla_unabsorb(self, out: torch.Tensor, w_v) -> torch.Tensor:
        if isinstance(w_v, tuple):
            base_w_v, lora_a, value_lora_b, lora_scaling = w_v
            result = torch.einsum("bshk,hdk->bshd", out, base_w_v)
            low_rank = torch.einsum("bshk,rk->bshr", out, lora_a)
            value_update = torch.einsum("bshr,hdr->bshd", low_rank, value_lora_b)
            value_update.mul_(lora_scaling)
            result.add_(value_update)
            return result
        return torch.einsum("bshk,hdk->bshd", out, w_v)

    def output_proj(self, attn_output: torch.Tensor, w_v: torch.Tensor) -> torch.Tensor:
        attn_output = self._mla_unabsorb(attn_output, w_v)
        batch_size, total_tokens = attn_output.shape[:2]
        attn_output = attn_output.reshape(batch_size, total_tokens, -1)
        return self.o_proj(attn_output)

    @phase_span(
        lambda: (
            "attention_recompute"
            if current_reentrant_block_phase() == "recompute"
            else "attention_forward"
        )
    )
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ks: torch.Tensor | None = None,
        ke: torch.Tensor | None = None,
        cached_indices: torch.Tensor | None = None,
        block_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_latent, k_compressed_normed, k_rope = self.attn_projections(hidden_states)

        if self.cp_enabled:
            k_compressed_normed = gather_for_cp(k_compressed_normed, self._cp_group)
            k_rope = gather_for_cp(k_rope, self._cp_group)

        if block_mask is not None:
            sparse_q, sparse_kv, w_v = self.mla_up_proj(
                q_latent_local=q_latent,
                k_compressed_normed_full=k_compressed_normed,
                k_rope_full=k_rope,
                position_embeddings_full=position_embeddings,
            )
            out = _flex_attention(
                sparse_q.transpose(1, 2),
                sparse_kv[:, :-1].transpose(1, 2),
                sparse_kv[:, :-1, :, : self.kv_lora_rank].transpose(1, 2),
                block_mask=block_mask,
                scale=self.scaling,
                enable_gqa=True,
                kernel_options=_ADA_FLEX_OPTIONS,
            ).transpose(1, 2).to(w_v.dtype)
            return self.output_proj(out, w_v), None

        indices = cached_indices
        if not self.skip_topk:
            indices = self.indexer.compute_sparse_indices(
                hidden_states_local=hidden_states,
                q_latent_local=q_latent,
                ks=ks,
                ke=ke,
                index_topk=self.args.index_topk,
                position_embeddings_full=position_embeddings,
                cp_group=self._cp_group,
                cp_world_size=self._cp_world_size,
                cp_rank=self._cp_rank,
            )

        sparse_q, sparse_kv, w_v = self.mla_up_proj(
            q_latent_local=q_latent,
            k_compressed_normed_full=k_compressed_normed,
            k_rope_full=k_rope,
            position_embeddings_full=position_embeddings,
        )

        out = _SparseMLA.apply(sparse_q, sparse_kv, indices, self.scaling)
        cached_indices = indices if self.use_index_cache else None
        return self.output_proj(out, w_v), cached_indices
