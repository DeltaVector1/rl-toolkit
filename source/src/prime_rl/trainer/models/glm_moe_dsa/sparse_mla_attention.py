import weakref
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
from prime_rl.trainer.models.layers.norms import LayerNorm, RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import rotate_half
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


class _PackedIndexerProjection(nn.Module):
    input_size = 6144
    key_size = 128
    head_count = 32
    output_size = key_size + head_count

    def __init__(self) -> None:
        super().__init__()
        self.in_features = self.input_size
        self.out_features = self.output_size
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size,
                self.input_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.bias = None

    def _validate(self, x: torch.Tensor) -> None:
        if tuple(self.weight.shape) != (self.output_size, self.input_size):
            raise ValueError(
                f"packed indexer weight has shape {tuple(self.weight.shape)}; "
                f"expected {(self.output_size, self.input_size)}"
            )
        if self.weight.dtype != torch.bfloat16:
            raise TypeError(
                f"packed indexer weight must be BF16; got {self.weight.dtype}"
            )
        if x.dtype != torch.bfloat16:
            raise TypeError(f"packed indexer input must be BF16; got {x.dtype}")
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"packed indexer input width is {x.shape[-1]}; expected {self.input_size}"
            )
        if x.device != self.weight.device:
            raise ValueError("packed indexer input and weight must use one device")
        if self.bias is not None:
            raise ValueError("packed indexer projection does not support bias")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(x)
        combined = torch.nn.functional.linear(x, self.weight)
        if combined.dtype != torch.bfloat16:
            raise TypeError(f"packed indexer output must be BF16; got {combined.dtype}")
        return combined.split((self.key_size, self.head_count), dim=-1)


class _PackedIndexerWeightSlice(nn.Module):
    def __init__(self, owner: _PackedIndexerProjection) -> None:
        super().__init__()
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        self.in_features = owner.input_size
        self.out_features = owner.head_count
        self.bias = None

    @property
    def owner(self) -> _PackedIndexerProjection:
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("packed indexer projection owner no longer exists")
        return owner

    @property
    def weight(self) -> torch.Tensor:
        return self.owner.weight.narrow(0, self.owner.key_size, self.owner.head_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "indexer weights projection must run through the packed indexer call"
        )


class Indexer(nn.Module):
    def __init__(self, args: SparseMlaAttentionArgs):
        super().__init__()
        expected = {
            "hidden_size": _PackedIndexerProjection.input_size,
            "index_head_dim": _PackedIndexerProjection.key_size,
            "index_n_heads": _PackedIndexerProjection.head_count,
        }
        actual = {
            "hidden_size": args.hidden_size,
            "index_head_dim": args.index_head_dim,
            "index_n_heads": args.index_n_heads,
        }
        if actual != expected:
            raise ValueError(
                f"packed GLM indexer dimensions are {actual}; expected {expected}"
            )
        if args.attention_bias:
            raise ValueError("packed GLM indexer does not support attention bias")
        self.n_head = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_head * self.head_dim, bias=False)
        self.wk = _PackedIndexerProjection()
        self.k_norm = LayerNorm(dim=self.head_dim, eps=1e-6)
        self.weights_proj = _PackedIndexerWeightSlice(self.wk)
        self.weight_scale = (self.head_dim**-0.5) * (self.n_head**-0.5)
        self._register_state_dict_hook(self._split_packed_indexer_state)
        self._register_load_state_dict_pre_hook(
            self._pack_indexer_load_state, with_module=True
        )

    @staticmethod
    def _split_packed_indexer_state(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
    ) -> None:
        del local_metadata
        if not isinstance(module, Indexer):
            raise TypeError("packed indexer state hook received the wrong module")
        wk_key = f"{prefix}wk.weight"
        weights_key = f"{prefix}weights_proj.weight"
        if wk_key not in state_dict or weights_key in state_dict:
            raise KeyError("packed indexer state keys are not in their internal form")
        packed = state_dict[wk_key]
        if tuple(packed.shape) != (
            _PackedIndexerProjection.output_size,
            _PackedIndexerProjection.input_size,
        ):
            raise ValueError(f"packed indexer state has shape {tuple(packed.shape)}")
        if packed.dtype != torch.bfloat16:
            raise TypeError(f"packed indexer state must be BF16; got {packed.dtype}")
        if module.wk.bias is not None or module.weights_proj.bias is not None:
            raise ValueError("packed indexer state does not support bias")
        state_dict[wk_key] = packed.narrow(0, 0, _PackedIndexerProjection.key_size)
        state_dict[weights_key] = packed.narrow(
            0,
            _PackedIndexerProjection.key_size,
            _PackedIndexerProjection.head_count,
        )

    @staticmethod
    def _pack_indexer_load_state(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        if not isinstance(module, Indexer):
            raise TypeError("packed indexer load hook received the wrong module")
        wk_key = f"{prefix}wk.weight"
        weights_key = f"{prefix}weights_proj.weight"
        if wk_key not in state_dict or weights_key not in state_dict:
            raise KeyError(
                "packed indexer load requires wk.weight and weights_proj.weight"
            )
        wk = state_dict[wk_key]
        weights = state_dict.pop(weights_key)
        expected_wk = (
            _PackedIndexerProjection.key_size,
            _PackedIndexerProjection.input_size,
        )
        expected_weights = (
            _PackedIndexerProjection.head_count,
            _PackedIndexerProjection.input_size,
        )
        if tuple(wk.shape) != expected_wk or tuple(weights.shape) != expected_weights:
            raise ValueError(
                "packed indexer load shapes differ: "
                f"wk={tuple(wk.shape)}, weights={tuple(weights.shape)}"
            )
        if wk.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
            raise TypeError(
                "packed indexer load requires BF16 tensors; "
                f"got wk={wk.dtype}, weights={weights.dtype}"
            )
        state_dict[wk_key] = torch.cat((wk, weights), dim=0)

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
        k_idx_local, w = self.wk(hidden_states_local[0])
        k_idx_local = self.k_norm(k_idx_local)

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
