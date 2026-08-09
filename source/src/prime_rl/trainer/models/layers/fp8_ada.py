import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DTensor
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, _online_logsumexp_and_weighted_update
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    get_default_config,
    moe_kernel_quantize_input,
)


block_shape = [128, 128]
_DIAGNOSE_LINEAR_BACKWARD = False
_REPORTED_LINEAR_BACKWARD: set[tuple[tuple[int, ...], bool, bool]] = set()
_REPORTED_INPUT_GRAD: set[tuple[int, ...]] = set()


def set_diagnose_linear_backward(enabled: bool) -> None:
    global _DIAGNOSE_LINEAR_BACKWARD
    _DIAGNOSE_LINEAR_BACKWARD = enabled


def _linear(x, weight, scale):
    shape = (*x.shape[:-1], weight.shape[0])
    x = x.reshape(-1, x.shape[-1]).contiguous()
    offsets = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int32)
    return _grouped_mm(x, weight.unsqueeze(0), scale.unsqueeze(0), offsets).view(shape)


def _linear_input_grad(grad, weight, scale, chunk_rows=1024):
    """Compute dX = dY @ dequant(W) without a transposed FP8 kernel.

    The forward fused kernel accepts checkpoint layout [out, in]. Its
    transposed-weight use in backward is not a supported checkpoint layout and
    produced non-finite LoRA gradients on the 64-GPU Ada job. Dequantize bounded
    row chunks and accumulate in FP32 instead.
    """
    shape = (*grad.shape[:-1], weight.shape[1])
    grad_2d = grad.reshape(-1, grad.shape[-1])
    compute_dtype = (
        torch.float64
        if grad.dtype == torch.float64 or weight.dtype == torch.float64
        else torch.float32
    )
    result = torch.zeros(
        grad_2d.shape[0],
        weight.shape[1],
        device=grad.device,
        dtype=compute_dtype,
    )
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        dequantized = _dequant_weight(weight, scale, start, end, compute_dtype)
        result.addmm_(grad_2d[:, start:end].to(compute_dtype), dequantized)
    converted = result.to(grad.dtype)
    if (
        (_DIAGNOSE_LINEAR_BACKWARD or os.environ.get("PRIME_FP8_DIAG_LINEAR_BACKWARD", "") == "1")
        and not bool(torch.isfinite(converted).all())
        and tuple(weight.shape) not in _REPORTED_INPUT_GRAD
    ):
        _REPORTED_INPUT_GRAD.add(tuple(weight.shape))
        finite_result = result[torch.isfinite(result)]
        weight_float = weight.float()
        finite_weight = weight_float[torch.isfinite(weight_float)]
        print(
            "[fp8-linear-input-grad] "
            f"weight={tuple(weight.shape)} weight_finite={bool(torch.isfinite(weight_float).all())} "
            f"weight_absmax={float(finite_weight.abs().max()) if finite_weight.numel() else float('nan'):.6e} "
            f"scale_finite={bool(torch.isfinite(scale).all())} "
            f"scale_min={float(scale[torch.isfinite(scale)].min()):.6e} "
            f"scale_max={float(scale[torch.isfinite(scale)].max()):.6e} "
            f"result_finite={bool(torch.isfinite(result).all())} "
            f"result_nan={int(torch.isnan(result).sum())} result_inf={int(torch.isinf(result).sum())} "
            f"result_absmax={float(finite_result.abs().max()) if finite_result.numel() else float('nan'):.6e} "
            f"converted_nan={int(torch.isnan(converted).sum())} converted_inf={int(torch.isinf(converted).sum())}",
            flush=True,
        )
    return converted.view(shape)


def _current_module_tensor(module: nn.Module, name: str) -> torch.Tensor:
    value = getattr(module, name)
    return value.to_local() if isinstance(value, DTensor) else value


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, weight_name, scale_name):
        # Do not save FSDP-managed frozen tensors. With RAF enabled their
        # all-gather storage is released after forward. Saving the tensor alias
        # left backward reading the released FP8/scale allocation. Store the
        # stable module accessor and read its rematerialized parameters after
        # FSDP's pre-backward unshard instead.
        ctx.module = module
        ctx.weight_name = weight_name
        ctx.scale_name = scale_name
        weight = _current_module_tensor(module, weight_name)
        scale = _current_module_tensor(module, scale_name)
        ctx.weight_shape = tuple(weight.shape)
        ctx.scale_shape = tuple(scale.shape)
        return _linear(x, weight, scale)

    @staticmethod
    def backward(ctx, grad):
        weight = _current_module_tensor(ctx.module, ctx.weight_name)
        scale = _current_module_tensor(ctx.module, ctx.scale_name)
        if tuple(weight.shape) != ctx.weight_shape or tuple(scale.shape) != ctx.scale_shape:
            raise RuntimeError(
                "FSDP did not rematerialize the native-FP8 linear before backward: "
                f"weight {tuple(weight.shape)} != {ctx.weight_shape}, "
                f"scale {tuple(scale.shape)} != {ctx.scale_shape}"
            )
        input_grad = _linear_input_grad(grad, weight, scale)
        if _DIAGNOSE_LINEAR_BACKWARD or os.environ.get(
            "PRIME_FP8_DIAG_LINEAR_BACKWARD", ""
        ) == "1":
            incoming_finite = bool(torch.isfinite(grad).all())
            output_finite = bool(torch.isfinite(input_grad).all())
            report_key = (tuple(weight.shape), incoming_finite, output_finite)
            if (
                (not incoming_finite or not output_finite)
                and report_key not in _REPORTED_LINEAR_BACKWARD
            ):
                _REPORTED_LINEAR_BACKWARD.add(report_key)
                finite_grad = grad[torch.isfinite(grad)]
                finite_output = input_grad[torch.isfinite(input_grad)]
                print(
                    "[fp8-linear-bwd] "
                    f"weight={tuple(weight.shape)} grad={tuple(grad.shape)} "
                    f"incoming_finite={incoming_finite} "
                    f"incoming_absmax={float(finite_grad.abs().max()) if finite_grad.numel() else float('nan'):.6e} "
                    f"output_finite={output_finite} "
                    f"output_absmax={float(finite_output.abs().max()) if finite_output.numel() else float('nan'):.6e}",
                    flush=True,
                )
        return input_grad, None, None, None


class FP8CheckpointLinear(nn.Linear):
    def forward(self, x):
        out = _Linear.apply(x, self, "weight", "weight_scale_inv")
        return out if self.bias is None else out + self.bias

    @classmethod
    def from_linear(cls, linear):
        with torch.device("meta"):
            module = cls(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
            )
        module.weight = nn.Parameter(
            torch.empty_like(linear.weight, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module.weight_scale_inv = nn.Parameter(
            torch.empty(
                linear.out_features // 128,
                linear.in_features // 128,
                device=linear.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        module.bias = linear.bias
        return module


class FP8CheckpointEmbedding(nn.Embedding):
    def forward(self, input):
        quantized = self.weight.view(torch.float8_e4m3fn).view(
            self.num_embeddings, self.embedding_dim
        ).index_select(0, input.flatten()).view(
            *input.shape, self.embedding_dim
        ).to(torch.bfloat16)
        scales = self.weight_scale_inv.index_select(0, input.flatten()).view(input.shape)
        return quantized * scales.to(quantized.dtype).unsqueeze(-1)

    @classmethod
    def from_embedding(cls, embedding):
        with torch.device("meta"):
            module = cls(
                embedding.num_embeddings,
                embedding.embedding_dim,
                embedding.padding_idx,
                embedding.max_norm,
                embedding.norm_type,
                embedding.scale_grad_by_freq,
                embedding.sparse,
            )
        module.weight = nn.Parameter(
            torch.empty(
                embedding.num_embeddings,
                embedding.embedding_dim // 2,
                device=embedding.weight.device,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        module.weight_scale_inv = nn.Parameter(
            torch.empty(embedding.num_embeddings, device=embedding.weight.device, dtype=torch.float32),
            requires_grad=False,
        )
        return module


def _dequant_weight(weight, scale, start, end, dtype):
    return (
        weight[start:end]
        .to(dtype)
        .view(-1, 128, weight.shape[1] // 128, 128)
        .mul(scale[start // 128 : end // 128].to(dtype)[:, None, :, None])
        .flatten(0, 1)
        .flatten(1, 2)
    )


def _scaled_mm(x, weight, weight_scale):
    scale = x.float().abs().amax(1).clamp_min(1e-12).div(torch.finfo(torch.float8_e4m3fn).max)
    quantized = (
        x.float()
        .div(scale[:, None])
        .clamp(torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max)
        .to(torch.float8_e4m3fn)
    )
    matrix = weight.t()
    if matrix.stride(0) != 1:
        matrix = matrix.t().contiguous().t()
    if weight_scale is None:
        weight_scale = torch.ones(matrix.shape[1], device=x.device, dtype=torch.float32)
    return torch._scaled_mm(
        quantized,
        matrix,
        scale[:, None],
        weight_scale[None],
        out_dtype=x.dtype,
        use_fast_accum=True,
    )


class _FP8LogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, scale, labels, inv_temperature, chunk_size):
        n = hidden.shape[0]
        vocab = weight.shape[0]
        logprobs = torch.empty(n, device=hidden.device, dtype=torch.float32)
        entropy = torch.empty_like(logprobs)
        logz = torch.empty_like(logprobs)
        mean_scaled = torch.empty_like(logprobs)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t = inv_temperature[start:end].unsqueeze(-1)
            m = torch.full((end - start,), float("-inf"), device=hidden.device, dtype=torch.float32)
            s = torch.zeros_like(m)
            t = torch.zeros_like(m)
            target = torch.zeros_like(m)
            for vocab_start in range(0, vocab, 16384):
                vocab_end = min(vocab_start + 16384, vocab)
                logits = _scaled_mm(
                    hidden_chunk,
                    weight[vocab_start:vocab_end],
                    scale[vocab_start:vocab_end],
                )
                scaled = logits.float() * inv_t
                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled)
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                row = torch.arange(end - start, device=hidden.device)
                index = (labels_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1).long()
                target = torch.where(mask, scaled[row, index], target)
            logz[start:end] = m + torch.log(s)
            logprobs[start:end] = target - logz[start:end]
            mean_scaled[start:end] = t / s
            entropy[start:end] = logz[start:end] - mean_scaled[start:end]
        ctx.save_for_backward(
            hidden, weight, scale, labels, inv_temperature, logz, mean_scaled
        )
        ctx.chunk_size = chunk_size
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs, grad_entropy):
        hidden, weight, scale, labels, inv_temperature, logz, mean_scaled = (
            ctx.saved_tensors
        )
        grad_hidden = torch.zeros_like(hidden)
        for start in range(0, hidden.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, hidden.shape[0])
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_logprob_chunk = (
                torch.zeros(end - start, device=hidden.device, dtype=torch.float32)
                if grad_logprobs is None
                else grad_logprobs[start:end].float()
            )
            grad_entropy_chunk = (
                torch.zeros(end - start, device=hidden.device, dtype=torch.float32)
                if grad_entropy is None
                else grad_entropy[start:end].float()
            )
            inv_t = inv_temperature[start:end].unsqueeze(-1)
            for vocab_start in range(0, weight.shape[0], 16384):
                vocab_end = min(vocab_start + 16384, weight.shape[0])
                weight_chunk = weight[vocab_start:vocab_end]
                scale_chunk = scale[vocab_start:vocab_end]
                scaled = _scaled_mm(hidden_chunk, weight_chunk, scale_chunk).float() * inv_t
                probability = torch.exp(scaled - logz[start:end].unsqueeze(-1))
                # d log p_y / dz = one_hot(y) - p and
                # d H / dz = p * (E[z] - z), where z is the
                # temperature-scaled logit.
                grad_logits = probability * (
                    -grad_logprob_chunk.unsqueeze(-1)
                    + grad_entropy_chunk.unsqueeze(-1)
                    * (mean_scaled[start:end].unsqueeze(-1) - scaled)
                )
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                row = torch.arange(end - start, device=hidden.device)
                index = (labels_chunk - vocab_start).clamp(0, vocab_end - vocab_start - 1).long()
                grad_logits[row, index] += grad_logprob_chunk * mask
                grad_hidden[start:end].add_(
                    _scaled_mm(
                        (grad_logits * inv_t * scale_chunk).to(hidden.dtype),
                        weight_chunk.t(),
                        None,
                    )
                )
        return grad_hidden, None, None, None, None, None


class _AllToAllRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group):
        out = x.new_empty((sum(output_splits),) + x.shape[1:])
        dist.all_to_all_single(out, x.contiguous(), output_splits, input_splits, group=group)
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((sum(ctx.input_splits),) + grad.shape[1:])
        dist.all_to_all_single(
            out,
            grad.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            group=ctx.group,
        )
        return out, None, None, None


def _balanced_splits(counts, rank):
    world = len(counts)
    total = sum(counts)
    targets = [total // world + (i < total % world) for i in range(world)]
    source_start = sum(counts[:rank])
    source_end = source_start + counts[rank]
    target_starts = [sum(targets[:i]) for i in range(world)]
    sends = [
        max(0, min(source_end, start + size) - max(source_start, start))
        for start, size in zip(target_starts, targets)
    ]
    target_start = target_starts[rank]
    target_end = target_start + targets[rank]
    source_starts = [sum(counts[:i]) for i in range(world)]
    receives = [
        max(0, min(target_end, start + size) - max(target_start, start))
        for start, size in zip(source_starts, counts)
    ]
    return sends, receives


def _fp8_lm_head_forward(self, hidden_states, labels=None, temperature=None):
    b, s, h = hidden_states.shape
    hidden_states = hidden_states.reshape(b * s, h)
    labels = labels.reshape(b * s)
    temperature = temperature.reshape(b * s)
    valid = labels != -100
    hidden = hidden_states[valid].contiguous()
    target = labels[valid].contiguous()
    inv_temperature = (1.0 / temperature[valid]).contiguous()
    sends = receives = None
    if getattr(self, "cp_group", None) is not None:
        count = torch.tensor([hidden.shape[0]], device=hidden.device)
        counts = [torch.empty_like(count) for _ in range(self.cp_size)]
        dist.all_gather(counts, count, group=self.cp_group)
        sends, receives = _balanced_splits([int(x.item()) for x in counts], self.cp_rank)
        hidden = _AllToAllRows.apply(hidden, receives, sends, self.cp_group)
        moved_target = target.new_empty(sum(receives))
        moved_temperature = inv_temperature.new_empty(sum(receives))
        dist.all_to_all_single(moved_target, target, receives, sends, group=self.cp_group)
        dist.all_to_all_single(moved_temperature, inv_temperature, receives, sends, group=self.cp_group)
        target = moved_target
        inv_temperature = moved_temperature
    scored, scored_entropy = _FP8LogProb.apply(
        hidden,
        self.weight.view(torch.float8_e4m3fn).view(self.out_features, self.in_features),
        self.weight_scale_inv,
        target,
        inv_temperature,
        self.chunk_size,
    )
    if sends is not None:
        scored = _AllToAllRows.apply(scored, sends, receives, self.cp_group)
        restored_entropy = scored_entropy.new_empty(sum(sends))
        dist.all_to_all_single(restored_entropy, scored_entropy, sends, receives, group=self.cp_group)
        scored_entropy = restored_entropy
    logprobs = torch.zeros(b * s, device=hidden_states.device, dtype=torch.float32) + scored.sum() * 0
    entropy = torch.zeros_like(logprobs) + scored_entropy.sum() * 0
    logprobs[valid], entropy[valid] = scored, scored_entropy
    return PrimeLmOutput(logprobs=logprobs.view(b, s), entropy=entropy.view(b, s))


def replace_linear(model, embed=False, lm_head=False):
    exclusions = set(model.config.quantization_config["modules_to_not_convert"])
    for name, module in list(model.named_modules()):
        if (
            not isinstance(module, nn.Linear)
            or name in exclusions
            or name.replace(".mlp.router.gate", ".mlp.gate") in exclusions
            or name.endswith((".self_attn.kv_b_proj", ".self_attn.indexer.wk"))
        ):
            continue
        if module.in_features % 128 or module.out_features % 128:
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, FP8CheckpointLinear.from_linear(module))
    if embed:
        model.model.embed_tokens = FP8CheckpointEmbedding.from_embedding(model.model.embed_tokens)
    if not lm_head:
        return
    model.lm_head.weight = nn.Parameter(
        torch.empty(
            model.lm_head.out_features,
            model.lm_head.in_features // 2,
            device=model.lm_head.weight.device,
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    model.lm_head.weight_scale_inv = nn.Parameter(
        torch.empty(model.lm_head.out_features, device=model.lm_head.weight.device, dtype=torch.float32),
        requires_grad=False,
    )
    model.lm_head.forward = _fp8_lm_head_forward.__get__(model.lm_head)


@torch.compiler.disable
def _grouped_mm(x, weight, scale, offsets):
    experts, output_size, input_size = weight.shape
    config = get_default_config(
        x.shape[0], experts, output_size, input_size, 1, "fp8_w8a8", block_shape
    )
    rows = torch.arange(x.shape[0], device=x.device)
    ids = torch.searchsorted(offsets, rows, right=True).clamp_max(experts - 1).int().view(-1, 1)
    sorted_ids, expert_ids, padded = _prepare_expert_assignment(
        ids,
        config,
        x.shape[0],
        1,
        experts,
        None,
        block_shape=block_shape,
        ignore_invalid_experts=True,
    )
    quantized, activation_scale = moe_kernel_quantize_input(
        x,
        None,
        torch.float8_e4m3fn,
        False,
        block_shape,
    )
    out = torch.empty(x.shape[0], 1, output_size, device=x.device, dtype=x.dtype)
    dispatch_fused_moe_kernel(
        quantized,
        weight,
        out,
        activation_scale,
        scale,
        None,
        None,
        sorted_ids,
        expert_ids,
        padded,
        False,
        1,
        config,
        tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16,
        True,
        False,
        False,
        False,
        False,
        block_shape,
    )
    # Expert-parallel dispatch can append rows after the last real token.  The
    # assignment above must give every row an expert id for the fused kernel,
    # but padded rows are not tokens and must not contribute in forward or
    # backward.  The dense expert implementation has the same zero-padding
    # contract.
    return out[:, 0].masked_fill((rows >= offsets[-1]).unsqueeze(-1), 0)


class _Experts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, s1, w2, s2, w3, s3, offsets):
        z1 = _grouped_mm(x, w1, s1, offsets)
        z3 = _grouped_mm(x, w3, s3, offsets)
        q1, qs1 = ops.scaled_fp8_quant(z1, use_per_token_if_dynamic=True)
        q3, qs3 = ops.scaled_fp8_quant(z3, use_per_token_if_dynamic=True)
        ctx.save_for_backward(q1, qs1, q3, qs3, w1, s1, w2, s2, w3, s3, offsets)
        F.silu(z1, inplace=True)
        z1.mul_(z3)
        del z3
        return _grouped_mm(z1, w2, s2, offsets)

    @staticmethod
    def backward(ctx, grad):
        q1, qs1, q3, qs3, w1, s1, w2, s2, w3, s3, offsets = ctx.saved_tensors
        z1 = q1.to(grad.dtype).mul_(qs1.to(grad.dtype))
        z3 = q3.to(grad.dtype).mul_(qs3.to(grad.dtype))
        hidden_grad = _grouped_mm(
            grad, w2.transpose(-2, -1), s2.transpose(-2, -1), offsets
        )
        sigmoid = torch.sigmoid(z1)
        z1_grad = hidden_grad * z3 * sigmoid * (1 + z1 * (1 - sigmoid))
        z3_grad = hidden_grad * F.silu(z1)
        x_grad = _grouped_mm(
            z1_grad, w1.transpose(-2, -1), s1.transpose(-2, -1), offsets
        )
        del hidden_grad, sigmoid, z1, z3, z1_grad
        x_grad.add_(
            _grouped_mm(
                z3_grad, w3.transpose(-2, -1), s3.transpose(-2, -1), offsets
            )
        )
        return x_grad, None, None, None, None, None, None, None


def fp8_experts(w1, w2, w3, s1, s2, s3, x, num_tokens_per_expert):
    """Run checkpoint-FP8 experts with TorchTitan's EP row permutation.

    After the EP all-to-all, rows are grouped first by sending rank and then
    by this rank's local experts.  ``num_tokens_per_expert`` consequently has
    ``ep_size * local_experts`` entries while the local weight shard has only
    ``local_experts`` matrices.  The ordinary grouped-expert path uses
    TorchTitan's ``expert_parallel`` decorator to transpose that logical
    layout, combine counts for each local expert, and add aligned padding.
    The native FP8 path used to unwrap the DTensors and call the kernel
    directly.  Most rows were then assigned expert ids beyond the local
    weight array and were ignored by the fused kernel.
    """
    if isinstance(w1, DTensor):
        w1, w2, w3, s1, s2, s3 = (
            value.to_local() for value in (w1, w2, w3, s1, s2, s3)
        )

    local_experts = w1.shape[0]
    if num_tokens_per_expert.shape[0] == local_experts:
        return _Experts.apply(
            x,
            w1,
            s1,
            w2,
            s2,
            w3,
            s3,
            torch.cumsum(num_tokens_per_expert, 0, dtype=torch.int32),
        )

    if num_tokens_per_expert.shape[0] % local_experts:
        raise ValueError(
            "EP token counts must be a whole number of local expert groups: "
            f"{num_tokens_per_expert.shape[0]} counts for {local_experts} experts"
        )

    from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    ep_size = num_tokens_per_expert.shape[0] // local_experts
    with torch.no_grad():
        permuted_indices, local_counts, _ = generate_permute_indices(
            num_tokens_per_expert,
            local_experts,
            ep_size,
            x.shape[0] + local_experts * TOKEN_GROUP_ALIGN_SIZE_M,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    # The last row is a shared zero sentinel selected for padding positions.
    padded_x = torch.vstack((x, x.new_zeros((1, x.shape[-1]))))
    permuted_x = padded_x[permuted_indices, :]
    out = _Experts.apply(
        permuted_x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        torch.cumsum(local_counts, 0, dtype=torch.int32),
    )
    out_unpermuted = out.new_empty(padded_x.shape)
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted[:-1]
