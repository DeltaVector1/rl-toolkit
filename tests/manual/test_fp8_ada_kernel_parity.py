"""One-GPU parity and latency probe for serialized GLM FP8 matrices on Ada."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from prime_rl.trainer.models.layers.fp8_ada import (
    _grouped_mm,
    _linear,
    _linear_input_grad,
    _linear_input_grad_oracle,
    _native_block_fp8_mm,
)

MODEL=Path("/model")
NAMES=[
 "model.layers.0.self_attn.q_a_proj.weight",
 "model.layers.0.self_attn.o_proj.weight",
]

def dequant(w,s):
    return (w.float().view(w.shape[0]//128,128,w.shape[1]//128,128)
            * s.float()[:,None,:,None]).flatten(0,1).flatten(1,2)

def profiled(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    return start.elapsed_time(end) / iterations, peak / (1 << 20)

index=json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]
torch.manual_seed(1234)
for name in NAMES:
    scale_name=name.removesuffix(".weight")+".weight_scale_inv"
    with safe_open(MODEL/index[name],framework="pt") as f:
        w=f.get_tensor(name)
    with safe_open(MODEL/index[scale_name],framework="pt") as f:
        s=f.get_tensor(scale_name)
    ref_w=dequant(w,s).cuda()
    w=w.cuda()
    s=s.cuda()
    x=(torch.randn(64,w.shape[1],device="cuda",dtype=torch.bfloat16)*0.1)
    got=_linear(x,w,s)
    forward_offsets=torch.tensor([x.shape[0]],device="cuda",dtype=torch.int32)
    fused_f=_grouped_mm(
        x,w.unsqueeze(0),s.unsqueeze(0),forward_offsets
    )
    direct_f=_native_block_fp8_mm(
        x,w,s,weight_is_output_major=True
    )
    ref=F.linear(x.float(),ref_w).to(x.dtype)
    diff=(got.float()-ref.float())
    fused_diff=(fused_f.float()-ref.float())
    direct_diff=(direct_f.float()-ref.float())
    fused_forward_ms,fused_forward_peak=profiled(
        lambda:_grouped_mm(
            x,w.unsqueeze(0),s.unsqueeze(0),forward_offsets
        )
    )
    direct_forward_ms,direct_forward_peak=profiled(
        lambda:_native_block_fp8_mm(x,w,s,weight_is_output_major=True)
    )
    print(name,tuple(w.shape),tuple(s.shape),
          "forward_rel_rmse",float(diff.square().mean().sqrt()/ref.float().square().mean().sqrt()),
          "forward_cos",float(F.cosine_similarity(got.float().flatten(),ref.float().flatten(),dim=0)),
          "fused_forward_rel_rmse",float(fused_diff.square().mean().sqrt()/ref.float().square().mean().sqrt()),
          "fused_forward_cos",float(F.cosine_similarity(fused_f.float().flatten(),ref.float().flatten(),dim=0)),
          "direct_forward_rel_rmse",float(direct_diff.square().mean().sqrt()/ref.float().square().mean().sqrt()),
          "direct_forward_cos",float(F.cosine_similarity(direct_f.float().flatten(),ref.float().flatten(),dim=0)),
          "fused_forward_ms",fused_forward_ms,"direct_forward_ms",direct_forward_ms,
          "fused_forward_peak_mib",fused_forward_peak,"direct_forward_peak_mib",direct_forward_peak,
          "got_std",float(got.float().std()),"ref_std",float(ref.float().std()))
    g=(torch.randn(64,w.shape[0],device="cuda",dtype=torch.bfloat16)*0.1)
    got_b=_linear_input_grad(g,w,s,fqn=name)
    ref_b=g.float() @ ref_w
    diff_b=got_b.float()-ref_b.float()
    offsets=torch.tensor([g.shape[0]],device="cuda",dtype=torch.int32)
    fused_b=_grouped_mm(
        g,w.t().unsqueeze(0),s.t().unsqueeze(0),offsets
    )
    fused_diff=fused_b.float()-ref_b.float()
    native_ms,native_peak=profiled(
        lambda:_native_block_fp8_mm(g,w,s,weight_is_output_major=False)
    )
    fused_ms,fused_peak=profiled(
        lambda:_grouped_mm(g,w.t().unsqueeze(0),s.t().unsqueeze(0),offsets)
    )
    oracle_ms,oracle_peak=profiled(
        lambda:_linear_input_grad_oracle(g,w,s,fqn=name),
        warmup=1,
        iterations=3,
    )
    print(name,"backward_rel_rmse",float(diff_b.square().mean().sqrt()/ref_b.float().square().mean().sqrt()),
          "backward_cos",float(F.cosine_similarity(got_b.float().flatten(),ref_b.float().flatten(),dim=0)),
          "fused_transposed_rel_rmse",float(fused_diff.square().mean().sqrt()/ref_b.float().square().mean().sqrt()),
          "fused_transposed_cos",float(F.cosine_similarity(fused_b.float().flatten(),ref_b.float().flatten(),dim=0)),
          "native_ms",native_ms,"fused_transposed_ms",fused_ms,"oracle_ms",oracle_ms,
          "native_peak_mib",native_peak,"fused_transposed_peak_mib",fused_peak,
          "oracle_peak_mib",oracle_peak)
