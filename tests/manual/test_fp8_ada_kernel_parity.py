"""One-GPU parity and latency probe for serialized GLM FP8 matrices on Ada."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from prime_rl.trainer.models.layers.fp8_ada import (
    _linear,
    _linear_input_grad,
    _linear_input_grad_oracle,
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
    path=MODEL/index[name]
    with safe_open(path,framework="pt") as f:
        w=f.get_tensor(name)
        s=f.get_tensor(name.removesuffix(".weight")+".weight_scale_inv")
    ref_w=dequant(w,s).cuda()
    w=w.cuda()
    s=s.cuda()
    x=(torch.randn(64,w.shape[1],device="cuda",dtype=torch.bfloat16)*0.1)
    got=_linear(x,w,s)
    ref=F.linear(x.float(),ref_w).to(x.dtype)
    diff=(got.float()-ref.float())
    print(name,tuple(w.shape),tuple(s.shape),
          "forward_rel_rmse",float(diff.square().mean().sqrt()/ref.float().square().mean().sqrt()),
          "forward_cos",float(F.cosine_similarity(got.float().flatten(),ref.float().flatten(),dim=0)),
          "got_std",float(got.float().std()),"ref_std",float(ref.float().std()))
    g=(torch.randn(64,w.shape[0],device="cuda",dtype=torch.bfloat16)*0.1)
    got_b=_linear_input_grad(g,w,s,fqn=name)
    ref_b=g.float() @ ref_w
    diff_b=got_b.float()-ref_b.float()
    native_ms,native_peak=profiled(lambda:_linear_input_grad(g,w,s,fqn=name))
    oracle_ms,oracle_peak=profiled(
        lambda:_linear_input_grad_oracle(g,w,s,fqn=name),
        warmup=1,
        iterations=3,
    )
    print(name,"backward_rel_rmse",float(diff_b.square().mean().sqrt()/ref_b.float().square().mean().sqrt()),
          "backward_cos",float(F.cosine_similarity(got_b.float().flatten(),ref_b.float().flatten(),dim=0)),
          "native_ms",native_ms,"oracle_ms",oracle_ms,
          "native_peak_mib",native_peak,"oracle_peak_mib",oracle_peak)
