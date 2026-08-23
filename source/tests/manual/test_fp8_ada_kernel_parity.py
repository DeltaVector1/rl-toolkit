"""One-GPU parity probe for serialized GLM FP8 dense matrices on Ada."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from prime_rl.trainer.models.layers.fp8_ada import _linear

MODEL=Path("/model")
NAMES=[
 "model.layers.0.self_attn.q_a_proj.weight",
 "model.layers.0.self_attn.o_proj.weight",
]

def dequant(w,s):
    return (w.float().view(w.shape[0]//128,128,w.shape[1]//128,128)
            * s.float()[:,None,:,None]).flatten(0,1).flatten(1,2)

index=json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]
torch.manual_seed(1234)
for name in NAMES:
    path=MODEL/index[name]
    with safe_open(path,framework="pt") as f:
        w=f.get_tensor(name)
        s=f.get_tensor(name.removesuffix(".weight")+".weight_scale_inv")
    ref_w=dequant(w,s).cuda().bfloat16()
    w=w.cuda()
    s=s.cuda()
    x=(torch.randn(64,w.shape[1],device="cuda",dtype=torch.bfloat16)*0.1)
    got=_linear(x,w,s)
    ref=F.linear(x,ref_w)
    diff=(got.float()-ref.float())
    print(name,tuple(w.shape),tuple(s.shape),
          "forward_rel_rmse",float(diff.square().mean().sqrt()/ref.float().square().mean().sqrt()),
          "forward_cos",float(F.cosine_similarity(got.float().flatten(),ref.float().flatten(),dim=0)),
          "got_std",float(got.float().std()),"ref_std",float(ref.float().std()))
    g=(torch.randn(64,w.shape[0],device="cuda",dtype=torch.bfloat16)*0.1)
    got_b=_linear(g,w.T,s.T)
    ref_b=F.linear(g,ref_w.T)
    diff_b=got_b.float()-ref_b.float()
    print(name,"backward_rel_rmse",float(diff_b.square().mean().sqrt()/ref_b.float().square().mean().sqrt()),
          "backward_cos",float(F.cosine_similarity(got_b.float().flatten(),ref_b.float().flatten(),dim=0)))
