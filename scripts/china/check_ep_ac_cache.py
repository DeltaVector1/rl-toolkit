"""Verify the EP activation-checkpoint cache skips the recompute all-to-all
AND is numerically exact.

Runs the same checkpointed dispatch region twice -- once with the cache
disabled (cache=None, original behaviour) and once enabled -- comparing:
  1. the number of real all_to_all_single calls (cached must be strictly fewer),
  2. that the cache is actually populated by the forward pass,
  3. outputs and gradients (must be bitwise identical).
"""

import os

import torch
import torch.distributed as dist

import prime_rl.trainer.distributed.expert_parallel as ep

_real_a2a = dist.all_to_all_single
COUNT = {"n": 0}


def _counting_a2a(*args, **kwargs):
    COUNT["n"] += 1
    return _real_a2a(*args, **kwargs)


def run_region(x, out_splits, in_splits, group_name, cache):
    from torch.utils.checkpoint import checkpoint

    def block(inp):
        y = ep._INT4Dispatch.apply(
            inp,
            out_splits,
            in_splits,
            group_name,
            "dispatch",
            cache,
            None,
        )
        return torch.nn.functional.silu(y) * 1.5

    return checkpoint(block, x, use_reentrant=False, preserve_rng_state=False)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    dist.all_to_all_single = _counting_a2a
    ep.dist.all_to_all_single = _counting_a2a

    torch.manual_seed(1234 + rank)
    rows_per_peer, cols = 64, 6144
    rows = rows_per_peer * world
    splits = [rows_per_peer] * world
    base = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)

    results, counts, populated = {}, {}, {}
    for label, cache in (("uncached", None), ("cached", [None])):
        x = base.clone().detach().requires_grad_(True)
        COUNT["n"] = 0
        out = run_region(x, splits, splits, group_name, cache)
        populated[label] = cache is not None and cache[0] is not None
        loss = (out.float() ** 2).sum()
        loss.backward()
        counts[label] = COUNT["n"]
        results[label] = (out.detach().clone(), x.grad.detach().clone())
        if cache is not None:
            assert cache[0] is None, f"rank{rank}: cache not consumed by recompute"

    o_u, g_u = results["uncached"]
    o_c, g_c = results["cached"]
    out_exact = torch.equal(o_u, o_c)
    grad_exact = torch.equal(g_u, g_c)
    fewer = counts["cached"] < counts["uncached"]

    checks = torch.tensor(
        [float(out_exact), float(grad_exact), float(fewer), float(populated["cached"])],
        device="cuda",
    )
    dist.all_reduce(checks, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"world={world} rows={rows} cols={cols}")
        print(
            f"all_to_all_single calls: uncached={counts['uncached']} cached={counts['cached']}"
        )
        print(f"cache populated by forward: {bool(checks[3])}")
        print(f"outputs bitwise identical:  {bool(checks[0])}")
        print(f"grads   bitwise identical:  {bool(checks[1])}")
        print(f"cached did fewer a2a calls: {bool(checks[2])}")
        print("RESULT:", "PASS" if checks.min().item() == 1.0 else "FAIL")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
