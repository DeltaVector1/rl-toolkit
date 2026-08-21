#!/usr/bin/env python3
"""Verify PP state dictionaries gather only to rank 0 over a CPU group."""

import torch
import torch.distributed as dist

from prime_rl.trainer.weights import merge_pp_state_dict, set_pp_gather_context


dist.init_process_group("gloo")
rank = dist.get_rank()
world = dist.get_world_size()
assert world == 8

stage_master_ranks = [0, 2, 4, 6]
group = dist.new_group(ranks=stage_master_ranks, backend="gloo")
is_stage_master = rank in stage_master_ranks
set_pp_gather_context(group, is_stage_master=is_stage_master)

local = {f"stage_{rank}": torch.tensor([rank])} if is_stage_master else {}
merged = merge_pp_state_dict(local, is_master=rank == 0)
if rank == 0:
    assert sorted(merged) == ["stage_0", "stage_2", "stage_4", "stage_6"]
    print("PP weight gather to rank 0 over Gloo: PASS", flush=True)
elif is_stage_master:
    assert merged == local
else:
    assert merged == {}

dist.barrier()
dist.destroy_process_group()
