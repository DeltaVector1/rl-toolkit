"""All-to-all benchmark that reproduces the trainer's SM contention.

The earlier version ran the all-to-all on otherwise-idle GPUs and overstated the
benefit of extra NCCL channels by ~1.9x against ~1.06x measured in the real
trainer. NCCL channels occupy SMs, so with no compute running they are free;
in training they compete with the model's kernels.

This version keeps a configurable amount of GEMM work in flight on a separate
stream while timing the all-to-all, so channel counts can be compared under
conditions resembling the real forward/backward pass. Set CONTEND=0 to
reproduce the old idle-GPU numbers for comparison.
"""

import os
import statistics
import threading
import time

import torch
import torch.distributed as dist

ranks_per_node = 8
link_bytes_per_second = 1.25e9  # 10 Gb/s


class Contender:
    """Keeps the SMs busy with GEMMs on a side stream, like the model's compute."""

    def __init__(self, size: int):
        self.stream = torch.cuda.Stream()
        self.a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
        self.b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
        self.stop = threading.Event()
        self.thread = None

    def _loop(self):
        with torch.cuda.stream(self.stream):
            while not self.stop.is_set():
                for _ in range(20):
                    torch.mm(self.a, self.b)
                self.stream.synchronize()

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        time.sleep(1.0)  # let it saturate before timing
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=10)
        torch.cuda.synchronize()
        return False


def make_splits(rows, world, skew):
    """Uniform when skew=1. Higher skew concentrates rows on some peers, like
    real MoE routing where a few experts are hot. Total rows is preserved."""
    base = rows // world
    if skew <= 1.0:
        return [base] * world, base * world
    hot = world // 8 or 1
    hot_rows = int(base * skew)
    rest = (rows - hot_rows * hot) // (world - hot)
    sp = [hot_rows] * hot + [rest] * (world - hot)
    return sp, sum(sp)


def bench(group, rows, cols, iters=10):
    world = dist.get_world_size(group)
    rows -= rows % world
    skew = float(os.environ.get("SKEW", "1"))
    rank = dist.get_rank(group)
    # Every rank sends in_splits[j] rows to rank j, so rank j receives
    # in_splits[j] from each peer: its out_splits is that value repeated.
    in_splits, total = make_splits(rows, world, skew)
    out_splits = [in_splits[rank]] * world
    send = torch.randint(0, 127, (total, cols), device="cuda", dtype=torch.uint8)
    recv = send.new_empty((sum(out_splits), cols))
    for _ in range(3):
        dist.all_to_all_single(recv, send, out_splits, in_splits, group=group)
    torch.cuda.synchronize()
    dist.barrier(group)
    s = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_to_all_single(recv, send, out_splits, in_splits, group=group)
        torch.cuda.synchronize()
        s.append(time.perf_counter() - t0)
    return statistics.median(s), total * cols


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    g = dist.group.WORLD

    T = int(os.environ.get("BENCH_T", "1024"))
    contend = int(os.environ.get("CONTEND", "1"))
    gemm = int(os.environ.get("GEMM_SIZE", "4096"))
    dim, rows = 6144, T * 8

    ctx = Contender(gemm) if contend else None
    if ctx:
        ctx.__enter__()
    try:
        t_int4, b_int4 = bench(g, rows, dim // 2)
        t_fp8, b_fp8 = bench(g, rows, dim)
    finally:
        if ctx:
            ctx.__exit__()

    per_layer_t = 2 * t_int4 + 2 * t_fp8
    per_layer_b = 2 * b_int4 + 2 * b_fp8
    offnode = per_layer_b * ranks_per_node * (1 - ranks_per_node / world)
    ideal = offnode / link_bytes_per_second

    if rank == 0:
        print(
            f"RESULT {os.environ.get('BENCH_LABEL','')} contend={contend} skew={os.environ.get('SKEW','1')} "
            f"int4={t_int4*1000:.0f}ms fp8={t_fp8*1000:.0f}ms "
            f"per_layer={per_layer_t*1000:.0f}ms ideal={ideal*1000:.0f}ms "
            f"eff={100*ideal/per_layer_t:.0f}% | a2a_per_microbatch={per_layer_t*75:.0f}s",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
