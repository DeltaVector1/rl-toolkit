"""Single-node NCCL SHM all-to-all audit.

The script is run in a disposable trainer image.  It uses exact 384 MiB BF16
and 192 MiB FP8-equivalent payloads and records splits and byte checksums.
"""

import hashlib
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


ROWS = 32768
COLS = 6144
WARMUPS = int(os.environ.get("AUDIT_WARMUPS", "3"))
ITERS = int(os.environ.get("AUDIT_ITERS", "7"))


def pin_cpu(local_rank: int, world: int) -> list[int]:
    """Keep each rank on the NUMA node attached to its GPU group."""
    requested = os.environ.get("AUDIT_NUMA", "auto")
    if os.environ.get("AUDIT_CROSS4") == "1" and world == 4:
        node = 0 if local_rank < 2 else 1
    elif requested in {"0", "1"}:
        node = int(requested)
    else:
        node = 0 if world == 8 and local_rank < 4 else 1
    # nvidia-smi reports 0-31,64-95 for node 0 and 32-63,96-127 for node 1.
    cpu_base = 0 if node == 0 else 32
    slot = local_rank % 4
    cpus = list(range(cpu_base + slot * 8, cpu_base + slot * 8 + 8))
    os.sched_setaffinity(0, cpus)
    return cpus


def timed_all_to_all(send: torch.Tensor, recv: torch.Tensor) -> tuple[list[float], list[int]]:
    world = dist.get_world_size()
    rows_per_peer = ROWS // world
    splits = [rows_per_peer] * world
    for _ in range(WARMUPS):
        dist.all_to_all_single(recv, send, splits, splits)
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_to_all_single(recv, send, splits, splits)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return samples, splits


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact contiguous device bytes without dtype-specific NumPy support."""
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if os.environ.get("AUDIT_NO_PIN") == "1":
        cpus = sorted(os.sched_getaffinity(0))
    else:
        cpus = pin_cpu(local_rank, world)
    torch.cuda.set_device(local_rank)
    dtype_name = os.environ.get("AUDIT_DTYPE", "bf16")
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.uint8
    element_size = torch.empty((), dtype=dtype).element_size()
    dist.init_process_group("nccl")
    try:
        # The per-rank input is exactly 384 MiB BF16 or 192 MiB FP8 bytes.
        send = torch.empty((ROWS, COLS), dtype=dtype, device="cuda")
        recv = torch.empty_like(send)
        send.fill_(1)
        recv.zero_()
        input_sha256 = tensor_sha256(send)
        torch.cuda.reset_peak_memory_stats()
        samples, splits = timed_all_to_all(send, recv)
        dist.barrier()
        output_sha256 = tensor_sha256(recv)
        rank_integrity = {
            "local_rank": local_rank,
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
            "send_splits": splits,
            "recv_splits": splits,
        }
        per_rank_integrity = [None] * world
        dist.all_gather_object(per_rank_integrity, rank_integrity)
        # Validate a byte value after the timed operations.  This catches a
        # completed collective that returned corrupt data.
        checksum = int(recv[0, 0].item())
        expected = 1
        payload = ROWS * COLS * element_size
        result = {
            "world": world,
            "local_rank": local_rank,
            "numa": os.environ.get("AUDIT_NUMA", "auto"),
            "trial": os.environ.get("AUDIT_TRIAL", ""),
            "cpus": cpus,
            "send_splits": splits,
            "recv_splits": splits,
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
            "per_rank_integrity": per_rank_integrity,
            "min_channels": os.environ.get("NCCL_MIN_NCHANNELS"),
            "max_channels": os.environ.get("NCCL_MAX_NCHANNELS"),
            "dtype": dtype_name,
            "payload_mib": payload / (1024 * 1024),
            "median_ms": statistics.median(samples) * 1000,
            "min_ms": min(samples) * 1000,
            "max_ms": max(samples) * 1000,
            # Aggregate send bytes across all ranks.  Every rank sends the
            # payload once, split evenly among all peers.
            "aggregate_send_gb_s": (payload * world) / statistics.median(samples) / 1e9,
            "aggregate_send_gib_s": (payload * world) / statistics.median(samples) / (1024**3),
            "checksum": checksum,
            "expected": expected,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            "free_gib_after": torch.cuda.mem_get_info()[0] / (1024**3),
        }
        if local_rank == 0:
            print("AUDIT_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
