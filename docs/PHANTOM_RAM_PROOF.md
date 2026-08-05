# RETRACTED — do not circulate

> **This document's central claim is not supported and should not be sent to
> anyone.** It argued that the OOM kills were an accounting artifact — that the
> memory was not real. Three things contradict that, all found 2026-08-05:
>
> 1. The kernel call trace in the OOM is `shmem_fault -> shmem_alloc_folio ->
>    __alloc_pages -> out_of_memory`. That is the kernel allocating **real**
>    page-cache folios for a shared mapping and failing to find pages. It is
>    not a double-count of GPU memory.
> 2. The correlation this document rests on does not hold at rest: with ~29 GB
>    per GPU in use, host `shared` reads 1 GB, not ~230 GB.
> 3. When a 16K step finally ran (x8, 18:05), it died with 481 GB available,
>    `shared` at 1 GB and zero swap — no memory pressure of any kind. The
>    actual cause was a dtype error (`Promotion for Float8 Types is not
>    supported`) in `RMSNorm.forward`, unrelated to memory.
>
> What survives: the per-rank `shmem-rss` figures below are real and did roughly
> track GPU usage, and a plain 160 GB `cudaMalloc` produced no host charge. So
> *something* creates large shared mappings under the trainer workload, with
> `/dev/shm` holding only 609 MB and every CPU-offload flag set to `false`.
> That remains unexplained. The open question is what owns those mappings —
> answerable with a per-mapping `smaps` dump from a live rank during a step,
> which has never been captured because no 16K step has completed.
>
> The raw `dmesg` in `logs/oom-n82/` is the honest artifact. Use that instead.

---

## Original claim (unsupported, retained for reference)

Claim under test: on these nodes (Ubuntu, kernel 6.8, NVIDIA open kernel
module), memory allocated ON THE GPU is additionally counted by the Linux
kernel as host RAM belonging to the allocating process. At 16K context the
8 ranks' GPU usage (~46 GB x 8 = ~368 GB) plus real host usage exceeds the
503 GB of physical RAM in the kernel's accounting, causing OOM kills and
swap thrash, even though real host memory usage is small.

## Evidence 1 — kernel OOM-kill records (dmesg, verbatim)

The kernel logs what a process held when killed. `shmem-rss` is the shared
memory charged to the process. The trainer's REAL shared memory use is a few
hundred MB (NCCL buffers). The kernel recorded:

    [Tue Aug  4 11:35:40 2026] Out of memory: Killed process 1115946 (PRIME-RL::Train)
        total-vm:1131619100kB, anon-rss:7621784kB, file-rss:1807400kB, shmem-rss:34509292kB   (= 34.5 GB)
    [Tue Aug  4 13:36:21 2026] Out of memory: Killed process 1159130 (PRIME-RL::Train)
        anon-rss:1647124kB, file-rss:1298088kB, shmem-rss:49451008kB                           (= 49.4 GB)
    [Tue Aug  4 17:39:54 2026] Killed process ... shmem-rss:51160064kB                         (= 51.2 GB)

34.5-51.2 GB of "shared memory" per rank matches that rank's GPU usage at the
moment of death (GPU cards are 47.4 GB; measured per-rank GPU usage at 16K is
44-47 GB). Real shared memory (/dev/shm) measured at the same moments: under
1 GB. The number the kernel used to justify the kill is the GPU's contents.

## Evidence 2 — node-level accounting during a step (free -g, verbatim)

    Mem:  total 503   used 502   free 2   shared 474
    Swap: total 135   used 94    free 41

"shared 474" GB on a node whose processes' combined REAL host memory
(anon+file from ps/smaps) totals ~30 GB. 474 GB is within measurement noise
of 8 ranks x ~46 GB GPU usage + ~100 GB legitimate shared use. /dev/shm at
the same moment held 0.5 GB (df /dev/shm), so the 474 GB is not tmpfs files.

## Evidence 3 — the phantom pages do not swap

Under pressure the kernel swapped out 94 GB (Evidence 2, Swap used). The
"shared" figure did NOT decrease — the kernel evicted the processes' REAL
pages (orchestrator, sshd; hence nodes stopped answering SSH) while the
470+ GB charge remained. Pages that cannot be swapped and vanish instantly
when the GPU program exits are driver-pinned mirrors of device allocations,
not application data.

## Evidence 4 — controlled experiment (before fix)

Procedure: on an otherwise idle node, allocate exactly 20 GB on each of the
8 GPUs (160 GB device total) from one process whose host footprint is <2 GB.
If host accounting is honest, "used" rises by ~2 GB. If the double-count
exists, "used"/"shared" rise by ~160 GB.

Not runnable before the fix (the fleet was already saturated), but the same
relationship is measured in Evidence 1-3 from the live run: per-rank
"shmem-rss" in the kill records equals that rank's GPU usage, and node-level
"shared" equals the sum of the 8 GPUs' usage.

## Evidence 5 — same experiment after the fix (nvidia_uvm HMM disabled)

Node n83, kernel 6.8.0-134-generic, uvm_disable_hmm=Y, verbatim:

    NODE n83  kernel=6.8.0-134-generic  uvm_disable_hmm=Y
    --- BASELINE (GPUs idle) ---
    Mem:  total 503  used 4  free 500  shared 0  buff/cache 0  available 498
    GPU MiB: 1 1 1 1 1 1 1 1

    --- WITH 20GB x 8 GPUs = 160GB ON DEVICE ---
    Mem:  total 503  used 6  free 498  shared 0  buff/cache 1  available 497
    GPU MiB: 19470 19470 19470 19470 19470 19470 19470 19470

160 GB placed on the GPUs. Host "used" moved 4 GB -> 6 GB (the +2 GB is the
Python/CUDA process itself) and "shared" stayed at 0. Before the fix the same
160 GB would have appeared as ~160 GB of host "shared". The double-count is
gone, and this is the direct experimental confirmation of the diagnosis.

## The fix applied

`/etc/modprobe.d/glm52-uvm-nohmm.conf`: `options nvidia_uvm uvm_disable_hmm=1`
HMM (Heterogeneous Memory Management) registers GPU memory with the kernel's
page tables so CPU and GPU can share address spaces; a side effect on this
kernel/driver combination is that device pages appear in host memory
accounting. Training does not use the shared-address-space feature, so
disabling it removes the double-count with no functional loss.
