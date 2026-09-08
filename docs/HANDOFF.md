# Current handoff

Updated 2026-09-08 UTC. Use `configs/operations.toml` for every host, path,
port, container, and profile value.

## Current state

- `glm52-sparse-stability` completed 20/20 synthetic steps at 04:37:59 UTC.
  Warm forward/backward time was 177.66-179.92 seconds (median 178.39).
  Peak reserved memory stayed at 42.93 GiB for steps 3-20. All eight trainer
  containers exited with code 0 and no OOM flag by 04:38:09.
- The test uses 524,288 fresh synthetic tokens (16 x 32,768), PP8/CP8/EP8,
  learning rate 0, and the same saved initial adapter as the baseline. It uses
  an FP8 base with BF16 rank-32 LoRA on attention, shared-expert, and
  routed-expert linear weights. The router is frozen.
- Steps 1 and 2 match the baseline loss, entropy, gradient norm, and peak
  memory. Candidate step 2 is 14.8% faster on the reported forward/backward
  metric. Peak memory was 42.65 GiB and no OOM occurred. Detailed timings are
  in `docs/PERFORMANCE.md`.
- This is a controlled synthetic check, not a production qualification. Real
  production remains at 0 optimizer steps; the 20-step requirement is not
  passed. Root owns live monitoring.

## Repository and operating references

- The active config is `configs/qualification/glm52-sparse-stability.toml`.
  Do not use the older `configs/current/` decensor profile to resume this run.
- The next draft RL profile is `configs/general-agent/{trainer,orchestrator}.toml`.
- `configs/operations.toml` is the source for hosts, paths, ports, containers,
  and profiles. Do not copy these values into another launcher.
- `docs/RUNBOOK.md` contains run procedures. `docs/PERFORMANCE.md` contains
  measured timings and historical production results.
- `docs/PINS.md` records the reviewed and fetched upstream versions and the
  general-agent package and dataset revisions.
- The latest local `prime-rl.patch` refresh is `3759+sentinel`: two source
  blocks changed and the other 49 are unchanged. The current apply-check covers
  those two blocks; earlier full source/package apply-checks are historical.
  Dependency blocks are preserved and were not reapplied.
- The training-ops skill is under `.agents/skills/training-ops/`; the Prime
  Agent uses the matching skill under `.prime/agent/skills/training-ops/`.
- The normal read-only checks are:

  ```python
  import training_ops
  training_ops.status()
  training_ops.reward()
  training_ops.steps(limit=20)
  training_ops.watch(hours=2)
  ```

  `watch()` is a foreground monitor. Keep its interval at or below ten
  minutes. Root owns any live start, stop, or restart.

## Adapter and transport status

- The native-aware trained-adapter loader is deployed in trainer dependencies
  through qualification `901` on all eight hosts. It passed the two-rank
  CPU/Gloo checks for nonzero B restoration, exact export, invalid rejection,
  and source-order checks. Nonzero-trained live resume is not qualified.
- Cross-run resume still needs broadcast history and metadata handling in
  training-ops. The step-0 initial adapter path is not proof that trained
  adapter resume is ready.
- Adapter cache `3759` is committed locally but not deployed. H200 engines
  still run old code; both READY old step-0 caches remain available. Positive
  step adapter delivery is not measured.
- Shared prefetch `3971` and the partition prototype are isolated and
  untested. Do not claim a transport improvement from them.
- Never copy the full GLM weights casually: the H200 model is
  755,632,050,320 bytes (about 704 GiB).

## Known faults and completed repairs

- The latest production-shaped attempt stopped at 0 optimizer steps after a
  rollout over the 720-second limit and unfinished compute over the 600-second
  limit. It is historical, not a pass; detailed measurements are in
  `docs/PERFORMANCE.md`.
- Earlier H200 broadcast directories were root-owned. Repairs `8bc0602` and
  `7912cc0` create and repair the run directories for UID/GID 1000. Confirm
  ownership and source SHA before any new live run.
- The null harness retry storm was repaired in `eb5ffad`: inner timeout and
  retries are disabled, and a harness timeout retries only after a recorded
  failed policy call. Focused tests and shell checks passed.
- Native KV offload is not deployed. CPU-only validation passed with
  `kv_offloading_size=64` and `enable_cumem_allocator=True`; a live sparse-MLA,
  MTP, LoRA, cache, and throughput comparison is still required.
- General-agent Hub 0.1.4 is local under `environments/general-agent`.
  Its 4,413 usable tasks have not established a deployed evaluation. The
  official dataset repository revision is
  `e3f0ec777c02b734ad1a7d246cf06a709dc3e6e2`. Budget fix `92f06f4c` passed
  five actual-image CPU tests, but the real probe remains blocked by an
  output-directory permission error.

## Current inference and parity constraints

- H200 inference remains MTP 1 with EP8, FULL CUDA graph capture at 256
  scheduled tokens, and up to 128 requests. Context is 32K and the output
  limit is 16K. The MTP and reload measurements are component checks only;
  they are not full rollout-throughput approval. See `docs/PERFORMANCE.md`.
- Compare parity at equal selected-token counts and with the same activation
  allocation. Existing direct parity is diagnostic; no trained nonzero-adapter
  parity result is established. See `docs/PERFORMANCE.md`.

## Remaining checks

1. Run the separate two-step adapter-update check at LR `8e-6`. The completed
   LR0 test does not prove behavior after weights change. Its rollout,
   adapter, judging, and save times are not applicable; measure these in a
   real run. The earlier real batch had 17.2 times more tokens than this test.
2. Keep the 600-second trainer and 720-second rollout limits. Stop and trace a
   run that exceeds them; busy GPUs do not prove progress.
3. Before a production retry, verify configured source files, recursive
   imports, launcher mounts, patch targets, remote modes, broadcast ownership,
   and adapter source SHA as one set.
4. Deploy and qualify trained-adapter resume only after broadcast history and
   metadata handling are repaired. Preserve nonzero B through reset hooks.
5. Resolve the general-agent output-directory permission issue before claiming
   a real probe. Do not change neutral, decensor, or reward settings for it.

No Decensor environment or reward edits were made for these measurements.
