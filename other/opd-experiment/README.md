# OPD / ECHO experiment archive

Two abandoned training experiments, kept here as recoverable patches so they don't
clutter the main `patches/prime-rl/` set. Neither is needed for a normal Trinity RL
run — the running config sets none of their flags, so all their code paths are dormant.

## OPD — off-policy distillation (cross-tokenizer)

Distill from a DeepSeek-V4 teacher into Trinity across tokenizers (DPCA logprob
projection), based on paper `2606.09456` (in `paper/`). Activated only by
`training_mode = "opd"` + a `teacher` config.

- `opd-teacher.patch` — the teacher-side prime-rl `src/` changes that are *cleanly
  decoupled* from a normal run: vLLM `/recycle` endpoint, teacher prompt-logprob
  sanitizing, DeepSeek-V4 inference patches, and their tests. Apply on top of the
  main patch to restore full teacher support.
- `configs/` — `opd.toml`, `teacher.toml`, `protocol_recovery.toml`.
- `prepare_cuda_wheel_home.py`, `wait_for_gpu_release.py` — teacher-server dev helpers.
- `paper/` — the source paper.

Note: the *orchestrator-side* OPD scaffolding (DPCA stack in `orchestrator/utils.py`,
teacher hooks in `orchestrator.py`) rides along dormant inside the main
`patches/prime-rl/` patch — it's coupled to the audit/token-usage code and only runs
under `training_mode="opd"`. So restoring OPD = main patch + `opd-teacher.patch`.

## ECHO — per-env supervised loss on observation tokens

A second experiment: supervised loss on observation tokens, per training env.
Activated only when a train env sets `echo = true`. Its trainer scaffolding
(`loss.py` `echo_loss_fn`, `environment_mask` threading through
`batch.py`/`data.py`/`packer.py`/`transport/types.py`/`trajectories.py`) is threaded
into every step with dormant all-False masks, so it's **carried whole inside the main
patch** rather than split out — reverting it piecemeal would break the trainer. No
separate patch needed; just never set `echo = true`.
