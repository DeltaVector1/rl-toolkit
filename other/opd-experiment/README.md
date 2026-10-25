# OPD / ECHO experiment archive

This directory stores recoverable patches for two abandoned training experiments
outside the main `patches/prime-rl/` set. Neither is needed for a normal Trinity RL
run. The running config sets none of their flags, so their code paths are inactive.

## OPD — off-policy distillation (cross-tokenizer)

Distill from a DeepSeek-V4 teacher into Trinity across tokenizers (DPCA logprob
projection), based on paper `2606.09456` (in `paper/`). Activated only by
`training_mode = "opd"` + a `teacher` config.

- `opd-teacher.patch` contains the teacher-side prime-rl `src/` changes that can be
  applied separately: vLLM `/recycle` endpoint, teacher prompt-logprob
  sanitizing, DeepSeek-V4 inference patches, and their tests. Apply on top of the
  main patch to restore full teacher support.
- `configs/` — `opd.toml`, `teacher.toml`, `protocol_recovery.toml`.
- `prepare_cuda_wheel_home.py`, `wait_for_gpu_release.py` — teacher-server dev helpers.
- `paper/` — the source paper.

The orchestrator-side OPD code (DPCA stack in `orchestrator/utils.py`, teacher hooks
in `orchestrator.py`) remains in the main `patches/prime-rl/` patch because it shares
code with the audit/token-usage functions. It only runs under `training_mode="opd"`.
To restore OPD, apply the main patch and `opd-teacher.patch`.

## ECHO — per-env supervised loss on observation tokens

A second experiment: supervised loss on observation tokens, per training env.
Activated only when a train env sets `echo = true`. Its trainer scaffolding
(`loss.py` `echo_loss_fn`, `environment_mask` threading through
`batch.py`/`data.py`/`packer.py`/`transport/types.py`/`trajectories.py`) is threaded
into every step with inactive all-False masks. All of this code remains in the main
patch because partial removal would break the trainer. No separate patch is needed.
Do not set `echo = true` to leave ECHO inactive.
