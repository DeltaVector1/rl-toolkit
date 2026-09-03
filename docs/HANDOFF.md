# Engineering handoff

This file records decisions that are easy to lose. Use `git log` for the full
change history.

## Rollout service

The rollout timeout is absolute. A harness timeout is terminal and does not
enter the network retry path. This rule fixed an unlimited retry that held a
batch at 1008 of 1024 for more than 90 minutes.

The orchestrator addresses every replica directly. Do not add a shared proxy
to the production request path. It would remove group-to-replica ownership.

The refusal evaluation is held out. Training uses the balanced 4,000-prompt
`training-eval-set`. The `final-eval-set` runs after training and reports each
source separately.

## Trainer and inference comparison

Compare these four paths with the same tokens and adapter:

1. trainer scoring;
2. inference prefill;
3. inference eager decode;
4. inference CUDA-graph decode.

Compare selected-token FP32 log probabilities. Repeat the request alone and at
several batch sizes. Test long positions and adapter reload. Do not use an
objective change to hide a numeric difference.

## Current GLM-5.2 design

GLM qualification and production files use the same repository and remote
deployment path. `training_ops` copies every mounted Prime-RL file before a
run starts.

The trainer targets the attention projections, shared experts, and routed
experts. The router stays frozen.

The native FP8 expert path must keep its low-memory backward. A past step failed
when one rank had 1.5 MiB free and needed a 24 MiB output. The current source
releases expert buffers at their last use and retries one output allocation
after releasing unused CUDA cache.

The GLM trainer uses PP8, CP8, and node-local EP8. Keep source-major expert rows
before the local fused expert call. Keep FP8 weights and scales as explicit
autograd inputs. Keep backward snapshots outside checkpoint saved-tensor
hooks.

Do not add attention projections to native-FP8 activation checkpoint targets.
Their FSDP recomputation produced nonfinite sparse-query activations.

Native MTP is disabled. A one-token trial made FlashMLA receive a stride above
the signed 32-bit limit at 32K context and stopped every worker.

## Faults that must not return

- Do not turn a terminal timeout into a transient retry.
- Do not accept a partial GRPO group after a policy update.
- Do not start a new run while an old supervisor still has the old run name.
- Do not report an H200 API response as healthy without a real completion.
- Do not report a distributed trainer as alive when only the head container is
  alive.
- Do not use a hidden metric host fallback. Name the host or return an error.
- Do not use generated copies of old run configurations.
- Do not weaken parity limits, batch size, context length, or precision to get
  a passing result.

The known fault table is
`.agents/skills/training-ops/references/errors.md`.
