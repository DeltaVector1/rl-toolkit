# GLM52 Prime-RL v0.8.0 patch series

This directory contains a source-maintenance port. It is not a deployment
approval.

## Exact base

Apply this series to Prime-RL v0.8.0 at:

```text
60bc29547a8824ad1de7b9af8d265e2b27b2a72d
```

The current project pin is
`f50ec5c4c8c45864ec535dead2c3b630de50d068`. Prime-RL v0.8.0 is 32
commits after that pin. The public development head that was audited was
`6e33f3f82fea6a9014dcaad5503b2b35a0a12c97`. Do not use that development
head for the first port.

Prime-RL v0.8.0 changes the packing, transport, LoRA, environment, and
configuration interfaces. It also changes vLLM from 0.24 to 0.26. A clean
source apply does not show that these runtime changes are safe.

## Inventory result

The project override differs from the old Prime-RL pin in 38 implementation
paths and 9 test paths. The portable v0.8.0 result has 48 paths because it
also carries the deterministic FP8 parity tool and updates one upstream LoRA
client test.

No local patch has an exact upstream patch-id match. Upstream commit
`06eb91d7c88828c75d466db6c4a2feebd4ff482d` overlaps the GLM sequence-length
contract. This port keeps the v0.8.0 contract and adds the local CP behavior.
The port also keeps upstream data-position checkpoint and resume behavior
from `e229a286f29e506af46a3b90a6ea9dc203899155`.

The port does not include:

- MXFP4 or other INT4 checkpoint weights.
- Temporary FP8 component-bisection controls.
- Unconditional GLM synchronization prints.
- The temporary EP CUDA-event profiler.
- Patch-only DeepEP and environment changes that have no current production
  requirement.

The port keeps:

- PP8, CP8, and node-local EP8.
- Serialized GLM 128 by 128 block-FP8 loading.
- The Ada native-FP8 linear and expert path.
- Source-major EP row permutation.
- Explicit FP8 weight and scale autograd inputs.
- Owned backward snapshots outside saved-tensor hooks.
- Module-FQN nonfinite backward attribution.
- Finite fully masked sparse-MLA behavior.
- LoRA policy initialization and complete PP adapter export.
- Filesystem rollout and adapter transport.
- Queue-inclusive off-policy rejection.
- Absolute LoRA administration deadlines and exact visibility checks.
- Nonfinite gradient refusal and adapter broadcast safety.

The INT4 expert-activation transport remains. It is not an INT4 weight path.

## Ordered patches

1. `0001`: configuration, CP helpers, and monitor integration.
2. `0002`: PP8, CP8, node-local EP8, and pipeline execution.
3. `0003`: native GLM block-FP8 loading, Ada backward, LoRA, and expert
   execution.
4. `0004`: sparse MLA forward and backward parity and fully masked-row safety.
5. `0005`: trainer safety, policy continuity, microbatch data, and filesystem
   adapter broadcast.
6. `0006`: durable rollout transport, group completeness, off-policy
   enforcement, and LoRA administration recovery.
7. `0007`: transport and topology regression tests.
8. `0008`: FP8, LoRA, and deterministic parity regression tests.

`SHA256SUMS` records the exact patch payloads.

## Apply in a disposable clone

```bash
git checkout 60bc29547a8824ad1de7b9af8d265e2b27b2a72d
git am /path/to/v0.8.0-glm52/000*.patch
git submodule update --init --recursive
```

The v0.8.0 submodule revisions are:

```text
renderers  d4707862ac83aa3773c21f4096aec72bd17b91e4
verifiers  d30a3f48e5f14b06b3081b2102ec32cc3149b849
```

Apply these project patches inside the matching submodule roots:

```text
patches/renderers/v0.8.0-disconnected-and-tokenizer.patch
patches/verifiers/v0.8.0-disconnected-provider-retry.patch
```

The old dependency patches remain for the old project pin. Do not apply both
old and v0.8.0 dependency patches.

## Validation completed

- All eight patches apply with `git am` to the exact v0.8.0 base.
- Each cumulative patch revision passes `compileall` and `git diff --check`.
- The focused Prime-RL suite passes: 43 tests.
- The full renderers suite passes: 2,589 tests, with 153 skips and 1 expected
  failure.
- The targeted verifiers relay suite passes: 25 tests.
- The added verifiers disconnect-retry test passes.
- Both dependency patches pass a clean `git apply --check` at their exact
  v0.8.0 submodule revisions.
- The x56p trainer TOML parses after removal of the two obsolete false-only
  bisection keys. It preserves PP8, CP8, EP8, LR `8e-6`, maximum grad norm
  `1.0`, weight decay `0.0`, and native FP8.

A broad Prime-RL unit run was not clean. The observed failures included
uninstalled Environment Hub tasksets, two upstream Nemotron-H FlashAttention
test failures in this local wheel environment, and test-order contamination
from FP8 import stubs. The focused FP8 tests pass when isolated. These results
do not authorize a runtime gate.

## Required semantic work before deployment

1. Rewrite the production combined RL configuration for the v0.8.0
   `trainer` and `orchestrator` sections. The current x56p orchestrator TOML
   fails v0.8.0 validation with 14 schema errors.
2. Verify v0.8.0 orchestrator-side packing for PP8 and CP8. The packer uses
   the DP mesh size as its worker count. Measure microbatch count, PP stage
   balance, and token ownership before a trainer run.
3. Verify the flattened v0.8.0 output and transport records against the
   filesystem edge protocol. Verify restart position and immutable batch
   identity.
4. Verify the singleton v0.8.0 LoRA state model against policy-only restart,
   run-slot reset, PP adapter export, and exact H200 visibility.
5. Port the Decensor v0.8 environment package and keep final-assistant-turn
   judging.
6. Build a new image with the v0.8.0 lock and vLLM 0.26. Do not reuse the
   x56p containers.
7. Run the staged gates in `TRAINING_EXPECTATIONS.md`: one matrix, one GLM
   layer, one node, one full-topology zero-LR step, three performance steps,
   batch 2048 capacity, and only then the final 50-step run.

Do not deploy this series to x56p. Do not change a trainer or orchestrator
until the configuration, packing, transport, LoRA, precision, memory, and
performance gates pass.
