# GLM-5.2 port for Prime-RL v0.8.0

This is a source port. It is not approved for deployment.

## Base

Apply the series to commit:

```text
60bc29547a8824ad1de7b9af8d265e2b27b2a72d
```

The current deployment uses the older commit named in `docs/PINS.md`. Do not
mix the two source trees.

Prime-RL v0.8.0 changes packing, transport, LoRA, environment, configuration,
and vLLM interfaces. A clean patch apply does not prove runtime safety.

## Contents

1. Configuration, context-parallel helpers, and monitoring.
2. PP8, CP8, node-local EP8, and pipeline execution.
3. Native block-FP8 loading, Ada backward, LoRA, and expert execution.
4. Sparse MLA forward and backward agreement, including fully masked rows.
5. Trainer safety, policy continuity, microbatch data, and filesystem adapter
   broadcast.
6. Durable rollout transport, complete groups, off-policy rejection, and LoRA
   administration recovery.
7. Transport and topology tests.
8. FP8, LoRA, and deterministic comparison tests.

The port keeps INT4 expert-activation transport. It does not use INT4 model
weights or QLoRA.

It does not include temporary profiling controls, debug prints, or experiments
that have no current production use.

## Apply

Use a disposable clone:

```bash
git checkout 60bc29547a8824ad1de7b9af8d265e2b27b2a72d
git am /path/to/v0.8.0-glm52/000*.patch
git submodule update --init --recursive
```

Dependency revisions:

```text
renderers  d4707862ac83aa3773c21f4096aec72bd17b91e4
verifiers  d30a3f48e5f14b06b3081b2102ec32cc3149b849
```

Apply these files inside the matching dependency roots:

```text
patches/renderers/v0.8.0-disconnected-and-tokenizer.patch
patches/verifiers/v0.8.0-disconnected-provider-retry.patch
```

## Completed source checks

- All eight files apply with `git am` to the exact base.
- Every cumulative revision passes `compileall` and `git diff --check`.
- The focused Prime-RL suite passed 43 tests.
- The renderers suite passed 2,589 tests, with 153 skips and one expected
  failure.
- The selected verifiers relay suite passed 25 tests.
- Both dependency patches passed `git apply --check` at their named revisions.
- Controlled one-turn and two-turn Decensor v1 evaluations completed against
  the external judge.

These results do not approve a production run.

## Work required before deployment

1. Migrate the v0.8 configuration schema.
2. Measure PP8 and CP8 packing and token ownership.
3. Verify filesystem batch identity and restart position.
4. Verify LoRA restart, export, and inference visibility.
5. Run the production-size environment server and transport checks.
6. Build a new image with the v0.8 lock and vLLM 0.26.
7. Complete the direct checks in `TRAINING_EXPECTATIONS.md`.
8. Complete a 50-step stable run.

Do not deploy this series to an active run.
