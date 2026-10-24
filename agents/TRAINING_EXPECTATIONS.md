# Training requirements

Read this file before each trainer qualification or production run. These are
acceptance requirements. Do not change them without an owner decision.

## Readiness rules

- Use the native FP8 trainer path and measure it against H200 FP8 inference.
  Use trainer-only fake data to debug trainer execution and deterministic
  Hugging Face output to check log probabilities. Start rollout production
  only after these direct checks pass.
- A small test does not prove production speed or stability. Compare token
  count, sequence length, packed microbatch count, and training settings before
  comparing times. State exactly what passed: a diagnostic, one real step, or
  a full run.
- A 524,288-token synthetic check took about 178 seconds. An 11,960,320-token
  production-shaped step took 2,877 seconds (47m57s). The small result did not
  predict the large result and did not pass production readiness.
- Starting containers is not proof that training works. Do not call a run
  ready while required checks are unfinished.
- Measure a newly trained adapter. An unchanged cached adapter does not prove
  that a new adapter can be saved, transferred, loaded, or compared for
  parity.

## GLM-5.2 settings and limits

- Use batch 1,024, group 16, context 32,768, maximum output 16,384, learning
  rate `8e-6`, maximum gradient norm 1.0, and weight decay 0.
- Combined forward and backward elapsed time, from forward start to backward
  end, must stay within 600 seconds. If it exceeds 600 seconds, stop the test,
  preserve logs, and trace the cause before retrying. GPU utilization does not
  prove progress.
- Active rollout generation should take about 720 seconds per batch. Validate
  the path to batch 2,048 before using that batch in production.

## Measurement rules

- Report trainer compute, rollout collection, waiting, judging, adapter
  preparation, weight gathering, file writing, compression, network transfer,
  adapter loading, and full-weight transfer as separate measurements. Do not
  add overlapping times. A full-model copy is not adapter or rollout time.
- A network test that times out failed. Report received bytes, elapsed time,
  and errors. Do not present a short traffic sample as a completed sync.
- Watch live output and state whether monitoring is active. Do not use buffered
  output that hides failures. Keep estimates separate from measurements.
- If the reward command says `inside the noise`, the measured change is too
  small to show improvement or decline.

## Final acceptance

- Run the full topology for at least 50 completed trainer steps. Keep
  `training_ops.watch()` in the foreground with an interval of 600 seconds or
  less.
- An isolated unscored rollout after three attempts is replaced with its
  complete GRPO group. It is not alone a run failure while both judges are
  healthy and batch progress continues. Stop on a judge outage, eight
  consecutive errored groups, or stopped batch progress.
- Stop acceptance for an OOM, nonfinite value, failed adapter load, transport
  failure, an environment failure beyond an isolated replaced rollout, or
  measured reward regression.
- A short run, step 1, cached adapter, or reward change inside the noise does
  not pass. Report unfinished adapter, parity, rollout, judging, save, or
  transfer checks as unfinished.
