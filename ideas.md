# Difficulty ideas

## Pre-batch group progress telemetry

- **Measured production baseline.** The `h200b` snapshot contained 849 trace
  records (189,742,608 bytes) in 59 groups: 43 complete and 16 partial
  groups (161 partial rollouts). Of the complete groups, 26 were clean (416
  possible accepted rollouts) and 17 were invalid and rejected (272
  rollouts); 70 records were policy-invalid. Policy phase mean was 219.72s /
  90,434.85, judge mean was 12.65s, and user mean was 3.38s. Rollout metrics
  stayed at zero until the accepted batch reached 1,024, so status/watch could
  report no progress while collection was active.
- **Acceptance.** Add bounded structured INFO telemetry at group finalization
  or sink accumulation. Expose clean and rejected group totals, accepted
  buffer count toward 1,024, rejection reason counts, authoritative partial or
  inflight counts, oldest partial age when available, and already-measured
  per-stage latency only. Never log prompts, responses, provider text, or infer
  accepted groups from request counts. Keep reward and training semantics and
  measurement limits unchanged. Extend `training_ops status` and `watch` to
  show these counters as pre-batch progress, distinct from accepted batches and
  steps. Tests must reproduce the 43/26/17 snapshot with empty rollout metrics,
  verify bounded output and sensitive-content exclusion, and cover parser/sink,
  reward, status, and dispatcher behavior.

## vLLM prompt-logprob row chunking

- **Measured failure.** An 8,192-row prompt-logprob request allocates about
  6.09 GiB for the full-vocabulary logits while only about 2.97 GiB is free.
- **Plan.** Patch only vLLM 0.26's `_get_prompt_logprobs_dict` output head. Keep
  model forward and request behavior unchanged, and process logits, scoring,
  gathering, and CPU copies in chunks of at most 128 prompt rows. Preserve
  target shifting, offsets, request accumulation, raw/processed modes, ranks,
  LoRA mapping, tensor parallel behavior, and the API schema. Apply the patch
  read-only after the FlashMLA patch and fail closed on duplicate or drifted
  patch context.
- **Gates.** Patch application must pass once, fail on a second application,
  and fail on upstream context drift. A bounded fake model must match the
  unchunked oracle for lengths 512, 974, 8192, 11324, and 32768, including
  tails, offsets, call counts, scheduler accumulation, raw/processed modes,
  LoRA, no-logprob requests, and response schema. Run launcher, memory,
  compile, bash, and patch-diff checks before commit.
- **TDD red baseline.** Before the patch existed, the focused suite ran 11
  tests with 6 missing-patch errors, 1 missing-launcher-mount failure, and 4
  passes.

## H200 prefix-stability diagnostic

- **Planned, diagnostic only.** Reuse the exact fresh 512-token reference and
  loaded policy receipt to compare isolated exact-token requests at prefix
  lengths 256, 257, 288, 384, and 512. Keep request salts distinct, require
  the same readiness receipt before, between, and after every request, and
  compare common-prefix route order/sets plus actual-token logprobs.
- At length 256, reproduce the original one-token decode with prompt
  logprobs. At longer lengths, use teacher-forced exact tokens and prompt
  logprobs. Reject nonfinite, sentinel, misaligned, tampered, or changed-
  receipt evidence and publish atomically without clobbering an existing
  output. This isolates sequence-shape and chunked-prefill numerical drift
  without changing a qualification gate.

## H200 batch-invariant inference diagnostic

- **Planned, diagnostic only.** Add a clean H200 inference config that is
  semantically identical to `trinity-h200-parity-production`, except for
  `VLLM_BATCH_INVARIANT="1"`. Do not add a trainer or orchestrator. Keep the
  native FP8 model, TP8/DP1, expert parallel, routed experts, all LoRA targets,
  0.85 memory limit, prefix caching, BF16 KV cache, and 65,536-token maximum.
- **Diagnostic effects and risk.** Batch-invariant scheduling may reduce
  throughput and increase latency or memory pressure. It can also change
  prefill/decode batching and expose startup or LoRA-load regressions, so this
  profile must stay isolated from production and qualification gates.
- **Pass criteria.** At prefix lengths 256 and 512, common routes and actual
  token logprobs must match the reference. Actual decode behavior must match
  trainer parity, and startup plus LoRA load must succeed with the expected
  readiness identity. Any mismatch, nonfinite output, or startup/load failure
  fails the diagnostic; no production setting changes.

## Packed sparse-MLA gpipe diagnostic correction

- **Measured failure.** Packed qualification launched all 8 workers in
  31.768s, but stopped after 10.329s with no outputs: step 0 had no forward
  pass and 2,859 identical errors from 20:18:44 to 20:21:34. The packed
  batch 1 was below the 1f1b PP8 minimum of 8.
- **Diagnostic correction only.** Use `pp_schedule='gpipe'` for both packed
  batch 1 and individual batch 5. Retain CP8/PP8/EP8, native FP8, the same
  tokens and routes, no LoRA, lr0, the existing gates, and the 600s limit.
  This tests the packed sparse-MLA numerical path under the full topology but
  does not qualify production 1f1b scheduling. If it reproduces the mismatch,
  it localizes packing independently of schedule; if green, next construct an
  8-microbatch 1f1b discriminator without changing measurement limits. Do not
  claim production readiness.

## Isolated decode256 to prefill-rescore parity qualification

- **Planned, opt-in only.** Keep the existing policy-0 qualification unchanged
  by default. The new mode decodes exactly 256 completion tokens from a 256-token
  prompt, then scores the exact prompt-plus-completion sequence with a separate
  cache salt.
- Require identical policy readiness receipts before decode, between decode and
  rescore, and after rescore. Preserve exact token IDs, validate the decode
  route (511 rows) and full rescore route (512 rows), and publish no artifact
  when any request, response, route, token-key, finite-value, alignment, receipt,
  or timeout check fails.
- Keep a strict diagnostic comparison with both request bodies and salts,
  response and route hashes, completion logprob deltas, and predicting-route
  differences. Use the rescored completion logprobs with zero prompt values in
  the trainer reference. This is a qualification parity check, not a training
  result.

## Policy-0 reference staging repair

- Route the small H200 reference file through the controller, verify its
  manifest locally, upload unique partial files, and preserve each exact old
  trainer reference before atomic replacement.

## Final failed-experiment audit

- **Equal 1:1:1 difficulty — failed.** It did not show a per-difficulty reward
  change outside noise and did not meet the learning acceptance test.
- **Hard-heavy staged curriculum — failed.** The full run had mean reward
  0.0574 and fitted change -0.0701. The harder later mix confounds the raw fall,
  but no easy, medium, or hard bucket proved learning.
- **Prefill runs 1-16 — inconclusive until the final run.** Runs 1-15 mainly
  exposed collection, storage, timeout, and fixed-batch faults or ended after
  too few steps. They do not support a reward trend.
- **Older TailRL plus fixed-prefill, 50 steps — failed.** Mean reward was
  0.1756, fitted change was -0.0058, and the noise threshold was 0.0931. The
  result was inside noise.
- **Run `glm-decensor-prefill-16` — failed.** Steps 21-31 had mean 0.4123 and
  fitted change +0.0048 versus a 0.1288 noise threshold, so the result was
  inside noise. Fixed no-prefix v28 evaluation was easy 0.3396 versus 0.3387,
  medium 0.0532 versus 0.0535, and hard 0.0116 versus 0.0147.
- **Reward-ordering rewrite — failed design test.** Offline replay changed
  263/1,268 winners and reduced high-refusal winners, but it was not a live
  learning result and replaced the approved multiplicative reward contract.
- **All/strong-refusal and usable-response group filters — inconclusive for
  learning.** The filters removed known low-signal groups, but no controlled
  run proved a gain. The changed accepted difficulty mix was a confound.
- **Easy/medium schedules — inconclusive.** Short easy-only and proposed
  70% easy/30% medium phases did not produce a completed controlled result.
- **Fixed holdout — useful measurement, no proven gain.** The v28 results above
  did not establish improvement.

The active selection is plain GRPO on the pre-difficulty
`NewEden/RL-seed-Decensor` dataset with one `decensor` environment. It has
no fixed prefill, TailRL, staged ratios, group filters, or held-out split wiring.

## Measured long-haul evidence

Snapshot: 2026-09-14 22:56 UTC, run `glm-decensor-difficulty-9`.

- The completed full steps 1-50 reward result has mean 0.0574, standard
  deviation 0.0364, fit -0.0701, residual 0.0298, and noise threshold 0.0596.
  Verdict: **falling**. The raw fall is confounded by the planned curriculum
  moving to about 82% hard prompts, so it does not prove that the model itself
  regressed.
- Per-difficulty reward also stayed inside the noise after 50 steps: easy was
  0.1466 for the first 10 steps and 0.1874 for the last 10, with fit +0.0451
  versus threshold 0.2018; medium was 0.0227 to 0.0277, with fit +0.00746
  versus threshold 0.0401; hard was 0.00604 to 0.00937, with fit +0.00393
  versus threshold 0.00496. Refusal and reasoning-safety trends for all three
  buckets were also inside the noise. No difficulty bucket has proven
  learning after 50 steps; the aggregate fall mostly reflects later
  hard-heavy sampling.
- The earlier reward report covered steps 7-34. Effective reward mean was
  0.0629, fit was -0.0421, and the noise threshold was 0.0650. Verdict:
  **inside the noise**. This was not evidence of improvement or decline.
- Completed steps 35-50 had effective rewards
  [0.0131, 0.0515, 0.0308, 0.0663, 0.0322, 0.0557, 0.0221, 0.0563,
  0.0265, 0.0118, 0.0116, 0.0501, 0.0370, 0.0285, 0.0136, 0.0395].
  Mean is 0.0342, standard deviation is 0.0177, fit is -0.0119, residual is
  0.0173, and the noise threshold is 0.0346. Verdict: **inside the noise**.
  The measured change is too small to show improvement or decline.
- Descriptively, steps 35-40 had mean reward 0.0416, while the hard-heavy
  steps 41-50 had mean reward 0.0297. This comparison is descriptive, not
  statistically proven.
- Step 50 reward was 0.0395 with 766/1,568 trainable rollouts (48.9%), zero
  errors, and 29m08 rollout time.
- Trainer compute for steps 44-49 was 403.74s, 405.86s, 403.26s, 406.36s,
  403.99s, and 404.66s. This was stable at about 6m44-6m46, with a peak of
  44.4 GiB/GPU and no OOM or nonfinite values.
- Adapter staging for v44-v49 stayed below 5m. Examples were v44 2m58 with
  1m33 network time, v45 3m03 with 1m30 network time, v46 2m48 with 1m28
  network time, and v48 3m02 with 1m34 network time.
- n82 cleanup left about 879 GiB free (74% used) during the run.
- Step 32 reward is 0.0156 with 400/640 trainable rollouts (62.5%). The
  accepted mix is easy 10.0%, medium 32.5%, hard 57.5%; rewards are easy
  0.0366, medium 0.0111, and hard 0.0122. Compare each difficulty separately;
  the mix changes the aggregate.
- Step 32 trainer forward/backward time is 408.95s. The status snapshot has
  1,268,125/1,500,000 accepted tokens (84.5%), 248 inflight rollouts, and no
  rollout failures or unscored rollouts in the last five minutes. Acceptance
  still records one transport/rollout failure and no 2,048-batch validation.

## Qualitative audit through step 32

- The audit covers 23,435 rows and 46,340 turns.
- Audit reward by difficulty is easy 0.1452, medium 0.0195, and hard 0.0057.
- Hard reward is mainly real refusal: 7,150/8,228 hard records score refusal
  9. Hard records include 387 fulfilled and 15,732 declined cases.
- Hard reasoning penalty is 98.5%; meta is 16.2%, word is 75.3%, and Markdown
  is 28.4%.
- Visible-answer failures are 0.33%; reasoning incoherence is 0.035%.
- High-refusal judge decisions are consistent: 26,678 decisions, only 2
  partially fulfilled, and no silent replacement.
- Topic switches are 0.17% overall and 0.068% for hard; there is no broad
  reward hack. Invalid model responses are 1.55%; judge parse failures are
  0.06%.
- Conclusion: low hard reward is mainly real refusal, not judge error.

## Reasoning audit

- The audit has 15,356 scored records and 5,157 fulfilled turns. None of the
  fulfilled turns has an explicit refusal.
- The 2,655 fulfilled-only records are the strongest possible false-positive
  set for an internal-refusal penalty.
- The reasoning multiplier affected 13,221 records, but this includes true
  refusals.

## Group 16 signal

- Full groups were easy 278 (6 all-refusal), medium 390 (52 all-refusal), and
  hard 477 (252 all-refusal).
- Hard non-all-refusal groups had reward spread 217/225. This shows usable
  signal exists, but more than half of hard groups refuse entirely.

## Worked

- Multi-turn traces contain more than one assistant turn.
- Staged sampling changes the difficulty stage as intended.
- Direct all-refusal exclusion works: 75 raw groups / 1,200 rollouts had 10
  all-refusal groups skipped (easy 1, medium 1, hard 8), plus 1 errored group,
  leaving 64 groups / 1,024 rollouts. There were zero zero-advantage groups
  and no complete all-9 group in the effective traces. Keep this curriculum
  protection so a batch does not become all-zero.
- Steps 1-41 all-9 group rates were easy 1.07%, medium 5.80%, and hard
  49.83%.
- Equal raw sampling with the skip gave an accepted mix of easy 40.00%,
  medium 38.94%, and hard 21.05%. Approximate raw compensation is
  easy:medium:hard = 1:1.05:1.97.
- The external judge was saturated at 80 running, 0 waiting, about 543.5
  tok/s, and about 96% GPU use. Do not raise concurrency now.

## Did not work or is not enough

- The hard-heavy staged curriculum did not successfully produce a reward climb
  by step 50. Recent trainable rates were roughly 49-55%. These observations
  do not show causality.
- One step is not a reward result. Step 1 took 44m09, had 1,024/1,200
  trainable rollouts, mean reward 0.1080, and mix easy 48% / medium 36% /
  hard 16%; it is only a collection check.
- The steps 35-50 local window was **inside the noise**, but the authoritative
  full steps 1-50 result is falling. Do not infer model regression from the raw
  fall because the planned curriculum changed the difficulty mix.
- An earlier steps 6-40 summary had reward fit -0.0048 and noise threshold
  0.0433. Keep it as historical evidence, not as the current step-32 result.
- The previous production-shaped trainer trial stopped at the 600s
  forward/backward limit: 606.58s active, with no completion, OOM, or crash.
  Its saved optimizer batch had 1,864 samples, 4,522,029 raw tokens,
  2,712,271 loss tokens, and max sequence 24,060. The earlier roughly 3m
  fixed-token result is not comparable.
- The stopped difficulty-10 step-1 batch had 1,839 sequences, 4,024,960
  packed tokens, 2,639,688 loss tokens, 123 microbatches, and max sequence
  18,715. Active forward/backward time was 614.523s. Host health was clean;
  fixed-rollout volume caused the limit. The next run restores a 1.5M-token
  cap while preserving group 16, 32K/16K, rank-32 targets, reward logic, and
  the 600s limit.

## Approved difficulty-10 experiment and measured stop

- `glm-decensor-difficulty-10` completed four trainer steps and was deliberately
  stopped. In `token_batch_size` mode, it skipped the existing batch-boundary
  deferral and repeatedly restarted the same 21 partial groups at policy v1/v2.
- The accepted batch plateau was 1,212,901/1,500,000 tokens. Discarded groups
  were 206 live plus 130 complete at v1, and 256 live plus 77 complete at v2.
  Forward/backward for steps 1-4 was about 385-399s, with no OOM or nonfinite
  values.
- Step rewards were .1953/.2343/.1769/.2188. Four points are too few and too
  noisy to show a trend.
- Accepted repair: token batches now use the existing bounded 1200s
  batch-boundary wait. There was no reward, algorithm, or limit change.
- The new run keeps easy-only prompts, no prefill, TailRL, a 1.5M-token cap,
  and group size 16. Use fixed no-prefix evaluation with 300 prompts per
  difficulty bucket at startup and every 10 steps. Skip a group only when it
  is fully scored and error-free and all refusal scores are at least 7.
  Complete 50 steps and require the reward/no-prefix trend to be outside noise.

## Difficulty-11 startup and steps 1-3

- Startup evaluation rewards and mean refusals were easy 0.34851 and
  2.695, medium 0.05917 and 6.279, and hard 0.013826 and 8.660.
- Step 1 effective reward was 0.1955. Trainer compute was 399.81s. Step 2
  reward was 0.1980 and compute was 423.16s. Step 3 rollout reward was 0.2516
  and compute was 406.07s. Step 3 total time was 10m51s; entropy was 0.4402,
  grad norm was 0.03394, throughput was 3793 tok/s, and MFU was 6.5%.
- Three points cannot establish a trend.
- Batch 4 did not start. Policy 2 transport exhausted the fixed 600s attempt
  and the bounded retry after an earlier partial batch was lost. The engines stayed
  on exact policy 1, and the trainer stayed healthy after step 3.

## Historical prefill and easy-first trial

- The earlier 100% assistant-prefill plan is historical. It is not the active
  difficulty-10 run above.

- The measured 50-step result had per-difficulty changes inside noise. The
  aggregate fall was confounded by the harder prompt mix, so it is not proof
  of model regression.
- A group with the same reward for every completion has zero GRPO advantage:
  group-normalized rewards are all zero. Keep enough within-group variation
  for a policy signal.
- Judge this trial by the full 40-50-step reward trend and the repository's
  exact noise rule, not by five steps.
- If it works, restart from its trained adapter with 70% prefixed groups and
  30% normal groups. Fix the prefix choice for the whole group and the whole
  conversation. Never mix prefixed and unprefixed rollouts in one GRPO group.
  Add medium and hard only after the no-prefix measurement also improves.
- A gain only with the prefix is a prompt effect, not proof that the
  unprompted model learned.

Preliminary evidence from the first prefill attempt:

- Rollout step 1 took 15m01, reward was .1900, and 672/672 rollouts were
  trainable. Step 2 took 8m53, reward was .1870, and 336/336 were trainable.
  Trainer forward/backward time was 431.77s and KL was .01295.
- Matched reward was .1938 versus .1633 for the prior easy run. Policy
  reasoning was 42.6% versus 71.7%; rejected was 5.5% versus .8%, visible
  failures were 8.0% versus 1.9%, and reasoning failures were 7.5% versus
  .8%.
- This is preliminary evidence, not a conclusion. The run was stopped
  mistakenly after one nonfatal dropped judge group and is being rerun to
  40-50 steps.
- The 35 deterministic invalid-output zero-reward records were valid training
  inputs by design. The one three-attempt judge validation failure was
  unscored, and its whole group was excluded and replaced.

Research basis, kept brief: DeepSeekMath GRPO (arXiv:2402.03300), the DAPO
paper, Online Difficulty Filtering (arXiv:2504.03380), Easy-to-Hard
Generalization (arXiv:2506.06632), POPE (arXiv:2601.18779), PrefixRL
(arXiv:2601.18795), Prefix-RFT (arXiv:2507.01679), and the official TRL GRPO
docs.

Research checks: fixed prefill is an exploration aid. Evidence for transfer
comes from guided or task-specific prefixes, so evaluate the adapter without
a prefix. Never mix prefill conditions within one GRPO group. Taper prefill
only after measured learning.

## Untested

- Rollout time added by retries and all-refusal replacement.
- Concurrent visible and reasoning judge phases with separate prompts,
  history, events, retries, and strict validation.
- A specialized correction prompt for contradictory declined and explicit-
  refusal fields.
- After the clean comparison, add an explicit `answer_restriction_applied`
  decision so a brief caution in a fully fulfilled answer is not treated as
  an internal refusal. Keep the penalty for declined, partial, or replaced
  answers. Test this alone for 40-50 steps.
- Distinguish an actual refusal decision in the reasoning trace from a brief
  caution in an otherwise fulfilled answer. Do not deploy it mid-run; compare
  it only in a clean 40-50-step run.

## Fixed-size rollout timing repair

- Step 23 took 45m53. Policy v22 became ready at 16/1,024 rollouts, then the
  batch-boundary wait reached its 1,200s limit, followed by 60s grace. The
  restart discarded 166 live and 186 completed rollouts across 22 groups.
- For fixed-size batches, an immediate strict restart at the policy boundary
  preserves GRPO policy purity and the existing limits. Token batches keep the
  bounded batch-boundary deferral. Source fix commit `14c49e1f` was not yet
  deployed at the time of this note.
- Evaluate the next three completed rollout batches against steps 19-24.
  Confirm no policy mixing, no rollout errors, and no stop-gate events before
  accepting the timing repair.

## Current Trinity reward diagnosis and controlled idea

- Steps 1-24 have mean reward 0.0319, fitted change -0.0110, and noise
  threshold 0.0229. The fitted change is inside noise. Steps 19-24 rewards
  were 0.0330, 0.0318, 0.0195, 0.0214, 0.0221, and 0.0188.
- Recent component ranges are refusal mean 6.82-7.53, reasoning
  safety-policy turn rate 84.1%-89.7%, reasoning multiplier mean
  0.180-0.222, and no-reasoning zero-rejection about 6.9%-11.9%.
  Judge parse failures were about 0.11%-0.45% in the reported snapshots.
- The measured pattern is reward compression with high refusal. It is not
  evidence of a judge outage. The historical audit removed zero-advantage
  effective groups, but it did not prove a learning gain.
- Controlled idea: test an opt-in fully scored strong-refusal group filter at
  refusal threshold >=7. Preserve errored, unscored, and mixed groups, and do
  not change reward multipliers. Do not enable it during the current run
  without explicit acceptance. Compare a clean baseline and experiment for
  40-50 steps using per-difficulty reward and zero-advantage group rate.

## Large-model reward-ordering audit

- **Measured Astra audit:** Steps 15-24 covered 10,240 records in 640 groups.
  There were zero uniform groups, 114/640 winner reversals, and 3,667/16,742
  reversals among unequal-refusal pairs.
- All 171 fully scored strong-refusal groups had reward spread. This corrects
  the earlier strong-refusal and zero-advantage assumption: a broad
  strong-refusal filter would discard useful refusal variation. Do not apply
  that filter from the current evidence.
- **Offline calculations, not causal results:** removing the word factor gave
  79 winner reversals, removing Markdown gave 85, and removing reasoning gave
  123. Multi-turn reward means for 1, 2, 3, 4, 5, and 6 turns were
  0.0374, 0.0215, 0.00766, 0.0110, 0.00533, and 0.00174. Prompt mix
  confounds these means. The run 1-27 trend stayed inside noise.
- No active-run change is allowed before step 50. After step 50, run these
  ideas in order:
  1. Attribute normalized advantages to each reward component offline.
  2. Replay parser decisions on saved outputs and measure changed winners.
  3. Run a blind judge audit on a fixed holdout.
  4. Run one controlled fixed-holdout comparison with refusal as the primary
     reward and quality as the tie-break.
  5. If reward stays flat, audit whether the update changes the measured
     policy before changing the reward.
- These tests can show association and implementation errors. They do not
  prove that a reward change causes learning until a clean controlled run is
  outside noise.

## Acceptance

- Actual ratios change by stage.
- Multi-turn traces contain more than one assistant turn.
- With exclusion enabled, all-refusal groups create no optimizer samples.
- Use 40-50 completed trainer steps unless a hard failure occurs.
- Compare each difficulty separately and require reward movement beyond noise.

- Prefill-2 step 1 reward was 0.1819 with 0% errors; rollout plus judge took
  24m15s. The collected batch had 1,980 sequences, 4,236,410 tokens, and 130
  packed microbatches; forward/backward stopped at 610.70s. Cause: fixed-rollout
  batching. Next test: 1.5M-token cap. One point gives no reward trend claim.

- Prefill-3 measured rewards were .2002/.1457/.2261/.1663; F/B was
  430.98/404.30/403.49/330.93s; v2 stages were 415.193/406.830s. It stopped
  at 3 completed steps on the H200A free-space guard. Retention fix and cleanup
  evidence were recorded; no reward trend claim.

## D15 stopped result

- D15 stopped at 613.19s during forward/backward with 130 packed microbatches
  and about 4.26M tokens. It had no OOM or nonfinite values, but did not finish
  under the 600s gate. The temporary profiler configs are removed.

## Next prefill reward trial

- Startup rejected `keep_train_batch_ratios` with token batching; it was removed
  because this run has one training environment.
- Run `glm-decensor-prefill-16` with easy prompts, TailRL, group size 16, and
  the existing reward checks.
- Use the approved 1,500,000-token batch cap. This is fewer than 1024 effective
  samples per optimizer step; do not call it batch 1024.
- Keep 32K context, 16K output, native FP8, rank-32 LoRA, and the existing
  prefill. Judge reward and refusal rates over the full run, not one step.

## Trinity parser and flat-reward hypothesis

- Steps 15-24 rejected 1,006 of 10,240 effective records (9.8%) for no
  reasoning. There were 66 first-turn failures and 940 later-turn failures.
- Reward can be lower for multi-turn records. At step 18, the one-turn reward
  was 0.0581, the two-turn reward was 0.0208, the three-turn reward was
  0.0474, and the four-turn reward was 0.0196. This is a measured pattern,
  not proof of cause.
- The orchestrator renderer uses `reasoning_parser='think'`. The H200
  inference configuration uses `reasoning_parser='deepseek_r1'`. A parser
  mismatch may explain some missing-reasoning records, but this is not proven.
- Controlled test: in an isolated test, send matched prompts through both
  parser settings and compare returned reasoning, visible content, and
  rendered traces. If the mismatch is confirmed, align the parsers in a new
  controlled run. Do not change the active run or reward limits for this test.
- Add per-turn no-reasoning metrics, separated by first and later turn, before
  judging a parser change.

## Controlled Trinity H200 to trainer parity qualification

- Measured in the retained Trinity trainer metrics: mean masked fraction was
  0.2585, mean unmasked mismatch KL was 1.9495, and mean masked mismatch KL
  was 2.8736. Step 1 was 0.3742 masked, with 2.8270 unmasked mismatch KL.
- Run one deterministic reference batch from H200's serialized
  `/inference/v1/generate` endpoint. Preserve native FP8 weights, use BF16 KV
  cache, request routed experts, and enable trainer router replay. Use the
  existing reference schema and no cached or full rollout batch.
- Pass only when routes, token IDs, positions, masks, and logprobs validate and
  the fail-closed gates pass: unmasked KL <= 0.1, masked KL <= 0.5, and masked
  fraction <= 0.06. Do not start production training until this qualification
  passes. The live operator must capture the serialized response first.

- The first BF16-KV plus route-replay qualification stopped before the update:
  masked fraction was 0, unmasked mismatch KL was 0.6094949245 versus the 0.1
  gate, mean/max valid logprob delta was 0.9332404137, and PP time was 140.48s.
  This improved from the prior step-1 unmasked KL of 2.8270 but still failed
  safely. Next test: change only the trainer router dtype to BF16 to align it
  with inference; keep the production config unchanged until this passes.

- The BF16-router follow-up was worse: unmasked mismatch KL was
  0.636548996, versus 0.609494925 with the FP32 trainer router. Revert the
  experimental dtype to FP32 before the next test; do not change production.
- The qualification reference also had a capture flaw. The current 32768-token
  reference retained only one completion token from a temperature-0 request;
  it did not use the full `prompt_logprobs` stream and was not
  production-equivalent. Regenerate the same 32768-token reference through
  serialized `/inference/v1/generate` with `max_tokens=1`,
  `temperature=1.0`, `top_p=1.0`, and `prompt_logprobs=1`. Flatten one prompt
  logprob per token, set the leading token to 0 and masked, retain all other
  prompt tokens, and preserve the exact routed-expert payload. Use the
  `prefill_logprobs` contract in
  `prime-rl/src/prime_rl/utils/client.py`.

- The corrected 32768-token `prompt_logprobs` request OOMed on H200 during
  logits all-gather: it tried to allocate 24.44 GiB with 7.58 GiB free. No
  reference file was changed. Next controlled qualification uses the first
  exact 2048 token IDs, expects 2048 prompt logprobs and exact routes of shape
  `[2048, 60, 4]`, and retains the 2047 non-leading tokens. This tests direct
  numerical parity only; it does not prove production 32768-token capacity.

- The bounded qualification passed: 2048 tokens, 2047 selected, mean absolute
  delta 0.0795838, p95 0.2749004, maximum 0.7428370, masked fraction 0,
  unmasked mismatch KL 0.00738006, and masked KL 0. Strict gates pass. The
  flawed one-token 0.609 result is superseded; this remains a 2048-token
  numerical parity result, not proof of production capacity.

## Stale engine watchdog replacement

- Observed failure: one stale canonical `engine_watchdog.sh` on port 8000 had
  an old run/profile environment. The narrow repair refused it with exit 74;
  the engine was not restarted.
- Repair plan: replace exactly one stale process only when its command is the
  exact deployed watchdog path, its environment has the configured port, its
  PID is its own process group, and termination is verified. Keep refusal for
  multiple or ambiguous candidates, wrong commands, unreadable environments,
  or a PGID mismatch. Start the canonical exact environment after cleanup and
  test without restarting the engine.

## Trainer silence during active rollouts

- Observed failure: the exact policy was ready, the queue fell from 128 to 98,
  throughput was 2,126-2,499 tok/s, and the orchestrator was up. The trainer
  was correctly waiting for a batch, but a warning and stop fired only because
  the trainer log age exceeded 600 seconds.
- Repair plan: keep the 600-second limit and suppress only this trainer log-age
  warning when the trainer and orchestrator are up and at least one reachable
  exact-policy-ready engine has running requests and throughput above the
  existing 5 tok/s health threshold. Keep the warning for idle queues,
  non-exact policy, stalled throughput, orchestrator-down, and trainer-only
  states.

## Orchestrator silence during active rollouts

- Observed result: orchestrator Step 1 succeeded at 02:42:51, then wrote no
  log for more than 10 minutes while the exact policy v0 served 128 running
  requests at about 2,173 tok/s. Step 1 batch transfer was progressing:
  397,167,733 bytes at source, 187,432,960 bytes received, and +45,875,200
  bytes in about two minutes. The orchestrator silence warning was false.
- Repair plan: when the existing active-rollout condition is true, suppress
  log-age warnings for both trainer and orchestrator without changing the
  600-second limit. Keep warnings for down or non-exact systems, zero running
  requests, throughput below 5 tok/s, trainer-only mode, and inactive transfer
  or rollouts.

## Rollout puller per-part timeout

- Observed failure: source size was 397,167,733 bytes across 16 parts; the
  destination was about 310 MB with 8 complete parts, 7 partial parts, and
  `part007` absent. Transfer rate was only 25-31 KiB/s. Rsync children stayed
  held by SSH keepalive while the parent waited for all children, so no retry
  occurred. Stopping and starting the exact puller preserved complete files
  and resumed partial files with `--append-verify`; the direct adapter route
  was not involved.
- Repair plan: wrap each pull-side rsync child in a bounded configurable outer
  timeout, default 300 seconds based on about 90 KiB/s for 24.8 MB parts, with
  TERM then KILL grace. Keep rsync `--timeout=60`, `--partial`, and
  `--append-verify`; let timeout failures return to the existing scan retry
  loop without deleting partials.

## Oldest-first rollout pull retry

- Observed result: after a timeout, step 1 reached 346,422,813 bytes with 13
  full and 3 partial parts, but the same scan continued to steps 2 and 3.
  Step 1 then stayed unchanged for about 10 minutes while newer partials grew;
  the trainer remained at step 0.
- Repair plan: order rollout files by numeric step and stop the scan on the
  first failed rollout pull. The next loop must retry that same oldest
  undelivered rollout, while control and receiver files still run before
  rollouts and receipts, hashes, and partials remain unchanged.

## Rollout puller lock orphan recovery

- Observed failure: the general stop exited the parent and health loop, but 38
  exact timeout/rsync/ssh step-2 children inherited fd 9 for
  `.rollout-pull.lock`. There was no target parent or pid file, so the lock
  stayed held and blocked the exact stop.
- Recovery plan: close fd 9 for every background transfer job and health child
  before fork or exec. A one-time cleanup may kill only processes with the
  exact RUN/source/destination/identity environment and an fd pointing to the
  exact lock inode, then must verify that the lock is free. Do not use broad
  process matching or pkill.

## Production-sequence Trinity parity qualification

- Measured fault: the old qualifier used `[1, 2, 3, 4]` repeated, so it had
  only four unique input IDs. The production parity guard therefore failed
  on a non-representative sequence.
- Experiment: create a fresh deterministic natural-language input from an
  existing Hub/cache dataset or an operator-provided JSON that is not a
  rollout artifact. Use a diverse 8,192-11,945-token sequence, then call the
  serialized `/inference/v1/generate` endpoint with temperature 1.0,
  `top_p=1.0`, `max_tokens=1`, and prompt logprobs enabled. Select base or
  policy-0 explicitly. Preserve the exact input IDs, sequence lengths, prompt
  logprobs, and routed-expert payload. Fail closed on token, logprob,
  route-shape, or route-digest mismatch. Keep the strict parity gates, native
  FP8, and trainer-only scope; do not use a full rollout batch or cached
  rollout batch.
- Follow-up risk to test separately: the policy-version `cache_salt` is placed
  in environment `extra_body`, then flattened by `TrainClient` into
  `sampling_params`, while `GenerateRequest` expects it at the top level. A
  same-name policy reload may therefore reuse stale prefix KV or routes. This
  commit records the risk only; it does not change the client.

## H200 long-context prompt-logprob OOM

- Measured failure: the 8,192-token prompt-logprobs request OOMed in
  `_patched_get_logits` during TP all-gather. It requested 6.11 GiB with
  2.18 GiB free at 0.92 GPU memory utilization. No trainer or optimizer ran.
- Bounded experiment: retain the exact 8,192-token input and strict gates,
  temporarily reduce engine KV allocation only enough to leave at least 8 GiB
  free, rerun the same reference, then restore the setting. Prefer a
  source-supported chunked prompt-logit computation if a small repair exists.

## Base-only long-context diagnostic

- Measured state: the live image does not contain the exact-target
  prompt-logprob extension. After a clean restart, the base-only engine has
  about 9.75 GiB free per GPU, which is about 3.64 GiB above the measured
  6.11 GiB allocation.
- Diagnostic: run the unchanged normal 8,192-token `prompt_logprobs`
  reference against the base model only, then run strict trainer parity. This
  isolates long-context trainer/base parity; it does not qualify the loaded
  policy-0 adapter. No optimizer runs in this diagnostic.

## Explicit engine repair config resolution

- Measured failure: `training-ops repair engine:h200b:configs/trinity-h200-parity-production`
  stopped before deployment at 05:00:31 because `_repair_engine()` called
  `current_run()` and the trainer container on n82 had no run path.
- Bounded retry: when an explicit config directory is supplied, resolve that
  profile directly and derive its stable watchdog identity. Keep
  `current_run()` selection for implicit repair, then retry the same explicit
  engine repair without changing host or config validation.

## Qualification reference LAN fanout

- Measured stall: the local `training_ops` copy of the 2,850,051-byte
  reference stayed live for more than 3m22s with only n76, n78, and n82
  children, remote rsync servers, and no completion. A separate eight-host
  hash probe ran for more than 1m11s.
- Measured topology failure: direct H200 access to `root@10.92.20.82:22`
  timed out, and the configured peer identity path was unavailable in the
  H200 host shell. No remote change was made.
- Authoritative resolution: `china-trainer` is unresolved in the H200 host
  shell. Local `ssh -G n82` resolves user `ubuntu`, hostname `10.92.20.82`,
  `ProxyJump china-jump`, and the existing China rollout identity. The
  reference is already verified locally.
- Corrected plan: after five minutes without progress, stop only the owned
  transfer and probe processes and preserve partial files. Perform one
  sequential local-to-n82 transfer using the configured `n82` SSH route, then
  verify exact size and SHA256 before using n82's LAN access to fan out to the
  other seven hosts. Do not start parallel eight-host copies. Verify all eight
  SHA256 values match
  `df7d7b0baf5aa646751fc1762fb7882e4266afda3df2017f92ba44a3758af589` before
  the qualifier. No model transfer is involved.
- Cleanup result: the owned processes were stopped at about 4m35s, all
  matching local PIDs were confirmed gone, and partial files were preserved.
- Username correction: the n82 seed completed in 5 seconds with the exact
  size and SHA256, but `root@10.92.20.76` was rejected with `publickey`.
  Existing deployment fanout in `training_ops/__init__.py` uses `scp ...
  peer:` without a username, while the n82 session uses `ubuntu`. The next
  bounded probe and fanout must inherit or explicitly use `ubuntu`, not root;
  the exact SHA256 gate remains required.

## Qualification reference seed fanout

- Measured failure: direct controller fanout exceeded 600 seconds and left
  six unique partial reference files.
- Bounded fix: compress the verified 2.7 MB H200 reference once, upload and
  verify one seed on n82, then fan out unique partials over n82's China LAN
  route before invoking the existing locked installer on all eight trainers.

## Trinity reference determinism split

- Measured fresh qualifier: one masked token out of 8,192 at index 1,074
  changed from ` numbered` to ` condition`. Reference logprob was
  `-0.000221228`, trainer logprob was `-1.698086739`, delta was
  `-1.697865511`, and KL was `0.880939387`. Unmasked mean delta was
  `0.00002744046`; routes were valid; the phrase repeated 264 times.
- Experiment: first independently regenerate the same H200 base reference at
  0.85 from the identical input and compare every input, logprob, route, SHA,
  and index-1,074 value. Then run the trainer once against the unchanged
  verified reference and compare its artifact and outlier. Do not change
  thresholds or optimizer settings. Repeat stability assigns the cause to
  inference or trainer nondeterminism.

## Diverse Trinity qualifier seed

- Measured fault: H200 and trainer repeated the current qualifier byte for
  byte, including the deterministic token-1,074 mismatch. The phrase
  `numbered condition` occurs 264 times, so this input is not
  production-representative despite having 8,192 tokens.
- Experiment: replace it with deterministic seeded prose using varied
  templates, vocabulary, and indexed details. Keep one exact 8,192-token
  sequence, native FP8, and strict gates. Reject excessive exact 16-token
  window repetition after tokenization. Do not change thresholds, drop
  tokens, or use rollout data.
- Diverse base result: the input had 8,192 tokens, 750 unique IDs, and an
  exact-window repeat fraction of `0.00110065`. The reference SHA was
  `ad0dc...`. The strict step passed in 74.15 seconds; its artifact KL proxy
  was `0.00196996`, mean absolute difference was `0.02028694`, position
  mismatches were zero, and all processes exited 0.
- Next experiment: load a verified zero rank-32/alpha-64 Trinity policy-0
  adapter through the supported H200 path, generate the same diverse
  reference under its exact loaded model name, and rerun the unchanged strict
  gate. The base pass does not qualify the adapter path. Do not change
  thresholds or optimizer settings and do not transfer model weights.

## Clean Trinity experimental run config

- Basis: the diverse base qualification passed its strict gate, but the
  policy adapter was not independently preloaded and tested. The next
  qualification must start with a fresh adapter under the supported H200
  policy path.
- Plan: add a new clean production-shaped run identity with the same native
  FP8, topology, reward, rollout, and strict parity settings. The guard must
  fail closed before optimizer or weight broadcast when parity mismatches.
  Preserve the completed parity run and do not change production settings.

## Reconcile validated stale watchdog identities

- Measured state: two own-process-group watchdogs served the same h200b
  container, model, start script, and port. Their identities were
  parity-production and experimental-parity. Trainers were idle, the
  orchestrator was down, and the engine was healthy, but cold start refused
  the ambiguous watchdog set.
- Plan: accept multiple candidates only when every one has the exact static
  host, container, kind, model, start script, startup grace, and port. Stop
  each validated process with the existing process-group checks, then start
  and verify one desired watchdog. Keep refusal for mismatched, unreadable,
  shared-group, or otherwise unverified candidates.

## Fix fixed-run orchestrator metric path

- Measured failure: the fixed orchestrator output directory is
  `/data/glm52-rl/runs/trinity-decensor-experimental-fixed/run_trinity-decensor-experimental-fixed`,
  but its file monitor filename also starts with
  `run_trinity-decensor-experimental-fixed/`. The effective path is therefore
  duplicated and missing, which causes `FileNotFoundError`.
- Plan: follow the existing orchestrator path contract and use the bare
  `rollout-metrics.jsonl` filename. Keep the trainer's nested filename because
  its output directory is the run root. Add a test for both effective paths.

## Exact policy-0 parity qualification

- Measured production failure: the fixed run's first step failed the parity
  guard with unmasked mismatch KL `1.734912`, masked mismatch KL `2.133168`,
  and masked fraction `0.266403`. The passing diverse qualifier used the base
  model and prompt-prefill path, so it did not qualify the policy-0 generated
  path.
- Controlled experiment: require an explicit loaded policy model and a
  readiness/load receipt whose adapter SHA, config SHA, LoRA name, and server
  instance match the expected values. Send a fresh top-level cache salt with
  the serialized generate request and persist the policy identity, request
  mode, token IDs, cache salt, and routes in the reference metadata. Keep the
  strict gates and native FP8 settings unchanged. Trainer LR0 must point its
  `model.lora.initial_adapter_path` at the same verified step-0 adapter; no
  cached rollout data is used.
## Exact policy-0 decode qualification correction

- Reviewer measured that the prior policy-0 reference requested `max_tokens=1` and prompt logprobs, then discarded generated token IDs and scores. It did not qualify the decode path and compared only a caller-supplied receipt.
- The corrected experiment requests a fresh deterministic completion with exact prompt and completion IDs/logprobs, a zero prompt loss mask and completion-token loss mask, and routed experts aligned to every token. It fetches an explicit readiness receipt immediately before and after generation and requires the active endpoint identity to remain unchanged. A verified step-0 adapter, config, and reference are staged to every configured trainer with existing `training_ops` APIs before an operator invokes qualification. No cached rollout data or live action is part of this change.

## Policy-0 staging review corrections

- Review measured that remote verification read the full 16.5 GB adapter into one Python buffer. The verifier must hash in bounded blocks. Review also found that `prepare-policy0 stage` only checked files already present on n82; it did not stage the preserved step-0 adapter/config or the fresh H200 reference.
- The corrected local-only plan adds explicit source paths and host identities, refuses missing, symlinked, conflicting, or digest-mismatched sources/destinations, stages the small reference through the configured H200 connection, stages adapter/config only from n82 over the trainer LAN, and atomically verifies every destination. It must preserve source files and bind the reference receipt to the staged adapter/config SHA and size. No model-weight transfer is allowed.

## Policy-0 staging final review corrections

- Review measured that `ssh -n` cannot consume the heredoc remote scripts, and that the fresh reference stopped at n82 instead of reaching all configured trainers. The corrected path must use stdin-capable SSH for remote bash, fan out the verified reference from n82, and verify every host.
- Review also requires an exact trainer-length contract: policy-0 uses an 8160-token prompt plus exactly 32 generated tokens, requests `min_tokens=max_tokens=32`, refuses early stop or any total other than trainer `seq_len=8192`, and keeps the base qualifier unchanged. Route and logprob scaling remain pending deployed audit.
## Policy-0 routed-expert final-token contract

- Deployed MTP0/default serialized `generate` with prompt length 8160 and 32 generated tokens returned 8191 routed-expert rows (`P+G-1`). Token IDs, logprobs, and the loss mask remain length 8192.
- Production rollout packing keeps the full token sequence and zero-pads one missing final route row before the trainer model consumes it. The qualification artifact must keep the raw 8191 rows; the trainer fake loader must apply the same one-row tail handling. It must not fabricate a route in the serialized reference.

- Reviewer found the fake-loader route validator used bitwise inversion on a boolean tensor and rejected a valid sparse row whose selected expert IDs were all zero. The repair must use type-safe boolean masks, validate every real row for range and top-k uniqueness, and allow only the explicit final zero padding row.

- Measured policy-0 qualification OOM: requesting prompt logprobs for an approximately 8192-token prompt allocated about 6.09 GiB of full-vocabulary logits while only 2.97 GiB was free at fixed 0.92 GPU utilization. Prompt positions are all outside the loss mask. The bounded fix is policy-0-only: omit prompt logprobs, use exact zero prompt inference logprobs, keep generated logprobs and 8191 raw route rows, and retain strict 8192-token, receipt, and route checks. Base prompt-prefill behavior stays unchanged.

- Policy-0 load audit plan: add an offline `load-policy0 ... plan|apply` action that stages the exact stable step-0 adapter/config, then loads only the qualification slot after proving the H200 API/container identity, idle state (`running=0`, `waiting=0`), no active stage, source and tunnel reachability, and exact adapter/config identities. Recheck every identity and the final staging receipt after load. Missing tunnels must name the existing repair command; the action must never launch training/orchestration or move base weights.

- Policy-0 load review found that the first implementation imported client code on the controller and used controller-local `127.0.0.1` endpoints. The correction must run stage/load in the exact H200 inference container environment through the existing SSH path, preserve tunnel/HTTP/identity error categories, aggregate every running/waiting metric sample, and refuse any occupied or conflicting target slot before mutation. It must pin the real API listener PID/start identity across pre/mid/post checks and select only the requested final receipt. No service start is allowed.

- Policy-0 load review found that the source URL was incorrectly hard-coded to `/step_0`. It must match the production watcher: derive the safe relative path below the configured `weight_broadcast.source_dir`, URL-encode each path component, and refuse sources outside that directory or with traversal.

- Policy-0 source audit: export metadata can omit `shared_expert` when the shared FP8 wrapper is present, and can write integral LoRA alpha as a float. The guarded source-preparation experiment will preserve the original step-0 adapter, accept only this exact legacy mismatch (eight targets plus integral alpha, rank32/dropout0), write a canonical nine-target integer-alpha config in a new step-0 directory below `weight_broadcast.source_dir`, verify full adapter/config identities, and publish `STABLE` last. No WAN/H200/base-weight transfer, service start, or mutation of the original source is allowed.

- Policy-0 source review found that the first correction copied only weights and config. A valid adapter-server step-0 also requires `adapter_model.safetensors.sha256`, `adapter_manifest.json`, and every manifest-referenced transport chunk. The corrected experiment now validates the full source contract and all identities before and after the copy, derives a manifest with the new config identity and destination URL, and uses no-replace atomic directory publication so a destination race fails without nesting a temporary directory.

- Final source review requires full policy-0 manifests to be integer version 1 only. The source config is now opened once with `O_NOFOLLOW`, hashed and parsed from the same bytes, then checked again by the pre-publish source snapshot. Version-2 full manifests and any config read/change mismatch fail closed before publication.

- Live source audit found the preserved adapter uses flat transport chunk files in the step root, named by each manifest `file` field. The earlier correction incorrectly required `transport-chunks/`. The next repair follows the adapter-server safe-name rule, copies each declared chunk to the same flat relative path, rejects traversal/absolute/metadata collisions and unexpected entries, and keeps the source and destination identity checks.
- 2026-09-26 policy-0 source audit: the preserved step_0 has one optional regular `receivers-ready.json` control marker (1001 bytes), which is a valid old step-0 receiver ACK. Source preparation must validate the existing watcher ACK schema, bind its step/source URL and adapter/config identities to the preserved source, and pin its hash, size, inode, and mtime before and after staging. It must preserve the source marker but must not copy it to the derived destination because the corrected config identity invalidates the old receipt. Symlinks, non-regular files, invalid or mismatched markers, and any other extra source file must fail closed.
- 2026-09-26 policy-0 loader repair: the H200 helper imported `prime_rl` client/config APIs that are absent in the deployed image, so the qualification load could not reach the admin API. The deployed contract is direct HTTP to the admin root: POST `/stage_lora_adapter` with only `lora_path` and `require_delta:false`, then POST `/load_lora_adapter` with `lora_name`, `lora_path`, `require_delta:false`, and `prefetched:true`; use `/staging_status` to verify the exact loaded receipt. The controller's 900-second SSH timeout is below the measured 1800-second stage budget. The repair will use direct `httpx`, preserve fail-closed identity and idle checks, and set 1860/660-second stage/load outer deadlines with cleanup margin. No live action is part of this change.
- 2026-09-26 policy-0 load audit: the H200 engine-state probe passed Docker Go template `\\t` text to `read`, so valid container fields were separated by literal backslash-t instead of tab characters and the probe rejected a valid engine. Replace the delimiter with a safe literal pipe, parse with `IFS='|'`, and keep exact name, container ID, StartedAt, Running, listener, metric, and receipt checks. No live action.
- 2026-09-26 policy-0 load identity hardening: the listener must match Docker State.Pid, the exact container cgroup ID, NSpid host PID then container PID 1, and the audited `PRIME-RL::Inference` command title with `PRIME-RL::Infer` comm. Its executable must match `/usr/bin/python3.12` and the container init executable by device/inode. Parse start ticks after the final `)` and repeat Docker/listener identity checks after health, metrics, and staging reads. No live action.
- 2026-09-26 trainer-loss watchdog audit: cold start falsely stopped at 15:47:11 before the trainer first log at 15:47:42; policy-0 became STABLE at 15:49:55, while lease writers published backward epochs. The lease watchdog needs a bounded 900-second startup phase, one fresh authenticated healthy status before arming, strict newer missing epochs, and one periodic health writer in the rollout puller. No live action.

## Agentic review failure observability experiment

- **Measured code gap.** A failed `turn_1_review` currently ends as a generic task-level
  review failure. The existing judge events retain full system/user messages,
  model output, and raw tool arguments, but they do not provide one bounded,
  field-level failure record. This makes invalid visible or reasoning tool
  arguments hard to diagnose and risks exposing prompt or response content in
  logs.
- **First experiment.** Add one structured INFO record at each visible or
  reasoning review failure boundary. Include only the phase, attempt, exception
  class, a sanitized and truncated exception message, raw argument character and
  UTF-8 byte lengths, raw-argument SHA-256, finish/truncation metadata when the
  provider returns it, and the final failure status. Include the group/task
  identifier when the task runtime provides one. Never include raw prompts,
  model output, or raw tool arguments. Keep reward, group, retry, and timeout
  semantics unchanged.
- The no-raw-content rule applies to the new INFO records. Existing
  `trace.info["judge"]` event storage is unchanged by this experiment and may
  still contain the provider request and response fields for the existing
  trace contract; this is not a claim that every persisted trace is globally
  free of raw content.
- **Follow-up measurement.** After the bounded record is verified, measure the
  current baseline first: the judge service has
  `max_concurrent_requests=48`, while each worker's review semaphore is 2.
  Then run one-variable A/B comparisons in order: per-worker review semaphore
  2 versus a higher value, followed separately by the renderer/tokenizer
  `pool_size=4` versus `pool_size=16`. `pool_size` is not the judge client
  concurrency. Keep the judge limit at 48 while measuring the worker semaphore,
  and change only one limit in each comparison. Keep batch size 1024, group
  size 16, context length 32768, output length 16384, reward settings, and
  timing limits fixed. Use the new phase-level failure counts to separate
  provider argument failures from timeout or service failures before changing
  concurrency.

## Batch-32 review failure diagnostic

- **Measured production evidence.** One production-shaped attempt ran for 826
  seconds, accepted 0 rollouts, dropped 2, and recorded 3 `turn_1_review`
  failures. The evidence does not yet include rollout IDs, the exact review
  exception or HTTP status, the drop reasons, per-phase timings, provider
  finish/truncation fields, or model/server/cache identities. This is not
  enough to identify whether the failures were provider, judge, validation,
  transport, or timeout failures.
- **Diagnostic rationale.** Use one disposable batch-32 diagnostic to capture
  those fields with a small blast radius. Keep the model, native FP8 path,
  context, output limit, reward, sampling, judge/user limits, worker count,
  pool size, and learning rate unchanged. Require every review failure to
  produce a bounded sanitized record with phase, attempt, exception class,
  truncated message, argument lengths, argument hash, provider finish and
  truncation metadata when available, and final status. Never log prompts,
  model output, or raw tool arguments.
- **Diagnostic gates.** The run is valid only if the config parses, all paths
  and SwanLab identifiers are unique, no cached rollout is used, every dropped
  group has a reason, every `turn_1_review` failure has a phase record, and
  the trainer reports the final bounded outcome. Preserve the original
  failure counts and separate provider, judge, validation, transport, and
  timeout categories. Stop after the one-step diagnostic; do not infer a
  reward result from it.
- **Scope boundary.** This batch-32 run is a failure-capture diagnostic, not
  the later production-sized A/B experiment. A production-sized A/B must be
  planned separately after this evidence identifies the failure class. Do not
  reuse its run, output, checkpoint, broadcast, cache, LoRA, or SwanLab
  identity for production.

## Batch-32 diagnostic restart and cleanup plan

- Start this disposable profile with `training_ops.start(..., mode="restart")`
  and no resume adapter or resume step. This launches fresh diagnostic
  components without cold-start cleanup or the inference engine start script.
- **Measured H200 capacity.** The same H200 server instance previously held
  production and qualification receipts concurrently while its inference
  config had `max_loras=1` and `max_cpu_loras=2`. `max_loras` limits active GPU
  concurrency; this server has measured capacity for two receipts. The
  diagnostic is the second receipt. Do not unload production before the run.
- Before any live start, snapshot the full production staging receipt and
  server identity: exact source URL, model/config hashes and sizes, verified
  files, LoRA name and ID, status, container, listener, and start identity.
  Require the same server instance and engine identity, `max_loras=1`,
  `max_cpu_loras=2`, exactly one verified production receipt, no second receipt,
  and `running=0`; fail closed if any check differs.
- Before restart, execute the no-clobber gate on every configured
  trainer/orchestrator/H200 host. This is an executable gate, not only text:
  use `test ! -e` for the exact run parent, service, output, broadcast,
  control, rollout, and checkpoint paths, and archive the absent results before
  start. Abort if any path exists.

  ```sh
  gate_archive=/operator-artifacts/trinity-decensor-experimental-fixed-diag-d29a5958-no-clobber.tsv
  run_parent=/data/glm52-rl/runs/trinity-decensor-experimental-fixed-diag-d29a5958
  service_dir="$run_parent/run_trinity-decensor-experimental-fixed-diag-d29a5958"
  trainer_output="$run_parent"
  : > "$gate_archive"
  for path in "$run_parent" "$service_dir" "$trainer_output" \
      "$service_dir/broadcasts" "$service_dir/control" "$service_dir/rollouts" \
      "$run_parent/checkpoints" "$service_dir/checkpoints"; do
    if test ! -e "$path"; then
      printf '%s\tabsent\n' "$path" >> "$gate_archive"
    else
      printf '%s\tpresent\n' "$path" >> "$gate_archive"
      exit 1
    fi
  done
  ```

  Run this exact gate on every host selected from `configs/nodes.toml` and
  preserve the archive before invoking restart.
- After the diagnostic stops, POST to the root `/remove_lora_adapter` endpoint
  with the exact diagnostic LoRA name. Validate response status, name, and ID,
  then validate `/staging_status`. Never run raw `rm` or manually remove
  shared-cache files; cache files remain engine-managed. Require the
  production receipt and server identity to be byte-for-byte unchanged.
- If production disappears or drifts, fail closed and use only the reviewed
  production start/restore path with the snapshotted exact production name and
  source identity. Do not improvise a raw API restore.
- The diagnostic paths contain no resume adapter, cached rollout, or stale
  rollout-batch input. This plan must be reviewed before any live execution.

## Explicit fresh diagnostic start mode

- **Measured restart deadlock.** The existing restart-without-resume path starts
  the trainer and waits for a fresh broadcast before it starts the orchestrator.
  The trainer waits for orchestrator control, while the launcher waits for the
  broadcast that the orchestrator must produce. This can leave a new diagnostic
  run stopped before its first batch.
- **Design.** Add an explicit `mode="fresh"` for a disposable new run. Keep
  `cold`, `restart`, and adapter-resume behavior unchanged. Fresh mode must
  preflight every selected trainer, the configured orchestrator host, and the
  H200 host before dependency deployment, global stop, or any other remote
  mutation. The preflight must prove that the exact run parent, service,
  trainer output, broadcast, control, rollout, and checkpoint paths are absent
  and are not symlinks; the exact run-specific cache key derived from the
  configured diagnostic source URL is absent on inference hosts; and no global
  trainer, orchestrator, or adapter container is active. `n79` remains
  excluded.
- **Fresh order and exclusions.** After the first preflight, dependency
  deployment and the static image/config checks run. A second full fresh
  preflight then runs immediately before any launch. Both preflights derive
  the inference cache key from the exact source URL
  `${source_url.rstrip('/')}/broadcasts/step_0`, not only the run URL. Fresh
  mode skips `_cold_start_cleanup`, `_cold_adapter_cache_cleanup`, resume
  markers and drains, inference restart, and the unconditional global
  trainer/orchestrator/adapter stop block. If the second preflight sees a new
  path, symlink, cache entry, or fixed-name container, it refuses; it never
  removes that container. It starts adapter, transport, orchestrator, waits
  for control, then starts the trainer. Set the initial-health watchdog
  requirement when the existing launcher interface supports it. Do not change
  training parameters or the existing cold/restart/resume paths.
- **Gates.** Refuse before any mutation if one exact path exists, is a symlink,
  the source-URL-derived cache key exists, a global container is active, a
  selected host is blocked or missing, or the dependency and run identities do
  not match. Tests must prove fresh order and preflight-before-mutation, and
  must prove refusal for each path, symlink, cache, and container condition.
- **Fresh launcher guard.** Training-ops passes
  `FRESH_START_NO_REPLACE=1` to every fresh trainer, adapter, and
  orchestrator launch; the orchestrator passes it through to its relay path.
  `cluster.sh` and `start_orchestrator.sh` validate the flag, inspect fixed
  target names including stopped containers, refuse before mutation, skip
  fixed-container `docker rm -f` and watchdog `pkill`, and let a create-after-
  inspect race fail at `docker run` without removing the other container.
  A failed fresh launch can leave its own failed container for the next
  preflight to report; do not manually remove it as part of this mode. Fresh
  does not invoke a trainer-side orchestrator launcher.

## Fresh watchdog ownership and adapter-host gate

- **Confirmed risk.** The failure watchdog currently identifies the
  orchestrator by its fixed container name only. A stale watchdog can therefore
  see a fresh container that reused that name and remove it after a later
  trainer failure. Fresh preflight also uses the selected trainer list, while
  the adapter runs on the configured `trainer_head`; a custom list can omit
  that implicit host and leave it unchecked before mutation.
- **Acceptance criteria.** Persist the exact launched orchestrator container ID
  and launch identity, and require both to match before the watchdog performs
  any stop or removal. A mismatch must fail closed. Keep the normal launch path
  compatible. Fresh preflight must include the configured adapter host even
  when custom trainer nodes omit `trainer_head`; a test must prove this host is
  checked before deployment or launch mutation.

## Rollout puller stop lock evidence

- **Measured baseline.** At `404c75eab4a52623a4a973b21d89ab92f4fbb3d5`,
  `pull_rollout_batches.sh stop` scans `/proc` for the exact source and
  destination environment, stops only matching process groups, then runs
  `flock -n .rollout-pull.lock -c true`. A plain stale lock file is harmless
  and returned status 0 in a disposable run directory. If `flock` reports a
  lock holder while no exact-run puller is discoverable, the command returns
  status 1. This makes stop fail on stale or unresolvable lock evidence.
- **Acceptance.** Stop is idempotent success when no exact-run puller is
  live, including a stale `.rollout-pull.lock`, stale PID text, or an
  unresolvable orphan lock record. A live puller for another run must not be
  killed and its PID or lock evidence must not be removed; stop must fail
  closed. A live exact-run puller must be stopped by its verified process
  group, and its state must be cleaned only after the lock is safe. Path,
  permission, process, and cleanup errors must still return failure.
- **Test plan.** Add disposable local tests for stale lock/PID state,
  unresolvable lock evidence, another-run lock ownership, exact-run process
  shutdown, repeated stop, and injected cleanup errors. Run the focused
  rollout-puller tests, the broader training-ops test set, `bash -n` on the
  runtime script, Python compilation, and the repository diff check. No SSH,
  deployment, or live training action is allowed for this repair.

## Watchdog process-side-effect race

- **Measured race.** The watchdog checks the orchestrator identity, then kills
  the batch publisher and runs a broad `pkill` before it checks the container
  again. A deterministic container identity change between those checks can
  therefore leave the new container running but still kill publisher processes
  from the wrong launch.
- **Acceptance test.** A race harness must change the reported container ID or
  launch label between the ownership checks and assert zero `docker rm`, zero
  publisher `kill`, and zero broad `pkill`. The implementation must make
  successful removal of the exact immutable container ID the ownership claim,
  then touch only an exact, independently verified publisher PID with its
  start time, command, and run identity. If exact process ownership cannot be
  proved, it must skip that process action. The normal failure path must still
  remove its owned container and stop its owned publisher.

## Exact unloaded adapter-cache cleanup

- **Measured need.** The existing cold cleanup scans and removes every
  64-hex cache entry. A fresh production start needs one stale unloaded entry
  removed without changing another policy. The configured production profile
  is `configs/trinity-decensor-experimental-fixed`; its exact source object is
  `http://127.0.0.1:18100/trinity-decensor-experimental-fixed/broadcasts/step_0`.
  SHA-256 of that exact URL is
  `056ce0a30be48095dbd88b8c0e741146d5be162be82252eb2040ee1ec0cd9074`, so
  the only allowed target is
  `/data/glm52-rl/lora-cache/056ce0a30be48095dbd88b8c0e741146d5be162be82252eb2040ee1ec0cd9074`.
  `configs/nodes.toml` selects `h200b` with port `8000` for inference.
- **Acceptance criteria.** Add one supported `training_ops` API and CLI
  operation that accepts an explicit run/config or an exact source URL and
  LoRA name. Resolve the URL to one exact 64-hex key and never accept a
  wildcard, cache-root path, or generic cleanup request. Before deletion,
  query `/staging_status` and fail closed for running or waiting requests,
  unknown status or contents, an active stage task for the target, any loaded
  receipt (including another receipt pointing to the target path), invalid
  server identity, a path or symlink mismatch, invalid or mismatched manifest
  or key, or a held readers lock. Require a stable resolved directory identity
  before and after checks. Delete only that immutable resolved directory and
  preserve every other cache entry.
- **Tests and limits.** Use disposable local fixtures and mocked remote
  responses for success and every refusal boundary above, including the
  production key. Do not change start settings or run a live action. Compile,
  run focused tests, then run the broader training-ops suite and inspect the
  final diff for any broad cache deletion.

## Exact broadcast service URL identity correction

- **Measured discrepancy (2026-09-27).** The configured production base URL is
  `http://127.0.0.1:18100/trinity-decensor-experimental-fixed`, but the runtime
  filesystem service publishes the object below the run-specific service
  directory `run_trinity-decensor-experimental-fixed/broadcasts/step_0`.
  The actual source URL is therefore
  `http://127.0.0.1:18100/trinity-decensor-experimental-fixed/run_trinity-decensor-experimental-fixed/broadcasts/step_0`.
  Its SHA-256 cache key is
  `f72c3222b63fe4faf9e482d6b36fb6887521aac5a76f96655e2b94a18a630f8a`.
  The current exact-cache helper and fresh preflight append only
  `/broadcasts/step_0`, producing the incorrect key
  `056ce0a30be48095dbd88b8c0e741146d5be162be82252eb2040ee1ec0cd9074`.
- **Acceptance criteria.** Before implementation, record this discrepancy in
  this file. Then use one shared derivation for config/run mode in fresh
  preflight and `remove-adapter-cache`: configured base URL plus exact run and
  service identity must produce the nested runtime URL and `f72c...` key.
  Explicit raw exact URL/name mode may remain exact. Tests must prove the
  archived nested URL/key, prove fresh preflight checks `f72c...` and not
  `056ce...`, and prove cleanup removes only `f72c...` while preserving other
  entries. Loaded-receipt checks must canonicalize base, nested actual, and
  safe aliases consistently without allowing the target receipt to evade
  detection. Add `transport-manifest.json` only if its schema is fully
  validated as a known legacy member; unknown members must still refuse.

## Measured legacy transport manifest member

- **Measured production member (2026-09-27).** The archived production cache
  entry at key `f72c3222b63fe4faf9e482d6b36fb6887521aac5a76f96655e2b94a18a630f8a`
  contains an object named `transport-manifest.json`, size `313546` bytes,
  mode `0600`. Cross-checks against the other cache manifests passed. The
  observed document has version `1`, the nested production source URL, exact
  target and config hashes and sizes matching the main manifest, mode `full`,
  transport `chunked-full-zstd`, and `986` chunks.
- **Acceptance criteria before allowing the member.** Treat only the exact
  filename `transport-manifest.json` as a known legacy member. Require a real
  regular file with the observed mode and strict top-level key set. Validate
  every scalar value and cross-check source URL, target/config hashes, and
  sizes against `manifest.json` and the sidecar. Validate the chunk list as a
  uniform list with the known key set, integer indices/offsets/sizes, and
  lowercase 64-hex hashes; reject duplicate or non-contiguous indices,
  overlapping or inconsistent offsets/sizes, malformed entries, extra keys,
  symlinks, unknown files, and identity mismatches. Do not implement until
  the exact chunk key/type schema is captured from the measured member.

## Exact fixed-token production-step1 policy-v0 re-score

- **Measured evidence.** The preserved production step-1 artifact is
  `/data/glm52-rl/runs/trinity-decensor-experimental-fixed/run_trinity-decensor-experimental-fixed/rollouts/step_1/train_rollouts.bin`,
  size `326027085` bytes, SHA-256
  `29a55a5e1efd674fb8ce7237a83105efd2624340829721f7624fc1763f68c86c`.
  It contains `1068` samples and `1203614` tokens. The first packed
  microbatch has sequence lengths `[8187, 7006, 6543, 6436, 4596]`; the
  first four are complete, and the fifth is `4530` source tokens plus `66`
  masked right-padding tokens. This is the required packing and alignment
  contract.
- **Measured parity failure.** The parity-step-1 artifact SHA-256 is
  `284761f2e5c0f37b6ac45d701e7266aa63f12b779b26c5d2c3c81e3c9f5ad678`.
  It contains `32768` packed positions and `29420` valid positions. Valid
  logprob delta mean is `-1.011637` and absolute mean is `1.156809`.
  Mismatch KL is mean `0.979907`, p95 `4.254092`, and maximum `2595.608`;
  per-sequence means are `1.2541`, `0.9708`, `1.0013`, `0.6167`, and
  `0.9832`. The full guard aggregation was unmasked KL `1.147836`, masked
  fraction `0.213119`, and masked KL `2.193980`. The prior diagnostic had
  KL mean `0.694752` and absolute delta mean `0.953952`; the passing 8192
  qualification exercised only `32` active tokens.
- **Experiment.** Re-score exactly five fixed-token sequences from this
  artifact on a fresh H200 policy-v0 request. Identify each sequence by a
  SHA-256 hash, and record only hashes and numeric metadata, never raw
  prompts or responses. Compare fresh policy-v0 per-token logprobs against
  the stored rollout logprobs and trainer per-token logprobs. Record exact
  model, adapter, server-instance, request/cache-salt, route, and reference
  identities for every comparison. Keep the native FP8 model, packing,
  padding, masks, and existing parity gates unchanged.
- **Decision.** If fresh policy-v0 matches stored rollout but both differ
  from trainer, investigate trainer ingestion or logprob alignment. If fresh
  policy-v0 differs from stored rollout, investigate H200 routing, adapter,
  or cache identity. If fresh policy-v0 and trainer agree while stored data
  differs, quarantine the stored rollout path. Run the router replay with
  router replay off and on only if the first comparison cannot distinguish
  these cases; do not change production settings before the result.

## Exact policy-0 source binding

- **Measured evidence.** The n82 artifact is `16,537,503,808` bytes with
  SHA-256 prefix `192215f1...`; the H200 cache has the same size with SHA-256
  prefix `3d365b82...`. Configs are identical with SHA-256 prefix
  `4d4e48b0...`, and headers are identical with SHA-256 prefix
  `446a9925...`. The artifact has `41,952` B-factor tensors and
  `4,131,520,512` BF16 elements, all zero.
- **Proposal.** Publish only with a fail-closed source manifest bound to the
  target SHA, and reject a stale adapter that has the same shape but a
  different source identity. Keep existing publication and qualification
  gates unchanged.

## Repeated packed 1f1b parity qualification

- **Measured parser failure and retry correction.** At `2026-09-27
  21:00:11 UTC`, TrainerConfig rejected the extra top-level input
  `name='packed-1f1b-run'`. The run completed `0` steps and did no forward,
  forward/backward, or PP work. Its output directory exists but is empty.
  Remove the unsupported `name` field. Use planned qualification name
  `packed-1f1b-run-v2`, with output directory
  `/data/glm52-rl/qualifications/trinity-h200-synthetic-packed-1f1b-parity/packed-1f1b-run-v2`.
  Preserve all other settings and gates. This is a parser correction;
  it gives no numerical or timing evidence for 1f1b.

- **Measured baseline.** Packed gpipe forward/backward took `103.600719s`
  (PP `71.33s`). Against H200, `32763` selected positions had absolute
  delta mean `0.06331701576755229`, p95 `0.22156524658203125`, maximum
  `1.8082847595214844`, and mean mismatch KL `0.006326035675103694`.
  The packed-run/direct-parity.json artifact under
  `/data/glm52-rl/qualifications/trinity-h200-synthetic-packed-parity/`
  is `2694767` bytes, SHA-256
  `bcfc9d74347ef66974a16228a54400155d004ea9a18f5cc3f0703fa5fde21a09`.
  Individual gpipe took `105.949494s`, with mean KL
  `0.0065851526030371605`. The same directory's comparison-result.json
  has SHA-256
  `dd5a241d1a4309efcf18e316cb2d4f512641ac5ce2ca1dfc6b195fdde34d9abd`.
  Production failed with unmasked KL `1.147836`, masked fraction
  `0.213119`, and masked KL `2.193980`.
- **Hypothesis and variables.** Packing alone did not reproduce production
  error under gpipe. Repeat the same packed synthetic H200 reference eight
  times with FakeDataLoader batch_size `8` and PP schedule `1f1b`. This
  isolates the combined change in schedule and number of microbatches;
  it does not separate those two variables. Each microbatch keeps five
  sequences and `32768` positions. Keep native FP8, CP8/PP8/EP8, no LoRA,
  lr `0`, router replay, and sparse-index transfer `false`. Use a new
  trinity-h200-synthetic-packed-1f1b-parity output directory.
- **Checks and decision.** Require schema-v2 output with exactly eight
  ordered microbatches. Bind each record's tokens, positions, lengths,
  masks, and reference logprobs to the same packed H200 reference. Compare
  every record with H200 and optionally the preserved packed gpipe output;
  do not use the five-individual-sequence comparator. Keep the forward/
  backward limit at `600s` and parity gates at unmasked KL `0.1`, masked
  KL `0.5`, and masked fraction `0.06`. A failure isolates schedule or
  microbatch-count effects for further tests. A pass qualifies only this
  synthetic case. It does not prove production capacity or an update.

## Fresh zero-effect LoRA packed 1f1b parity

- **Measured baseline (2026-09-27).** `packed-1f1b-run-v2` completed step 1.
  Forward/backward took `94.874800s`; PP took `70.33s`. Its direct artifact
  is `19985591` bytes, SHA-256
  `9f9d1b6fe092f9c58c6d0ec05f60ec82b1ebd0757132a98a711d74d3aa0b25b`.
  Its comparison artifact is `27732` bytes, SHA-256
  `66473642ec820bd0ae1bf6fd47f2e0d05ae13a54a2080fe35aff32f4d8d6f929`.
  Across `262104` selected positions, trainer versus H200 absolute delta
  mean was `0.06331701576755229`, p95 `0.22156524658203125`, maximum
  `1.8082847595214844`, and mean mismatch KL `0.006326035675103694`.
  All trainer versus packed gpipe deltas were zero. Production guard
  measurements remain unmasked KL `1.147836`, masked fraction `0.213119`,
  and masked KL `2.193980`.
- **Hypothesis and experiment.** The production LoRA-enabled trainer model
  path can cause divergence even when policy-0 B weights are zero. Repeat
  the passing packed batch-8 1f1b synthetic test with fresh trainer LoRA:
  rank `32`, alpha `64` (scale `2`), dropout `0`, and the exact production
  targets from `configs/trinity-decensor-experimental-fixed/trainer.toml`:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`, `shared_expert`, and `experts`. Use normal fresh zero B
  initialization, no source adapter, no checkpoint resume, no adapter export,
  one step, and lr `0`. This does not test adapter loading or source identity.
- **Fixed settings and checks.** Keep the same exact H200 base reference:
  a mathematically zero LoRA must match the base. Keep native FP8,
  CP8/PP8/EP8, 1f1b, batch `8`, router replay, sparse-index transfer `false`,
  file monitoring, and all packing and mask bindings. Use output directory
  `/data/glm52-rl/qualifications/trinity-h200-synthetic-packed-1f1b-zero-lora-parity/packed-1f1b-zero-lora-run`.
  Reuse the strict eight-microbatch comparator unchanged. Keep the
  forward/backward limit `600s`, unmasked KL limit `0.1`, masked KL limit
  `0.5`, and masked fraction limit `0.06`. Validate the config with the exact
  repository TrainerConfig schema. Tests must show that only the output
  and LoRA settings differ from v2 and that resume and export are disabled.
- **Live-test blocker.** `create_trainer_only_run` resets LoRA, but the
  `fresh_lora_policy` condition in `prime-rl/src/prime_rl/trainer/rl/train.py`
  excludes trainer-only tests from the explicit distributed zero, finite,
  and scale checks. The config alone does not prove fresh zero LoRA.
  Do not run this live test until all-rank B-zero, finite, and scale evidence
  is added and reviewed or otherwise measured. Trainer changes are outside
  this config task.
- **Decision.** A parity failure implicates the LoRA-enabled trainer path
  for this synthetic case. A pass rules out that path only for fresh zero
  LoRA on these tokens. It does not qualify production capacity, an update,
  adapter loading, or policy-source identity. No live action is part of
  this config change.

## Production-density loss-mask compaction diagnostic

- **Measurements.** Fresh zero-LoRA packed batch-8 1f1b forward/backward
  took `109.649182s`, with PP `105.50s`. The direct artifact has `19987039`
  bytes and SHA-256
  `87553e22d668b0e307c98bc9982d9f2a6f35941c962657dc3971a2eed0d7012d`.
  Guard unmasked KL was `0.006344088818877935`, maximum
  `0.007684200070798397`; strict mean KL was `0.006420181652246836`.
  Production selected `29420/32768` positions with KL `1.147836`, first
  lengths `[8187,7006,6543,6436,4596]`, and `66` tail padding positions.
  The passing synthetic case selected `32763/32768` positions.
- **Hypothesis.** FP8 LM-head label compaction and all-to-all with uneven
  masks can cause misalignment or numerical error before loss calculation.
  Derive a new reference from the exact existing `packed-reference.json`.
  Keep tokens, inference scores, positions, sequence lengths, routes, and
  source/model identity unchanged. For lengths `[4096,5632,6656,7680,8704]`,
  mask prefixes `[410,563,666,768,870]` including sequence starts, plus the
  last `71` positions of the last sequence. Assert no overlap, `3348` false
  and `29420` true positions; eight copies select `235360` positions.
  This matches production density, not the exact production mask or lengths.
- **Method.** Record the source byte size and SHA-256 with the mask spec.
  Generate identical derived bytes independently on each node later; make
  no inference request, rollout, or network transfer. Use fresh zero LoRA
  with exact production rank and targets, batch `8`, 1f1b, native FP8,
  CP8/PP8/EP8, lr `0`, and sparse-index transfer `false`. Use the new
  config `trinity-h200-synthetic-packed-1f1b-masked-parity` and output run
  `packed-1f1b-masked-run`. Live use requires the merged all-rank fresh-zero
  verifier. No live action is part of this change.
- **Checks and limits.** Require eight ordered schema-v2 records bound to
  the derived mask and float32 reference scores, finite per-copy,
  per-sequence, and total metrics, and trainer logs. Optional comparison
  with the passing zero-LoRA dense artifact uses only sparse-selected
  positions and permits only the prescribed mask difference. Keep the
  `600s` forward/backward limit and trainer gates `0.1` unmasked KL, `0.5`
  masked KL, and `0.06` masked fraction. Report `qualification_pass=null`;
  trainer guard evidence remains separate. A pass applies only to this
  synthetic diagnostic, not production capacity or an update.

## Fresh synthetic packed sparse-MLA parity

- **Measured evidence.** Production packed parity reported unmasked KL
  `1.147836`, masked KL `2.193980`, and masked fraction `0.213119`. Fresh
  H200 versus stored logprobs had mean absolute error `0.05346337`; fresh
  H200 versus trainer had `1.15468605`, about `21.6x` larger.
- **Proposal.** Generate five new deterministic nonuniform sequences totaling
  `32768` tokens from the native FP8 base with no LoRA. Compare individual and
  packed requests under CP8/PP8/EP8 using the same routes. Do not use cached
  rollouts or a full rollout batch. Only if the result is red, toggle
  `pp_transfer_sparse_indices=false`, then isolate CP and PP and assert
  `ks`/`ke` and index ranges. Keep existing parity gates unchanged.

## Fresh production-shape synthetic parity

- **Measurements.** The production-density mask diagnostic took
  `109.279052s` forward/backward, with PP `105.46s`. The trainer delta from
  the dense reference was exactly zero on `235360` selected positions.
  H200 KL was `0.005689727327266585`. The direct artifact SHA-256 was
  `a3634e11d9ad0daf037fa44f0349890278abce184c393b990ae3c3de513c139b`;
  the comparison SHA-256 was
  `63128fee00cfa51621594bfc683a01067dc9f87bfabc64dd4351dfe7110314cd`.
  Production unmasked KL remains `1.147836`.
- **Experiment.** Generate fresh deterministic native-FP8 H200 base requests
  with lengths `[8187,7006,6543,6436,4530]`, unique cache salts, and no LoRA.
  Use no cached reference or rollout. Append `66` padding tokens to the last
  sequence for packed lengths `[8187,7006,6543,6436,4596]` and total `32768`.
  Preserve the exact production tokens, positions, logprobs, loss mask,
  route rows, and route mask for padding. Bind new artifacts to exact source
  and model identities, and refuse to overwrite outputs. Compare eight
  ordered schema-v2 trainer records to H200 on selected positions, per
  sequence and in total. A dense baseline is not required.
- **Planned settings.** Use fresh zero LoRA, batch `8`, 1f1b, CP8/PP8/EP8,
  native FP8, lr `0`, and sparse-index transfer `false`. Keep the `600s`
  forward/backward limit and gates `0.1` unmasked KL, `0.5` masked KL, and
  `0.06` masked fraction. Reserve config
  `trinity-h200-synthetic-production-shape-parity` and output run
  `production-shape-run`. No live action is part of this proposal.
- **Blocked before implementation.** At commit
  `3045045bc73a4b6b4336d75be3145b83e3ca5b4c`, the requested production
  `_pad_packed_micro_batch` function is absent from checked-in sources.
  The available `pad_micro_batch_to_length` in
  `prime-rl/src/prime_rl/trainer/rl/train.py` pads token IDs with `1`,
  positions with `0`, inference logprobs with `0`, loss mask with `false`,
  and route rows with zeros, then extends the final sequence length.
  `FakeDataLoader._get_reference_micro_batch` instead rebuilds positions
  with `arange(length)` and ignores supplied position IDs. Its route decoder
  permits one zero final route row, but rejects repeated zero expert IDs in
  the preceding padding rows. Thus the allowed runtime/config changes cannot
  reproduce even this available padding contract for `66` tail positions.
  The exact production padding and route-mask contract must be supplied and
  loader support must be resolved before implementation. No trainer edits,
  config, generator, or live test were made for this blocked proposal.

### Production-shape construction after loader review

- **Measured code contract.** The PP trainer takes the maximum microbatch
  length and calls `pad_micro_batch_to_length` before CP sharding. Seven
  real `32702`-token records and one `32768`-token control therefore produce
  exactly `66` tail padding positions on each of the seven short records.
  The existing fake loader rejects mixed lengths before this call.
- **Revised implementation.** Add a default-false `allow_variable_length`
  fake-data option. Only schema-v2 references may use it. Require every
  length to be positive and at most `seq_len`, and at least one record to
  equal `seq_len`. Preserve the default exact-length requirement. Generate
  five fresh production-length H200 requests and one fresh `32768`-token
  control request with distinct cache salts. Pack the five real sequences
  once and repeat them seven times; use the full control as record seven.
  The trainer applies its own padding, so the reference has no zero padding
  routes to decode. Test the actual loader and padding functions together.
- **Checks.** Bind requests, responses, tokens, scores, and route bytes.
  Require eight ordered direct records, exact padding token `1`, position
  `0`, score `0`, loss mask `false`, and zero route rows on the seven short
  records. Their final lengths become `[8187,7006,6543,6436,4596]`.
  Each short record selects `32697` positions, for `228879` across seven;
  the control selects `32767`. Report these groups separately. Direct dumps
  do not expose routes; route padding is checked through loader/pad tests.
  Keep prior gates, run name, and diagnostic-only qualification status.

### Production-shape H200 response contract correction

- **Review measurements.** The first implementation checked equal route
  dimensions across six requests but accepted the test shape `[tokens,8,2]`.
  Trinity requires `[tokens,60,4]`. The shared score parser also accepted a
  numeric score at position zero, where no preceding token exists.
- **Correction.** Require every H200 route shape to be exactly
  `[request_token_count,60,4]`. Require the first prompt-logprob entry to be
  null and every later entry to contain the exact token-key numeric finite
  score. Apply these checks during both reference construction and replay
  validation. Change fixtures to the Trinity shape and add rejection tests
  for `[tokens,8,2]` routes and numeric position-zero entries. Keep all token
  lengths, padding, masks, settings, and parity gates unchanged. No live
  action is part of this correction.

## Bind pipeline packed boundaries through checkpoint recompute

- **Measurements.** The production-shape diagnostic took `131.18s`
  forward/backward and `108.34s` PP. Control KL was `0.008315`; packed
  sequence KL was `[0.005716,0.278246,0.754759,0.493240,1.090153]`.
- **Code evidence.** Training prepares all microbatches before the PP
  schedule. The final control leaves Ulysses boundaries `[0,32768]` in
  process-global state, while each production record needs
  `[0,8187,15193,21736,28172,32768]`. The image-provided
  `ulysses_attn.py` reads that state at each attention invocation. Identical
  microbatch layouts hid this difference. A stage-forward refresh alone is
  insufficient: mounted `checkpointing.py` later calls the captured attention
  method during backward after another microbatch has changed the state.
- **Fix and experiment.** Bind packed CP state to each consumed PP side-input
  record. Publish it within that stage call and restore the prior state on
  exit. Capture that same state when each checkpoint is created and publish
  it again inside every checkpoint invocation, including delayed recompute.
  Keep the state and context helpers in mounted `cp.py`; use the existing
  mounted `pp_runner.py` and `checkpointing.py`. Do not change attention
  kernels, packing, routes, gates, or model settings. Test alternating packed
  and control layouts, delayed reentrant and non-reentrant recompute, nested
  contexts, restoration on errors, and missing or invalid boundaries. A
  later authorized operator can repeat the same production-shape diagnostic;
  this code task includes no live action.

## Bounded rollout transport connection test

- **Measurements.** A `350074342`-byte rollout used `16` rsync-over-SSH
  streams. Connections reset or closed; the transfer reached the `60s`
  timeout with `44302336` bytes preserved. The adapter route already holds
  `64` separate WAN SSH owners.
- **Experiment.** Create one fresh disposable incompressible file of exactly
  `350074342` bytes. Transfer it from the H200 source to `n82` only, using
  the current endpoint and host-key security. Keep chunk boundaries fixed.
  Run one trial each with `1`, then `2`, then `4` workers. For each trial,
  record exact source and destination bytes and SHA-256, elapsed time,
  goodput, errors, and bytes transferred again. Stop escalation on the first
  timeout, connection reset, or hash fault. Keep the `60s` timeout and all
  other measurement limits unchanged.
- **Scope and decision.** Use no trainer, model, or rollout data. Preserve
  evidence, then remove only the disposable test artifacts. Select the
  lowest reliable connection count for rollout transport from these bounded
  trials. Adapter transport design remains a separate task. This record
  precedes live testing; this documentation change performs no live action.

## Temporary authenticated HTTPS transport experiment

- **Test limits.** Use one fresh disposable incompressible `16777216`-byte
  file, one direct HTTPS GET, and one attempt. Set a `300s` total limit and
  a `60s` no-progress limit. Use no redirects, resume, or SSH data tunnel.
- **User-approved experiment.** Provision a temporary H200 HTTPS endpoint
  reachable only from `n82` if available controls permit that restriction.
  Verify the TLS server identity and require client authentication scoped to
  this test. Prefer an ephemeral CA, server certificate, and client
  certificate with restrictive file modes. Do not expose the existing
  unauthenticated loopback relay or touch production artifacts.
- **Measurements and stop rules.** Record source and destination byte counts
  and SHA-256, TLS handshake time, connect time, time to first byte, transfer
  time, goodput, and errors. Stop on any authentication, TLS, redirect,
  connection reset, timeout, size, or hash fault. A pass qualifies only a
  later full-size test; it does not qualify production transport.
- **Cleanup.** After preserving evidence, remove the exact temporary endpoint,
  certificates and private keys, source file, and destination file. Verify
  that no test process or listener remains. This entry precedes live action;
  this documentation task performs no live or network action.
