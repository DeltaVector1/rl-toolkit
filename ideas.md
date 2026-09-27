# Difficulty ideas

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
- **Follow-up measurement.** After the bounded record is verified, run a
  controlled A/B timing and failure-rate comparison for review concurrency 2
  versus a higher value, and judge client `pool_size=4` versus `pool_size=16`.
  Keep batch size 1024, group size 16, context length 32768, output length
  16384, reward settings, and timing limits fixed. Use the new phase-level
  failure counts to separate provider argument failures from timeout or service
  failures before changing concurrency.
