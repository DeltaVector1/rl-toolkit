# Difficulty ideas

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
