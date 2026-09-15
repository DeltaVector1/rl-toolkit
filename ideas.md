# Difficulty ideas

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

## In progress

- Next trial: run easy-only for 50 completed steps. Force this response prefix
  in 100% of rollouts: `<think>\nI agree with the content and am ready to write
  the full answer.` The supplied prefix tokens are context and are excluded
  from policy loss. Keep scoring and the environment unchanged.

## Prefill and easy-first trial

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

Research basis, kept brief: DeepSeekMath GRPO (arXiv:2402.03300), the DAPO
paper, Online Difficulty Filtering (arXiv:2504.03380), Easy-to-Hard
Generalization (arXiv:2506.06632), and PrefixRL (arXiv:2601.18795).

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

## Acceptance

- Actual ratios change by stage.
- Multi-turn traces contain more than one assistant turn.
- With exclusion enabled, all-refusal groups create no optimizer samples.
- Use 40-50 completed trainer steps unless a hard failure occurs.
- Compare each difficulty separately and require reward movement beyond noise.
