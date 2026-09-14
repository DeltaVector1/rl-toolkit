# Difficulty ideas

## Measured long-haul evidence

Snapshot: 2026-09-14 15:14 UTC, active run `glm-decensor-difficulty-9`.

- The latest reward report covers steps 7-34. Effective reward mean is
  0.0629, fit is -0.0421, and the noise threshold is 0.0650. Verdict:
  **inside the noise**. This is not evidence of improvement or decline.
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

- One step is not a reward result. Step 1 took 44m09, had 1,024/1,200
  trainable rollouts, mean reward 0.1080, and mix easy 48% / medium 36% /
  hard 16%; it is only a collection check.
- The current long-haul reward is **inside the noise**, not a proven trend.
  Do not judge an idea from five steps. Use 40-50 completed trainer steps,
  unless a hard failure ends the experiment.
- An earlier steps 6-40 summary had reward fit -0.0048 and noise threshold
  0.0433. Keep it as historical evidence, not as the current step-32 result.
- The previous production-shaped trainer trial stopped at the 600s
  forward/backward limit: 606.58s active, with no completion, OOM, or crash.
  Its saved optimizer batch had 1,864 samples, 4,522,029 raw tokens,
  2,712,271 loss tokens, and max sequence 24,060. The earlier roughly 3m
  fixed-token result is not comparable.

## In progress

- The current run continues unchanged to 50 completed trainer steps.
- Finish the active long-haul run to 40-50 completed trainer steps, then
  compare easy, medium, and hard rewards separately and require movement
  beyond the measured noise before claiming improvement.
- Keep the measured raw-mix control and its approximate compensation:
  easy:medium:hard = 1:1.05:1.97.
- Do not change steps 41-50 mid-run.
- If the current run remains inside the noise at 50, run a clean comparison
  first, using constant 1:1:1 raw easy:medium:hard sampling and no other
  mid-run changes.
- Establish a fresh production-shaped trainer baseline below 600s with the
  same measured token and sequence workload. Do not replay the saved rollout
  batch or loosen the limit.

## Untested

- Rollout time added by retries and all-refusal replacement.
- Concurrent visible and reasoning judge phases with separate prompts,
  history, events, retries, and strict validation.
- A specialized correction prompt for contradictory declined and explicit-
  refusal fields.
- Add an explicit `answer_restriction_applied` decision so a brief caution in
  a fully fulfilled answer is not treated as an internal refusal. Keep the
  penalty for declined, partial, or replaced answers. Test this alone for
  40-50 steps.
- Distinguish an actual refusal decision in the reasoning trace from a brief
  caution in an otherwise fulfilled answer. Do not deploy it mid-run; compare
  it only in a clean 40-50-step run.

## Acceptance

- Actual ratios change by stage.
- Multi-turn traces contain more than one assistant turn.
- With exclusion enabled, all-refusal groups create no optimizer samples.
- Use 40-50 completed trainer steps unless a hard failure occurs.
- Compare each difficulty separately and require reward movement beyond noise.
