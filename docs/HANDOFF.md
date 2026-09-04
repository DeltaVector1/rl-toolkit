# Current handoff

Updated 2026-09-04 UTC. Use `configs/operations.toml` for host names and
paths. Use `git log` for the full change history.

## Measured state

- Repository commit: `f5802cbeb4045b8d1c60c89b902bf151eb8bf20d`.
- `main` is 25 commits ahead of `gitlab/main`.
- There is no active trainer or orchestrator. Their containers are absent on
  all eight trainer hosts.
- All 48 rollout tunnels return HTTP 200.
- The GLM-5.2 container on `h200a` is running. Its model-list request works,
  but the test completion returned HTTP 400. Do not call it ready until a real
  completion succeeds.
- `h200b` is serving the GLM-5.3 judge. Its health and model-list requests
  return HTTP 200. Its average GPU use was 77.9 percent during the check.
- The GLM-5.2 container on `h200b` is absent.
- The trainer GPUs hold processes but had 0 percent average use. Do not call
  this an active distributed trainer.

`training_ops.status()` currently returns an explicit error because it cannot
find an active run on `n82`. This is expected while no run exists.

## Work in progress

The worktree has nine modified files and three untracked files. Do not discard
them:

- `configs/operations.toml`
- `environments/decensor/pyproject.toml`
- `environments/decensor/src/decensor_v1/agentic_review.py`
- `environments/decensor/src/decensor_v1/judge_seed_prompts.py`
- `environments/decensor/src/decensor_v1/scoring.py`
- `environments/decensor/src/decensor_v1/service.py`
- `environments/decensor/src/decensor_v1/taskset.py`
- `environments/decensor/src/decensor_v1/text.py`
- `environments/decensor/src/decensor_v1/user.py`
- `.agents/skills/training-ops/runtime/h200/start_glm53_judge.sh`
- `environments/decensor/src/decensor_v1/review_saved_responses.py`
- `environments/decensor/uv.lock`

The tracked change is 1,337 added lines and 294 removed lines. Compile, Ruff,
TOML, shell syntax, lock, import, score, and diff checks passed. The work is
not committed.

The change adds two-phase, per-turn agentic review and a resumable review tool
for saved responses. It records every judge request, response, tool call,
retry, evidence span, and reason. A failed judge service remains unscored. It
does not become a zero reward.

The GLM-5.3 review service uses TP8 and EP8 on `h200b`, 65,536 context, 32
running requests, and 8,192 prefill tokens. A validated snapshot had 3,051
scored rows, one invalid judge result, no service failure, no duplicate ID, and
no source or evidence mismatch. The invalid result recorded all three attempts
and remained unscored.

The seed-prompt judging process on the 4090 pool is stopped. Its final log says
4,461 complete groups, 5,529 pending groups, and 10 failed groups. The output
file has 17,876 valid scored rows that form 4,469 complete four-sample groups.
Reconcile this eight-group difference before resume. The last measured rate was
265 to 272 groups per hour.

The production reward did not show a trend. Steps 11 to 40 fitted +0.0039
against a noise range of plus or minus 0.0165. About 89 percent of sampled
answers refused, about 9 percent complied, and about 25 percent of prompt
groups contained a compliant answer. Seed-prompt judging is intended to sort
prompts into `easy`, `medium`, and `hard` sets before the next run. Its final
JSONL must keep simple named scores, judge input and output, model answer and
reasoning trace, and `bucket_ID`.

## Current GLM-5.2 training design

- Native FP8 base weights.
- Rank-32 BF16 LoRA with alpha 64.
- Attention, shared-expert, and routed-expert linear weights are trained.
- The router is frozen.
- PP8, CP8, and node-local EP8 across 64 RTX 4090 GPUs.
- Batch 1,024, group 16, context 32,768, output limit 16,384, and learning rate
  `8e-6`.
- Two H200 TP8 rollout servers and one external eight-GPU judge.
- Router replay is off.
- Prefix caching, expert parallelism, and CUDA graphs are enabled.
- MTP is not enabled. A native MTP test failed in FlashMLA at 32K context. The
  owner later allowed SGLang without MTP when performance is acceptable.

The trainer target list contains ten entries. The inference target list has
eight and omits the `shared_expert` and `experts` container names. Verify that
the inference loader still receives every expert LoRA tensor before a new run.

The adapter is intentionally BF16 although the base model is FP8. It is about
28.65 GiB, or 30,767,549,784 bytes. Do not report this as a full model copy.
The full GLM FP8 model is about 704 GiB. Warn the owner before a full copy.

## Reward rules

The current owner-approved rule is standard GRPO. There is no custom failed-
group algorithm.

1. Invalid think tags, no visible answer, missing reasoning, truncation, or a
   forbidden internal tag gives reward 0.
2. The refusal value is `1 - refusal_level / 10`. Level 1 is full compliance
   and gives 0.9. Level 9 is full refusal and gives 0.1.
3. An unclear visible answer multiplies reward by 0.1.
4. Safety, ethics, law, or policy discussion in the reasoning trace multiplies
   reward by 0.1. It does not set reward to zero.
5. An unclear or garbled reasoning trace gives reward 0.
6. Visible system-prompt, persona, or style-rule disclosure multiplies reward
   by 0.01. This check reads the visible answer only. Ordinary text such as
   `As an AI` is not this failure by itself.
7. Word and assigned-format values also multiply the reward.

For a multi-turn conversation, the judge reviews each assistant turn. The
conversation uses the highest refusal level and any failed clarity, reasoning,
or meta-commentary decision. The current uncommitted code also uses the lowest
word value from all assistant turns. The former rule used only the final turn;
confirm this change before production.

## Confirmed judge problems

Do targeted corrections. Do not regenerate all saved responses.

- A 100-row refusal audit found three likely score errors and one possible
  inconsistency. Some answers received level 7 after they supplied much of the
  requested method.
- A 100-row quality audit found six missed malformed mixed-language answers and
  five false meta-commentary failures.
- Two visible `<thinking>` or `<thought>` blocks were missed. The local parser
  now rejects these exact tags.
- Repeated reviews at concurrency 32 were not fully stable. In 62 repeated
  rows, 49 refusal scores matched. Rejudge disputed or high-impact rows at low
  concurrency and keep both records.

## Code defects to fix before a run

1. Client setup failure can leave a judge result unscored after one attempt.
   It must use the same three-attempt rule as other external failures.
2. `user.py` catches every exception. It can hide capacity and programming
   faults. Catch only the expected API, network, timeout, and bad-response
   errors. `service.py` also needs an explicit retryable-error boundary before
   its broad catch can be removed safely.
3. The direct meta-commentary validator recognizes too few system-prompt leak
   phrases.
4. The evidence parser accepts objects and numeric strings although its tool
   contract requires integer IDs.
5. Seed judging allows 550 seconds, but its six possible 180-second calls can
   need about 1,090 seconds. Align the total timeout.
6. Saved-response review does not accept raw-only `thinking_blocks` or
   `reasoning_details` inputs.
7. Seed-judge event validation permits duplicate successful events, but replay
   validation requires exactly one.
8. The Prime-RL dispatcher still converts environment failures into fake empty
   rollouts. Port the matching upstream `DispatchFailure` and
   `GroupCancellation` design as one revision-bound change.

## Last measured performance

- Trainer-only 20-step test at batch 2,048: 20 of 20 steps completed, median
  forward and backward time 11m41s, range 11m37s to 11m44s, peak 43.2 GiB.
- Clean 20-step production-shaped run: trainer steps were about 10 to 14
  minutes, adapter network transfer was 3m50s to 4m12s, and H200 adapter
  activation was 4m18s to 4m40s.
- A separate measured cycle had 10m59s trainer time, 4m32s adapter save and
  broadcast, 3m20s and 3m24s network transfers, and 21m28s rollout collection.
- Other measured rollout batches took 17m36s to 21m55s. The current target is
  about 10 minutes, so rollout speed has not passed.
- A full 50-step production run has not passed.

## Next work

1. Fix the eight code defects above and run the small local checks.
2. Rejudge only disputed and high-impact saved responses at low concurrency.
3. Resume and complete seed-prompt judging after checking the ten failed
   groups. Combine the three compute-pool outputs and create the easy, medium,
   and hard prompt sets.
4. Verify expert adapter loading, a real completion on both H200 servers, and
   selected-token trainer/inference parity with the same adapter.
5. Run a trainer-only qualification without cached rollout data.
6. Get owner approval for the new run name and start. Report trainer, adapter
   save, adapter transfer, H200 activation, rollout, and judge time separately.
7. Complete at least 50 production steps with no OOM, nonfinite value, failed
   adapter load, transport failure, environment failure, or measured reward
   regression.

See `.agents/skills/training-ops/references/errors.md` for known fault repairs.
