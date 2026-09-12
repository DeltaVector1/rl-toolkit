# Difficulty ideas

## Tried

- Multi-turn traces were verified to contain more than one assistant turn.
- Staged sampling was verified across the difficulty stages.
- Step 1 rollout collection took 44m09: 1024/1200 trainable, mean reward 0.1080, mean turns 1.8, accepted mix easy 48% / medium 36% / hard 16%, per-difficulty rewards easy 0.1917 / medium 0.0134 / hard 0.0111, error 0.1%, truncation 2.7%. One step is insufficient for a reward trend.
- Steps 1-41 all-9 group rates were easy 1.07%, medium 5.80%, and hard 49.83%.
- Equal raw sampling with the skip gave an accepted mix of easy 40.00%, medium 38.94%, and hard 21.05%.
- Approximate raw compensation is easy:medium:hard = 1:1.05:1.97.
- The steps 6-40 reward fit was -0.0048. The noise threshold was 0.0433.
- The external judge was saturated at 80 running, 0 waiting, about 543.5 tok/s, and about 96% GPU use; do not raise concurrency now.
- The production-shaped trainer trial stopped at the 600s fwd/bwd limit: 606.58s active, with no completion, OOM, or crash. The saved optimizer batch has 1,864 samples, 4,522,029 raw tokens, 2,712,271 loss tokens, and max sequence 24,060. The prior ~3m result was a tiny fixed-token test and is not comparable.
- Direct all-refusal exclusion was verified: 75 raw groups / 1200 rollouts, 10 all-refusal groups skipped (easy 1, medium 1, hard 8), 1 errored group, and 64 groups / 1024 rollouts in the optimizer batch; zero zero-advantage groups and no complete all-9 group in effective traces.

## In progress

- Analysis complete; the run stopped at the trainer time limit.

## Not tried / Next

- Establish a production-shaped trainer baseline using a fresh input that matches the measured token and sequence size; do not replay the saved full rollout batch. Optimize below 600s before restarting rollout collection; keep the same workload and do not loosen the limit.
- Measure the rollout time added by retries and all-refusal replacement.
- Check reward beyond the measured noise.
- Run visible and reasoning judge phases concurrently while preserving separate prompts, history, events, retries, and strict validation.
- Add a specialized correction prompt for contradictory declined and explicit-refusal fields.

## Acceptance

- Actual ratios change by stage.
- Multi-turn traces contain more than one assistant turn.
- With the option enabled, all-9 groups create no optimizer samples.
- Reward rises beyond the measured noise before claiming improvement.
