# Difficulty ideas

## Tried

- The equal-ratio run used fixed easy/medium/hard ratios of 1:1:1. It failed to test a curriculum.
- Reward change from steps 6-36 was +0.0070. The noise threshold was 0.0439, so this is not an improvement.
- Recent rewards were easy 0.2205, medium 0.0393, and hard 0.0093.
- 61 of 102 hard groups were all refusal 9. These groups still had reward variation.
- At step 36, multi-turn was assigned to 512/1024 groups, but none completed. The deployed old Verifiers has `DecensorTask.user=None`.

## In progress

- Curriculum: change the easy/medium/hard ratios by stage.
- Multi-turn repair: fix the old Verifiers task user and verify completed traces.
- Optional all-9 group skip: create no optimizer samples for an all-refusal group.

## Not tried

- No stage-varying run has passed acceptance yet.

## Acceptance

- Actual ratios change by stage.
- Multi-turn traces contain more than one assistant turn.
- With the option enabled, all-9 groups create no optimizer samples.
- Reward rises beyond the measured noise before claiming improvement.
