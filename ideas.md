# Difficulty ideas

## Tried

- The multi-turn and all-9 group fixes are implemented and independently passed.
- Steps 1-41 all-9 group rates were easy 1.07%, medium 5.80%, and hard 49.83%.
- Equal raw sampling with the skip gave an accepted mix of easy 40.00%, medium 38.94%, and hard 21.05%.
- Approximate raw compensation is easy:medium:hard = 1:1.05:1.97.
- The steps 6-40 reward fit was -0.0048. The noise threshold was 0.0433.

## In progress

- Curriculum: use the compensated stage ratios.

## Not tried / Next

- Run the staged curriculum.
- Verify that multi-turn traces have more than one actual assistant turn.
- Measure that skipped all-9 groups create no optimizer samples.
- Measure the added timing cost.
- Check reward beyond the measured noise.

## Acceptance

- Actual ratios change by stage.
- Multi-turn traces contain more than one assistant turn.
- With the option enabled, all-9 groups create no optimizer samples.
- Reward rises beyond the measured noise before claiming improvement.
