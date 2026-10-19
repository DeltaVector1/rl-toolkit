# Review 5: the review 4 fixes at `6d1030cd`

Branch `Xiaomi-Envs`, commits `83d17b2d`, `c163f87a`, `6d1030cd` (compared with `4b5f3251`).

## Verdict

**Good. Ready to use.** Nothing to fix before merging. All seven review 4 findings are
fixed, and the fixes hold on every Code image, not only on the eight rows that were
re-checked.

## What this change does

- Removes the last line of the Code start-up file, so login profiles load once per command
  and child shells inherit the result.
- Stops the cache clean-up from deleting `__pycache__`.
- Turns Git hooks off for the `git checkout --detach` in the history strip.
- Decides the "history is truncated" check by exit status, not by printed text.
- Adds `format-code-task-001217` to the exclusion list (six excluded, 2,692 eligible).
- Puts the exception type into General grading errors.
- Shortens `docs/code.md` and moves the evidence into `docs/evaluation.md`.

## What I did

The report says the full sweep was not repeated, so I repeated it. I ran the real `setup`
and then the real `verified` on the untouched tree on **all 2,698 Code images** at
`6d1030cd`, with no patches, and compared the result with the review 3 baseline and the
review 4 pass. I also ran `uv run pytest -q` (107 passed) and read the diff and the tests.

## Results on all 2,698 images

| Check | Review 4 (`4b5f3251`) | Now (`6d1030cd`) |
| --- | --- | --- |
| Setup passes | 2,694 | **2,697** (only `002295` fails, as documented) |
| `HEAD`, tree, `git describe` changed by setup | 0 | 0 |
| `git status` changed by setup | 1 | 0 |
| Commits left outside the base | 0 | 0 |
| Unreachable commits (images with `.git` under 400 MB) | 0 of 2,604 | 0 of 2,607 |
| Child bash loses a `PATH` entry set by its parent | 897 | **0** |
| Child bash has a parent variable overwritten | 485 | **0** |
| `bash -u` child prints an error | 1,797 | **0** |
| `bash -eu` child exits with an error | 4 | **0** |
| Cost of starting a child shell | up to 60 s on pyenv images | 0 to 3 ms |
| pyenv rows (118) taking 124 s or more to grade | 77 | 2 |
| New "command not found" or missing-module errors | 2 rows | 0 |

The verifier exit code matches the review 3 baseline on every row except the ones that are
meant to differ:

- The 11 repaired rows (9 Rust, Java `003041`, Go `002099`) now run their tests instead of
  exiting 127. Two of the Rust rows (`000797`, `001540`) did not finish inside my 300 s
  cap; they needed 394 s and 664 s in review 4.
- `002643` and `002606` are long-running rows that hit my cap. They did the same in review 4.
- `000400`, `001939` and `000815` differ from the baseline in the same way they did in
  review 4. The fixes did not change them.

The five rows that review 4 found broken:

| Row | Review 4 | Now |
| --- | --- | --- |
| `001840` (compiled-only pytest) | `No module named pytest.__main__` | `3 failed, 3 passed`, same as baseline |
| `000050` (compiled-only pytest) | `No module named pytest.__main__` | exit 2, same as baseline |
| `001166` (failing `post-checkout` hook) | setup failed | setup passes, tests run |
| `002071` (failing `post-checkout` hook) | setup failed | setup passes, tests run |
| `001621` (profile prints a banner) | setup failed | setup passes, tests run |

The exclusion list behaves as documented: `001225` and `001597` pass untouched, `001227` and
`002879` exit 127, `002295` fails closed, and `001217` failed this time (it is the flaky one).

**Setup time.** The full pass shows a median of 22 s for setup, against 3.9 s in review 4.
This is the sandbox provider being slower at 128 parallel sandboxes, not the code: the
verifier step shows the same extra 16 to 18 s on rows that take a second to test. I re-ran
ten pyenv rows at 10 in parallel with nothing but the port's own setup and verifier. Setup
took 1.9 to 4.4 s and grading took 2 to 103 s, with no pyenv lock messages.

**The agent run in the report.** I opened the trace in
`outputs/review4-committed-000484-c163f87/`. It records reward 1.0, 20 turns, no errors, no
truncation and no timeout, as reported.

## Must fix

None.

## Should fix

None.

## Nice to have

**1. `docs/evaluation.md` can now state the full result.**

- Where: `docs/evaluation.md`, section "Review 4 committed-source validation".
- Problem: the last sentence says the 2,698-row sweep was not repeated at the committed
  source. It has now been repeated, with the numbers above. The section also refers to "the
  supplied Review 4 full-pass file", which a reader of the repo does not have.
- Fix: replace that sentence with the full-pass numbers, or leave it. Either is fine.
- If we skip it: the doc understates the evidence. Nothing breaks.

**2. A short judge key still redacts ordinary words.**

- Where: `xiaomi_general/taskset.py:470-474` (`redact`).
- Problem: this was a side note in review 4 finding 6 and was not changed. If
  `GA_JUDGE_KEY` is a short word such as `dummy`, every `dummy` in the verifier output
  becomes `<redacted>`. Real keys are long, so this only shows up in tests.
- Fix: none needed. Mention it in a comment if you like.
- If we skip it: nothing.

## Checked, no finding

- **Hooks are only turned off for the checkout.** The later `git update-ref`, `git remote
  remove` and `git gc` could run a `reference-transaction` hook. No image has one (85 have
  `post-checkout`), and setup passes on every image, so there is nothing to fix.
- **The history check fails closed.** If `git rev-list` itself fails, the command exits
  non-zero and setup raises. The new test covers the clean case, a leftover commit and a Git
  failure, and the old code fails its clean case.
- **Profiles that print.** Eight images print something from a profile (seven to stderr,
  `001621` to stdout). All eight pass setup and run their tests.
- **The new tests fail on the old code.** I ran `tests/test_code.py` against the
  `4b5f3251` version of `taskset.py`: five tests fail, one for each of the four Code fixes
  plus the test that pins the new history command. The child-shell test counts profile
  loads, the clean-up test imports a relocated `.pyc`, and the strip test installs a failing
  `post-checkout` hook.
- **General error message.** The type name is now in the message, the chain is still cut,
  and the key is still redacted. Nothing else in the repo matches on the old message text.
- **Docs.** `docs/code.md` no longer carries review history or points outside the repo. The
  README matches the code.
- **Resources.** Every sandbox in the full pass resolved to 8 GB memory from the task.

## Not checked

- Unreachable objects on the 90 images whose `.git` is 400 MB or larger.
- The six rows that did not finish inside my 300 s cap: `000252`, `000348`, `002606`,
  `002643` (long-running, unrelated to this change) and `000797`, `001540` (slow Rust rows
  that finished in review 4 with a longer cap).
- An agent run of my own at this commit. I only checked the other agent's trace for `000484`.
- Whether a correct solution passes on rows other than the ones run with a model so far.
- General and Music were not run live this round. The General change is one line and its
  unit tests pass.
- General with an independent judge, and the `abcmidi` version, as in earlier reviews.

## Housekeeping

- This review used about 2,710 Prime sandboxes. All are deleted; the list is empty.
- I changed nothing in `rl-toolkit`. It is at `6d1030cd` with a clean working tree.
- Raw results: `claude-review-probe-tooling/results/review5_full.jsonl` (one record per
  image) and `review5_clean_10.jsonl` (the ten-row timing check). The scripts are the review
  4 ones, unchanged; see the "Review 5" section of the tooling README.
