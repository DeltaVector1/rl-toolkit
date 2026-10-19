# Review 4: Xiaomi env ports after the review 3 fixes

Date: 2026-10-11. Reviewed: `rl-toolkit` branch `Xiaomi-Envs` at `4b5f3251`
(commits `db04188e`, `8298ed46`, `2b214f9c`, `47914330`, `4b5f3251` on top of
`58615037`). Read-and-test review only. No code was changed and nothing was
committed.

All paths are relative to `rl-toolkit/` unless they start with
`claude-review-probe-tooling/`.

## Verdict

Not ready yet. The six review 3 findings are addressed and the main parts hold
up on every image, but the fixes brought in four new defects that only show
when the new code runs on all images. Each has a small fix, and I tested all
four fixes together on 85 affected rows.

The other agent's summary said "No full 2,698-image rerun was performed". I ran
one: the real new `setup`, then the real verifier on the untouched tree, on all
2,698 Code images. Everything below comes from that run and targeted re-runs.

## What this change does

- Code `setup` now deletes git history after the task's starting commit,
  deletes some cache folders, and writes a small shell start-up file
  (`BASH_ENV`) so the agent's shell and the grader's shell get the image's
  login environment.
- That start-up file also puts Rust, Java and one Go tool back on `PATH` when
  they are installed but not found.
- Code tasks now ask for 8 GB of memory.
- The General grader hides judge API keys in stored output and in error text.
- Five Code rows are documented as excluded.

## Summary

| # | Severity | Env | One line |
|---|----------|-----|----------|
| 1 | Must fix | Code | The start-up file re-applies login profiles in every child shell. On 77 pyenv rows grading takes 4 to 10 minutes instead of seconds, and child shells lose the parent's `PATH` on 897 images |
| 2 | Must fix | Code | The cache clean-up deletes the only copy of the test dependencies on 2 rows. They now always score 0 |
| 3 | Must fix | Code | A git hook makes the history strip fail on 2 rows. One of them worked before this change |
| 4 | Must fix | Code | Text printed by a login profile is mistaken for leftover history on 1 row that worked before |
| 5 | Should fix | Code | Row `001217` passes with no change made in 3 of 11 runs |
| 6 | Nice to have | General | Grading errors lose their original type; some end up with an empty message |
| 7 | Nice to have | Docs | `docs/code.md` contains review history and points at a folder outside the repo |

Net effect on rows compared with `58615037`: the strip, the login environment
and the tool repairs bring back about 650 rows. Findings 2 to 4 take away 4 rows
that worked, and finding 1 makes 77 rows very slow.

## Review 3 findings: status

| R3 # | Fix | How I checked | Result |
|------|-----|---------------|--------|
| 1 | History strip runs in every setup | Real `setup` on all 2,698 images | Setup passes on 2,694. Of the 618 rows from review 3, 616 now pass; `002071` (finding 3) and `002295` (documented) do not. Two rows that passed before now fail: `001166` (finding 3) and `001621` (finding 4) |
| 2 | Same strip removes unreachable objects | `git fsck --unreachable` after setup on the 2,604 images with `.git` under 400 MB | 0 unreachable commits on all 2,604. 0 commits outside the base on every image that passed setup |
| 3 | `BASH_ENV` start-up file | Real verifier on all 29 rows from `login_shell_required_ids.txt` | All 29 run their tests. No `command not found`, no missing module. But see finding 1 |
| 4 | 11 rows repaired, 5 excluded | Real verifier on all of them | `003041` (Java), `002099` (Go) and 7 of the 9 Rust rows compile and run their tests within my 300 s cap. The other two Rust rows (`000797`, `001540`) were still compiling at the cap; in the fix check they finish in 394 s and 664 s with normal test failures. `001227` and `002879` still exit 127. `001225` and `001597` still pass untouched. `002295` fails closed. The exclusion list is right apart from finding 5 |
| 5 | `resources=vf.TaskResources(memory=8)` | 13 rows from `sandbox_died_4gb_10gb_ids.txt` with **no runtime flags** | All 13 finish. The sandbox resolves to 1 CPU, 8 GB, 5 GB disk. The 8 largest repositories (`.git` 3 to 7 GB) also finish on those defaults |
| 6 | Judge key redaction | Read the code and tests; read `outputs/review3-general-redaction-fixed` | Stored output shows `key=<redacted>`. No other `key=` text in the trace |

What the strip does to the working tree, on all 2,694 images that passed setup:

| Check | Result |
|-------|--------|
| `HEAD` or tree hash changed | 0 |
| `git describe` tag or distance changed | 0 |
| `git status` changed | 1 (`000939`: two folders that held only `__pycache__` are gone; harmless) |
| `git diff HEAD` text changed | 129, all with uncommitted changes. On the two I opened only the short hashes in `index` lines got shorter; `--full-index` output is identical |
| Setup time | median 3.9 s, 95th percentile 23.5 s, worst 168 s (budget 600 s) |
| Git versions in the images | 2.30 to 2.47, so every git option the strip uses exists |

Other checks:

- `uv run pytest -q`: 103 passed.
- The other agent's Code trace (`outputs/review3-code-001138-fixed-full`)
  matches their report: 49 calls, scoring ran 8 tests, reward 0.

## Must fix

### 1. The start-up file re-applies login profiles in every child shell

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:37`, the last line of
`BASH_ENV_SCRIPT`: `export BASH_ENV=/tmp/_xiaomi_code_bash_env`.

- Review 3 proposed a three-line file that unsets `BASH_ENV` and loads the
  profiles once. That copies upstream's `bash -lc`: the outermost shell is a
  login shell, and shells started inside it are not.
- The shipped file adds a last line that sets `BASH_ENV` again. So every bash
  started inside the agent's or grader's shell loads the login profiles again:
  every `bash script.sh`, every script with a bash shebang, every pyenv shim.
- `tests/test_code.py::test_code_bootstrap_reaches_each_nested_shell_without_recursing`
  asserts this behaviour, so it was done on purpose. I could not find a row
  that needs it (see "Fix" below).
- The numbers I gave in review 3 for option B were measured on the three-line
  file. They do not cover this variant.

**Effect 1: grading stalls on pyenv images.**

- 118 images set up pyenv in a login profile. `pyenv init` runs `pyenv rehash`,
  which takes a lock file. With `BASH_ENV` still set, a login shell's rehash
  starts child scripts that load the profile again and try to take the same
  lock. Each waits 60 seconds and gives up.
- Every Code test script is started as `bash <script>`, and 843 scripts then
  run `exec bash -lc '... bash -eo pipefail ...'`. So the grader hits this
  without the agent doing anything.
- On `format-code-task-000484`, in a fresh sandbox with the shipped file:
  `bash -lc true` takes **120 seconds** and prints
  `pyenv: cannot rehash: couldn't acquire lock ... for 60 seconds`. With the
  last line removed it takes 0 seconds.

Grading time on the untouched tree, only the port's own `setup` and `verified`
running in the sandbox (`results/review4_clean_*.jsonl`):

| Row | Shipped file | Without the last line |
|-----|--------------|-----------------------|
| `000484` | 243 s | 3 s |
| `002409` | 243 s | 2 s |
| `001264` | 246 s | 4 s |
| `002370` | 248 s | 4 s |
| `002557` | 256 s | 16 s |
| `000540` | 262 s | 20 s |
| `000223` | 485 s | 3 s |
| `000573` | 488 s | 4 s |
| `000420` | 488 s | 4 s |
| `001267` | 491 s | 8 s |

- In the full pass, 77 of the 118 pyenv rows took 124 s or more to grade
  (median 259 s; five never answered within my cap). The other 41 took 32 s or
  less. All 62 pyenv rows whose script uses `bash -lc` are among the 77.
  List: `results/review4_pyenv_slow_ids.txt`.
- On 25 of those rows re-run with a long cap, grading took 352 to 988 s with
  the shipped file and 2 to 122 s without the last line
  (`results/review4_slow_*.jsonl`).
- Exit codes are the same in both cases, and all 77 rows have a 1,800 s
  verifier timeout, so I saw no reward change on untouched trees. I did not
  test with an agent's solution applied, where more test processes run.
- For the agent, any command that starts a login shell on these images stalls
  for minutes.

**Effect 2: child shells do not keep the parent's environment.**

Measured on all 2,698 images, by setting a value in the agent's shell and
reading it in a child bash:

| What happens in a child bash | Images |
|------------------------------|--------|
| A `PATH` entry added by the parent is gone (the profile resets `PATH`) | 897 |
| The parent's entry is kept but pushed behind pyenv or venv entries | 211 |
| `NODE_OPTIONS`, `GRADLE_OPTS`, `MAVEN_OPTS` set by the parent are overwritten | 485 |
| `VIRTUAL_ENV` set by the parent is overwritten | 86 |
| A `bash -u` child prints `PS1: unbound variable` or similar to stderr | 1,797 |
| A `bash -eu` child exits with an error before running anything | 4 |

Example on `format-code-task-000820`. A tool is put on `PATH`, then a script
with a bash shebang calls it:

```
port (shipped):   mytool: command not found
upstream bash -lc: mytool ok
```

On `format-code-task-001072` a `python` placed first on `PATH` is used directly
but the script gets the profile's venv Python instead. Upstream uses the
parent's in both places.

- This is what `npm run` does (it adds `node_modules/.bin` to `PATH`, then runs
  the script), and what `tox`, `make` and an activated virtualenv rely on.
- I found no verifier whose exit code changed because of it on untouched trees.

Fix: delete the last line of `BASH_ENV_SCRIPT` and change the test to assert
that a child shell does **not** load the profile again. Nothing else needs it:

- The tool repairs are exported, so child processes inherit them.
- With the line removed, all 11 repaired rows and all 29 login-shell rows still
  run their tests (`results/review4_noreexport_40.jsonl`), no child shell loses
  `PATH`, and there is no stderr noise.

If we skip it: 77 rows spend minutes per grading run, agents on pyenv images
stall whenever they start a login shell, and scripts that rely on an inherited
`PATH` break on a third of the images. None of that happens upstream.

### 2. The cache clean-up deletes the test dependencies on 2 rows

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:81-88` (`CACHE_CLEANUP`),
called from `setup` at line 197.

- This one starts with me. Review 3 suggested deleting `__pycache__` folders
  as a nice-to-have and said it was untested. It is wrong for some images.
- Some images ship their virtualenv **without source files**. For example
  `/testbed/.venv/lib/python3.11/site-packages/pytest/` holds only
  `__pycache__/*.pyc`. The row's own test script copies those `.pyc` files next
  to the packages before it runs pytest.
- The clean-up deletes every untracked `__pycache__`, including those. After
  that the packages are empty folders and pytest cannot start.

| Row | Before this change | At `4b5f3251` |
|-----|--------------------|----------------|
| `format-code-task-001840` | `3 failed, 3 passed` | `No module named pytest.__main__; 'pytest' is a package and cannot be directly executed` |
| `format-code-task-000050` | exit 2, tests collected | same pytest error, exit 1 |

- Confirmed by running both rows with and without the clean-up step
  (`results/review4_pyc_*.jsonl`). Three other rows whose scripts read `.pyc`
  files (`001232`, `001981`, `000401`) end the same way either way.
- The leak this step was meant to close is small: review 3 found hidden-test
  names in such caches on 35 rows, partial matches only.

Fix: stop deleting `__pycache__`. Either drop `-name __pycache__ -o` from the
`find` expression and keep the `.pytest_cache` and `.jest/cache` removal, or
remove the whole step. Update the README sentence and
`test_cleanup_preserves_tracked_caches_and_build_environment` to match.

If we skip it: two rows that worked now return 0 whatever the model does.

### 3. A git hook makes the history strip fail on 2 rows

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:63`
(`git checkout --detach {base_ref}` inside a `set -euo pipefail` script).

- `git checkout` runs the repository's `post-checkout` hook and returns the
  hook's exit code. Two images have a hook that fails (husky and a lint tool
  whose Node modules are missing or that expects a reflog).
- With `set -e` the strip stops there and setup raises
  `Cannot sanitize task Git history`.

| Row | Hook error | Before this change |
|-----|-----------|--------------------|
| `format-code-task-001166` | `git diff-tree ... 'HEAD@{1}' HEAD` → `fatal: log for HEAD is empty` | Passed setup |
| `format-code-task-002071` | `Cannot find module '/workspace/repo/node_modules/cosmiconfig/dist/index.js'` | Failed setup (one of the 618), still fails |

Fix: `git -c core.hooksPath=/dev/null checkout --detach {base_ref}`. Every image
has git 2.30 or newer, so the option exists. With that one change both rows
pass setup and their verifiers run normally.

If we skip it: one row that worked is lost and one stays lost.

### 4. Text printed by a login profile is mistaken for leftover history

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:194-196`.

- The check runs `git rev-list --all --not <base>` and raises if stdout is not
  empty.
- Since the start-up file exists, every shell command loads the image's login
  profile first. On `format-code-task-001621` the profile prints three lines:

  ```
  JAVA_HOME=
  java debugger:     /opt/arthas/as.sh
  maven settings:    /root/.m2/settings.xml
  ```

- So stdout is never empty and setup raises
  `Image history is not truncated at the captured task base commit`. The
  repository itself is clean after the strip (0 commits outside the base).
- This row passed setup before the change. It is the only image whose profile
  prints to stdout.

Fix: decide by exit status, not by stdout. For example run
`[ -z "$(git rev-list -n 1 --all --not <base>)" ]` and raise on a non-zero exit
code. The base-commit capture a few lines above already copes with extra
output by matching a 40-character hash.

If we skip it: one row that worked is lost, and any future image with a chatty
profile fails the same way with a misleading message.

#### The four fixes together

`claude-review-probe-tooling/review4_fix_check.py` applies fixes 1 to 4 as
in-memory patches (the repository is not touched) and runs the real `setup` and
`verified`. On 85 rows (every row named in findings 1 to 4, the 11 repaired
rows, the 29 login-shell rows, the 30 rows that were slow in the full pass, 4
history rows from review 3 and `002295`):

- Setup passes on all except `002295`, which fails closed as documented.
- `001621`, `001166`, `002071`, `001840` and `000050` all reach their tests.
- No pyenv lock message on any row. Grading on the 48 pyenv rows: median 5 s,
  worst 159 s.
- No `command not found` or missing module, apart from the import the hidden
  test of `000050` is meant to fail on.

Raw records: `results/review4_fix_check.jsonl`.

## Should fix

### 5. Row `001217` passes with no change made in 3 of 11 runs

Where: the dataset. `docs/code.md` lists the excluded rows.

- `format-code-task-001217` is fsnotify. Its hidden test `TestRemoveRace` is a
  race test. On the untouched tree it sometimes passes and sometimes crashes
  with a segmentation fault.
- Review 3's pass saw it fail once and counted it as healthy. This pass saw it
  pass. Ten more runs in one sandbox gave rewards
  `0, 0, 0, 1, 1, 0, 0, 0, 0, 0`.
- This is not caused by the change.

Fix: add it to the exclusion table and both example commands in `docs/code.md`
as a flaky free positive. That makes six excluded rows and 2,692 eligible.

If we skip it: one row pays out about a quarter of the time for doing nothing.

## Nice to have

### 6. General grading errors lose their original type

Where: `environments/xiaomi-general/xiaomi_general/taskset.py:513-518`.

- Any error that is not a Verifiers `RolloutError` is replaced by
  `RuntimeError(redact(str(exc)))` with the cause chain cut.
- The trace then records type `RuntimeError` and a five-line traceback that
  points at the `raise`, not at what failed. A `TimeoutError` becomes
  `RuntimeError('')`; a `KeyError('reward_file')` becomes
  `RuntimeError("'reward_file'")`.
- Cutting the chain is reasonable: the trace stores full tracebacks, and a
  sandbox error could carry the command's environment. The cost is that these
  failures are now hard to tell apart.
- Smaller: full-key replacement also rewrites ordinary words when the key is
  short. With `GA_JUDGE_KEY=dummy`, which the smoke runs use, every "dummy" in
  verifier output becomes `<redacted>`. Rewards are not affected.

Fix: keep the type name in the message, for example
`RuntimeError(f"{type(exc).__name__}: {redact(str(exc))}")`.

### 7. `docs/code.md` carries review history

Where: `docs/code.md`, section "Evidence and limits".

- It argues with review 3's wording ("The report's '13 unverifiable' count
  conflicts with its narrative…") and points readers at
  `claude-review-probe-tooling/results/`, which is outside the repository.
- A user of the env needs the exclusion list, the 8 GB note and the known
  limits. The rest is already in `docs/evaluation.md` under "Review 3 live
  evidence".

Fix: keep the exclusion table, the commands and a short limits list in
`docs/code.md`; drop the paragraphs about the review.

## Checked, no finding

- **Rust repair on non-Rust images.** The `/testbed` images share a base that
  has rustup in `/usr/local/cargo`, so the repair adds `/usr/local/cargo/bin`
  to the front of `PATH` on 1,556 images, not only on the 9 Rust rows. That
  folder holds only rustup's own proxies (`cargo`, `rustc`, `rustfmt`, …), so
  nothing in the images is shadowed. Agents on Go and Python rows now have a
  working `cargo`, which they did not have upstream. Harmless as far as I can
  see; appending instead of prepending would be the more careful choice.
- **Strip safety checks.** The refusals for shared object stores, `.keep`
  packs and linked worktrees never fire on the published images. Only the
  promisor check does, on `002295`. They are cheap and fail closed.
- **Cache clean-up cost.** 948 images have cache folders; the largest has
  10,302 and setup still takes 18 s.
- **Profiles are trusted at grading.** An agent can edit `~/.bashrc` and the
  grader's shell will load it. Upstream's `bash -lc` has the same property and
  the README says so.
- **Memory precedence.** No flag gives 8 GB. An explicit
  `--env.agent.runtime.memory 4` gives 4 GB, as the README says.

## Corrections to my earlier reviews

- Review 3 said "ten Rust rows". The ID list has nine. The other agent caught
  this and `docs/code.md` is right.
- Review 3's nice-to-have about deleting `__pycache__` was wrong for images
  with compiled-only packages. Finding 2 is the result.
- Review 3 counted `001217` as healthy from a single run (finding 5).

## Not checked

- The four fixes are tested as in-memory patches, not as commits, and only on
  the 85 rows listed above, not on all images.
- No agent run with the fixes applied. Finding 1's effect on a working agent is
  inferred from shell behaviour, not seen in a transcript.
- Unreachable objects on the 90 images whose `.git` is 400 MB or larger. I
  skipped `fsck` there because it killed sandboxes in review 3. The strip ran
  and the history check passed on all of them.
- Whether a correct solution still passes on the 77 slow pyenv rows within the
  1,800 s timeout. Untouched trees finish in at most 988 s.
- Rows whose verifier runs for 6 to 10 minutes for reasons unrelated to this
  change (`000252`, `000348`, `002606`, `002643`). They take the same time with
  and without the last line of the start-up file.
- A General eval with an independent judge. The other agent's run again used
  the policy model as its own judge.
- Whether Debian's abcmidi version matches Xiaomi's (unchanged since review 1).

## Reproduce

Scripts and raw results are in `claude-review-probe-tooling/`; its README has a
"Review 4" section with the commands.

| Check | Script | Raw results |
|-------|--------|-------------|
| Real setup and verifier on all 2,698 images | `review4_full_probe.py`, `summarize_review4.py` | `results/review4_full.jsonl` |
| Finding 1, grading time with and without the last line | `review4_full_probe.py` with `PROBE_CLEAN=1`, `PROBE_NO_REEXPORT=1` | `results/review4_clean_*.jsonl`, `results/review4_slow_*.jsonl`, `results/review4_pyenv_slow_ids.txt`, `results/review4_child_path_lost_ids.txt` |
| Finding 1, repaired and login-shell rows without the last line | same, `PROBE_NO_REEXPORT=1` | `results/review4_noreexport_40.jsonl` |
| Finding 2, with and without the clean-up | same, `PROBE_NO_CLEANUP=1` | `results/review4_pyc_*.jsonl` |
| Findings 3 and 4, and all four fixes together | `review4_fix_check.py` | `results/review4_fix_check.jsonl` |
| Review 3 finding 5 with no runtime flags | `review4_full_probe.py` with `PROBE_DEFAULTS=1` | `results/review4_no_flags_21.jsonl` |

This review used about 3,100 Prime sandboxes. All were deleted afterwards.
