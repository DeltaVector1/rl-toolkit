# Review 3: Xiaomi env ports, full pass over the Code images

Date: 2026-10-11. Reviewed: `rl-toolkit` branch `Xiaomi-Envs` at `58615037`
(commits `da65e4b2`, `ee2887e6`, `58615037` on top of `66860c24`).
Read-and-test review only. No code was changed and nothing was committed.

All paths are relative to `rl-toolkit/` unless they start with `.references/`
or `claude-review-probe-tooling/`.

This document was first written from samples. It has since been updated with a
pass over all 2,698 Code images, which was the first item in its "Not checked"
list. Numbers below are exact counts unless marked as a sample. The four
follow-up checks and their results are in their own section near the end.

## Summary

The review 2 follow-ups are done correctly. The Code environment has problems
that predate these commits and that reviews 1 and 2 missed, because they only
sampled `base && new` rows.

| # | Severity | Env | One line |
|---|----------|-----|----------|
| 1 | Must fix | Code | 618 rows (23%) abort in setup: their images still carry git history past the task's starting commit |
| 2 | Must fix | Code | On 2,163 rows (80%) the future history, including the reference fix, is still in the git object store and can be read back |
| 3 | Must fix | Code | 29 rows always score 0: the verifier needs a login shell to find the image's Python environment |
| 4 | Should fix | Code | 15 rows are broken whatever the model does: 13 cannot run their verifier, 2 pass on an untouched tree |
| 5 | Should fix | Code | Code tasks declare no resources; 13 rows kill a 4 GB / 10 GB sandbox during verification |
| 6 | Should fix | General | The first 12 characters of the judge API key are saved in every general-agent trace |

Findings 1 and 2 share one fix. Fix 1 to 3 before training on Code.

Row lists for every finding are in `claude-review-probe-tooling/results/`.

## Corrections to my earlier reviews

- **Review 2 overstated coverage.** It said the post-fix Code probe included
  "every row that failed on a missing tool in review 1". It covered 12 of the
  17 flagged rows. The other agent caught this; their count is 11 of 16, the
  difference being how one row is classified. The five missed rows
  (`000169`, `000170`, `000305`, `000947`, `001576`) are now re-run through the
  real task code and none reports a missing tool. `CLAUDE-REVIEW-2.md` is corrected.
- **Review 1 was wrong that `bash -c` and `bash -lc` are equivalent.** It said
  the two gave identical results on 41 images and only the comment needed
  fixing. That sample did not contain an image that sets up its environment in
  a shell profile. Finding 3 is the result.
- **Why findings 1 and 2 were missed.** The first two reviews sampled only from
  the 276 `base && new` rows, all of which use `/testbed`. Almost every row
  that fails the history check uses `/workspace/repo`. Finding 2 needed
  `git fsck`, which I had not run.
- **The first version of this document** estimated finding 1 at about 570 rows
  and finding 3 at about 70. The exact counts are 618 and 29. It also listed
  build residue as "checked, no finding"; the full pass changed that, see
  finding 2 and follow-up check 4.

## Review 2 follow-ups: status

| Item | Verified how | Result |
|------|--------------|--------|
| Terminal Bench tasks picklable (`_no_toolsets`) | Read the diff; new test covers pickle, `with_data` and `deepcopy` | Done |
| Three unsolvable Terminal Bench rows documented with exclusions | Ran the exclusion through the real taskset; validated the TOML against the pinned Prime RL `TrainSourceConfig` | 989 rows load, 986 remain (925 + 61). `--select.exclude.ids` exists in both pinned Verifiers versions. Exclusion wins over inclusion, as the docs say |
| General eval with a judge | Read `outputs/review2-general-accounting-audit` | Reward 0.875, 5 of 6 checks, submission validated, no errors, empty stderr |
| `uv run pytest -q` | Ran it | 73 passed |

One note on the General eval: the judge was the policy model itself
(`zai-org/GLM-5.2-FP8` on the same endpoint, with `GA_JUDGE_KEY=dummy`). That
proves the judge plumbing works after the validator change. It is not an
independent grade, so the 0.875 should not be compared with Xiaomi's numbers.

## Must fix

### 1. 618 Code rows abort in setup

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:121-123`.

- Setup runs `git rev-list --all --not <base>` and raises
  `Image history is not truncated at the captured task base commit` if anything
  comes back. That mirrors upstream's assertion.
- Many published images fail it. They still hold branches, remote-tracking refs
  and tags that are ahead of the task's starting commit, so the reference fix
  can be read with `git log` or `git diff`.
- Two shapes:
  - `/workspace/repo` images: `HEAD` is a synthetic `task base` commit, with
    the original repository's `origin/*` refs still present. Example
    `format-code-task-000820`: 182 refs, 526 extra commits.
  - Some `/testbed` images: release tags or a `main` branch ahead of `HEAD`.
    Example `format-code-task-002737`: `refs/heads/main` is 18 commits ahead.

Counts, all images (`full_pass_probe.py`, 2,688 of 2,698 measured):

| Working directory | Rows | History not truncated |
|-------------------|------|-----------------------|
| `/workspace/repo` | 519 | 498 |
| `/testbed` | 2,169 | 120 |
| Total | 2,688 | 618 (23.0%) |

- The list is `results/history_not_truncated_ids.txt`.
- In the 60-row sweep through the real `setup`, 10 rows raised.
- Upstream fails the same way on these images with its default setting. Its
  error message names the remedy: "set `git_leak_prevention: strip` to truncate
  it per rollout". The port dropped that option and that hint.
- Failing closed is the right default, so this is lost tasks and wasted
  sandboxes, not wrong rewards.

Fix: see "The strip" under finding 2. It fixes all 618.

### 2. The reference fix can be read back from unreachable git objects

Where: the published images. The port's part is
`environments/xiaomi-code/xiaomi_code/taskset.py:121-123`: the history check
only looks at commits reachable from a ref, and
`environments/xiaomi-code/README.md` says so.

- Most `/testbed` images were truncated by deleting refs without garbage
  collection. The commits after the base are still in `.git`, unreachable but
  intact.
- `git fsck --unreachable` lists them and `git show <sha>` prints them. Two
  commands, no special access.
- Upstream's own image requirements say unreachable objects must be removed,
  "otherwise the agent can read the reference fix out of `git log`". These
  images do not meet that requirement, and neither upstream's check nor the
  port's catches it.

Concrete case, `format-code-task-000121` (the task is an imbalance in Sarama's
round-robin balance strategy). It passes the history check. Among its 1,309
unreachable commits are the direct children of the base commit:

```
0cdf4a6c 2020-12-24 Merge pull request #1788 from kzinglzy/even-round-robin-balance-strategy
 balance_strategy.go      | 102 ++++++++++++++++++++++++++++++++++++++++-------
 balance_strategy_test.go |  56 +++++++++++++++++++++-----
```

That is the fix. This is the Code row the other agent's eval scored 1.0 on.
Its transcript does not run `git fsck`, and neither do the other four eval
transcripts under `outputs/`, so I see no sign this has been exploited yet.

Counts, all images:

| Measure | Rows |
|---------|------|
| Unreachable commits outside the base's ancestry | 2,186 (81.3%) |
| Some of them newer than the base commit | 2,163 (80.5%) |
| Future history present, reachable (finding 1) or unreachable | 2,662 (99.0%) |
| Median unreachable commits on affected rows | about 2,000 |

- In at least 248 rows an unreachable blob contains one or more of the hidden
  tests' names. The two I opened were later versions of the repository's own
  test file, sharing one or two test names with the hidden tests. That fits
  "future history is present" rather than a verbatim copy of the hidden tests.
  The scan stopped at 400 blobs per image, so 248 is a lower bound.
- `fsck` timed out on 11 rows and killed the sandbox on 10 very large ones
  (`.git` of 1 to 10 GB), so those 21 are unmeasured.

Fix: the strip below, run on every row, not only on rows that fail the check.
Its `gc --prune=now` deletes the unreachable objects.

#### The strip (fixes findings 1 and 2)

Run this in `setup`, after capturing the base commit and before the existing
assertion. It is upstream's `_strip_future_commits`
(`.references/xiaomi/verl/third_party/mimoagent-osr/src/mimoagent/environments/datasets/base.py:826`)
with one added line and one changed behaviour:

```sh
git checkout --detach $base
git for-each-ref --format="%(refname)" refs/heads | while read r; do git update-ref -d "$r"; done
git remote | xargs -r -n1 git remote remove
git for-each-ref --format="%(refname)" refs/replace refs/notes refs/stash refs/remotes refs/pull | while read r; do git update-ref -d "$r"; done
rm -rf .git/refs/remotes .git/refs/pull .git/refs/replace .git/refs/notes .git/refs/stash   # added
git for-each-ref --format="%(refname)" refs/tags | while read t; do git merge-base --is-ancestor "$t" $base || git tag -d "${t#refs/tags/}"; done
git reflog expire --expire=now --all
git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --prune=now                          # fail setup if this fails
```

- **The added `rm -rf` line is required.** 93 images carry a broken symbolic
  ref `refs/remotes/origin/HEAD` that points at a branch that no longer
  exists. `for-each-ref` does not list it, so upstream's sequence never deletes
  it, and `git gc` then dies with `fatal: bad object refs/remotes/origin/HEAD`.
  Upstream swallows that error, so its strip silently prunes nothing on those
  rows. With the added line, gc succeeds on all 93.
- **Check gc's exit code.** Upstream ignores it. Raise instead, as setup does
  for other failures.
- Do not add `git fsck` to setup as a check. It is slow on large repositories
  and killed the sandbox on the ten largest.

What I measured with this sequence:

| Check | Result |
|-------|--------|
| Rows failing the history check afterwards | 0 of 618 |
| `HEAD`, tree hash or `git status` changed | 0 of 2,688 |
| `git describe` tag or commit distance changed | 0 of 2,688 |
| Unreachable commits left, with the added line | 0 on every row where `fsck` finished; 1 exception below; 11 rows unmeasured (`fsck` timed out) |
| Real `setup` passes and verifier exit code is unchanged | 69 of 69 rows tested |
| Time for the strip step | median 1 s, 95th percentile 17 s, worst 187 s (includes an extra `fsck` of mine) |
| The ten largest repositories (`.git` 1 to 10 GB) | strip completes in 22 s or less |

- Exception: `format-code-task-002295` is a partial clone. Its packs are marked
  `.promisor`, and git never prunes those. After the strip it passes the
  history check but still holds unreachable commits. Exclude it, or remove the
  `.promisor` marker files before gc (not tested).
- The 69 rows are 10 from the first sample plus 59 chosen because they have
  ancestor tags, which is where version detection could be affected. Exit
  codes match on all 69. Test summaries differ on 9, all benign: a `(cached)`
  suffix, timestamps, a warning count, or which lines fall inside the output
  window.

Check after fixing: `format-code-task-000820`, `format-code-task-001525` and
`format-code-task-002737` must pass setup. On `format-code-task-000121`,
`git fsck --unreachable --no-reflogs | grep -c commit` must print 0 after setup.

If the strip is not wanted, the alternative is to exclude the 618 rows in
`results/history_not_truncated_ids.txt`. That does nothing for finding 2.

### 3. 29 Code rows always score 0 without a login shell

Where: `environments/xiaomi-common/xiaomi_common/runtime.py:26`
(`argv = ["bash", "-c", command]`), used by `XiaomiCodeTask.verified`.

- Upstream's Kubernetes backend runs every command as `bash -lc`
  (`.references/xiaomi/verl/third_party/mimoagent-osr/src/mimoagent/environments/kubernetes.py:474`).
  The Code images were built for that.
- Some images set up their Python environment in a shell profile. Example
  `format-code-task-001072`: `/root/.bashrc` contains
  `source /testbed/.venv/bin/activate` under the comment "Activate
  openfisca-france venv for all shells". Others initialise pyenv, or add
  `~/.cargo/bin` or `~/.local/bin`.
- Under `bash -c` none of that runs. The test script then either cannot find
  `pytest` (exit 127) or runs the system Python, which lacks the project's
  dependencies. Both become reward 0.

Counts, all images:

- In 224 rows (8.3%) a tool resolves differently in a login shell. In 102 a
  tool exists only there (`pytest` 89, `bundle`/`rake`/`ruby` 6, `uv` 5,
  `cargo` 3). In no row is a tool lost in a login shell.
- I ran the real test command under both shells on all 224
  (`login_verifier_probe.py`). 29 fail for an environment reason under
  `bash -c` and run their tests normally under `bash -lc`: 14 exit 127
  (`pytest`, `py.test`, `nosetests`, `python2` not found), 14 hit a missing
  module under the system Python, and 1 cannot collect its test there. No row
  is worse under `bash -lc`.
- The list is `results/login_shell_required_ids.txt`.
- 485 `/workspace/repo` images also set `GRADLE_OPTS`, `MAVEN_OPTS`,
  `NODE_OPTIONS` and `TEST_JVM_OPTS` (heap sizes) only in login shells. On 35
  of their Java, JavaScript and TypeScript rows the verifier's exit code was
  the same under both shells, so I found no effect from those.

Seen in live evals (`outputs/review3-code-001072`, `outputs/review3-code-001138`):

| Row | What the agent did | Verifier | Reward |
|-----|--------------------|----------|--------|
| `001072` | Found and used `.venv/bin/python` by itself; ran out of turns | exit 127, `pytest: command not found` | 0.0 |
| `001138` | Found `/root/.pyenv/versions/2.7.18/bin/python` by itself; finished | `ModuleNotFoundError: No module named 'django'` | 0.0 |

Fix, two options:

- **A, smallest.** Run the Code verifier's test command with `bash -lc`: a
  `login: bool = False` parameter on `run_shell` and `login=True` at the one
  call in `XiaomiCodeTask.verified`. This fixes the rewards. The agent's own
  shell stays non-login.
- **B, also fixes the agent's shell.** Leave `run_shell` alone. In `setup`,
  write a three-line file into the sandbox, and add `BASH_ENV` pointing at it
  to `XiaomiCodeTask.runtime_env`:

  ```sh
  unset BASH_ENV
  [ -r /etc/profile ] && . /etc/profile
  if [ -r ~/.bash_profile ]; then . ~/.bash_profile; elif [ -r ~/.bash_login ]; then . ~/.bash_login; elif [ -r ~/.profile ]; then . ~/.profile; fi
  ```

  Non-interactive `bash -c` sources `$BASH_ENV`, and both the stock harness's
  agent shell (`verifiers/v1/harnesses/utils/core.py:163`) and the verifier
  inherit the task's runtime environment. So both get what upstream's
  `bash -lc` gave them. A missing file is ignored, so commands that run before
  the file is written are unaffected.

  I tested B with probes, not through a live eval. On 2,687 of 2,688 images the
  resulting environment and tool lookup are identical to `bash -lc` (the one
  difference is a randomly named temp directory). On all 224 rows above the
  verifier ends the same way as under `bash -lc`.

I recommend B because it restores what the agent had upstream as well. A is
enough to stop the wrong rewards. Either way leave General as it is: its images
define their environment with `ENV`.

Check after fixing: `format-code-task-001072` and `format-code-task-001138` on
untouched images must report test failures, not `command not found` or a
missing module.

## Should fix

### 4. 15 Code rows are broken whatever the model does

Where: the dataset and images. The port loads them
(`environments/xiaomi-code/xiaomi_code/taskset.py`, `XiaomiCodeTaskset.load`).

I ran every row's verifier once on its untouched image, with the Docker PATH
and a login shell (`verifier_health_probe.py`). The hidden tests should fail.

| Outcome | Rows |
|---------|------|
| Tests fail, as expected | 2,663 |
| Test command still running at the 240 s cap | 7 |
| Verifier cannot run: exit 127 | 13 |
| Verifier passes with no change made | 2 |
| Sandbox died at 4 GB / 10 GB (finding 5) | 13 |
| Test patch failed to apply | 0 |

**13 rows cannot run their verifier** (`results/verifier_cannot_run_ids.txt`):

- Ten are Rust rows: `cargo` or `rustup` not found. The binaries are in
  `/usr/local/cargo/bin` and need `RUSTUP_HOME=/usr/local/rustup` and
  `CARGO_HOME=/usr/local/cargo`. That is the official Rust image's `ENV`, lost
  when the images were flattened. With those three set, the three rows I tried
  (`001083`, `001649`, `001652`) run their tests.
- `003041`: `javac` not found. It exists at `/opt/java/openjdk/bin`.
- `002879`: `swagger` not found. `002099`: `okexchaincli` not found (it is at
  `/root/go/bin`). `001227`: "Required offline Ruby gems are not installed".

**2 rows give a free reward** (`results/verifier_passes_untouched_ids.txt`):
`format-code-task-001225` and `format-code-task-001597` exit 0 on an untouched
tree in four runs out of four.

Fix: document these 15 as known-bad and exclude them from training and reported
scores, the same way as the three Terminal Bench rows. Optionally repair the
ten Rust rows by setting the three variables when `cargo` is missing and
`/usr/local/cargo/bin/cargo` exists. Do not set them unconditionally: three
images keep Rust in `/root/.cargo` instead.

If we skip it: 13 rows return a constant 0 and 2 a constant 1.

### 5. Code tasks declare no resources

Where: `environments/xiaomi-code/xiaomi_code/taskset.py`, `XiaomiCodeTaskset.load`
(no `resources=` on `CodeData`).

- Without flags, Prime's defaults apply: 1 CPU, 2 GB memory, 5 GB disk. The
  docs' example command passes 2 CPU, 4 GB, 10 GB.
- Upstream's Code config allows up to 4 CPU and 8 GiB
  (`.references/xiaomi/verl/config/agent/code/mini-bash.yaml`).
- In the verifier pass at 2 CPU / 4 GB / 10 GB, the sandbox died on 13 rows
  while the tests ran, twice each (`results/sandbox_died_4gb_10gb_ids.txt`).
  They are large Go and Gradle builds.
- At 4 GB with a 20 GB disk the same 13 died again. At 8 GB / 20 GB all 13 ran
  normally. So memory is the limit, and 8 GB matches upstream.

Fix: set `resources=vf.TaskResources(memory=8)` on Code tasks, or state 8 GB as
the minimum in the docs. Explicit runtime flags still override.

If we skip it: those rows end as sandbox errors, and more will at the 2 GB default.

### 6. Part of the judge API key is saved in every general-agent trace

Where: `environments/xiaomi-general/xiaomi_general/taskset.py:475-476`.

- Xiaomi's released `verify.py` prints its judge configuration, including the
  first 12 characters of the key:
  `[judge] chain[0]: model=… api=chat key=<first 12 chars>...`
  It prints the same for a fallback key when it switches.
- The port stores the last 10,000 characters of verifier stdout in
  `trace.info["verifier"]["stdout"]`. Traces are written under `outputs/` and
  are pushed unless `--no-push` is given.
- Seen in `outputs/review2-general-accounting-audit`: the stdout begins
  `[judge] chain[0]: model=zai-org/GLM-5.2-FP8 api=chat key=dummy...`.
- With a real key, 12 characters is a small part of a long provider key but a
  large part of a short self-hosted one. Either way it is a credential fragment
  in a file meant to be shared.

Fix: redact before storing, for example
`re.sub(r"key=\S+", "key=<redacted>", text)` applied to the verifier's stdout
and stderr in the general-agent branch. One line and one test.

## Follow-up checks

These are the four items the first version of this document listed as not
checked.

### Check 1: a full pass over all 2,698 Code images

Done, in two passes.

- `full_pass_probe.py`: git history, the strip, shell environments and leftover
  traces of the hidden tests, on every image. 2,688 measured. On the other 10
  my probe's `git fsck` killed the sandbox (`.git` of 1 to 10 GB); the strip
  was tested on those separately and completes.
- `verifier_health_probe.py`: every row's verifier, once, on its untouched
  image. All 2,698 rows have a result (13 needed a bigger sandbox).

What changed against the sample estimates:

| | First version (sample) | Full pass |
|---|---|---|
| Finding 1, rows aborting in setup | about 570 | 618 |
| Finding 3, rows needing a login shell | about 70 | 29 |
| Unreachable future history | not looked for | 2,163 rows |
| Rows broken regardless | not looked for | 15 |

### Check 2: does the strip break tag-based version detection?

No effect found.

- `git describe --tags --long` gives the same tag and the same commit distance
  before and after the strip on all 2,688 images. The strip only deletes tags
  that are not ancestors of the base, and `describe` only uses ancestors.
- On 1,442 images the abbreviated hash in the describe output gets one
  character shorter, for example `v1.27.2-7-g0d9d5725` becomes
  `v1.27.2-7-g0d9d572`. Garbage collection leaves fewer objects, so git needs
  fewer characters. A version string built from `git describe` at test time
  would change by that one character.
- On 59 rows picked for having ancestor tags, the verifier's exit code is the
  same before and after.
- `format-code-task-000971` is the hardest case I could find: its hidden tests
  are a versioneer test suite. The same four tests fail before and after the
  strip, across three runs.
- 519 images sit on a named branch (`master`) before the strip and on a
  detached `HEAD` after. Nothing I ran depended on the branch name. A test that
  reads it would see the difference.

### Check 3: the agent's own shell

Confirmed, and it can be fixed from the port.

- The stock bash harness runs each agent command as
  `subprocess.run(["bash", "-c", command])`
  (`verifiers/v1/harnesses/utils/core.py:163`), inheriting the harness
  program's environment. No login profile is read.
- In the two live evals on affected images the agent coped: it found the right
  interpreter by itself in both. So the cost on the agent's side is some wasted
  turns, not a blocked task. The reward was then zeroed by the verifier
  (finding 3).
- The earlier statement that the port "does not control" the agent's shell was
  too strong. The harness inherits the task's `runtime_env`, so option B in
  finding 3 (`BASH_ENV`) reaches it.

### Check 4: upstream's build-residue cleanup

Partly a finding. The serious part is finding 2.

- Xiaomi's shipped Code configs (`.references/xiaomi/verl/config/agent/code/*.yaml`)
  set `anti_hack_cleanup: true`. The port has no equivalent.
- What I looked for: the names of hidden tests (taken from each row's test
  patch, and absent from every tracked file) inside files git does not track,
  and inside unreachable git objects. 1,982 rows have at least one usable name.
- Unreachable objects: this is finding 2.
- Untracked or ignored files: 35 rows have a hit. By file: Python bytecode 38,
  `.pytest_cache` 18, Jest cache 7, `.build_env/test_command.sh` 3, other 15.
  These are leftovers from tests run while the image was built. In the two rows
  I opened, the file held 1 of 25 and 3 of 9 hidden-test names, so they look
  like an earlier or upstream version of the tests, not a full copy.
- No image has `/tests`, `/logs` or `.build_env/task_description.md`.
- Porting upstream's purge as written would be unsafe. Its `git clean -fdx`
  would remove something on 1,815 images, including
  `.build_env/test_command.sh`, which 843 test scripts call, and untracked
  build outputs.

Fix, nice to have: at setup, delete `__pycache__` directories, `.pytest_cache`
and `.jest/cache` under the repository. They are regenerated on demand. Not
tested.

## Checked, no finding

- **No test patch fails to apply.** All 2,698 apply cleanly to their image.
- **PATH fix.** Under the Docker PATH and a login shell, the only missing-tool
  failures left are the 13 rows in finding 4.

## Not checked

- The fixes themselves. Nothing here was applied to the port. Option B of
  finding 3 was tested with probes, not through a live eval.
- The seven rows whose test command runs longer than 240 seconds on an
  untouched tree. Re-run with a 1,700 second cap at 4 GB: `000815` and `000252`
  still had not finished (the verifier's own limit is 1,800 seconds), `001939`
  was killed, `002606` segfaults in a scipy test, `000183` and `002643` fail
  normally, and `000348` lost its sandbox. See
  `results/verifier_health_reruns.jsonl`.
- Whether 8 GB is enough for every row with an agent working in the sandbox.
  Finding 5 only ran the verifier.
- A General eval with an independent judge model.
- Whether Debian's abcmidi version matches the one Xiaomi trained with. Music
  code is unchanged since review 1.

## Reproduce

Scripts and raw results are in `claude-review-probe-tooling/`; its README has a
"Review 3" section with the exact commands.

| Check | Script | Raw results |
|-------|--------|-------------|
| Findings 1 and 2, checks 2 and 4: every image | `full_pass_probe.py`, `summarize_full_pass.py` | `results/full_pass.jsonl`, `results/history_not_truncated_ids.txt` |
| The strip's added line | `strip_fixed_probe.py` | `results/strip_fixed.jsonl` |
| The strip on the ten largest repositories | `strip_heavy_probe.py` | printed by the script |
| The strip does not change verifier results | `strip_effect_probe.py` | printed by the script |
| Finding 3: verifier under `bash -c`, `bash -lc` and `BASH_ENV` | `login_verifier_probe.py` | `results/login_verifier_all.jsonl`, `results/login_verifier_opts.jsonl`, `results/login_shell_required_ids.txt` |
| Finding 3 and check 3: live evals | eval command in the README | `rl-toolkit/outputs/review3-code-001072`, `rl-toolkit/outputs/review3-code-001138` |
| Findings 4 and 5: every verifier | `verifier_health_probe.py` | `results/verifier_health.jsonl`, `results/verifier_health_reruns.jsonl`, `results/verifier_cannot_run_ids.txt`, `results/verifier_passes_untouched_ids.txt`, `results/sandbox_died_4gb_10gb_ids.txt` |
| Finding 6 | read from `rl-toolkit/outputs/review2-general-accounting-audit` | the trace |
| First samples (superseded by the full pass) | `postfix_probe.py`, `hygiene_probe.py`, `login_env_probe.py`, `pyc_residue_probe.py` | `results/postfix_code3.jsonl`, `results/hygiene.jsonl`, `results/login_env.jsonl`, `results/login_verifier.jsonl` |

All probe sandboxes were deleted afterwards.
