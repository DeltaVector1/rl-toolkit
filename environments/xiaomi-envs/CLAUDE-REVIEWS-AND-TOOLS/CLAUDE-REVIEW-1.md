# Review 1: Xiaomi env ports (Code, General, Music)

Date: 2026-10-11. Reviewed: `rl-toolkit` branch `Xiaomi-Envs` at `596e07d0`.
Read-and-test review only. No code was changed and nothing was committed.

All paths below are relative to `rl-toolkit/` unless they start with `.references/`.

## Summary

Two bugs make large parts of Code and General return reward 0 no matter what
the model does. Fix findings 1 and 2 first. Music is correct.

| # | Severity | Env | One line |
|---|----------|-----|----------|
| 1 | Must fix | Code | Verifier tools in `/usr/local/bin` are not on PATH in Prime sandboxes |
| 2 | Must fix | General (Terminal Bench) | The port's own install in `/app` trips the tasks' anti-cheat guard; all 64 rows score 0 |
| 3 | Should fix | General (general_agent) | A virtualenv in the workspace zeroes the whole submission |
| 4 | Should fix | General (Terminal Bench) | Declared limits ignored: no-internet, agent timeout, CPU, memory, disk |
| 5 | Should fix | General (Terminal Bench) | Hidden tests are reachable by a root agent through the tool server |
| 6 | Nice to have | General | `install_common` depends on private Verifiers functions |
| 7 | Nice to have | Code | Verifier stdout/stderr stored untruncated in the trace |
| 8 | Nice to have | Code | One row has a test file with no `diff --git` header, so it is never reset |

## How this was tested

- `uv run pytest -q`: 59 passed.
- Upstream sources compared: `.references/xiaomi/verl/third_party/mimoagent-osr/src/mimoagent/environments/datasets/{opensource_code,base}.py`,
  `.references/xiaomi/verl/recipes/general/general_agent/environment.py`,
  `.references/xiaomi/verl/recipes/design/music/scorer/`.
- Prime sandboxes on real task images, driven through `verifiers.v1.runtimes.provision_runtime`
  with `PrimeConfig`, so commands ran exactly the way the port runs them (`runtime.run(argv, env)`).
- One full live eval of a Terminal Bench row against GLM-5.2. Output is in
  `outputs/review-tb-0036/` (gitignored).
- Docker Hub registry API for image configs (no pulls of layers).
- All probe sandboxes were deleted afterwards (`prime sandbox list` shows 0).

The probe scripts and raw results are in `claude-review-probe-tooling/`, next to
this file. Its `README.md` maps each script to a finding and gives the exact
command. Each finding below names the script that backs it.

| Finding | Script | Raw results |
|---------|--------|-------------|
| 1 | `path_probe.py`, `summarize_path_probe.py`, `shell_probe.py`, `image_probe.py`, `registry_env_probe.py`, `tool_count_probe.py` | `results/path_probe.jsonl`, `results/shell_probe.jsonl` |
| 2 | `guard_probe.py`, plus the live eval command in the README | `rl-toolkit/outputs/review-tb-0036/` |
| 3 | `validator_probe.py` | printed by the script |
| 8 | `code_probe.py`, `code_probe2.py` | printed by the scripts |
| Music | `music_probe.py` | printed by the script |

Findings 4 to 7 came from reading code and data and have no script.

## Must fix

### 1. Code verifiers cannot find their tools

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:73-78` (the Go repair),
`environments/xiaomi-common/xiaomi_common/runtime.py:25` (comment).

Cause:

- The published Code images are flattened: one layer, no `ENV` at all. Checked
  25 image configs through the registry API; all 25 have one layer and no `Env`.
- Inside the images, `go`, `gofmt`, `node`, `npm`, `npx`, `yarn`, `dev-switch`
  and `yq` live in `/usr/local/bin`.
- Docker and Kubernetes (what Xiaomi ran on) give an image with no PATH the
  default `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`.
- Prime gives `/usr/sbin:/usr/bin:/sbin:/bin`. `/usr/local/bin` is missing.
- The existing repair only adds `/usr/local/go/bin` when `go` is missing. Every
  other tool stays missing.
- The verifier then exits 127, which the port scores as reward 0 (a wrong
  answer), not as an infrastructure error. The agent's own shell has the same
  broken PATH during the rollout.

Evidence. I applied each task's test patch to an untouched image and ran the
`base` half of its test command (`bash …/mimo_test_command.sh base`), which
should pass before any fix. 42 rows sampled from the 276 `base && new` rows,
deliberately weighted toward non-Python languages; 41 ran (one sandbox returned
HTTP 502).

| Mode | Pass | Tests fail | Tool not found |
|------|------|------------|----------------|
| Port as written (`bash -c`, Go repair) | 23 | 2 | 16 |
| Docker default PATH, `bash -c` | 34 | 7 | 0 |
| Docker default PATH, `bash -lc` | 34 | 7 | 0 |

- "Tool not found" = 14 rows exiting 127 plus 2 rows that exit 1 after a
  missing `dev-switch` / `julia`.
- Missing tools seen: `npx` 6, `npm` 5, `dev-switch` 2, `bundle` 2, `node` 1,
  `yarn` 1, `julia` 1.
- 11 rows fail as written and pass with the Docker PATH. No row goes the other way.
- The 7 remaining failures under the Docker PATH are real test failures on an
  unfixed tree (for example Go packages that no longer compile once the new
  tests are applied). They are not environment problems.
- Dataset-wide, at least 295 of the 2,698 test scripts call
  `npm`/`npx`/`node`/`yarn`/`pnpm`/`dev-switch` directly with no PATH setup of
  their own. Another 843 scripts hand off to an in-image `.build_env` script
  that I could not analyse statically.
- Example rows that fail as written and pass with the fix:
  `format-code-task-000604` (npm, npx), `format-code-task-000601` (npm),
  `format-code-task-002905` (npx), `format-code-task-002679` (node),
  `format-code-task-000020` (julia).

Fix:

- Add to `XiaomiCodeTask`:

  ```python
  def runtime_env(self):
      # Published Code images carry no ENV; Docker supplies this default, Prime does not.
      return {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}
  ```

  `GeneralTask.runtime_env` already uses this hook. The value reaches both the
  harness (agent) and the verifier through `Runtime.process_env`.
- Delete the Go repair (`taskset.py:73-78`), its test
  (`tests/test_code.py::test_setup_repairs_only_missing_installed_go_for_agent_and_grader`),
  and the Go paragraph in `environments/xiaomi-code/README.md`. With the Docker
  PATH, `go` resolves to `/usr/local/bin/go` on its own.
- Correct the comment at `runtime.py:25`. It says `bash -c` matches upstream,
  but upstream runs `timeout N /bin/bash -lc …`
  (`.references/xiaomi/verl/third_party/mimoagent-osr/src/mimoagent/environments/kubernetes.py:474`).
  The two flags gave identical results on all 41 images, so the flag itself can
  stay; only the comment is wrong.
- Update `tests/test_runtime.py` and `docs/evaluation.md` where they describe
  the Go repair.

Check after fixing: re-run `format-code-task-000604` and
`format-code-task-000121` live. The verifier output must contain no
`command not found`.

Reproduce: `claude-review-probe-tooling/path_probe.py` with the IDs in
`results/path_probe_ids.txt`, then `summarize_path_probe.py` for the table.
The script's `port` mode hard-codes the old Go repair, so after the fix compare
against its `docker-c` column rather than re-reading `port`.

General and Music are not affected: their images define `ENV PATH`, and Prime
honours it (checked on `general-agent-env-1`: PATH starts with `/usr/local/bin`).

### 2. All 64 Terminal Bench tasks score 0

Where: `environments/xiaomi-general/xiaomi_general/taskset.py:271-274`
(setup calls `install_common`), `:367-383` (terminal scoring), `:226-227`
(`toolsets` is the same for every row).

Cause:

- Every Terminal Bench row ships `anti_hack_guard.py`, run at the start of
  `test.sh`. It walks all of `/app` with `rglob("*")`, hidden directories
  included, and writes reward 0 if it finds a file named `conftest.py`,
  `sitecustomize.py`, `usercustomize.py`, `pytest.ini`, `.coveragerc`,
  `tox.ini`, `pyproject.toml` or `*.pth` that is not in the task's pristine
  manifest.
- These rows use workdir `/app`. Verifiers installs a colocated tool server
  under the workdir: `/app/.vf-src`, `/app/.vf-tmp`, `/app/.vf-uv-cache`,
  `/app/.vf-venv`. `install_common` adds to the same venv.
- A venv and a uv cache are full of `*.pth`, `conftest.py` and
  `pyproject.toml` files, so the guard always fires.
- The tool server is pointless here: `GeneralTools.register` registers nothing
  for `terminal_bench` rows.
- No Terminal Bench row had been run live before this review.

Evidence:

- Live eval of `candidate-0036-software-data-engineering`
  (`outputs/review-tb-0036`): rollout completed with no errors, 
  `verifier_reward = 0.0`, verifier return code 0, stderr:
  `anti_hack_guard: REJECT planted_interpreter_hook:.vf-uv-cache/archive-v0/L_21nGkBuBs_InAb/sympy/conftest.py`
- Same image, clean `/app`, tests staged under `/tests`:
  `python3 /tests/anti_hack_guard.py` exits 0.
- Same image with only an empty `/app/.vf-venv/lib/site-packages/_virtualenv.pth`
  added: exits 1 with `REJECT planted_interpreter_hook:.vf-venv/lib/site-packages/_virtualenv.pth`.

Fix, smallest:

- In the terminal branch of `verifier_reward`, before `bash /tests/test.sh`,
  remove the four directories:
  `rm -rf /app/.vf-src /app/.vf-tmp /app/.vf-uv-cache /app/.vf-venv`
  (use the task's workdir, not a literal `/app`). The harness and tool servers
  are already closed when scoring starts (`Rollout.close` in
  `verifiers/v1/rollout.py`).

Fix, better (also solves finding 5 and removes about 20 seconds of setup per rollout):

- Start no tool server for `terminal_bench` rows. The rollout calls
  `self.task.toolsets(self.task.config)` on the task instance
  (`verifiers/v1/rollout.py:284`), so a `GeneralTask` subclass for terminal
  rows that returns `[]` from `toolsets` is enough at rollout level. I did not
  check whether the taskset-level code accepts a subclass instance; verify that
  before relying on it.
- Then drop the `install_common` call in the terminal branch of `setup`.

Check after fixing: re-run `candidate-0036-software-data-engineering` live. The
verifier stderr must not contain `anti_hack_guard: REJECT`, and
`/logs/verifier/reward.txt` must come from pytest.

Reproduce: `claude-review-probe-tooling/guard_probe.py` for the clean-versus-venv
guard check; the live eval command is in that folder's README.

## Should fix

### 3. A virtualenv in the workspace zeroes a General submission

Where: `environments/xiaomi-general/xiaomi_general/taskset.py:77-101`
(`VALIDATE_SUBMISSION`, the failing checks are at `:94-98`).

- The validator runs in `finalize` as `xiaomi_agent` and walks
  `/work/workspace`. Any symlink that resolves outside the workspace, any
  dangling symlink, any unreadable file and any special file makes it exit 42,
  which becomes reward 0 with `invalid_submission`.
- `python3 -m venv .venv` creates `bin/python` symlinks that point outside the
  workspace. Agents do this often, for example after a pip
  "externally-managed-environment" error.
- Reproduced locally by running `VALIDATE_SUBMISSION` on a temp workspace
  (`claude-review-probe-tooling/validator_probe.py`):
  - with a venv: exit 42, `Submission path is outside the workspace or unreadable: .venv/bin/python3`
  - with one dangling symlink: exit 42, `[Errno 2] No such file or directory`
- Upstream has no such check. Upstream's verifier runs in a separate sidecar
  container, where a link to `/usr/bin/python3` would simply resolve inside the
  sidecar or dangle.
- The check exists for a real reason: here the verifier runs as root in the
  same sandbox, so a link into `/work/_verifier` could feed the answer key to
  the judge. Keep that protection.

Fix: for symlinks that leave the workspace or dangle, remove the link and
continue, instead of failing the whole submission. The validator runs as the
owner of those files, so it can unlink them. That matches what the upstream
sidecar would have seen and still blocks reads of protected files. Update the
tests `test_submission_paths_reject_private_aliases_including_hidden_names` and
`test_invalid_submission_scores_zero_without_privileged_upload_or_fallback`
to match.

### 4. Terminal Bench rows ignore their declared limits

Where: `environments/xiaomi-general/xiaomi_general/taskset.py:476-492`.

- All 64 rows declare `allow_internet: false`, `agent_timeout_sec: 900.0`,
  `cpus: 1`, `memory_mb: 2048`, `storage_mb: 10240`.
- The port sets only `TaskTimeout(setup=900, scoring=…)`. Nothing else is mapped.
- Concrete case: `candidate-0036-software-data-engineering` asks the agent to
  repair a "frozen, pre-fix Snakemake source tree" in an "offline integration
  replay". With internet, the agent can download the fixed release.

Fix: map the fields onto `vf.TaskTimeout(agent=…)`, `vf.TaskResources(cpu, memory, disk)`
(memory and disk are in GB) and the task's `network_allow` / `network_block`.
I did not check how Verifiers expresses "no internet" on Prime; read
`TaskData.network_allow` in `verifiers/v1/task.py` first. Tool-server and model
routes must stay reachable; `runtime.prepare_execution` handles the framework
routes.

### 5. Hidden tests are reachable on Terminal Bench rows

Where: same code as finding 2.

- `GeneralData.instance` carries `tests_files`: the base64 hidden tests and
  expected fixtures.
- A colocated tool server gets `VF_STATE_SECRET` in its environment
  (`verifiers/v1/mcp/launch.py:320`). With that secret, `GET /task` returns the
  full task data (`verifiers/v1/interception/server.py`, `handle_task_get`).
- On Terminal Bench rows the agent runs as root in the same sandbox as that
  server, so it can read `/proc/<pid>/environ` and fetch the tests.
- `general_agent` rows are not exposed: the agent is `xiaomi_agent`, the server
  is root, and those rows carry no answers in task data.
- This comes from reading the code. I did not exploit it.

Fix: the "better" option in finding 2 (no tool server on terminal rows) removes it.

## Nice to have

### 6. `install_common` depends on private Verifiers functions

Where: `environments/xiaomi-general/xiaomi_general/taskset.py:178-197`.

- It imports `_cached_package` and `_package_dir` from `verifiers.v1.mcp.launch`.
- `docs/evaluation.md` says live runs used Prime RL's Verifiers at `2cb0a6d`,
  while the workspace pins `bf2834c`. Private names can change between them.
- The function exists only because `xiaomi-common` is a separate, unpublished
  package, and Verifiers bundles only the toolset's own package into the sandbox.
- Same cause: the three envs cannot be installed standalone from a hub.

Fix: none needed now. If it breaks on a Verifiers bump, copy the roughly 100
lines of `xiaomi-common` into `xiaomi-general` and delete `install_common`.

### 7. Code verifier output is stored whole

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:120-121`.

- Full stdout and stderr go into `trace.info["verifier"]`. Verbose runs
  (`go test -v`, Jest) can be very large.
- General already keeps `stdout[-10000:]` and `stderr[-4000:]`.

Fix: use the same tails.

### 8. One Code row has a test file with no `diff --git` header

Where: `environments/xiaomi-code/xiaomi_code/taskset.py:22-39`, row
`format-code-task-000989`.

- That patch starts with a file that has only `--- /dev/null` / `+++ b/pkg/common/moerr/error_wrapping_test.go`.
  `touched_files` reads `diff --git` lines only, so this path is never reset.
- Upstream has the same gap.
- If the agent creates that file, `git apply` fails with "already exists" and
  the rollout raises an infrastructure error.

Fix: also collect `+++ b/<path>` lines in `touched_files`.

## What held up

- Music scorer: all six files (`__init__.py`, `core.py`, `feats.py`,
  `pipeline.py`, `score.py`, `baselines/ref_full4k.json`) are byte-identical to
  `.references/xiaomi/verl/recipes/design/music/scorer/`.
- Music end to end: upstream's health-check tune
  (`X:1\nT:t\nM:4/4\nL:1/8\nK:C\nCDEF|GABc|c4|z4|`, from
  `recipes/design/music/scorer_precheck.py`) scores 0.198 through
  `MusicTask.setup` + `MusicTask.music_score` in a Prime sandbox. Upstream
  expects "around 0.2". The same tune in a fenced block also scores 0.198;
  a reply with no ABC and an empty reply both score 0.0. Setup took 2.8 s and
  installs abcmidi `20250216+ds-1`.
- Code reward logic matches upstream's reset → `git apply` → run → exit-code
  contract. The documented differences are deliberate: infrastructure failures
  raise instead of scoring 0, and a failed `git checkout` during reset is an
  error instead of being ignored.
- Code patch parsing: across all 2,698 rows, `touched_files` agrees with the
  `---`/`+++` lines except for empty new files (correctly included) and the
  single row in finding 8. No patch has a path containing spaces. The one
  `diff --git ab/…` header (row `format-code-task-000121`) is handled.
- General reward parsing, env forwarding and the binary full-score threshold
  match upstream. The released verifier helpers skip hidden directories, so
  `.vf-*` in `/work/workspace` does not affect `general_agent` grading.
- `download_general_task` is fast once cached (0.02 to 0.09 s per call), so the
  repeated calls per rollout are not a problem.
- Style: the code is mostly tight. Formatting is uneven (250-character lines in
  the Code taskset, 88-column formatting in General) and there is no lint config.

## Not checked

- A `general_agent` row live in this review. `GA_JUDGE_KEY` was not set in my shell.
- The other roughly 2,650 Code images. The PATH sample was 41 rows, weighted
  toward non-Python languages, so 16 of 41 is not a dataset-wide rate.
- The 843 Code scripts that delegate to an in-image `.build_env` script.
- Whether Debian's abcmidi version matches the one Xiaomi trained with. The
  version changes the warning counts that feed the scorer's reject gate.
- The fixes themselves. Nothing here was applied or re-run.
