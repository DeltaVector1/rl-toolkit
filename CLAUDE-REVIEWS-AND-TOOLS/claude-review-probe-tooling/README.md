# Probe tooling for the CLAUDE-REVIEW documents

The scripts used to produce the evidence in `../CLAUDE-REVIEW-1.md`,
`../CLAUDE-REVIEW-2.md` and `../CLAUDE-REVIEW-3.md`, plus the raw results. Reviews 2
and 3 each have their own section near the end. They are throwaway probes, not part of the port: nothing here is imported
by the environments or run by the test suite.

Run everything from `rl-toolkit/` so the workspace venv (pinned Verifiers, the
`xiaomi_*` packages, pyarrow) is used. `DATA` below is `../.references/xiaomi/data`.

```sh
cd rl-toolkit
P=../claude-review-probe-tooling
DATA=../.references/xiaomi/data
```

## Cost and safety

- Scripts marked **sandbox** create Prime sandboxes, one per task image, labelled
  `xiaomi-review`. They are torn down when the script exits. Check afterwards with
  `prime sandbox list --plain`.
- `path_probe.py` runs up to 10 sandboxes at once and each test command up to three
  times with a 1200 s limit, so a full 42-row run takes several minutes.
- `registry_env_probe.py` makes anonymous Docker Hub requests, which are rate limited.
- Nothing here needs a model endpoint or a judge key.

## Finding 1 (Code: tools not on PATH)

| Script | Needs | What it shows |
|--------|-------|---------------|
| `path_probe.py` | sandbox | Runs the `base` half of each test command on an untouched image three ways: the port as written, Docker default PATH with `bash -c`, and with `bash -lc`. The main evidence. |
| `summarize_path_probe.py` | nothing | Turns a `path_probe.py` result file into the table in the review. |
| `shell_probe.py` | sandbox | Earlier, narrower run: `bash -c` vs `bash -lc` with Prime's own PATH. Shows the login shell does not help. |
| `image_probe.py` | sandbox | Prints, inside an image: PID 1 environment, `/usr/local/bin` contents, where each toolchain lives, and what the profile files set. |
| `registry_env_probe.py` | network | Reads image configs from Docker Hub: Code images are one layer with no `ENV`. |
| `tool_count_probe.py` | local | Counts test scripts that call `/usr/local/bin` tools with no PATH setup (295 of 2,698). |

```sh
uv run python $P/path_probe.py $DATA /tmp/path_probe.jsonl $(cat $P/results/path_probe_ids.txt)
python3 $P/summarize_path_probe.py $P/results/path_probe.jsonl
uv run python $P/image_probe.py format-code-task-000604 format-code-task-000121
python3 $P/registry_env_probe.py $DATA/image-mapping.jsonl 20 general-agent-env-1
uv run python $P/tool_count_probe.py $DATA
```

In zsh, `$(cat …)` is split into separate arguments, but a plain `$IDS` variable is
not; use `${=IDS}` if you put the IDs in a variable.

After fixing finding 1, re-run `path_probe.py`: its `port` mode hard-codes the old
Go repair, so compare the fixed environment against the `docker-c` column instead,
or run a live eval on `format-code-task-000604`.

## Finding 2 (Terminal Bench: anti-cheat guard)

| Script | Needs | What it shows |
|--------|-------|---------------|
| `guard_probe.py` | sandbox | Stages a row's hidden tests under `/tests`, then runs `anti_hack_guard.py` on a clean `/app` (exit 0) and again with only `/app/.vf-venv/.../_virtualenv.pth` added (exit 1, `REJECT planted_interpreter_hook`). |

```sh
uv run python $P/guard_probe.py $DATA/general/train.parquet candidate-0036-software-data-engineering
```

The live confirmation was a full eval, not a script. From `rl-toolkit/`:

```sh
PATH="$PWD/../.eval-venv/bin:$PATH" ../.eval-venv/bin/eval xiaomi-general -n 1 -r 1 -c 1 \
  -m zai-org/GLM-5.2-FP8 --client.base-url http://38.255.28.19:20003/v1 \
  --client.api-key-var GLM_API_KEY \
  --select.include.ids candidate-0036-software-data-engineering \
  --env.agent.runtime.type prime --env.agent.runtime.cpu 2 \
  --env.agent.runtime.memory 4 --env.agent.runtime.disk 10 \
  --env.agent.max-turns 40 --sampling.max-tokens 16000 --run.name review-tb-0036
```

The verifier result is in `outputs/review-tb-0036/monitors/file/traces/stream/00000.jsonl.zst`
under `traces[0].info.verifier` (decompress with `zstd -dc`).

## Finding 3 (General: virtualenv zeroes the submission)

| Script | Needs | What it shows |
|--------|-------|---------------|
| `validator_probe.py` | local | Runs `VALIDATE_SUBMISSION` on temp workspaces: plain files and internal links exit 0; a venv and a dangling link exit 42. |

```sh
uv run python $P/validator_probe.py
```

## Music check

| Script | Needs | What it shows |
|--------|-------|---------------|
| `music_probe.py` | sandbox | Runs upstream's health-check tune through `MusicTask.setup` and `MusicTask.music_score`. Expect about 0.2 (observed 0.198), and 0.0 for a reply with no ABC. |

```sh
uv run python $P/music_probe.py
```

## Dataset inspection

| Script | Needs | What it shows |
|--------|-------|---------------|
| `rows_probe.py` | local | Row counts, schemas, instance keys, working directories, timeouts and prompts for all three parquet files. |
| `code_probe.py` | local | Compares `touched_files` with the `---`/`+++` lines of every Code test patch (finding 8), and surveys the test scripts. |
| `code_probe2.py` | local | Prints the odd patch headers (rows `000989`, `000121`) and the `bash -lc` hand-off lines. |

```sh
uv run python $P/rows_probe.py $DATA
uv run python $P/code_probe.py $DATA
uv run python $P/code_probe2.py $DATA
```

## Results

- `results/path_probe.jsonl`: one record per task from the 42-row `path_probe.py` run
  (41 ran; `format-code-task-001965` hit a sandbox HTTP 502). Each record holds the exit
  code, the missing commands and the last 400 characters of output for every mode.
- `results/path_probe_ids.txt`: the 42 task IDs, one per line.
- `results/shell_probe.jsonl`: the earlier 11-row `shell_probe.py` run, including the
  PATH and resolved tool locations under `bash -c` and `bash -lc`.

Findings 4 to 7 of review 1 came from reading code and data; they have no script.

## Review 2 (re-check after the fixes at `66860c24`)

These drive the real task classes, so they test the code as it now is rather than
a hand-written imitation of it.

| Script | Needs | What it shows |
|--------|-------|---------------|
| `postfix_probe.py` | sandbox | Builds tasks from the real tasksets, resolves the runtime config the way Verifiers does, then runs `setup` and the reward method on an untouched image. `code` mode: no `command not found`. `terminal` mode: guard not tripped, no `.vf-*` in `/app`, egress open in setup and blocked in execution, declared CPU/memory/disk applied. |
| `tb_import_probe.py` | sandbox | For Terminal Bench rows whose tests cannot import a module on an untouched image: the failing import, whether the module is installed, whether the image's requirements list it. |
| `general_finalize_probe.py` | sandbox | Real `general_agent` setup, then the agent user creates a venv, a dangling link and links to `/work/system` and `/work/_verifier`, then real `finalize`. Shows which links survive and that the submission is accepted. Needs `GA_JUDGE_KEY` set to any value; no judge call is made. |
| `patch_paths_probe.py` | local | Runs `touched_files` on all 2,698 Code test patches and compares with `git apply --numstat`. |
| `manifest_audit_probe.py` | network | Checks all 925 `general_agent` manifests against the published file tree: every upload source exists, and the setup/verifier settings the port relies on are uniform. |

```sh
uv run python $P/postfix_probe.py $DATA /tmp/postfix_code.jsonl code format-code-task-000604 format-code-task-000601
uv run python $P/postfix_probe.py $DATA /tmp/postfix_tb.jsonl terminal candidate-0036-software-data-engineering
uv run python $P/tb_import_probe.py $DATA/general/train.parquet cryptography=candidate-2376-security-cryptography
GA_JUDGE_KEY=dummy uv run python $P/general_finalize_probe.py $DATA/general/train.parquet s3k_0000_accounting_audit_tax_en_t1_rl_008
uv run python $P/patch_paths_probe.py $DATA
uv run python $P/manifest_audit_probe.py
```

`postfix_probe.py` runs no model, so hidden tests are expected to fail and every
reward is 0. A reward above 0 on an untouched image would itself be a bug.

Results from review 2:

- `results/postfix_code.jsonl`: 14 Code rows (the rows that failed on missing tools in
  review 1, plus `000989`, `000478` and `000121`).
- `results/postfix_tb.jsonl`: all 64 Terminal Bench rows.

## Review 3 (wider sweep of the Code environment at `58615037`)

| Script | Needs | What it shows |
|--------|-------|---------------|
| `postfix_probe.py` (`code` mode) | sandbox | Re-used on 65 rows: the 5 rows review 2 had not re-run, plus 60 random rows from the whole dataset. This is where setup first raised "Image history is not truncated". Now also records failure signatures. |
| `hygiene_probe.py` | sandbox | Finding 1. On a sample split by working directory: commits reachable outside `HEAD`'s ancestry (non-zero means setup aborts), what upstream's `anti_hack_cleanup` would delete, and whether upstream's history strip fixes the row without changing `HEAD`, the tree or `git status`. |
| `strip_effect_probe.py` | sandbox | Finding 1 fix. Runs the hidden tests before the strip, then the real `setup` and reward after it, and compares the two verifier runs. |
| `login_env_probe.py` | sandbox | Finding 2. Diffs `env` and tool resolution between `bash -c` (the port) and `bash -lc` (upstream Kubernetes). |
| `login_verifier_probe.py` | sandbox | Finding 2. Runs each row's real test command under both shells and records tool-not-found and module-not-found failures. |
| `pyc_residue_probe.py` | sandbox | Checked, no finding. Looks for bytecode of the hidden test files left in Python images. |

```sh
uv run python $P/hygiene_probe.py $DATA /tmp/hygiene.jsonl 50 100
uv run python $P/strip_effect_probe.py $DATA format-code-task-000820 format-code-task-002737
uv run python $P/login_env_probe.py $DATA /tmp/login_env.jsonl 150
uv run python $P/login_verifier_probe.py $DATA /tmp/login_verifier.jsonl format-code-task-001072 format-code-task-000966
uv run python $P/pyc_residue_probe.py $DATA format-code-task-000412 format-code-task-001811
```

Results from review 3:

- `results/postfix_code3.jsonl`: the 65-row sweep through the real task code.
- `results/hygiene.jsonl`: 150 images (50 with working directory `/workspace/repo`, 100 with `/testbed`).
- `results/login_env.jsonl`: 150 random images, environment and tool differences between the two shells.
- `results/login_verifier.jsonl`: 18 rows run under both shells.

### Review 3 follow-up: full pass over every Code image

These back the exact counts in `CLAUDE-REVIEW-3.md`. They create one sandbox per image, so a
full run is 2,698 sandboxes. Both big probes are resumable and take a concurrency argument.

| Script | Needs | What it shows |
|--------|-------|---------------|
| `full_pass_probe.py` | sandbox, all images | Per image, without running tests: commits outside the base's ancestry before and after the strip; whether the strip changes `HEAD`, the tree, `git status` or `git describe`; unreachable commits (history deleted without garbage collection); `env` and tool lookup under `bash -c`, `bash -lc` and `bash -c` with `BASH_ENV`; hidden-test names in untracked files and unreachable blobs. Its `git fsck` kills the sandbox on the ten largest repositories. |
| `summarize_full_pass.py` | nothing | Tallies a `full_pass_probe.py` result file. |
| `strip_fixed_probe.py` | sandbox | Runs the strip with the one added line (`rm -rf .git/refs/remotes …`) on the images where upstream's sequence leaves `git gc` failing. |
| `strip_heavy_probe.py` | sandbox | Runs the strip on very large repositories at 4 GB with a bounded gc and at 8 GB with the default gc. |
| `strip_effect_probe.py` | sandbox | Now uses the corrected strip and takes `PROBE_CONCURRENCY`. |
| `login_verifier_probe.py` | sandbox | Now has a third mode, `bash -c` with `BASH_ENV`, and takes `PROBE_CONCURRENCY`. |
| `verifier_health_probe.py` | sandbox, all images | Applies each row's test patch and runs its test command once on the untouched image, with the Docker PATH and a login shell and a time cap. Finds rows whose patch does not apply, whose verifier cannot run, or whose verifier already passes. `PROBE_IDS`, `PROBE_MEMORY` and `PROBE_DISK` restrict or resize it. |

```sh
uv run python $P/full_pass_probe.py $DATA /tmp/full_pass.jsonl 72
python3 $P/summarize_full_pass.py $P/results/full_pass.jsonl
uv run python $P/strip_fixed_probe.py $DATA format-code-task-001692 format-code-task-003001
uv run python $P/strip_heavy_probe.py $DATA format-code-task-000630 format-code-task-000647
PROBE_CONCURRENCY=30 uv run python $P/login_verifier_probe.py $DATA /tmp/login_verifier_all.jsonl $(cat $P/results/login_shell_required_ids.txt)
uv run python $P/verifier_health_probe.py $DATA /tmp/verifier_health.jsonl 80 240
PROBE_IDS="$(cat $P/results/sandbox_died_4gb_10gb_ids.txt | tr '\n' ' ')" PROBE_MEMORY=8 PROBE_DISK=20 \
  uv run python $P/verifier_health_probe.py $DATA /tmp/verifier_health_8gb.jsonl 13 600
```

If one of the big probes is interrupted, its sandboxes are not always torn down. They all
carry the label `xiaomi-review`: list them with `prime sandbox list --label xiaomi-review`
and remove them with `prime sandbox delete --label xiaomi-review --only-mine --yes`.

The two live evals used the same command as in the finding 2 section above, with
`xiaomi-code`, `--select.include.ids format-code-task-001072` (and `001138`),
`--env.agent.max-turns 64` and `--run.name review3-code-001072`.

Results:

- `results/full_pass.jsonl`: one record per Code image (10 with an `error`: the sandbox died).
- `results/strip_fixed.jsonl`: the corrected strip on the 93 images where gc failed.
- `results/login_verifier_all.jsonl`: all 224 rows where a tool resolves differently, three modes each.
- `results/login_verifier_opts.jsonl`: 36 rows whose images set JVM and Node heap options only in login shells.
- `results/verifier_health.jsonl`: one record per Code row; `results/verifier_health_reruns.jsonl`: the errored, capped and free-reward rows re-run.
- Row lists, one ID per line:
  - `history_not_truncated_ids.txt` (618): setup aborts today.
  - `login_shell_required_ids.txt` (29): verifier fails for an environment reason under `bash -c` only.
  - `verifier_cannot_run_ids.txt` (13): verifier exits 127 even with the Docker PATH and a login shell.
  - `verifier_passes_untouched_ids.txt` (2): verifier passes with no change made.
  - `sandbox_died_4gb_10gb_ids.txt` (13): sandbox died during verification at 4 GB.

## Review 4 (the review 3 fixes at `4b5f3251`, on every Code image)

`review4_full_probe.py` drives the port's real `setup` and `verified` on untouched images, the
way an eval would, and records what the new start-up file does to child shells.
`review4_fix_check.py` applies the four fixes proposed in `CLAUDE-REVIEW-4.md` as in-memory
patches and runs the same two steps. Neither touches the repository.

```sh
# full pass: about 2,700 sandboxes, 25 minutes at concurrency 128
uv run python $P/review4_full_probe.py $DATA /tmp/review4_full.jsonl 128 300
python3 $P/summarize_review4.py /tmp/review4_full.jsonl $P/results

# finding 1: only the port's own setup and verifier, shipped file vs without its last line
IDS="$(head -10 $P/results/review4_pyenv_slow_ids.txt | tr '\n' ' ')"
PROBE_CLEAN=1 PROBE_IDS="$IDS" uv run python $P/review4_full_probe.py $DATA /tmp/clean_shipped.jsonl 10 2000
PROBE_CLEAN=1 PROBE_NO_REEXPORT=1 PROBE_IDS="$IDS" uv run python $P/review4_full_probe.py $DATA /tmp/clean_noreexport.jsonl 10 2000

# finding 2: with and without the cache clean-up
IDS="format-code-task-001840 format-code-task-000050"
PROBE_CLEAN=1 PROBE_NO_REEXPORT=1 PROBE_IDS="$IDS" uv run python $P/review4_full_probe.py $DATA /tmp/pyc_cleanup.jsonl 2 900
PROBE_CLEAN=1 PROBE_NO_REEXPORT=1 PROBE_NO_CLEANUP=1 PROBE_IDS="$IDS" uv run python $P/review4_full_probe.py $DATA /tmp/pyc_nocleanup.jsonl 2 900

# findings 3 and 4, and all four fixes together
uv run python $P/review4_fix_check.py $DATA /tmp/fix_check.jsonl format-code-task-001621 \
  format-code-task-001166 format-code-task-002071 format-code-task-001840 format-code-task-000050

# review 3 finding 5 with no runtime flags at all
PROBE_DEFAULTS=1 PROBE_IDS="$(cat $P/results/sandbox_died_4gb_10gb_ids.txt | tr '\n' ' ')" \
  uv run python $P/review4_full_probe.py $DATA /tmp/no_flags.jsonl 13 600
```

One caveat about `review4_full_probe.py`'s `env_diff` field: Prime starts each command from a
bash wrapper that has already loaded the start-up file, so variables it exports (other than
`PATH`, which the runtime sets again) leak into the `bash -lc` side of the comparison. Only
the `PATH` difference in that field is reliable.

Results:

- `results/review4_full.jsonl`: one record per Code image. `review4_rerun_001002.jsonl`: the one
  row that hit a transient sandbox error, re-run.
- `results/review4_clean_shipped.jsonl`, `review4_clean_noreexport.jsonl`: ten pyenv rows, nothing
  but the port's own setup and verifier.
- `results/review4_slow_shipped.jsonl`, `review4_slow_noreexport.jsonl`: the rows that hit the
  300 s cap in the full pass, re-run with a 1,500 s cap.
- `results/review4_noreexport_40.jsonl`: the 11 repaired rows and 29 login-shell rows without the
  start-up file's last line.
- `results/review4_pyc_with_cleanup.jsonl`, `review4_pyc_without_cleanup.jsonl`: the five rows
  whose test scripts read `.pyc` files.
- `results/review4_no_flags_21.jsonl`: the 13 memory-hungry rows and the 8 largest repositories
  with Prime's defaults plus what the task declares.
- `results/review4_fix_check.jsonl`: setup and verifier with the four proposed fixes applied.
- Row lists, one ID per line:
  - `review4_pyenv_ids.txt` (118): images that set up pyenv in a login profile.
  - `review4_pyenv_slow_ids.txt` (77): of those, grading took 124 s or more with the shipped file.
  - `review4_child_path_lost_ids.txt` (897): a child bash drops a `PATH` entry set by its parent.

## Review 5 (the review 4 fixes at `6d1030cd`, on every Code image)

No new scripts. The review 4 full pass was repeated on the committed fixes with no patches:

```sh
uv run python $P/review4_full_probe.py $DATA /tmp/review5_full.jsonl 128 300
python3 $P/summarize_review4.py /tmp/review5_full.jsonl $P/results

# timing without provider load: ten pyenv rows, only the port's own setup and verifier
IDS="$(head -10 $P/results/review4_pyenv_slow_ids.txt | tr '\n' ' ')"
PROBE_CLEAN=1 PROBE_IDS="$IDS" uv run python $P/review4_full_probe.py $DATA /tmp/review5_clean.jsonl 10 2000
```

The full pass took about 40 minutes this time. Setup and verifier times in it carry roughly
16 to 18 s of provider latency per step at concurrency 128; use the ten-row run for timing.

Results:

- `results/review5_full.jsonl`: one record per Code image at `6d1030cd`.
- `results/review5_clean_10.jsonl`: the ten-row timing check.

## Review 6 (leftover checks at `27c2acd8`)

Three new scripts. All run the port's real setup on the untouched repo; none runs a model.

```sh
# git fsck to completion on every image, no size limit, exit status recorded
uv run python $P/review6_fsck_probe.py $DATA /tmp/review6_fsck.jsonl 96 1500

# files in .git that still name pruned commits (ORIG_HEAD, FETCH_HEAD, commit-graph, ...)
uv run python $P/review6_residue_probe.py $DATA /tmp/review6_residue.jsonl 96

# the proposed fix (delete ORIG_HEAD, FETCH_HEAD and the commit-graph before gc), applied in
# memory, plus the verifier exit code for comparison with review5_full.jsonl
PROBE_FIX=1 PROBE_VERIFY=1 PROBE_IDS="$(tr '\n' ' ' < $P/results/review6_fix_check_ids.txt)" \
    uv run python $P/review6_residue_probe.py $DATA /tmp/review6_fix_check.jsonl 77

# the six rows that hit the 300 s cap in review 5, with the task's own 1,800 s timeout
PROBE_CLEAN=1 PROBE_IDS="format-code-task-000252 format-code-task-000348 format-code-task-002606 format-code-task-002643 format-code-task-000797 format-code-task-001540" \
    uv run python $P/review4_full_probe.py $DATA /tmp/review6_capped_6.jsonl 6 2400

# the same Music replies scored under several abcmidi builds (replies.json: [{"id", "reply"}])
uv run python $P/review6_abcmidi_versions.py replies.json /tmp/abcmidi.json \
    python:3.11-slim python:3.11-slim-bookworm ubuntu:22.04 ubuntu:24.04
```

In the residue results, `objects/info/packs` (pack names) and `./index` (tracked paths) are
false positives. `002295` fails setup on purpose, so its rows show the unstripped repo.

Results:

- `results/review6_fsck.jsonl`: one record per Code image.
- `results/review6_residue.jsonl`: one record per Code image.
- `results/review6_fix_check_ids.txt`, `results/review6_fix_check.jsonl`: the 77-row fix test.
- `results/review6_capped_6.jsonl`: the six long rows.
- `results/review6_000252_reruns.jsonl`: two more runs of `000252` (same command, one row).
- `results/review6_abcmidi_60.json`: 58 GLM-5.2 Music replies plus the upstream health-check
  tune, scored in four images; `why` holds the scorer's reject reasons per reply.

## Review 7 (the review 6 fixes at `76084bed`)

`review6_residue_probe.py` now also records HEAD, tree, `describe` (hash suffix stripped) and
status before and after setup (`state_changed`), unreachable objects, submodule Git folder
history and leftovers, and a verifier that runs past 600 s as `exit: "cap600"`. One pass covers
the Git checks and the verifier:

```sh
PROBE_VERIFY=1 uv run python $P/review6_residue_probe.py $DATA /tmp/review7_full.jsonl 128

# submodules: future task-repo commits that moved a submodule pointer to a commit still in
# .git/modules (checked on the untouched image; base commits come from the full pass)
uv run python $P/review7_submodule_probe.py $DATA /tmp/review7_full.jsonl /tmp/review7_submodules.jsonl 40
```

`state_changed` on `001621` is the login-profile banner in the raw text, not a Git change (its
`head_is_base` and `status` are unchanged).

Results:

- `results/review7_full.jsonl`: one record per Code image at `76084bed`.
- `results/review7_submodules.jsonl`: the 40 images with submodule Git folders.

## Review 8 (standalone packages at `8567350a`)

`xiaomi-common` was removed, so every probe now imports `XiaomiTasksetConfig` and
`resolve_image` from `xiaomi_code._data`. Run `uv sync --locked` in `rl-toolkit` first so a
stale `xiaomi-common` install cannot hide a missing import.

```sh
# 80 Code rows (ids in results/review8_code_ids.txt), compared with review7_full.jsonl
PROBE_VERIFY=1 PROBE_IDS="$(tr '\n' ' ' < $P/results/review8_code_ids.txt)" \
    uv run python $P/review6_residue_probe.py $DATA /tmp/review8_code.jsonl 80

# standalone install: copy one environment out of the repo and install it from PyPI
cp -R environments/xiaomi-music /tmp/solo/ && uv venv /tmp/solo/v --python 3.12 && \
    uv pip install --python /tmp/solo/v/bin/python /tmp/solo/xiaomi-music
```

Results:

- `results/review8_code.jsonl`: the 80-row sample at `8567350a`.
