# Review 2: Xiaomi env ports after the review 1 fixes

Date: 2026-10-11. Reviewed: `rl-toolkit` branch `Xiaomi-Envs` at `66860c24`
(commits `1afbd32f`, `0b6caef4`, `66860c24` on top of `596e07d0`).
Read-and-test review only. No code was changed and nothing was committed.

All paths are relative to `rl-toolkit/` unless they start with `.references/`
or `claude-review-probe-tooling/`.

## Summary

The seven fixes from review 1 are correct and hold up under independent testing.
Nothing blocks shipping. Two small follow-ups remain, and one of them is a defect
in the published data, not in the port.

| # | Severity | Env | One line |
|---|----------|-----|----------|
| 1 | Should fix | General (Terminal Bench) | Three rows cannot pass in their published images: a Python module the tests need is not installed |
| 2 | Nice to have | General (Terminal Bench) | Terminal Bench task objects can no longer be pickled |

Still open from review 1 by agreement: finding 6 (`install_common` uses private
Verifiers functions). It now carries a `shortcut:` comment and a docs note.

## Review 1 findings: status

| R1 # | Fix | Verified how | Result |
|------|-----|--------------|--------|
| 1 | `XiaomiCodeTask.runtime_env` supplies Docker's default PATH; Go repair deleted | Real `setup` + `verified` on 14 untouched images, including 12 of the 17 rows flagged for a missing tool in review 1 (corrected in review 3: an earlier version of this line said "every row"; the other five were re-run in review 3 and are clean) | No `command not found` on any row. `go`, `node`, `npm`, `npx` resolve to `/usr/local/bin` |
| 2 | Terminal rows start no tool server and skip `install_common` | Real `setup` + `verifier_reward` on all 64 untouched images | Guard never fires. No `.vf-*` entry in `/app` on any row |
| 3 | Validator removes symlinks that leave the workspace or dangle | Local validator probe, plus real `setup` + `finalize` in a sandbox | Venv no longer fails the submission. Links to `/work/system` and `/work/_verifier` are removed. Internal links are kept |
| 4 | Agent timeout, CPU, memory, disk and no-internet mapped from the row | All 64 rows | Runtime resolves to 1 CPU, 2 GB, 10 GB, `allow=[]`, `block=["*"]`. Egress is open during setup and blocked once the execution policy is applied, on all 64 |
| 5 | No tool server on terminal rows, so no state secret in the sandbox | Read the auth code in both pinned Verifiers versions; `toolsets` returns `[]` on all 64 | Closed. `/task` is keyed by the state secret, which is only handed to tool servers |
| 7 | Code verifier output kept to the last 10,000 / 4,000 characters | Read the diff | Done |
| 8 | `touched_files` also reads `---`/`+++` headers and tracks hunk lengths | All 2,698 patches, compared with `git apply --numstat` | No parse errors. Every path git touches is in the reset set. Only row `000989` changes, as intended |

Other checks:

- `uv run pytest -q`: 72 passed.
- The other agent's three eval traces match what they reported:
  `outputs/review1-tb-0036-postfix` (reward 1.0, `5 passed`, empty stderr),
  `outputs/review1-code-000121` (reward 1.0, base and four new Go tests pass),
  `outputs/review1-code-000604` (reward 0.0).
- The `000604` zero is the model's doing, not the environment's. On an untouched
  image the `base` half passes; in their run two new Jest suites fail to compile
  with `TS2345` against the model's own types.
- No Terminal Bench row gives a reward above 0 on an untouched tree (64 of 64
  score 0 before any agent work), so no row is free.
- All 925 `general_agent` manifests agree with the published file tree: no
  upload source is missing, and the setup command, verifier command, reward file
  and (null) verifier timeout are identical across all of them. This matters
  because `upload()` raises on a missing source where upstream skips it.
- The corrected comment at `xiaomi_common/runtime.py:25` is accurate: upstream's
  Docker backend runs `bash -c`, its Kubernetes backend `bash -lc`.

## Should fix

### 1. Three Terminal Bench rows cannot pass in their published images

Where: the dataset and images, not the port. The port's part is only that it
loads these rows (`environments/xiaomi-general/xiaomi_general/taskset.py:486-518`).

- On an untouched image, pytest fails while collecting `/tests/test_outputs.py`
  because a third-party module is not installed. `test.sh` then writes reward 0.
- The module is absent from the image, from `/app/requirements-shared.txt` and
  from `/app/Dockerfile`. The tasks are offline (`allow_internet: false`), and
  the cryptography task also says "do not install dependencies or use the
  network". The agent has no way to make the import work.

| Row | Image | Failing import |
|-----|-------|----------------|
| `candidate-2376-security-cryptography` | `general-agent-env-50` | `cryptography` (needed by the vendored `authlib.jose`) |
| `candidate-0260-security-appsec` | `general-agent-env-3` | `stevedore` (needed by the vendored `bandit`) |
| `candidate-0674-ml-evaluation` | `general-agent-env-9` | `torch` (imported by the test file itself) |

- A fourth row, `candidate-1634-software-databases`, also fails at import on an
  untouched image (`from collections import Callable` in vendored peewee, which
  Python 3.12 removed). That file is in the task's mutable set, so the agent can
  repair it. I count that row as solvable.
- This is not a regression from the fixes. Before the no-internet fix an agent
  could have installed the module, but that was itself against the task's rules.
- Evidence: `claude-review-probe-tooling/tb_import_probe.py`; the four rows show
  up in `results/postfix_tb.jsonl` with an import error in `stdout_tail`.

Fix: list the three row IDs as known-unsolvable in `docs/general.md` and keep
them out of training and reported scores (a three-ID skip in `load()` with a
one-line comment, or a documented exclusion at run time). Do not install the
modules at setup; that would change the published image.

If we skip it: three of 64 rows return a constant 0, which costs rollouts and
can lower the Terminal Bench mean by up to about 5 points (3 of 64 rows) for reasons unrelated to the model.

## Nice to have

### 2. Terminal Bench task objects can no longer be pickled

Where: `environments/xiaomi-general/xiaomi_general/taskset.py:230-234`.

- `GeneralTask.__init__` sets `self.toolsets = lambda config: []` on terminal rows.
- A lambda defined inside a method cannot be pickled:
  `pickle.dumps(task)` raises `AttributeError: Can't pickle local object 'GeneralTask.__init__.<locals>.<lambda>'`.
  `general_agent` tasks still pickle.
- Nothing pickles tasks today. Both pinned Verifiers versions rebuild tasks from
  their data, and `with_data`, `copy.deepcopy` and `defer_scoring` all keep the
  override. So there is no failure now.

Fix: assign a module-level function instead of the lambda, for example
`def _no_toolsets(config): return []`. One line changes.

If we skip it: any later code that sends tasks across processes fails on the 64
terminal rows only, with an error that does not point at the cause.

## Not checked

- A full `general_agent` eval with the judge after the validator change. I have
  no `GA_JUDGE_KEY`. I ran real `setup` and `finalize` in a sandbox, but not the
  colocated tool-server install or `run_verify.py`. One live row with a real
  judge key would close this.
- The other roughly 2,640 Code images. Fourteen were re-run with the real task
  code; review 1's 41-row sample showed the Docker PATH resolves every tool.
- The 843 Code scripts that hand off to an in-image `.build_env` script.
- Whether Debian's abcmidi version matches the one Xiaomi trained with. Music
  code is unchanged since review 1.
- Whether upstream ran Terminal Bench tests in a separate image that had the
  missing modules. Upstream released no Terminal Bench wrapper, so this cannot
  be settled from the published material.

## Reproduce

Scripts and raw results are in `claude-review-probe-tooling/`; its README has a
"Review 2" section with the exact commands.

| Check | Script | Raw results |
|-------|--------|-------------|
| R1 findings 1, 2, 4, 5 on real task code | `postfix_probe.py` | `results/postfix_code.jsonl`, `results/postfix_tb.jsonl` |
| R1 finding 3 in a sandbox | `general_finalize_probe.py`, `validator_probe.py` | printed by the scripts |
| R1 finding 8 on all patches | `patch_paths_probe.py` | printed by the script |
| Manifest audit | `manifest_audit_probe.py` | printed by the script |
| Finding 1 of this review | `tb_import_probe.py` | printed by the script |

All probe sandboxes were deleted afterwards (`prime sandbox list` shows 0).
