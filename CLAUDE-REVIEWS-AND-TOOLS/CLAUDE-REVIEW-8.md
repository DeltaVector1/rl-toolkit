# Review 8: the review 7 fixes at `8567350a`

Branch `Xiaomi-Envs`, commits `46cece1e`, `b165c046`, `8567350a` (compared with `76084bed`).

## Verdict

**Good. Ready to use.** The must-fix from review 7 is done: `xiaomi-common` is gone, and each
of the three environments installs and loads on its own from PyPI, outside the repo. Both
review 7 doc notes are in. Nothing changed in how Code, General or Music behave.

## What this change does

- Copies the helpers each environment needs into that environment as `_data.py` and
  `_runtime.py`, and deletes the `xiaomi-common` package.
- Each `pyproject.toml` now depends on `verifiers>=0.3.2.dev206,<=0.3.2.dev208`,
  `huggingface-hub>=0.24` and `pyarrow>=18` instead of `xiaomi-common`.
- Deletes General's `install_common` workaround, the one that used private Verifiers helpers.
- Removes `xiaomi-common` from the root workspace and lock file. Points the tests at the copies
  and adds tests for each copy.
- Adds standalone install instructions to the README files, the submodule sentence to the Code
  README, and the `000915` network note to `docs/evaluation.md`.

## What I did

1. Read the diff. Compared each copied helper with the deleted original line by line.
2. Ran `uv lock --check`, `uv sync --locked` and `uv run pytest -q`.
3. Copied each environment folder out of the repo, installed it into a fresh venv from PyPI,
   and loaded its tasks with Python in isolated mode from an unrelated folder.
4. Opened the other agent's two General traces.
5. Ran a General MCP row live myself, with default sandbox resources.
6. Ran setup, the Git checks and the verifier on 80 Code rows and compared them with the review 7
   full pass.

## Results

**The copies are faithful.** Each `_data.py` and `_runtime.py` is a strict subset of the deleted
original. Nothing is changed, only the parts that environment does not use are left out.
`DATASET_ID`, `DATASET_REVISION` and the image-registry check are identical in all three, and
`test_each_loader_downloads_its_pinned_dataset` pins all three to the same revision.

**Standalone install works.** For each environment, in a fresh venv outside the repo:

| Environment | Installs from PyPI | `verifiers` | `xiaomi_common` importable | Tasks loaded |
| --- | --- | --- | --- | --- |
| Code | yes | 0.3.2.dev208 | no | 2,698 |
| General | yes | 0.3.2.dev208 | no | 989 |
| Music | yes | 0.3.2.dev208 | no | 1,000 |

The three allowed `verifiers` versions (dev206, dev207, dev208) all exist on PyPI.
`grep -rn xiaomi_common` finds only two sentences in `docs/evaluation.md` that describe the
removal.

**Tests.** `uv lock --check` passes and the suite passes 131 tests. My own venv still had the
old `xiaomi-common` installed from before; `uv sync --locked` removed it, and I ran the tests
after that, so they do not lean on the old package. Anyone with an older checkout should run
`uv sync` once for the same reason.

**General, the other agent's runs.** Both traces match the report:

- Terminal Bench `candidate-0036-software-data-engineering`: reward 1.0, `5 passed`, 21
  replies, ended normally.
- MCP `s3k_0000_accounting_audit_tax_en_t1_rl_008`: reward 0.8, 4 of 6 checks, 32 MCP tool
  calls, submission validated, judge key shown as `<redacted>`.

Their first MCP attempt (`outputs/review7-accounting-wheel/`) has no trace. Its log shows it
was stopped about 20 seconds into setup and restarted with explicit resources 17 seconds
later. It did not hang.

**General, my run.** Same MCP row, default sandbox resources (no CPU, memory or disk flags,
since MCP rows declare none):

- Setup took 20 s.
- 26 MCP tool calls, submission validated, no errors, no timeout.
- The judge key appears in none of the saved files.
- Reward 0.5. GLM's last reply hit my 16,000-token cap (`finish_reason=length`), so the
  judge failed 5 of 6 checks on a cut-off answer. That is my run setting, not the
  environment; at 32,000 tokens the other agent's run scored 0.8.

So the MCP tool server now installs from the General package alone, with no workaround.

**Code, 80 rows.** I used 14 rows that have mattered in earlier reviews (`001621`, `001840`,
`001166`, `002071`, `000050`, `000817`, `001920`, `001321`, `002990`, `002740`, `001802`,
`002295`, `003041`, `002099`) plus 66 random rows. All 80 match review 7 exactly:

- Setup passes on 79; `002295` fails closed as documented.
- HEAD equals the base on all 79.
- `git fsck` and commit-graph verification exit 0 on all 79.
- There are no unreachable objects.
- The verifier exit code, the `git status` hash, the leftover-file list and the submodule
  counts match review 7 on every row.

## Must fix

None.

## Should fix

None.

## Nice to have

**1. Add a test that keeps each environment standalone.**

- Where: `tests/` (a new small test).
- Problem: nothing stops someone from importing one environment's package from another, or
  adding a sibling `xiaomi-*` dependency again. The suite still passes in that case, because
  the workspace installs all three packages together.
- Fix: a test that reads each `environments/xiaomi-*/pyproject.toml` and checks:
  - no dependency starts with `xiaomi-`;
  - no file in `xiaomi_code/`, `xiaomi_general/` or `xiaomi_music/` imports another one of
    those packages.

  It is about ten lines.
- If we skip it: the standalone property is only protected by review.

**2. Note that General sandboxes install `pyarrow` and `huggingface-hub`.**

- Where: `docs/general.md`, the sentence about the native MCP package bundler.
- Problem: Verifiers runs `uv pip install` on the General package inside each General agent
  sandbox, with dependencies, to start the tool server. That now pulls `pyarrow` (about 40 MB)
  and `huggingface-hub`, which the tool server never uses; `verifiers` does not depend on
  either. In my run setup still took only 20 s, so the cost is small. It does mean the sandbox
  needs package-index access during setup.
- Fix: one sentence in the docs. Or leave it.
- If we skip it: nothing breaks.

## Checked, no finding

- **Workspace.** The leftover `environments/xiaomi-common/` folder in a checkout holds only
  ignored `__pycache__` files. `members = ["environments/*"]` does not trip on it:
  `uv lock --check` and `uv sync --locked` both pass.
- **The `.eval-venv` used for the live runs.** Its installed `xiaomi_general` files
  (`taskset.py`, `_data.py`, `_runtime.py`, `servers/tool.py`) are byte-identical to the repo
  at `8567350a`. The two later commits only change docs.
- **Eval instructions.** `docs/evaluation.md` installs the three environments with `--no-deps`
  into the Prime RL venv. That works because Prime RL already brings `pyarrow` and
  `huggingface-hub`; the live runs used exactly this setup.
- **Docs.** The submodule sentence is in `environments/xiaomi-code/README.md:20`, the `000915`
  note is in `docs/evaluation.md`, and the standalone install steps match what I ran.

## Not checked

- General with an independent judge. Still only one model endpoint (GLM) here.
- A full 2,698-image Code pass at this commit. The Code change is the import path only, and the
  80-row sample matched review 7 exactly.
- Publishing to the Prime hub itself. I tested the thing that blocked it: installing each
  environment alone from PyPI.

## Housekeeping

- This review used about 85 Prime sandboxes (80 Code rows, the MCP run and its tool server).
  All are deleted; the list is empty.
- I changed nothing in `rl-toolkit`. I fast-forwarded the local checkout to `8567350a`; `uv sync
  --locked` removed the stale `xiaomi-common` from my local `.venv` only. The working tree is
  clean.
- My probe scripts imported `xiaomi_common`. I changed them to import from `xiaomi_code._data`.
- Raw results: `claude-review-probe-tooling/results/review8_code.jsonl` and
  `review8_code_ids.txt`. The MCP trace is in my scratchpad under
  `evals/review8-general-mcp-default/`.
