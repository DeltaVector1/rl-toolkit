# Review 7: the review 6 fixes at `76084bed`

Branch `Xiaomi-Envs`, commits `0b8844d7`, `4d5f97a6`, `76084bed` (compared with `27c2acd8`).

## Verdict

**Not ready yet: one must-fix.** The three review 6 fixes are done correctly, and the full
pass on all 2,698 images is clean: no leftover pointer to a deleted commit anywhere, and
`git fsck` now exits 0 on every image. The must-fix
is a packaging problem that was there from the start: each environment depends on a local
helper package, `xiaomi-common`, which only installs from inside this repo. The project
owner wants every environment to stand on its own.

## What this change does

- Setup now deletes `ORIG_HEAD`, `FETCH_HEAD` and any old commit-graph before `git gc`
  (`taskset.py:76-77`). Adds a test for it with four cases.
- Adds `format-code-task-000252` to the Code exclusion list (seven excluded, 2,691 eligible).
- Adds a token-cap note to `docs/music.md`.
- Records the review 6 results and the other agent's own checks in `docs/evaluation.md`.

## What I did

1. Read the three commits and ran the tests, including the new test against the old code.
2. Ran the real `setup` and then the real `verified` on the untouched tree on **all 2,698
   Code images** at `76084bed`, with no patches. After setup, the same pass checked:
   - that HEAD, tree, `git describe` and `git status` are unchanged;
   - that no file in `.git` names a deleted commit;
   - full `git fsck --unreachable` and `git commit-graph verify`;
   - the submodule Git folders.

   The verifier cap was 600 s. I compared the exit codes with review 5.
3. For the 40 images with submodule Git folders, checked the untouched image for future
   submodule commits that setup would leave behind.
4. Tried installing an environment on its own, outside the repo (must fix 1).
5. Opened the other agent's `000816` trace.

## Results

### Full pass on all 2,698 images (with the fix)

| Check | Review 6 (`27c2acd8`) | Now (`76084bed`) |
| --- | --- | --- |
| Setup passes | 2,697 | **2,697** (only `002295` fails, as documented) |
| `HEAD` equals the base | 2,697 | 2,697 |
| Commits outside the base | 0 | 0 |
| Unreachable objects | 0 | 0 |
| `git fsck` exits 0 | 2,693 (4 exit 16) | **2,697** |
| `git commit-graph verify` exits 0 | 2,693 | **2,697** |
| `ORIG_HEAD` names a deleted commit | 608 | **0** |
| `FETCH_HEAD` present | 21 | **0** |
| Reflog bytes, bitmaps | 0 | 0 |
| HEAD, tree, `describe`, `status` changed by setup | 0 | 0 |

The only hits left in the `.git` scan are the two known false positives: pack names in
`objects/info/packs` (every row) and file names in `index` (54 rows).

One row, `001621`, showed a difference in my raw before/after text. Its HEAD equals the base
and its status hash is unchanged, and review 5's field-by-field check of the same row showed
no change. The most likely cause is the banner its login profile prints (review 5 lists
`001621` as the "profile prints a banner" row), which setup's start-up file makes appear in
later shells. Either way, it is not a Git change.

**Verifier exit codes** match review 5 on every row except these seven:

- `002643` (exit 9) and `000797` (exit 101) finished this time; in review 5 they hit my 300 s
  cap. Both match review 6.
- `000252`, `000348` and `001540` hit this pass's 600 s cap. `000252` is now excluded;
  the other two needed 659 s and 638 s in review 6.
- `002606` exited 139 (segfault in `test_kdtree.py`). It has flipped between 139 and 1 since
  review 3 (139 in review 4 and the review 3 rerun, 1 in review 6), so it is flaky upstream,
  not a change from this commit.
- `000915` exited 3 instead of 244. In every earlier run, `npm install -g npm@12.2.0`
  failed during grading; this time the download worked and the tests ran. This row depends
  on the network at grading time, like `000252` (see nice to have 2).

### Submodule Git folders (closes the last "not checked" item)

40 images have submodule Git folders under `.git/modules`. The strip does not touch them.

- 38 of them hold commits beyond each submodule's checked-out commit, 151,392 in total. That
  is the submodules' own upstream history (METIS, dlpack, nim libraries and so on).
- 9 still have `FETCH_HEAD` files inside `.git/modules/...`, naming submodule upstreams.
- The question that matters: did a future commit of the *task* repo move a submodule
  pointer to a commit that is still in the submodule folder? On the untouched images, that is
  true on 2 rows:
  - `002740`: 8 future pointers, all present. The nearest is 1 commit past the base ("bump
    nimbus-build-system").
  - `001802`: 1 of 10 present, 2 commits past the base ("Fix to the CEF external").

  Neither task is about those submodules. `002740` asks for a sync-committee cache in the main
  Nimbus code, and `001802` asks to fix the `mod` operator. So these are dependency bumps,
  not the answer. Upstream's strip does not touch submodules either (see nice to have 1).

## Must fix

**1. Remove `xiaomi-common`. Make each environment install on its own.**

- Where:
  - `environments/xiaomi-common/` (the package itself: `data.py`, 69 lines; `runtime.py`, 32
    lines).
  - `environments/xiaomi-{code,general,music}/pyproject.toml:6`
    (`dependencies = ["verifiers", "xiaomi-common"]`).
  - The imports at `xiaomi_code/taskset.py:15-16`, `xiaomi_general/taskset.py:13-21` and
    `xiaomi_music/taskset.py:7-8`.
  - `install_common` in `xiaomi_general/taskset.py:190-210` and its call at line 358.
  - Root `pyproject.toml:19` (`xiaomi-common = { workspace = true }`) and `uv.lock`.
  - Tests that import it: `tests/test_data.py:7`, `tests/test_runtime.py:7`,
    `tests/test_code.py:260`, and the `install_common` stub at `tests/test_general.py:314`.
  - Docs that mention it: `docs/evaluation.md:46`, `docs/general.md:6`, `:74`, `:138-141`,
    `docs/music.md:6`, `environments/xiaomi-general/README.md:4`,
    `environments/xiaomi-music/README.md:4`.
- Problem: all three environments list `xiaomi-common` as a dependency, but it is not
  published anywhere. It only resolves because the repo's uv workspace points at the
  local folder. Outside this repo, an environment cannot be installed. I copied
  `environments/xiaomi-music` out of the repo and ran `uv pip install --dry-run` on it:

  ```
  × No solution found when resolving dependencies:
  ╰─▶ Because xiaomi-common was not found in the package registry and
      xiaomi-music==0.1.0 depends on xiaomi-common, we can conclude that
      xiaomi-music==0.1.0 cannot be used.
  ```

  The same applies to Code and General, which have the same dependency line. So an
  environment pushed to the Prime hub, or installed by anyone outside this checkout, will
  not install.

  Two more costs:
  - General has to work around it at run time. `install_common` uses two *private*
    Verifiers helpers (`_cached_package`, `_package_dir`) to pack the local package and
    install it inside the sandbox. The code's own comment says to re-check it on every
    Verifiers upgrade.
  - The name `xiaomi-common` is free on PyPI today. If someone registers it, a standalone
    install would stop failing and quietly install their package instead.
- Fix: copy what each environment uses into that environment, then delete the shared
  package.
  - Code uses `XiaomiTasksetConfig`, `load_rows`, `load_instance`, `resolve_image`,
    `checked_run` and `run_shell`.
  - General uses the same six plus `DATASET_REVISION` and `download_general_task`.
  - Music uses only `XiaomiTasksetConfig`, `load_rows` and `checked_run`.

  Put them in a small module inside each package, for example
  `xiaomi_code/_data.py` and `xiaomi_code/_runtime.py`. Each environment copies only what it
  needs, so Music's copy is tiny. Keep the pinned `DATASET_ID`, `DATASET_REVISION` and the
  image-registry check exactly as they are.

  Then:
  - Change each `pyproject.toml` to depend on `verifiers` plus what the copied code
    imports directly: `huggingface-hub>=0.24` and `pyarrow>=18`.
  - Delete `install_common` and its call in General. The General package then ships as one
    package, so the private-helper workaround is no longer needed.
  - Remove `xiaomi-common` from the root workspace and lock file.
  - Point the tests at the per-environment copies. Keep one test per copy for the image
    mapping check and the `run_shell` timeout.
  - Update the docs and READMEs listed above.
- Check that proves it: for each of the three environments, copy its folder out of the
  repo and run `uv pip install` into a fresh venv. Then import its taskset and load one row
  from a local parquet. All three must work with no other folder present.
  `grep -rn xiaomi_common` must return nothing outside `outputs/`. Also re-run one General
  agent row (Terminal Bench) and one MCP-tool General row live, because removing
  `install_common` changes what gets staged into the sandbox.
- If we skip it: the environments only work from this repo checkout. They cannot be
  published or installed on their own. General keeps depending on private Verifiers
  internals.

## Should fix

None.

## Nice to have

**1. Say in the docs that submodule Git folders are left as shipped.**

- Where: `environments/xiaomi-code/README.md` (the paragraph about what setup strips), or
  `docs/code.md`.
- Problem: on 2 images (`002740`, `001802`), the submodule Git folders keep commits that
  future task-repo commits pointed to. On those two rows they are dependency bumps, not the
  task's answer. 9 images also keep `FETCH_HEAD` inside submodule folders. The README
  describes the superproject strip in detail and does not mention submodules, so a reader
  may assume they are stripped too.
- Fix: one sentence, for example "Submodule Git directories under `.git/modules` are left
  as the image ships them." Pruning them is possible, but it would need care for repos
  whose builds use `git submodule`, and upstream does not do it either.
- If we skip it: nothing breaks. A future dataset row whose fix is a submodule bump would
  leak through this path without anyone having written down that it could.

**2. Note that `000915` installs npm from the internet during grading.**

- Where: `docs/evaluation.md`, or the exclusion table in `docs/code.md` if it keeps failing.
- Problem: in review 3, review 4 and review 5, `000915`'s verifier failed on
  `npm install -g npm@12.2.0` (exit 244) before running any test. This time it worked and the
  tests ran (exit 3). It is the same kind of problem as `000252`, but it has worked once, so
  it is not a must-exclude yet.
- Fix: one line in the docs, or watch it in the next full pass and exclude it if it fails
  again.
- If we skip it: on a bad network day this row scores 0 for any solution.

## Checked, no finding

- **The Git clean-up fix is the proposed one.** The two `rm` lines sit just before
  `git gc`, as tested in review 6.
- **The new test catches the old code.** I ran
  `test_strip_removes_future_metadata_and_stale_commit_graphs` against the `27c2acd8`
  `taskset.py`. All four cases fail; on the new code all pass. The test covers a single
  commit-graph and a split one, with and without a surviving tag, and checks that the
  staged, working-tree and untracked edits survive.
- **Tests.** `uv run pytest -q`: 111 passed.
- **Exclusion list.** `000252` appears in the table and in the example command. The count
  says seven excluded and 2,691 eligible, and no "six" or "2,692" is left anywhere. The
  list exists only in `docs/code.md`.
- **The other agent's agent run.** I opened
  `outputs/review6-committed-000816/.../00000.jsonl.zst`. It records reward 1.0, `13 passed,
  13 total`, stop at `max_turns`, 64 assistant replies all ending in tool calls, and 83 tool
  calls (80 `bash`, 3 `edit`) with 82 tool replies, as their report says.
- **Their five-row check.** `001920`, `001321`, `000816`, `001800` and `002990` all exited 1
  untouched in my review 5 pass too, matching what `docs/evaluation.md` says.
- **Music docs.** The new sentence states the tested cap and does not overclaim.

## Not checked

- General with an independent judge. Still only one model endpoint (GLM) here.
- Whether a correct solution passes on Code rows beyond the ones run with a model so far.
- `000348` and `001540` at their full 1,800 s limit at this commit (my pass used 600 s;
  review 6 ran both to the end at the same code minus the clean-up lines).
- Whether removing `xiaomi-common` (must fix 1) breaks anything; that needs the change first.

## Housekeeping

- This review used about 2,750 Prime sandboxes (the full pass, a 4-row smoke test, the
  40-row submodule check and two detail runs). All are deleted; the list is empty.
- I changed nothing in `rl-toolkit`. I fast-forwarded the local checkout to `76084bed`, ran
  the new test against the old `taskset.py` by copying that file in and back, and left the
  working tree clean.
- Raw results: `claude-review-probe-tooling/results/review7_full.jsonl` (one record per image)
  and `review7_submodules.jsonl`. Scripts: `review6_residue_probe.py` (extended: before/after
  state, unreachable count, submodule checks, verifier timeout recorded as `cap600`) and the new
  `review7_submodule_probe.py`. See the "Review 7" section of the tooling README.
