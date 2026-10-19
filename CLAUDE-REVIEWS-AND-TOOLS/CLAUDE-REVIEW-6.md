# Review 6: closing the leftover checks at `27c2acd8`

Branch `Xiaomi-Envs` at `27c2acd8`. That commit only changes `docs/evaluation.md`. The code is
the same as `6d1030cd`, which review 5 passed.

## Verdict

**Still good. Still ready to use.** Nothing must be fixed. This round closes the checks that
reviews 4 and 5 listed as "not checked". It found one flaky row that should be excluded
(should fix 1), one optional hardening step (nice to have 1) and one doc tip (nice to have 2).

## What this change does

`27c2acd8` adds a "Review 5 full Code-image validation" section to `docs/evaluation.md`. I
read it against the review 5 results. It is accurate.

## What I did

Everything ran on the untouched repo at `27c2acd8` with no patches, except the one fix test
in nice to have 1, which patches the code in memory only.

1. Ran `git fsck` to the end on all 2,698 Code images after the real setup, with no size limit
   and the exit status recorded.
2. Scanned every image's `.git` after setup for files that still name commits the strip
   deleted.
3. Re-ran the six rows that hit my 300 s cap in review 5, this time with each task's own
   1,800 s timeout.
4. Ran GLM-5.2 as the agent on Code (six rows), General (two rows) and Music (60 rows).
5. Scored the same Music replies under four `abcmidi` versions.

## Results

### 1. `git fsck` on every image

This closes "90 large repos not measured" and "fsck exit status not recorded".

| Check | Result |
| --- | --- |
| Setup passes | 2,697 (only `002295` fails, as documented) |
| fsck timed out | 0 (longest run 10 s; largest `.git` 3,197 MB) |
| Unreachable objects of any kind | 0 on all 2,697 |
| Commits outside the base | 0 |
| `HEAD` equals the base | all 2,697 |
| Object store | one pack, no loose objects, no garbage, on all 2,697 |
| The 90 repos of 400 MB or more | all exit 0 |
| fsck exit 0 | 2,693 |
| fsck exit 16 | 4: `001321`, `000816`, `001800`, `002990` (see nice to have 1) |

### 2. The six rows that hit my 300 s cap

| Row | Time | Exit | Note |
| --- | --- | --- | --- |
| `000797` | 389 s | 101 | Rust tests run and fail on the untouched tree, as expected |
| `001540` | 638 s | 101 | same |
| `002643` | 478 s | 9 | same as the review 3 baseline |
| `000348` | 659 s | 1 | tests run |
| `002606` | 785 s | 1 | `1 failed, 230 passed` |
| `000252` | 1,800 s | 124 | hit its own timeout on a Maven download (should fix 1) |

I ran `000252` twice more. Both stalled on the same Maven download and timed out at 1,800 s
(exit 124). See should fix 1.

### 3. Agent runs with GLM-5.2

**Code.** Six rows: `001840`, `001166` and `001621` (the three that review 4 found broken
and that the fixes repaired), plus `000820`, `001083` and `000223`. Five scored 1.0 (three ended with `agent_completed`, two with
`max_turns`). `000223` scored 0.0: the agent used all 64 turns and left 1 of 65 tests failing
(`test_isort.py::test_shouldnt_add_lines`). The verifier ran normally there, so that 0 is
the agent's, not the env's. No errors on any row. The first `000223` run was cut off by the
overnight stop; this is a clean rerun.

**General.** `candidate-0036-software-data-engineering` (Terminal Bench) scored 1.0 and
`s3k_0000_accounting_audit_tax_en_t1_rl_008` (judged) scored 0.8. No errors; both ended with
`agent_completed`. GLM was also the judge, so this checks the plumbing, not judge quality.

**Music.** First 60 rows, with `--sampling.max-tokens 80000`. No errors; all 60 ended with
`agent_completed`. 57 replies finished normally and 3 hit the token cap. Mean reward 0.296,
best 0.852, 34 zeros.

Why the zeros: 2 replies were empty (cut off while still reasoning). The other 32 contain ABC
that upstream's scorer rejects. 27 have abc2midi errors or 10+ "Bar N has" warnings, and 5
make a MIDI file with no notes in it (`no_notes`). I checked that the port's vendored scorer
(`feats.py`, `core.py`, `score.py`, `pipeline.py`) is byte-identical to upstream's
`recipes/design/music/scorer/`, so these zeros are upstream's own scoring, not a port bug.

An earlier try at `--sampling.max-tokens 16000` was mostly cut off: 38 of 40 replies hit
the cap, 36 were empty, and 37 scored 0. GLM spends most of its budget reasoning. See nice to
have 2.

### 4. `abcmidi` version

Upstream does not pin a version: `recipes/design/run_music.sh` runs
`apt-get install -y abcmidi`. So the version depends on the image's distribution. I
scored the 58 non-empty replies above, plus upstream's health-check tune, in four images:

| Image | abcmidi | Python |
| --- | --- | --- |
| `python:3.11-slim` (the port's default) | 5.02, Feb 2025 (Debian trixie) | 3.11 |
| `python:3.11-slim-bookworm` | 4.84, Jan 2023 | 3.11 |
| `ubuntu:24.04` | 4.88, Feb 2024 | 3.12 |
| `ubuntu:22.04` | 4.68, Feb 2022 | 3.10 |

**All 59 scores are identical in all four images**, and they equal the live rewards from
the eval. This matters because the scorer rejects on abc2midi's error and warning text, so a
version change could have moved scores. On these replies it did not, including the 27 that
were rejected for errors or warnings. This closes the `abcmidi` question that has been open
since review 1.

## Must fix

None.

## Should fix

**1. Exclude `000252`: it times out most of the time on a Maven download.**

- Where: the exclusion list in `docs/code.md` and the matching docs in `docs/evaluation.md`.
- Problem: `000252`'s verifier downloads Maven files from `mirror.iscas.ac.cn/kunpeng` while
  it grades, and that download often hangs inside the sandbox. In all the runs I have for
  this row:

  | Run | Result |
  | --- | --- |
  | Review 3 baseline | exit 124, stalled on a `kunpeng` download |
  | Review 3 rerun | exit 124, still compiling 3,195 files at the timeout |
  | Review 4, two runs | finished in about 540 s, exit 1 |
  | Review 6, three runs | exit 124, all stalled on `.../geronimo/genesis/genesis/2.3/genesis-2.3.pom` |

  That is 5 timeouts out of 7 runs. From my own machine the same URL returns 404 in 2 s, so
  the hang is in the network path from the sandbox, not in the port's code. When it hangs, any
  solution scores 0, correct or not.
- Fix: add `format-code-task-000252` to the exclusion list, with one line saying why (outside
  Maven mirror, times out in 5 of 7 runs). That leaves 2,691 eligible rows. This is the same
  treatment as flaky `001217`.
- If we skip it: this row gives a 0 most of the time, whatever the agent does. It is one row
  out of 2,692, so it adds noise, not a wrong result elsewhere.

## Nice to have

**1. Delete `ORIG_HEAD`, `FETCH_HEAD` and the old commit-graph before `git gc`.**

- Where: `environments/xiaomi-code/xiaomi_code/taskset.py:75-76`, in `strip_history_command`.
- Problem: the strip removes the future commits, but three kinds of file can still name
  them after setup:
  - `ORIG_HEAD` names a deleted commit on 608 images. On the images I checked before setup
    it is a real later upstream commit: a branch tip 1 to 6,369 commits ahead of the base.
  - `FETCH_HEAD` survives on 21 images and holds the upstream URL. On 6 of them
    (`000965`, `001448`, `001920`, `002173`, `002462`, `002997`) it also names deleted
    commits.
  - On 4 images (`001321`, `000816`, `001800`, `002990`) no refs are left after the strip,
    so `git gc` does not rewrite the commit-graph. The image's old one survives and names
    one deleted commit. That is why `git fsck` exits 16 there. The commit is a "task base"
    root commit made when the image was built, so only its hash and date are left. `git
    log` and `git status` work normally.

  Only the hashes are left, not the contents. But Code sandboxes have open internet: the
  runtime resolves to `allow: ["*"]`, and `curl github.com` returns 200. On `001920` after
  setup, `git fetch --depth 1 <URL from FETCH_HEAD> <hash from ORIG_HEAD>` worked, and
  `git diff --stat HEAD <hash>` showed 162 changed files.
- Why this is not "must fix": upstream's own strip (`_strip_future_commits` in
  `mimoagent/environments/datasets/base.py`) leaves the same files, so the port already
  matches or beats upstream. With open internet an agent can also clone the upstream repo
  by name. Upstream's only guard is an optional `answer_leak_blocklist` that blackholes
  hosts in `/etc/hosts`; it is off by default and no shipped config turns it on.
- Fix: add two lines just before the `git -c gc.pruneExpire=now ...` line:

  ```sh
  rm -f -- "$git_dir/ORIG_HEAD" "$git_dir/FETCH_HEAD"
  rm -rf -- "$git_dir/objects/info/commit-graph" "$git_dir/objects/info/commit-graphs"
  ```

  I tested this as an in-memory patch on 77 rows: every `FETCH_HEAD` row, every
  commit-graph row, 37 `ORIG_HEAD` rows, and some controls. Setup passed on 76 (`002295`
  fails closed as before). No file named a missing hash. `git commit-graph verify` and
  `git fsck` exited 0 on all 76. The verifier exit codes matched review 5 on all 76.
  `git gc` writes a fresh commit-graph wherever refs remain. If you take it, a test that
  plants these three files and checks they are gone would pin it.
- Also worth knowing: if you want offline Code sandboxes, the Verifiers runtime has
  allow/block network options. That is a choice about the whole environment, not a bug in
  the port.
- If we skip it: we stay at upstream parity. An agent that reads `ORIG_HEAD` and fetches it
  can see the future upstream code, which it could also get by cloning the repo.

**2. Give a concrete token cap for Music.**

- Where: `docs/music.md` or the "Run an evaluation" part of `docs/evaluation.md`.
- Problem: `docs/evaluation.md:68` says the token cap includes reasoning and to raise it for
  long output. That is true, but the only example command uses 16,000. With GLM-5.2, Music at
  16,000 left 36 of 40 replies empty. At 80,000, 3 of 60 were cut off and 2 were empty.
- Fix: add one line to the Music docs, for example "reasoning models need a large
  `--sampling.max-tokens` for Music; GLM-5.2 needed about 80,000". Optional.
- If we skip it: someone runs Music with the example cap, sees mostly zeros, and may think
  the scorer is broken.

## Checked, no finding

- **The new docs section.** `docs/evaluation.md` at `27c2acd8` matches the review 5 results.
- **Nothing else in `.git` names deleted commits.** No reflog bytes, no bitmaps, and no
  `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD`, `BISECT_LOG`, `refs/original`, stash or
  grafts on any image. `config` keeps only submodule URLs and `url.insteadOf`; no `remote.`
  entries. Two kinds of hit in my scan are not leaks: `objects/info/packs` holds pack names
  (every row), and `index` holds tracked file names (54 rows).
- **Music zeros are upstream's.** The vendored scorer is byte-identical to upstream, and
  every zero traces to a scorer reject reason or an empty reply.
- **Speed.** fsck finished in 10 s or less everywhere, so a full fsck in setup would be
  cheap if you ever want one.

## Not checked

- The 40 images that have submodule Git directories under `.git/modules`. The strip does not
  touch them, and I did not scan them.
- General with an independent judge. Only one model endpoint (GLM) is available to me, and
  there are no judge credentials here.
- Whether a correct solution passes on Code rows beyond the ones run with a model so far.
- Why upstream's MIDI parser finds no notes in 5 tunes that look valid. The port matches
  upstream byte for byte, so this is a question about upstream's scorer, not the port.
- Music rows beyond the first 60, and a second Music model.

## Housekeeping

- This round used about 5,700 Prime sandboxes, mostly the two full passes over 2,698 images
  (fsck and the `.git` scan). All are deleted; the list is empty.
- The runs were stopped overnight at your request and restarted the next day. The killed
  runs were rerun from the start, not resumed, and the numbers above come from the reruns.
- I changed nothing in `rl-toolkit`. It is at `27c2acd8` with a clean working tree.
- Raw results are in `claude-review-probe-tooling/results/` (`review6_*`). The new scripts
  are `review6_fsck_probe.py`, `review6_residue_probe.py` and `review6_abcmidi_versions.py`.
  See the "Review 6" section of the tooling README.
