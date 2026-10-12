# Xiaomi Code

The published Code dataset contains 2,698 rows. Keep all rows loadable; apply
the exclusions below in the training or score-selection config until the
corresponding image/verifier issue is repaired. The source lists and raw probe
records are in [`claude-review-probe-tooling/results`](../../claude-review-probe-tooling/results/).

## Known-bad rows

The verifier health pass applied every test patch and ran the original verifier
on an untouched image with Docker's default `PATH` and a login shell. It found
13 rows whose verifier exits 127 regardless of the model:

- Rust tools missing (9): `format-code-task-000238`, `format-code-task-000681`,
  `format-code-task-000797`, `format-code-task-000798`,
  `format-code-task-001083`, `format-code-task-001540`,
  `format-code-task-001649`, `format-code-task-001652`,
  `format-code-task-002793`.
- Other missing prerequisites (4): `format-code-task-001227` (offline Ruby
  gems), `format-code-task-002099` (`okexchaincli`),
  `format-code-task-002879` (`swagger`), and `format-code-task-003041` (`javac`).

The raw ID file contains 13 rows. The Review 3 narrative says “Ten are Rust”
and separately lists four others, which would total 14; the raw verifier
records establish nine Rust rows plus four others. The counts above follow the
raw records. Three Rust rows checked with the needed `RUSTUP_HOME`,
`CARGO_HOME`, and cargo path ran their tests, so a targeted environment repair
may remove those IDs from the exclusion set once verified.

Two more rows pass untouched, which gives a free positive reward:
`format-code-task-001225` and `format-code-task-001597`. Each passed four runs
on the untouched image. Exclude them from both training and reported scores
until their test setup is repaired.

Use native Verifiers selection exclusions so the rows remain in the loaded
taskset and source dataset. Until the conditional Rust, Java, and Go path
repairs are validated, exclude these 16 rows (the 13 verifier failures, two
free-positive rows, and the partial clone):

```sh
uv run vf-eval xiaomi-code -n 1 -m MODEL \
  --select.exclude.ids format-code-task-000238 format-code-task-000681 \
    format-code-task-000797 format-code-task-000798 format-code-task-001083 \
    format-code-task-001227 format-code-task-001225 format-code-task-001597 \
    format-code-task-001540 format-code-task-001649 format-code-task-001652 \
    format-code-task-002099 format-code-task-002793 format-code-task-002879 \
    format-code-task-003041 format-code-task-002295 \
  --env.agent.runtime.type prime \
  --env.agent.runtime.memory 8 --env.agent.runtime.disk 20 --no-push
```

After the tool-path repairs are verified, remove the nine Rust IDs,
`format-code-task-002099`, and `format-code-task-003041` from the exclusion
list. Keep `format-code-task-001227` and `format-code-task-002879` excluded for
the missing gems and swagger, keep the two free-positive rows excluded, and
keep `format-code-task-002295` excluded while setup fails closed on its
promisor residue. Native exclusion also works with an include list and takes
precedence over inclusion.
For Prime RL training, use `orchestrator.train.source.select.exclude.ids`
with the same list; see [General's selection example](general.md) for the TOML
form. Do not delete rows from `code.parquet` or filter them out of
`XiaomiCodeTaskset.load`.

This partial clone retains unreachable commits in `.promisor` packs after
garbage collection, so the proposed strip does not remove its future history.
The Code implementation is planned to fail closed on it. Keep the row excluded
unless a safe promisor-aware repair removes the residue. The 618
reachable-history failures can be handled by the strip; an exclusion list
alone does not fix the 2,163 measured images with newer unreachable commits.

The 13 rows in `sandbox_died_4gb_10gb_ids.txt` are a separate resource issue,
not extra constant-zero rows. All 13 completed after memory increased to 8 GB
with 20 GB disk; the 4 GB / 20 GB rerun still died. The verifier-only rerun
does not establish memory needs while an agent is also running, so use 8 GB as
the documented minimum and recheck before reducing it.

## Running Code evaluations

Set Prime runtime memory to 8 GB for Code and use a bounded one-row smoke run
when validating changes. The review's source data were collected at commit
`58615037`. The selector and memory example above use the workspace's
`vf-eval` runner; the separate pinned Prime RL runner is documented in
[evaluation.md](evaluation.md#prime-rl-eval-runner).

Probe a small representative set after the Code implementation is accepted:

- `format-code-task-000820`, `format-code-task-001525`, and
  `format-code-task-002737`: setup should pass on the published Git layouts.
- `format-code-task-000121`: on the small ordinary clone, a bounded
  `git fsck --unreachable --no-reflogs` should find no remaining commits after
  setup. Do not run an unbounded `fsck` across all images; it timed out on 11
  measured rows and killed the sandbox on the ten largest repositories.
- `format-code-task-001072` and `format-code-task-001138`: untouched verifiers
  should reach normal test failures rather than a missing command or module.
- One row from `sandbox_died_4gb_10gb_ids.txt`: confirm it completes at 8 GB.
- `format-code-task-002295`: confirm setup fails closed while promisor residue
  remains.

The two Review 3 Code live evals used Prime RL `0.9.0` at commit
`ae28fb70c59197290defd3599dbf1cdddd3a1c40`, Verifiers `0.3.2.dev206`, a
Python 3.12 `.eval-venv`, and the GLM endpoint documented in
[`evaluation.md`](evaluation.md#prime-rl-eval-runner). This differs from the
workspace `.venv` (Python 3.11). Keep the model key unset when using the
unauthenticated endpoint; the General judge key is a separate setting and is
not needed for Code.

## Limits of the evidence

The full Git pass measured 2,688 rows; 10 sandboxes died during the probe's
`git fsck`. Its reachability results therefore describe measured rows only.
For strip comparison, the script reported 0 reachable future commits after
stripping all 2,688 measured images and unchanged `HEAD`, tree, and status.
Unreachable-object checks found newer commits on 2,163 measured images. One
partial clone (`002295`) is a known exception, and `fsck` timed out on 11 rows;
neither should be generalized into a success claim.

The strip's verifier comparison covered 69 selected rows and preserved exit
codes; it is not a full behavioral proof. Login-shell comparisons covered all
224 images whose tool resolution differed, and showed 29 verifier failures
under the current non-login command that work with a login shell. The first
two live examples (`001072`, `001138`) establish those failures in evaluation,
not the fix. Review 3 did not apply its proposed changes to the port.

Finally, the General self-judge reward of 0.875 described in
[evaluation.md](evaluation.md#verified-smoke-run) used the policy model as its
judge. It checks the execution path, not an independent benchmark score.
