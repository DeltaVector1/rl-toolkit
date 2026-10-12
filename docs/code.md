# Xiaomi Code

The published Code dataset has 2,698 rows. Keep every row loadable and apply
native Verifiers exclusions at selection time. The current verified exclusion
set has five rows, leaving 2,693 rows eligible for training and reported
scores:

| ID | Reason |
| --- | --- |
| `format-code-task-001227` | Required Ruby gems are absent from the offline image. |
| `format-code-task-002879` | `swagger` is absent from the image. |
| `format-code-task-001225` | Untouched verifier passes (free positive). |
| `format-code-task-001597` | Untouched verifier passes (free positive). |
| `format-code-task-002295` | Partial clone retains promisor residue; setup fails closed. |

Use the same exclusions for training and scoring. Keep the rows in
`code.parquet` and `XiaomiCodeTaskset.load`:

```sh
uv run vf-eval xiaomi-code -n 1 -m MODEL \
  --select.exclude.ids format-code-task-001227 format-code-task-002879 \
    format-code-task-001225 format-code-task-001597 format-code-task-002295 \
  --env.agent.runtime.type prime \
  --env.agent.runtime.memory 8 --env.agent.runtime.disk 20 --no-push
```

For Prime RL training, set `orchestrator.train.source.select.exclude.ids` to
the same list; see [General](general.md) for the TOML form. The selection
options are native Verifiers options, so excluded rows remain in the taskset.

Code setup repairs missing Rust, Java, and Go tool paths in the published
images. Executable/version checks passed for all nine Rust rows, Java row
`003041`, and Go row `002099`. Original verifiers were run on three Rust
examples (`001083`, `001649`, `001652`), Java `003041`, and Go `002099`; they
reached their normal task outcomes. Keep these rows eligible. Setup also strips unreachable
Git history for ordinary clones and preserves `HEAD`, tree, and status in
probes. A bounded `git fsck --unreachable --no-reflogs` on small clone
`000121` found no unreachable commits after setup. Partial clone `002295`
retains promisor objects and is rejected before pruning. The probe does not
establish that every promisor/cache residue can be sanitized, so such clones
remain fail-closed.

The native Code memory default is 8 GB; request 20 GB disk for these runs.
All 13 rows that died in the earlier 4 GB/10 GB probe completed their original
verifier with 8 GB memory and 20 GB disk requested; the 4 GB/20 GB comparison
still died. Gradle row `001739` completed its original test at native 8 GB,
with the sandbox reporting 8,218,440 kB
`MemTotal`. Keep 8 GB as the documented minimum.

## Evidence and limits

The Review 3 raw verifier records list 13 failures: nine Rust rows and four
other rows. The report's “13 unverifiable” count conflicts with its narrative
of ten Rust plus four others (14); the raw IDs establish nine plus four. The
four other rows are `001227`, `002099`, `002879`, and `003041`. Tool-path
repairs resolved the `002099` and `003041` startup failures. Three Rust
verifier examples reached their requested compile/test outcomes, including
ordinary task failures; this is not evidence of verifier incompatibility.

The full Git pass measured 2,688 rows; 10 sandboxes died during `git fsck` and
remain unmeasured. On measured rows, strip comparison reported zero reachable
future commits after stripping and unchanged `HEAD`, tree, and status.
Unreachable newer commits were present on 2,163 measured images. `fsck` timed
out on 11 rows. The partial clone `002295` is a known exception; these results
do not show that every repository is sanitized.

The strip's verifier comparison covered 69 selected rows and preserved exit
codes, not all task behavior. Login-shell comparisons covered 224 images with
different tool resolution and found 29 failures under the non-login command
that worked with a login shell. Raw probe records are in the companion
`claude-review-probe-tooling/results/` directory when that workspace is
available.

The post-fix Code evaluation on `001138` used the pinned Prime RL runner and
GLM model documented in [evaluation.md](evaluation.md#review-3-live-evidence).
At a 16k sampling cap the provider ended with `finish_reason=length` and no
final answer, so that attempt is incomplete. At 32k it completed normally in
49 calls with a final answer and no runtime errors, but scored 0: seven of the
original eight tests passed and the scalar `SimpleQueryString.fields` contract
failed. This is a model/task result, not a missing tool or Django failure.

The General reward of 0.875 in [evaluation.md](evaluation.md#verified-smoke-run)
used the policy model as its own judge; it is an execution-path check, not an
independent benchmark score.
