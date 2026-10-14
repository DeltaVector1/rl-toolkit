# Xiaomi Code

The published dataset has 2,698 rows. Keep every row loadable and apply these
native Verifiers exclusions when selecting training or scoring rows. The seven
excluded rows leave 2,691 eligible:

| ID | Reason |
| --- | --- |
| `format-code-task-001227` | Required Ruby gems are absent from the offline image. |
| `format-code-task-002879` | `swagger` is absent from the image. |
| `format-code-task-001225` | Untouched verifier passes (free positive). |
| `format-code-task-001597` | Untouched verifier passes (free positive). |
| `format-code-task-002295` | Partial clone retains promisor residue; setup fails closed. |
| `format-code-task-001217` | Untouched race test is flaky: it passed in 3 of 11 runs. |
| `format-code-task-000252` | External Maven mirror stalls; the verifier timed out in 5 of 7 reported runs. |

Use the same list for training and scoring:

```sh
uv run vf-eval xiaomi-code -n 1 -m MODEL \
  --select.exclude.ids format-code-task-001227 format-code-task-002879 \
    format-code-task-001225 format-code-task-001597 format-code-task-002295 \
    format-code-task-001217 format-code-task-000252 \
  --env.agent.runtime.type prime \
  --env.agent.runtime.memory 8 --env.agent.runtime.disk 20 --no-push
```

For Prime RL training, set `orchestrator.train.source.select.exclude.ids` to
the same seven IDs; see [General](general.md) for the TOML form. Excluded rows
remain in `code.parquet` and `XiaomiCodeTaskset.load`.

Code setup repairs missing Rust, Java, and Go tool paths in the published
images. Tool checks passed for all nine Rust rows, Java `003041`, and Go
`002099`; original verifiers reached normal outcomes on three Rust examples,
Java `003041`, and Go `002099`. The native memory default is 8 GB; request
20 GB disk. All 13 rows that died in the earlier 4 GB/10 GB probe completed
their original verifier with 8 GB memory and 20 GB disk requested.

Known compatibility limits: setup fails closed on partial clone `002295`,
and evaluation does not establish that every repository can be sanitized or
every verifier/task outcome preserved. See [evaluation evidence](evaluation.md)
for measured scope and historical results.
