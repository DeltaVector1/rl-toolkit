# xiaomi-general

Native Verifiers v1 `GeneralTaskset` for all 989 Xiaomi General rows. The taskset
and data helpers are contained in this package. Its direct dependencies are
Verifiers `>=0.3.2.dev206,<=0.3.2.dev208`, `huggingface-hub>=0.24`, and
`pyarrow>=18`. Install it in a fresh environment from this directory:

```sh
uv venv /tmp/xiaomi-general-venv --python 3.12
uv pip install --python /tmp/xiaomi-general-venv/bin/python .
```

No sibling Xiaomi package or repository checkout is required. The repository
workspace separately pins its Verifiers revision. Use the stock bash harness
and a container runtime supporting `agent_user`; set controller `GA_JUDGE_KEY`, `GA_JUDGE_URL`,
`GA_JUDGE_MODEL`, and `GA_JUDGE_API=chat` for grading.

See repository [docs/general.md](../../docs/general.md) for the MCP transport,
Unix-user isolation boundary, exact released verifier contracts, and the 64
Terminal Bench rows. Three terminal rows have missing verifier dependencies in
their published images; use the documented native exclusions for training and
reported scores (986 rows remain). All 989 rows remain loadable for reproduction.
Original source attribution is in NOTICE and LICENSE.
