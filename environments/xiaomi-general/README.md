# xiaomi-general

Native Verifiers v1 `GeneralTaskset` for all 989 Xiaomi General rows.
Install this package with the sibling `xiaomi-common` package and the repository's
pinned Verifiers revision. Use the stock bash harness and a container runtime
supporting `agent_user`; set controller `GA_JUDGE_KEY`, `GA_JUDGE_URL`,
`GA_JUDGE_MODEL`, and `GA_JUDGE_API=chat` for grading.

See repository `docs/general.md` for the MCP transport, Unix-user isolation
boundary, exact released verifier contracts, and the 64 Terminal Bench rows.
Original source attribution is in NOTICE and LICENSE.
