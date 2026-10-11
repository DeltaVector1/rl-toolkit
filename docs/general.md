# Xiaomi General

Native Verifiers v1 package `xiaomi-general` exports `GeneralTaskset`. It loads the
989 released General rows without filtering: 925 `general_agent` tasks and 64
`terminal_bench` tasks. Prompts remain exact dataset messages; published image
mapping and the pinned dataset revision come from `xiaomi-common`. Full General
assets download only during selected task setup. `task_root` optionally points
to an existing local General directory containing `envs/`.

General-agent tasks use each row's original workspace, SQLite state, tool module,
`mcp_http.py` signature inference/result-size caps, and verifier files. A native
colocated MCP Toolset registers tools as `mcp__<server>__<function>`. This replaces
the released sidecar HTTP process/stdio bridge, retaining the task function logic.
The stock bash harness supplies local command/edit tools. Task-specific setup
commands other than the released sidecar entrypoint fail explicitly.

The released two-container filesystem isolation is replaced by a Unix user
boundary in one sandbox: root owns the backing databases, tool code, native MCP
server, and grading files; the harness executes as `xiaomi_agent`. Setup checks
that this user cannot read the protected directories. The workspace root is
root-owned with its sticky bit set, protecting the native server's root-owned
installation directories from agent renames, while workspace files remain editable.
The pinned Prime runtime uploads files as root even for an agent-user view. A
per-runtime adapter writes only this trace's native harness transcript as
`xiaomi_agent`, so the harness can consume and delete it in the sticky directory.
Chunked writes also support resumed transcripts without shell argument-size limits;
all other uploads retain their native behavior.
Verifier paths are root-owned symlinks into a root-only directory, populated
only when grading. Original basenames are preserved inside that private directory
so verifiers resolving sibling metadata and helpers still work. Upload APIs cannot
expose payloads by replacing file modes. This is not equivalent to physical container isolation and
assumes the image provides a sound unprivileged-user boundary. A runtime without
user switching is rejected. Do not run these tasks on the host subprocess runtime.

Set `GA_JUDGE_KEY`, `GA_JUDGE_URL`, `GA_JUDGE_MODEL`, and `GA_JUDGE_API=chat` in the
controller environment for the hosted judge. Credentials are forwarded only to
the verifier process, never stored in task data. The original judge requirement
is enforced even for tasks with deterministic rubrics. An existing nonempty
`/work/workspace/answer.md` wins; otherwise the final assistant reply is saved
there. Original `run_verify.py` and `verify.py` calculate rewards. Missing,
malformed, nonfinite, or out-of-range rewards raise infrastructure errors rather
than silently grading the policy zero. Binary full-score threshold remains
`1 - 1e-6`. Verifier details and return code are retained in trace info.

The 64 terminal rows have no released environment wrapper. Their supplied inline
base64 `tests_files` are staged under `/tests` only for grading, then the original
`bash /tests/test.sh` runs in the task image. The adapter reads
`/logs/verifier/reward.txt`, independently of command exit status: the anti-hack
guard can exit zero while explicitly writing zero reward. No replacement tests
or inferred scores are used. This released inline-verifier contract is the port's
basis for those rows.

General uses the pinned native MCP package bundler to stage the local
`xiaomi-common` dependency before Verifiers stages its tool server; this is needed
because that sibling package has not been published to a package index.
