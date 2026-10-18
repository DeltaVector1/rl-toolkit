# Xiaomi General

Native Verifiers v1 package `xiaomi-general` exports `GeneralTaskset`. It loads the
989 released General rows without filtering: 925 `general_agent` tasks and 64
`terminal_bench` tasks. Prompts remain exact dataset messages; published image
mapping and the pinned dataset revision are included in this environment
package. Full General assets download only during selected task setup.
`task_root` optionally points to an existing local General directory containing
`envs/`.

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
the final-answer fallback uses the same unprivileged writer. Other uploads retain
their native behavior.
Verifier paths are root-owned symlinks into a root-only directory, populated
only when grading. Original basenames are preserved inside that private directory
so verifiers resolving sibling metadata and helpers still work. Upload APIs cannot
expose payloads by replacing file modes. This is not equivalent to physical container isolation and
assumes the image provides a sound unprivileged-user boundary. A runtime without
user switching is rejected. Do not run these tasks on the host subprocess runtime.

After the native harness and tool servers close, finalization stops remaining
agent-user processes before checking submission files as that user. Files must
be readable and resolve inside the workspace; internal symlinks are allowed.
External, dangling, or cyclic symlinks are removed before grading, so an ordinary
workspace virtualenv does not invalidate the answer or expose protected files.
Only the protected root-owned `.vf-src` and `.vf-venv` installation directories
are excluded. Invalid submissions receive zero with a reason in verifier trace
info; process-control or validation-execution failures remain infrastructure errors.
The grader's temporary scripts use `/work/_verifier/tmp`, overriding manifest
`TMPDIR`, so agent-created modules in shared `/tmp` cannot change their imports.
Grading also recreates `/tmp/agent_output` as an empty private directory, removing
agent-created session logs or directory aliases before the released runner reads it.

Set `GA_JUDGE_KEY`, `GA_JUDGE_URL`, `GA_JUDGE_MODEL`, and `GA_JUDGE_API=chat` in the
controller environment for the hosted judge. Credentials are forwarded only to
the verifier process, never stored in task data. The original judge requirement
is enforced even for tasks with deterministic rubrics. An existing nonempty
`/work/workspace/answer.md` wins; otherwise the final assistant reply is saved
there. Original `run_verify.py` and `verify.py` calculate rewards. Missing,
malformed, nonfinite, or out-of-range rewards raise infrastructure errors rather
than silently grading the policy zero. Binary full-score threshold remains
`1 - 1e-6`. Verifier details and return code are retained in trace info.
Before trace storage or output clipping, General-agent grading redacts the original
verifier's `key=` log fragments and full configured judge keys from stdout, stderr,
reward details, and grading exceptions. Native error types and retry status codes
are preserved. Judge credentials and original verifier inputs are unchanged. Terminal Bench output retains its original behavior.

The 64 terminal rows have no released environment wrapper and start no MCP server.
This choice is derived from each row when constructing `GeneralTask`, including
native worker and replay reconstruction. No state-server credential is installed
in the root agent's sandbox. The supplied inline
base64 `tests_files` are staged under `/tests` only for grading, then the original
`bash /tests/test.sh` runs in the task image. The adapter reads
`/logs/verifier/reward.txt`, independently of command exit status: the anti-hack
guard can exit zero while explicitly writing zero reward. No replacement tests
or inferred scores are used. This released inline-verifier contract is the port's
basis for those rows.

Terminal tasks stage the pinned Bash harness scripts under `/tmp/vf-scripts`,
with the task's uv cache and Python installs
under `/opt`, outside `/app`. It therefore leaves the task's pristine-tree guard
intact without deleting workspace files. Each terminal row declares one CPU,
2048 MiB memory, 10240 MiB disk, a 900-second agent timeout, and no internet.
These map to native task resources (1 CPU, 2 GB, 10 GB), timeout, and an empty
network allowlist. Prime applies that policy after trusted setup while retaining
framework model routes. Explicit native runtime resource and timeout overrides
retain their normal precedence; a runtime without network-policy support rejects
the task rather than executing it unrestricted.

Three terminal rows have known missing verifier dependencies in the published
images. On untouched images, the original pytest verifier fails during collection
with `ModuleNotFoundError` and writes reward zero:

| Task ID | Dataset image | Missing module |
| --- | --- | --- |
| `candidate-2376-security-cryptography` | `general-agent-env-50:oss` | `cryptography`, imported by vendored `authlib.jose` |
| `candidate-0260-security-appsec` | `general-agent-env-3:oss` | `stevedore`, imported by vendored `bandit` |
| `candidate-0674-ml-evaluation` | `general-agent-env-9:oss` | `torch`, imported by `/tests/test_outputs.py` |

Their prompts restrict dependency additions or installation, and their task
metadata disables internet access. Their intended mutable source files do not
supply these dependencies; this is separate from installing the controller's
Verifiers packages or staging the Bash harness. Do not install missing verifier
dependencies during setup, which would change the released image contract. These
are observed verifier dependency failures, not a proof that every possible source
edit fails. Upstream
released no terminal wrapper, so a separate upstream verifier image is unconfirmed.

Exclude these IDs from training and reported model scores. The loader deliberately
retains all 989 released rows for reproduction. Add this native selection option to
the `vf-eval xiaomi-general` or pinned Prime RL `eval xiaomi-general` command in
[evaluation.md](evaluation.md):

```sh
--select.exclude.ids candidate-2376-security-cryptography \
  candidate-0260-security-appsec candidate-0674-ml-evaluation
```

With no other selection limits, this retains 986 rows: 925 general-agent and 61
terminal rows. Record the exclusions and denominator with reported scores; the
remaining 61 rows have not all been demonstrated solvable. For pinned Prime RL
training, set the exclusion on the General source in the run's TOML:

```toml
[[orchestrator.train.source]]
[orchestrator.train.source.env.taskset]
id = "xiaomi-general"
[orchestrator.train.source.select.exclude]
ids = [
  "candidate-2376-security-cryptography",
  "candidate-0260-security-appsec",
  "candidate-0674-ml-evaluation",
]
```

Merge this into the existing General source rather than adding a duplicate source.
To reproduce a released row, remove its exclusion and use, for example,
`--select.include.ids candidate-2376-security-cryptography`; native exclusion takes
precedence over inclusion. `candidate-1634-software-databases` is not excluded:
its Python 3.12 `collections.Callable` import failure is in the mutable
`vendor/peewee/peewee.py`, so it can be repaired as part of the source task.

General-agent rows use the native MCP package bundler to stage their tool server.
