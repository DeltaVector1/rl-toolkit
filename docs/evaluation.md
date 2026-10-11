# Running evaluations

The workspace exposes the `xiaomi-code`, `xiaomi-general`, and `xiaomi-music`
tasksets as installable Verifiers v1 packages. From the repository root, install
the workspace and its development tools with `uv sync`; use Python 3.11 or newer.
The `verifiers` source is pinned in the workspace configuration.

Run a taskset with `uv run vf-eval`:

```sh
uv run vf-eval xiaomi-code -n 1 -m MODEL \
  --client.api-key-var GLM_API_KEY \
  --select.include.ids format-code-task-001457 \
  --env.agent.runtime.type prime \
  --env.agent.runtime.cpu 2 --env.agent.runtime.memory 4 --env.agent.runtime.disk 10 --no-push
```

The image is read from the selected task's metadata unless you set
`--env.agent.runtime.image`. For a hosted model endpoint, pass
`--client.base-url URL`; custom OpenAI-compatible endpoints can omit an API key
when the server permits unauthenticated requests. `vf-eval --help` lists the
sampling, selection, runtime, and client options. Keep smoke tests to one
rollout (`-n 1`) and a small number of agent turns.

The hosted Prime CLI does not run local evaluations. `prime eval run` is for
hosted Prime evaluations; use `uv run vf-eval` for local orchestration with a
Prime sandbox runtime.

## Prime RL eval runner

The live smoke evaluations for this port used Prime RL `0.9.0` at commit
`ae28fb70c59197290defd3599dbf1cdddd3a1c40` and its Verifiers submodule at
`2cb0a6d33a49de0752bbcf9e9a139104da9f5bc4` (`0.3.2.dev206`). This is a separate
Python 3.12 environment from the workspace's Verifiers pin. Keep the model
client key separate from Prime CLI credentials; set `GLM_API_KEY` only when
the model endpoint requires it. With a custom base URL, an unset key variable
resolves to `EMPTY` in the client. A clean setup is:

```sh
git clone https://github.com/PrimeIntellect-ai/prime-rl.git .references/prime-rl
git -C .references/prime-rl checkout ae28fb70c59197290defd3599dbf1cdddd3a1c40
git -C .references/prime-rl -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
uv venv --python 3.12 .eval-venv
uv pip install --python .eval-venv/bin/python -e .references/prime-rl
uv pip install --python .eval-venv/bin/python --no-deps \
  -e environments/xiaomi-common -e environments/xiaomi-code \
  -e environments/xiaomi-general -e environments/xiaomi-music
```

The Prime RL install uses its normal dependencies only; no GPU extra is needed
for controller-side evaluation. `--no-deps` on the Xiaomi packages keeps the
Verifiers revision supplied by Prime RL. Put the runner's `bin` directory on
`PATH` because it launches `env-server` by name:

```sh
PATH="$PWD/.eval-venv/bin:$PATH" .eval-venv/bin/eval xiaomi-code -n 1 -r 1 -c 1 \
  -m MODEL --client.base-url URL \
  --client.api-key-var GLM_API_KEY \
  --select.include.ids format-code-task-001457 \
  --env.agent.runtime.type prime --env.agent.runtime.cpu 2 \
  --env.agent.runtime.memory 4 --env.agent.runtime.disk 10 \
  --env.agent.max-turns 64 --sampling.max-tokens 16000
```

`-n 1 -r 1 -c 1` limits the smoke run to one rollout and one inference slot.
Omit `--env.agent.runtime.image` to use the image recorded for the selected
task. Choose a task whose image and required credentials are available. The
inference `max_tokens` cap includes model reasoning as well as the visible
answer; increase it for tasks that require long structured output, then inspect
the provider finish reason and final artifact for truncation.

## Verified smoke run

On 2026-10-11, `format-code-task-001457` completed against
`zai-org/GLM-5.2-FP8` with reward 1.0, no runtime errors or timeout, and no
turn truncation. The original task verifier returned 0 and its 26 Jest tests
passed. The run used one rollout, 22 turns, and the Prime image mapped to that
task. The output transcript and verifier log are under
`outputs/live-code-format-code-task-001457-final/`.

The integrated workspace regression suite passed all 39 tests with Verifiers
`bf2834cac27d882b6edc9388835791a395361954`. A separate metadata audit checked
all 925 General-agent manifests: every manifest matched its dataset task
directory and the adapter's upload/setup/verifier contract. This checked
metadata compatibility, not execution of all 925 task payloads.

After the Ponytail fixes, the integrated workspace suite passed all 59 tests.
The Code row `format-code-task-000121` completed in 53 turns with
`finish_reason=stop`, `agent_completed`, and reward 1.0. Its original verifier
returned 0; both the base and injected Go tests passed with empty stderr. The
replay transcript is under
`outputs/live-code-format-code-task-000121-ponytail/`.
An earlier replay of this row returned 127 because the image omitted Go from
`PATH`; the installed-toolchain setup repair made the final replay pass.

The Music row `gK-0768` completed with an untruncated ABC response containing
56 bars in each of the three voices (`finish_reason=stop`). Its task scorer
returned `music_score=0.0` with empty stderr, so completion confirms runtime and
output structure, not a positive musical score. A separate `pipeline.do()`
diagnostic on the same response reported 20 bar warnings (`err=0`, `blank=0`,
three MIDI channels, no channel conflicts); the scorer rejects responses at
10 or more warnings. The transcript is under
`outputs/live-music-gK-0768-full/`.

The General accounting review row
`s3k_0000_accounting_audit_tax_en_t1_rl_008` completed in 9 turns without
runtime errors, truncation, or timeout, with `finish_reason=stop` and
`agent_completed`. Its original verifier returned 0 and scored 0.875 (5 of 6
checks); `protection_statuses` failed. The first attempt exposed a
sticky-workspace prompt-file ownership error before any model turn; after the
General fix, the same row ran through the original verifier. The audited
transcript is under
`outputs/live-general-accounting-audit-ponytail/`.

For Xiaomi General, configure the external judge in the controller environment
before starting the run: `GA_JUDGE_KEY`, `GA_JUDGE_URL`, `GA_JUDGE_MODEL`, and
`GA_JUDGE_API=chat`. General tasks require a container runtime that supports an
unprivileged `agent_user`; see [general.md](general.md). Music scoring needs the
runtime's audio conversion/scoring tools; see [music.md](music.md).

Run output is written to `outputs/` by default. Each run stores the resolved
configuration, controller log, environment server log, episode transcript, and
metrics under its unique run directory. Confirm completion from the final
episode transcript and verifier reward fields, since a started process or
created sandbox alone does not establish a successful evaluation.
