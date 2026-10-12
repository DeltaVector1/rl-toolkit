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
  --env.agent.runtime.cpu 2 --env.agent.runtime.memory 8 --env.agent.runtime.disk 20 --no-push
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
  --env.agent.runtime.memory 8 --env.agent.runtime.disk 20 \
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
`PATH`. That run used a Go-only setup repair, later replaced by the Docker-default
`PATH` supplied to every Code task.

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

Review 2 integrated the picklable no-toolsets function and documented the three
Terminal Bench rows with missing verifier dependencies; all 989 rows remain
loadable, with explicit native exclusions recommended for training and reported
scores. The repo `.venv` suite passed 73 tests after the pickle fix. The post-fix
Code probe covered 14 rows, including 11 of the 16 rows flagged by the
older PATH probe; five were not rerun (`000169`, `000170`, `000305`, `000947`,
`001576`). No Code implementation changed in Review 2. A fresh run of
`s3k_0000_accounting_audit_tax_en_t1_rl_008` completed with final
`finish_reason=stop`, `agent_completed`, submission validation true, 9 agent model
calls, no timeout or trace errors, and original verifier return code 0 / reward
0.875 (5/6). The verifier produced five actual one-vote LLM results (four pass,
one fail on `protection_statuses`) plus one passing deterministic check, with
empty stderr. This confirms the configured hosted judge returned votes; the
nonempty `GA_JUDGE_KEY` requirement alone would only establish setup readiness.
The run is under `outputs/review2-general-accounting-audit/`.

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

## Claude review follow-up (2026-10-11)

The review fixes were integrated as commits `1afbd32f` and `0b6caef4`. The full
workspace suite passed **72 tests** using the repository's absolute
`.venv/bin/python`. Cross-review checked all 2,698 Code test patches against Git
path handling and reconstructed all 989 General rows through the worker path.
The 64 Terminal Bench guard checks used controlled task fixtures with the
installer stubbed; they were separate from the one live Terminal Bench replay.

The live `candidate-0036-software-data-engineering` replay completed with
`finish_reason=stop`, `agent_completed`, reward **1.0**, verifier return code 0,
five passing tests, and empty stderr. Its Prime sandbox reported 1 CPU, 2 GB
memory, 10 GB disk, and an allowlist containing only the model tunnel host. The
trace is under `outputs/review1-tb-0036-postfix/`.

Live Code row `format-code-task-000604` confirmed the Docker-default `PATH`:
both `npm` and `npx` resolved in the task image and all eight base tests passed.
The rollout ended normally, but reward was **0.0** because two new Jest suites
failed TypeScript compilation (`TS2345`: updater callbacks returned `void` where
the setter type requires state). The model's final message claimed all tests
passed, which contradicted the verifier. This run confirms tool availability,
not a successful solution. Its trace is under `outputs/review1-code-000604/`.

Live Code row `format-code-task-000121` completed in 33 turns with
`finish_reason=stop`, `agent_completed`, reward **1.0**, and no timeout or
runtime error. Its original verifier returned 0; the base test and all four
injected Go tests passed with empty stderr. Inside the sandbox, `go` resolved to
`/usr/local/bin/go` (Go 1.26.3). The trace is under
`outputs/review1-code-000121/`.

The older `path_probe.py` `port` mode still hard-codes the Go-only repair; after
the Docker-default `PATH` change, compare against its `docker-c` mode. The
General package's use of Verifiers' private `_cached_package` and `_package_dir`
helpers remains pinned debt; those helpers were present at both supported
Verifiers revisions and should be rechecked on upgrade.

## Review 3 live evidence

The Review 3 probes used the sibling Python 3.12 Prime RL environment at
`../.eval-venv` from this repository, Prime RL 0.9.0 commit
`ae28fb70c59197290defd3599dbf1cdddd3a1c40`, Verifiers
`0.3.2.dev206` commit `2cb0a6d33a49de0752bbcf9e9a139104da9f5bc4`, and
`zai-org/GLM-5.2-FP8` at `http://38.255.28.19:20003/v1`. The model key was
unset. This local workspace layout is specific to the review environment; the
clean-install recipe above uses in-repository `.eval-venv` paths.

Code setup passed for `000820`, `001525`, `002737`, and `000121`, with
`HEAD`/tree/status preserved. The bounded `fsck` on `000121` found no
unreachable commits. Executable/version checks passed for Rust rows `000238`,
`000681`, `000797`, `000798`, `001083`, `001540`, `001649`, `001652`, and
`002793`, Java `003041`, and Go `002099`. Original verifiers were run on Rust
examples `001083`, `001649`, `001652`, Java `003041`, and Go `002099` and
reached their normal task outcomes.
The examples `001072` and `001138` reached normal tests rather than missing
Python/Django errors. Free-positive rows `001225` and `001597` still passed
untouched; partial clone `002295` failed closed. Gradle row `001739` completed
at native 8 GB memory with 20 GB disk requested.

The 16k Code run on `001138` ended with provider `finish_reason=length` and an
empty final response; it is incomplete. The 32k rerun (`outputs/review3-code-001138-fixed-full/`)
completed in 49 calls with `finish_reason=stop`, final response present, no
trace errors, and no timeout or truncation. The eval requested runtime
configuration of 2 CPU, 8 GB memory, and 20 GB disk. It scored 0: the original
verifier ran eight tests, with seven passing and one failing because scalar
`SimpleQueryString.fields` was emitted as `['name']` instead of `'name'`.
This confirms a model contract failure, not a runtime or missing dependency issue.

The General redaction run on
`s3k_0000_accounting_audit_tax_en_t1_rl_008` completed in eight model calls
with `finish_reason=stop`, `agent_completed`, no timeout or trace errors, and
the final response present. The original verifier returned 0 with reward 0.8
(four of six checks passed); stderr was empty. The resolved runtime
configuration requested 2 CPU, 4 GB memory, and 10 GB disk. The run used a conspicuous fake judge-key
sentinel; a scan of the saved trace found it absent. No real judge credential
was supplied. The policy model also served as the judge, so this validates the
redaction path and judge integration, not an independent score. The trace is
under `outputs/review3-general-redaction-fixed/`. The integrated workspace
suite passed all 103 tests after the source fixes. The owned Prime sandboxes
were cleaned up; `prime sandbox list` reported none remaining.

## Review 4 committed-source validation

The supplied Review 4 full-pass file has 2,698 records: 2,693 immediate setup
passes, four actual setup failures, and one transient sandbox error whose
successful rerun brings the total to 2,694 setup passes. Its 85-row
combined-fix check has 84 rows reaching verification after successful setup
and the expected fail-closed `002295`. Both were against pre-fix source
`4b5f3251`; the four fixes in the
85-row check were in-memory patches. These results identify affected rows but
do not validate the committed fixes. After integrating `83d17b2d` and
`c163f87a`, the main Python 3.11 environment passed all **107 tests**.
`git diff --check` passed at source hash
`c163f87ac458d846357c783edf98f6442ce6f200`.

Real-source setup and untouched verification ran on eight targeted rows:
`000484`, `000820`, `001072`, `001166`, `002071`, `001621`, `001840`, and
`000050`. All reached their original verifiers and retained the task-base
`HEAD`; the pyc-only rows kept their two pytest bytecode files. Their reward
was 0 because the untouched task tests fail (or, for `000050`, fail during
pytest collection), as expected. The separate partial-clone row `002295`
still failed closed during setup. Post-setup shell checks on `000484`,
`000820`, and `001072` confirmed that child bash inherits a parent-added
`PATH`, `bash -eu` runs, and the bootstrap does not leave `BASH_ENV` exported.
On `000484`, `bash -lc true` took 0.4 seconds with no pyenv lock warning.

The committed-source Prime RL run
`outputs/review4-committed-000484-c163f87/` selected `000484` and completed one
GLM-5.2-FP8 rollout in 20 turns, with 2 CPU, 8 GB memory, 20 GB disk, a
32,000-token sampling cap, and a 64-turn limit requested. The final provider response had
`finish_reason=stop`; the assistant returned a final answer, with no timeout or
truncation. Reward was **1.0** and the original verifier exited 0 with all
three injected tests passing. The trace records 27 tool invocations across 20
provider responses (the harness exposes two tool schemas). Its sandbox is
terminated. This targeted source check does not repeat the 2,698-row sweep.
