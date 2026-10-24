# Xiaomi RL Environments

Native Verifiers v1 ports of Xiaomi MiMo's released Code, General, and Music
tasksets. Each environment declares Verifiers v1 and its direct data dependencies;
the workspace pins a compatible Verifiers revision and loads task data and images
from the published Xiaomi assets.

## Environments

- **Code** — repository tasks with the published command-based verifier.
- **General** — general-agent and Terminal Bench tasks with their original
  workspaces, tools, and verifier contracts. See [docs/general.md](docs/general.md).
- **Music** — ABC composition tasks scored through the published MIDI feature
  scorer. See [docs/music.md](docs/music.md).

## Run an evaluation

Install the workspace with Python 3.11 or newer:

```sh
uv sync
```

Each environment can also be installed on its own in a fresh virtual environment.
For example, install Code with:

```sh
uv venv /tmp/xiaomi-code-venv --python 3.12
uv pip install --python /tmp/xiaomi-code-venv/bin/python ./environments/xiaomi-code
```

Use `./environments/xiaomi-general` or `./environments/xiaomi-music` instead to
install either of the other packages. No repository workspace or sibling Xiaomi
package is needed. Each package accepts Verifiers `0.3.2.dev206` through
`0.3.2.dev208`; this range supports the standalone CLI and Prime RL's native MCP
worker.

Run one selected task against an OpenAI-compatible model endpoint:

```sh
uv run vf-eval xiaomi-code -n 1 -m MODEL \
  --client.base-url URL --client.api-key-var OPENAI_API_KEY \
  --select.include.ids format-code-task-001457 \
  --env.agent.runtime.type prime --env.agent.runtime.cpu 2 \
  --env.agent.runtime.memory 8 --env.agent.runtime.disk 20 --no-push
```

Set `OPENAI_API_KEY` only if the endpoint requires authentication. For Xiaomi
General, also configure the judge variables described in
[docs/general.md](docs/general.md). Task-specific runtime images are selected
from task metadata. See [docs/evaluation.md](docs/evaluation.md) for the
separate Prime RL runner and validated smoke runs.
