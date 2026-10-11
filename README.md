# Xiaomi RL Environments

Native Verifiers v1 ports of Xiaomi MiMo's released Code, General, and Music
tasksets. The workspace uses a pinned Verifiers revision and loads task data and
images from the published Xiaomi assets.

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

Run one selected task against an OpenAI-compatible model endpoint:

```sh
uv run vf-eval xiaomi-code -n 1 -m MODEL \
  --client.base-url URL --client.api-key-var OPENAI_API_KEY \
  --select.include.ids format-code-task-001457 \
  --env.agent.runtime.type prime --env.agent.runtime.cpu 2 \
  --env.agent.runtime.memory 4 --env.agent.runtime.disk 10
```

Set `OPENAI_API_KEY` only if the endpoint requires authentication. For Xiaomi
General, also configure the judge variables described in
[docs/general.md](docs/general.md). Task-specific runtime images are selected
from task metadata. See [docs/evaluation.md](docs/evaluation.md) for the
separate Prime RL runner and validated smoke runs.
