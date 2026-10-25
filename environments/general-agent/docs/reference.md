# General Agent

A two-phase environment for training LLM agents on tool-use tasks over structured databases.

In phase 1, an agent creates tasks. In phase 2, an agent solves them.

The synthesizer checks each new task with the local solver to measure difficulty.
Only tasks that pass gating are kept.

Further docs: [CLI](cli.md) | [Tests](tests.md)

## Phase 1: Synthesize

The synthesizer spawns an OpenCode agent in a sandbox. The agent follows the `synthesize-task` skill to design a new domain (DB schema, tools, verifier) and evolve it through difficulty tiers (t0-t4). Each tier is gated by two checks:

1. **Gold validation** — `general-agent validate` replays the gold solution and checks DB hashes + `verify(db)`.
2. **Pass-rate gating** — `vf-eval general-agent-solver-local` runs the local solver N times; the tier's pass rate must fall in its target band (t0: >=80%, t1: 60-80%, ..., t4: 0-20%).

The synthesizer uses the measured pass rate to adjust task difficulty.

```bash
# Single synthesis job (agent picks a novel task)
uv run vf-eval general-agent-synth

# 4 parallel synthesis jobs (each picks its own task)
uv run vf-eval general-agent-synth -r4

# Tier-0 only (faster, no difficulty evolution)
uv run vf-eval general-agent-synth -a '{"max_tier":0}'
```

`PRIME_API_KEY` is required — the local solver inside the sandbox calls a reference model for pass-rate gating.

## Phase 2: Solve

Three solver backends, each with a `general-agent-solver-*` entry point:

### Local solver (`general-agent` / `general-agent-solver-local`) — default, for synthesis gating + fast iteration

The default solver calls tools directly as Python functions. It needs no sandbox,
MCP server, or external agent binary. The synthesizer uses it for pass-rate gating;
it also supports local iteration.

```bash
uv run vf-eval general-agent
```

### OpenCode solver (`general-agent-solver-opencode`) — for training

Runs an OpenCode agent in a sandbox with task tools exposed via a local [MCP](https://modelcontextprotocol.io/) server. Each `@tool` method from the task's `tools.py` becomes a native MCP tool that OpenCode can call.

```bash
uv run vf-eval general-agent-solver-opencode
```

**Scoring**: after the rollout, the rubric downloads `db_final.json` from the sandbox (written by the MCP server after each tool call), replays the gold solution locally, and compares DB hashes + runs `verify(db)`.

### RLM solver (`general-agent-solver-rlm`) — for training

Runs an RLM agent in a sandbox with task tools exposed as per-tool skills. RLM calls tools from its IPython kernel using `await <tool>.run(...)`.

```bash
uv run vf-eval general-agent-solver-rlm
```

**Scoring**: same as OpenCode — downloads `db_final.json`, compares DB hashes + `verify(db)`.

## Architecture

```
Phase 1: Synthesize (vf-eval general-agent-synth)
  ComposableEnv + OpenCode harness
    sandbox:
      OpenCode agent follows synthesize-task skill
        ├── creates task: db.json, tools.py, instruction.md, gold.json, verify()
        ├── gates via: general-agent validate (gold replay)
        └── gates via: vf-eval general-agent-solver-local (pass-rate, 2-player)
    post-rollout:
      extract tasks → tasks/

Phase 2: Solve — local [default] (vf-eval general-agent / general-agent-solver-local)
  ToolComposableEnv (vf.ToolEnv)
    policy calls @tool methods directly as Python functions
    scoring: DB hash comparison + verify(db)

Phase 2: Solve — opencode (vf-eval general-agent-solver-opencode)
  ComposableEnv + OpenCode harness
    sandbox:
      OpenCode agent ← MCP → general-agent serve --task <name>
        tools exposed as native MCP tools in opencode.json
    scoring:
      download db_final.json → compare to gold hash + verify(db)

Phase 2: Solve — rlm (vf-eval general-agent-solver-rlm)
  ComposableEnv + RLM harness
    sandbox:
      RLM agent ← IPython → per-tool skill → Tools.call_tool()
        one rlm-skill-* per @tool method, agent calls await <tool>.run(...)
    scoring:
      download db_final.json → compare to gold hash + verify(db)

```

See [cli.md](cli.md) for the literal `general-agent` CLI commands.
