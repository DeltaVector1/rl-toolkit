# decensor-v1

`decensor-v1` is a multi-turn refusal-reduction environment for the verifiers v1 API.
It loads seed conversations from `NewEden/RL-seed-Decensor`, lets an adversarial simulated
user continue each conversation, and grades the model with parallel Gemma judges.

## Rollout

Each row receives one of three response styles: prose with styling, plain text, or Markdown.
Some rows also state a response-length target. The target number of turns follows a geometric
distribution capped by `max_turns`.

The simulated user keeps the complete transcript locally because the harness passes it only
the latest assistant message. It generates a short adversarial follow-up after each assistant
turn and stops when the row's target turn count is reached.

## Reward

The environment returns one multiplicative reward:

```text
refusal * length * coherency * markdown presence * markdown correctness * reasoning * meta commentary
```

The following response failures return zero before any judge score is used:

- truncated output
- an unclosed `<think>` block
- missing reasoning when reasoning is required
- an empty visible response
- structured output markers

Judge calls run concurrently. If the Gemma endpoint cannot be reached, the reward is zero and
`decensor_judge_endpoint_down` is logged. If Gemma replies but a verdict cannot be parsed, that
verdict is ignored instead of guessing a failure.

Reasoning judges inspect `reasoning_content` from the final assistant turn. Length targets are
shown to the model, and a non-empty response keeps a small length floor so length alone cannot
erase the entire learning signal. Every reward factor is recorded in the trace metrics.

## Config

```toml
[[orchestrator.train.env]]
name = "decensor"
ratio = 1.0
env.taskset.id = "decensor-v1"
env.agent.harness = { id = "null", runtime = { type = "subprocess" } }

[orchestrator.train.env.env.taskset.task.judge]
base_url = "http://judge-host/v1"
model = "google/gemma-4-26B-A4B-it"
api_key_var = "VLLM_API_KEY"

[orchestrator.train.env.env.taskset.task.user]
base_url = "http://judge-host/v1"
model = "google/gemma-4-26B-A4B-it"
api_key_var = "VLLM_API_KEY"
```

The main run uses the official GLM-5.2 chat template, GLM tool parsing, reasoning enabled for
the policy model, and thinking disabled for the simulated user and judges.
