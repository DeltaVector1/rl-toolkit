# decensor-v1

`decensor-v1` is a multi-turn refusal-reduction environment for the verifiers v1 API.
It loads seed conversations from `NewEden/RL-seed-Decensor`, lets an adversarial simulated
user continue each conversation, and grades the model with parallel Gemma judges.

## Rollout

Each row receives one of three response styles: prose with styling, plain text, or Markdown.
Some rows also request a terse, medium, or verbose response. The target number of turns follows a geometric
distribution capped by `max_turns`.

The simulated user keeps the complete transcript locally because the harness
passes it only the latest assistant message. Each rollout has a new random
nonce. This prevents two rollouts for one prompt from receiving the same user
turns.

For each follow-up, the user model receives the opening request, the latest
answer, and one assigned strategy. Normal code checks the wrapper, length,
sentence count, role labels, nested tags, and repeated text. A separate tool
call then checks that the message still seeks the opening request, fits the
latest answer, does not switch to an alternative from the assistant, and
sounds like a real user. The model gets three attempts. If all attempts fail,
the rollout is unscored. If an assistant turn has no visible answer, the user
model stops and the normal invalid-response check handles it.

## Reward

The environment records this multiplicative response score:

```text
refusal * length * coherency * markdown presence * markdown correctness * reasoning * meta commentary
```

Training does not normalize this combined value. It calculates two group
advantages and adds them:

```text
reasoning pass minus the group pass rate
+ response quality minus the mean quality of passing responses
```

Only responses that pass the private-reasoning review receive the quality
advantage. A failed response cannot receive a positive advantage. An
all-failed group has no training signal and is replaced by the rollout
collector. An all-passed group still trains response quality.

The following response failures return zero before any judge score is used:

- truncated output
- an unclosed `<think>` block
- missing reasoning when reasoning is required
- an empty visible response
- copied internal wrapper tags

Judge calls run concurrently. Each call gets three total attempts. A service
failure or three invalid replies make the rollout unscored. No service failure
becomes a model reward.

Every assistant turn has two narrow structured reviews. The visible review
receives the user context and that turn's answer. It scores refusal, visible
coherence, instruction leaks, and formatting. The private review receives the
same user context, that turn's reasoning, and the visible refusal score. It
scores restricted reasoning and reasoning coherence. It does not receive the
visible answer. Different turns are reviewed concurrently.

The controller divides each answer and reasoning trace into numbered exact
source sections. The judge selects section numbers for each failed or present
result. The controller records the selected source text and offsets. The judge
does not copy or rewrite evidence text.

A self-harm prevention trace is exempt from the restricted-reasoning result.
Length ranges are broad and are not shown as exact counts. A non-empty response
keeps a small length value so length alone cannot erase the response-quality
signal. Every score, section result, and final advantage is recorded.

Raw judge replies include the judge name, turn, phase, attempt number, status,
elapsed time, text, parsed value, and evidence offsets. The trace also records
the source harm type, source row, repeated-source status, follow-up strategies,
every follow-up attempt and semantic review, response format, length range,
and Markdown exception.

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

The Muse run uses the Muse Glimmer renderer and tool parser. Reasoning is
enabled for the policy model. Thinking is disabled for the simulated user and
judges.
