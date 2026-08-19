# LoRA rank and MoE adaptation

Last updated: 2026-08-14

This document records research evidence and project hypotheses. It is not a
production configuration. A hypothesis is not a measured result.

## Current project facts

- GLM-5.2 has 75 MoE layers, 256 routed experts in each MoE layer, and eight
  active routed experts for each token.
- The router is frozen. Router replay is off.
- Run x84p uses rank-32 LoRA on four attention projections and the always-on
  shared expert. It has 198,503,424 trainable BF16 parameters and exports an
  adapter of approximately 397 MB.
- The measured x84p peak GPU allocation is approximately 44.22 GiB on 48 GiB
  RTX 4090 GPUs. Short-lived CUDA workspaces need part of the remaining space.
- Rank-32 LoRA on every routed expert would add approximately 15.7 billion
  trainable parameters.
- The current native-FP8 routed-expert path does not have a correct fused LoRA
  implementation. The upstream grouped-expert LoRA class creates its LoRA
  tensors with the base-weight dtype and converts base expert weights to BF16
  in its forward path. For this checkpoint, that would create FP8 trainable
  LoRA tensors and remove the bounded native-FP8 memory behavior. Routed-expert
  LoRA therefore needs a new BF16-adapter/native-FP8-base fused path before a
  production configuration can enable it.

## What *LoRA Without Regret* shows

[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) reports rank
sweeps from 1 through 512 on Llama and Qwen models. Its experiments include a
Qwen MoE model, supervised instruction and reasoning training, and
policy-gradient reinforcement learning on mathematics tasks.

The main useful results are:

- Coverage is important. MLP-only LoRA was much better than attention-only
  LoRA. Adding attention LoRA to MLP LoRA gave little extra benefit in the
  reported experiments.
- Low rank can match full fine-tuning early in training. Lower ranks later
  leave the best learning curve when their capacity becomes limiting.
- Rank-1 LoRA matched full fine-tuning in the reported scalar-reward
  mathematics RL experiments, within the noise of those experiments.
- The article's MoE comparison gives each expert its own LoRA. It divides the
  nominal total rank by the number of active experts. With top-8 routing, a
  nominal total rank of 16 means rank 2 for each expert, not rank 16 for each
  expert.
- The authors state that the best way to apply LoRA to MoE layers, including
  its interaction with expert parallelism, is still an open question.
- LoRA was less tolerant of large supervised batches than full fine-tuning in
  one experiment. The gap was independent of rank. This result is relevant to
  the project's batch size of 1,024, but it was not measured on GRPO and must
  not be treated as a project result.

Rank 1 is rank 1 for each adapted matrix. It is not one global trainable
direction. In this GLM model, rank 1 on the routed experts alone is still
approximately 491 million trainable parameters. Parameter count alone does not
prove that it can express the required policy change.

The article's information argument also has a limited scope. It treats a
policy-gradient episode as receiving a scalar advantage and therefore only a
bounded amount of reward information. This supports low-rank RL, but it does
not prove that all scalar-reward tasks have the same intrinsic update rank.
Prompt diversity, sparse expert visitation, reward noise, optimization, and
the number of distinct behavioral modes can still change the required rank.

## Related papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
  introduced LoRA and found low intrinsic rank in several adaptation tasks.
  It does not establish one rank that is sufficient for all tasks.
- [QLoRA](https://arxiv.org/abs/2305.14314) found that applying LoRA to all
  transformer linear layers was important for matching full-fine-tuning
  quality in its instruction-tuning experiments. This supports MLP coverage.
  This project does not use QLoRA, NF4, or INT4 base weights.
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) found
  that standard low-rank LoRA underperformed full fine-tuning for mathematics,
  code, and continued pretraining, while retaining more base-model behavior.
  It found much higher-rank full-fine-tuning updates. This is evidence that
  task and data complexity can make low rank insufficient.
- [MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2405.12130)
  reports that low-rank updates can limit new-knowledge learning and
  memorization. It found smaller differences on ordinary instruction and
  reasoning tasks. This separates behavioral adaptation from knowledge-heavy
  adaptation.
- [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)
  argues that standard `alpha / rank` scaling can weaken higher-rank learning
  and proposes `alpha / sqrt(rank)`. A change to this scaling would change the
  optimization experiment and needs its own gate.
- [AdaLoRA](https://arxiv.org/abs/2303.10512) shows that a uniform rank budget
  can be inefficient because different matrices have different importance.
  This supports different ranks for dense/shared modules and routed experts.
- [A Kernel-Based View of Language Model Fine-Tuning](https://arxiv.org/abs/2210.05643)
  gives a theoretical reason why broad layer coverage can make LoRA dynamics
  closer to full fine-tuning.
- [MixLoRA](https://arxiv.org/abs/2404.15159) reports benefits from multiple
  LoRA-based experts on multi-task training. It builds an adapter MoE on a
  dense model, so it is related evidence and not a direct result for adapting
  the native GLM routed experts.
- [Rethinking the Rank Threshold for LoRA Fine-Tuning](https://arxiv.org/abs/2605.03724)
  is a 2026 preprint. It reports that rank 1 can be sufficient in a restricted
  binary-classification regime, while multi-class tasks move the preferred
  rank above 1. Its assumptions do not cover long-sequence GRPO.
- [Procedural Knowledge Is Not Low-Rank](https://arxiv.org/abs/2607.21612) is a
  2026 preprint. It reports a substantial LoRA/full-fine-tuning gap on complex
  branching procedures even at ranks 32 through 128. The task differs from
  decensor training, but it is evidence against a universal rank-1 claim.

## Project size estimates

These estimates use the measured project counts. They use rank 16 on the
current attention and shared-expert targets. The routed-expert rank changes in
each row. Dense MLP LoRA in the first dense layers is small relative to these
numbers and is not included.

| Routed rank for each expert | Approximate total trainable parameters | BF16 adapter file | Minimum trainer state for each GPU |
|---:|---:|---:|---:|
| 1 | 0.59 B | 1.18 GB | 0.07 GiB |
| 2 | 1.08 B | 2.16 GB | 0.13 GiB |
| 4 | 2.06 B | 4.12 GB | 0.24 GiB |
| 8 | 4.02 B | 8.05 GB | 0.47 GiB |
| 16 | 7.95 B | 15.90 GB | 0.93 GiB |

The minimum trainer state assumes even sharding over 64 GPUs, BF16 adapter
weights, BF16 gradients, and one FP32 optimizer buffer. It excludes LoRA saved
activations, allocator fragmentation, export buffers, and CUDA workspaces.
Actual peak memory will be higher.

With eight active experts, routed rank 16 gives a token up to 128 routed LoRA
directions before the expert outputs are combined. This is not directly equal
to a dense rank-128 update, but it explains why rank 16 for every expert is
much larger than the MoE scaling used in *LoRA Without Regret*.

The adapter file is also operational state. A 15.9 GB adapter is about 40 times
the current file size. It must be exported, copied over the high-latency link,
loaded on both H200 hosts, and reconciled after an engine restart for every
policy version.

## Current hypotheses

### Muse-Glimmer rank-128 coverage decision

Muse-Glimmer-30B is a dense language model with 52 language-model layers. It
does not have routed experts or a router. Inspection of the installed model
code found 416 linear modules in these language-model layers. The rank-128
configuration targets all of them: 260 attention projections and gates, and
156 MLP gate, up, and down projections. It targets no vision-tower module.

This configuration has 838,336,512 BF16 LoRA parameters, or approximately
1.56 GiB of adapter weights. Rank 128 and alpha 256 keep the standard
`alpha / rank` multiplier at 2. This matches the multiplier used by the prior
rank-32, alpha-64 project configuration. It is a recorded optimization choice,
not evidence that this multiplier is optimal for Muse-Glimmer.

Hypothesis: broad language-layer coverage is more important for this task than
an attention-only adapter. Measure finite, nonzero gradients for attention and
MLP LoRA matrices on every pipeline stage. Then use the reward verdict after
at least 10 completed steps and the 50-step stability gate.

### H1: MLP coverage is more important than keeping attention rank 32

The present policy adapts attention and the shared expert, but not the 256
routed experts. The research evidence predicts better learning when LoRA
reaches the MLP/MoE path. A long attention-heavy run that has no reward trend
is consistent with this hypothesis, but does not prove it.

Measure: routed-expert LoRA gradients must be finite and nonzero on all
pipeline stages. After at least 10 completed steps, use the project reward
verdict. Complete 50 stable steps before acceptance.

### H2: Multiple directives can require more rank, but directive count is not rank

The decensor task contains several behavior and quality requirements. This can
increase the intrinsic complexity of the desired update. It does not follow
that each directive needs one LoRA rank. The directives can share features,
and rank is local to each matrix.

Measure: compare rank configurations by learning rate of the effective reward,
reward-factor distributions, held-out prompt categories, KL, and refusal
behavior. Do not decide from total reward alone.

### H3: Uniform rank 16 on every expert is likely capacity-rich but
operationally expensive

It probably has enough representational capacity. The main risks are GPU peak
memory, saved LoRA activations, the 15.9 GB policy export, and H200 dynamic LoRA
load time. It is not a configuration-only change because the native-FP8 fused
routed-expert LoRA path does not exist yet.

Measure: full-shape native-FP8 forward/backward, export, H200 load/unload, and
one fresh full-topology run. No cached batch is valid.

### H4: Rank 16 on dense/shared modules and rank 4 on every routed expert is a
stronger first engineering point

This keeps all experts trainable. With top-8 routing, it provides up to 32
routed low-rank directions for a token, uses an approximately 4.12 GB adapter,
and has much lower trainer state than routed rank 16. It is more conservative
than the article's nominal-rank-16 rule, which would use rank 2 for each expert.

This needs per-target rank and scaling support. If rank 4 reaches a measured
capacity plateau while the run remains stable, rank 8 is the next controlled
test. Rank 16 for each expert should follow measured rank-8 saturation, not an
assumption.

### H5: Keep the router frozen during the first expert-LoRA experiment

Changing expert weights while preserving routing isolates the effect of expert
adaptation and avoids adding route-policy drift to an already off-policy RL
system. There is no direct published result that proves this is optimal for
GLM-5.2 decensor GRPO. Record it as an experiment-control decision, not a
general MoE rule.

Measure: expert utilization, route entropy, mismatch KL, off-policy version
distance, and reward distributions. Consider router adaptation only if these
measurements show that frozen routing prevents relevant prompts from reaching
useful experts.

## Configuration warning

The current run uses rank 32 and alpha 64, so its standard LoRA multiplier is
`alpha / rank = 2`. Changing to rank 16 while leaving alpha at 64 makes the
multiplier 4. A rank comparison must set alpha deliberately. Rank, alpha,
learning rate, initialization, and target coverage must not change together
without recording the resulting experiment.
