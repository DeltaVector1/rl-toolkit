# GLM-5.2 FP8 PARITY PROBE

This probe locates the first numerical difference between the Ada trainer and
H200 vLLM. Run it only in separate probe containers. Do not attach hooks to
the active x38 processes.

## CURRENT FACTS

The active x38 trainer does not use the native FP8 path. It dequantizes the
checkpoint to BF16. The H200 engines keep block-FP8 weights. Use x38 only as a
BF16 control.

The rejected x32, x35, and x36 gates did not identify the first bad layer.
Route replay reduced masked mismatch KL from 3.8317 to 3.1711. The forced H200
Triton MoE run still measured masked mismatch KL 6.5098. These results exclude
router selection and the automatic MoE backend as sole causes. They do not
prove that either component has parity.

The repository pins vLLM 0.24.0, but it does not record one common image
digest for the two H200 hosts. Record the probe image digest and the selected
attention and MoE classes before you compare results.

## THE ARTIFACT CONTRACT

`scripts/fp8_parity.py` supplies three operations.

```bash
python scripts/fp8_parity.py fixture /probe/fixture \
  --vocab-size 154880 --length 2304 --seed 20260808

python scripts/fp8_parity.py audit-checkpoint /model \
  --output /probe/checkpoint-audit.json

python scripts/fp8_parity.py compare /probe/ada /probe/h200 \
  --json /probe/comparison.json
```

The fixture uses an integer linear congruential generator. It does not depend
on a tokenizer, PyTorch, CUDA, or a backend random-number generator.

The checkpoint audit reads safetensors headers and payload hashes without
allocating tensors. It checks every E4M3 weight against its FP32
`weight_scale_inv`. It accepts a partial final 128-row or 128-column block by
using ceiling division.

The trace format stores floating tensors as FP32 NumPy arrays and indices as
INT64 NumPy arrays. Each file has a SHA-256 value in `manifest.json`.
`TraceWriter` refuses duplicate tensor names. The comparison refuses a
different fixture, model revision, source shape, or sampled-row set.

Use these rows for full-model traces:

```text
0, 1, 15, 16, 17, 63, 127, 128, 255, 1023, 2047, 2048, 2303
```

These rows cover the vLLM router batch boundary and the 128-element FP8 block
boundary. Capture all rows for router IDs, router weights, indexer IDs, and
expert counts.

The GPU probe code must instantiate `TraceWriter` with these fields:

```python
writer = TraceWriter(
    output_dir,
    side="ada",  # use "h200" on the other host
    fixture_sha256=fixture_digest(Path("/probe/fixture")),
    model_revision="ba978f7d347eaf65d22f1a86833408afdb953541",
    run={
        "image_digest": IMAGE_DIGEST,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(),
        "attention_impl": type(layer.self_attn.mla_attn.impl).__qualname__,
        "moe_impl": type(layer.mlp.experts.quant_method).__qualname__,
        "kv_cache_dtype": ACTUAL_CACHE_DTYPE,
        "tp": TP_SIZE,
        "ep": EP_SIZE,
    },
)
```

Call `writer.record(name, tensor, rows=ROWS)` at each boundary below. Flatten
the batch and token axes before row sampling. Do not sample the hidden axis.
Use `semantic="indices"` for selected keys and selected experts. Use
`semantic="count"` for expert counts. Close the writer only after CUDA has
finished all copies.

## REQUIRED CONTROLS

Use one request and one sequence. Disable LoRA for the first base-model pass.
Load the same adapter SHA for the final pass. Disable dropout, CUDA graphs,
prefix-cache hits, compilation, TF32, and sampling. Clear both the MLA cache
and the indexer cache before each H200 pass. Use temperature 1.

Use the same checkpoint shard hashes on both hosts. Keep EPLB off. Record TP
and EP logical-to-physical expert mappings. Fix the vLLM attention backend and
MoE backend for each pass. Do not rely on automatic selection without
recording the selected classes.

Run these three base modes:

1. BF16-dequantized Ada and BF16-dequantized H200 reference modules.
2. Native Ada block FP8 and H200 block FP8 with a BF16 MLA cache.
3. Native Ada block FP8 and H200 block FP8 with the production FP8 MLA cache.

Mode 1 tests algebra and loader layout. Mode 2 adds projection and expert FP8
arithmetic. Mode 3 adds query and cache quantization.

## PASS 1: LOADER PARITY

Run `audit-checkpoint` against the trainer copy and each H200 copy. The JSON
files must have the same index SHA, weight count, weight payload hashes, and
scale payload hashes.

Dump loaded tensors before a kernel changes their layout. Cover these tensors
in layers 0, 2, and 3:

```text
q_a, q_b, kv_a, kv_b, o
indexer.wq_b, indexer.wk, indexer.weights_proj
router gate, correction bias
dense gate/up/down
routed experts 0, 31, 32, 63, 64, 127, 128, 255
shared expert gate/up/down
```

For vLLM, gather TP and EP shards. Split fused q_a and kv_a. Split W13 into
gate and up. Undo the backend W31 shuffle when it exists. Compare the raw E4M3
bytes and FP32 scale bytes before you compare dequantized values.

Use this dequantization reference:

```python
reference = (weight_e4m3.float() * expanded_scale.float()).bfloat16()
```

Do not cast the FP32 scale to BF16 before multiplication. The previous trainer
loader did that and added an avoidable rounding. The loader and its unit test
now use the FP32 multiplication order.

Pay special attention to the following matrices:

* `kv_a` has shape `[576, 6144]` and scale shape `[5, 48]`. The Ada native
  path dequantizes it because its last output block is partial.
* `kv_b` has shape `[28672, 512]`, but the Ada path dequantizes it for MLA
  absorption.
* `indexer.wk` has shape `[128, 6144]`. vLLM dequantizes it before it fuses the
  tensor with BF16 `weights_proj`.

Do not continue to layer comparison if the canonical loaded tensors differ.

## PASS 2: EVERY LAYER BOUNDARY

Run a full prefill with the fixture. Record these canonical names for all 78
layers:

```text
layer.NN.input
layer.NN.attention_input
layer.NN.attention_output
layer.NN.attention_residual
layer.NN.moe_input
layer.NN.router_logits
layer.NN.router_ids
layer.NN.router_weights
layer.NN.moe_output
layer.NN.output
```

Also record embedding output, final norm output, and logits for 64 fixed token
IDs. Do not write the full 154880-entry logits tensor.

vLLM keeps the residual in two tensors. Before a layer, its canonical input is
`hidden_states` when `residual is None`, otherwise
`hidden_states + residual`. After its MoE, its canonical layer output is also
`hidden_states + residual`. Use the actual activation dtype for these adds.
Do not compare only the first tuple element against the trainer residual.

Find the first layer whose relative RMSE increases above the mode-1 control
envelope. Do not use one global tolerance as a parity claim. Report relative
RMSE, cosine, maximum absolute error, and finite status at each boundary.
Require exact equality for integer routes and indices during a forced-route
or forced-index pass.

## PASS 3: ATTENTION REPLAY

Use layer 2, which computes sparse indices. Then use layer 3, which reuses the
layer-2 indices. Feed the identical serialized BF16 layer input to both
implementations. Do not use the naturally accumulated input from each model.

Record these values in order:

```text
input RMSNorm
q_a base, LoRA, sum, and q_a RMSNorm
kv_a base, LoRA, sum, compressed split, RoPE split, and kv RMSNorm
q_b base, LoRA, sum, q_nope, and q_rope
indexer wq_b
indexer fused wk/weights_proj split and k LayerNorm
indexer q and k before quantization
indexer q/k E4M3 bytes and UE8M0 scales
indexer head weights and one FP32 score row
sorted top-2048 indices and the rank-2048 margin
absorbed Q and compressed KV
sparse attention LSE and latent output
W_UV output
o base, LoRA, and sum
```

The trainer groups the even and odd RoPE coordinates before it uses a
split-half rotation. vLLM keeps adjacent interleaved pairs. Apply the inverse
fixed permutation before you compare raw RoPE tensors. Also compare QK dot
products, which do not depend on that permutation.

Run the following attention bisections:

1. BF16 projections, BF16 cache, and forced identical top-2048 indices.
2. FP8 projections, BF16 cache, and forced identical indices.
3. Identical projections and indices with the production FP8 MLA cache.
4. Independently selected indices.
5. Layer-3 IndexShare reuse.
6. One complete prefill and token-by-token decode from an empty cache.

Report index exact fraction, Jaccard overlap, and the score margin at rank
2048. This separates indexer changes from sparse-attention changes.

## PASS 4: MOE REPLAY

Use layer 3, the first MoE layer. Feed the same serialized BF16
post-attention-normalized rows to both implementations.

Run the router with batch sizes 1, 16, 17, and 256. Also run the same selected
row alone and as part of each batch. Pinned vLLM 0.24 uses the specialized
GLM router GEMM at 16 tokens or fewer and its general linear path above 16.
Capture FP32 and native router logits before sigmoid.

Record these values:

```text
router logits and sigmoid scores
top-8 IDs, unbiased top-8 weights, and eighth-to-ninth margin
per-expert counts
logical-to-physical expert mapping
routed input
gate GEMM and up GEMM
SiLU(gate) multiplied by up
second activation E4M3 bytes and scales
down GEMM
routing-weighted output
shared-expert output
final scatter sum
```

First replay identical expert IDs and weights. Then enable each native router.
Include experts from each EP shard. Include a zero-token expert and three
trailing padding rows. All trailing padding outputs must be zero.

Compare each stage against an FP32 reference made from canonical dequantized
weights. The existing dense probe covers only q_a and o. It does not establish
MoE or layer parity.

## RESULT

Store the fixture, both trace directories, checkpoint audit JSON, comparison
JSON, image digest, adapter SHA, backend names, and the command line in one
read-only result directory. Report the first divergent boundary. Keep loader,
attention, MoE, trainer step time, and rollout collection time as separate
items.
