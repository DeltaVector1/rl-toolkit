# Future GLM-5.2 GRPO Training and Serving Design

## Decision summary and evidence boundary

The attached staging report is directionally correct, and the upstream review strengthens its central recommendation: **do not replace the validated RTX 4090 block-FP8 expert path. Add a BF16 routed-expert LoRA path beside it, borrowing ScatterMoE’s index-driven data movement and backward-buffer reuse rather than adopting ScatterMoE as the expert implementation.** The active Muse run should remain untouched. fileciteturn0file0

The relevant evidence falls into four distinct categories:

| Category | What should be treated as true for this design |
|---|---|
| **Project measurements** | 64 × 48 GiB RTX 4090 trainers; PP8/CP8/node-local EP8; two TP8 H200 inference hosts; 44–46.5 GiB historical peaks; a 24 MiB allocation failure at only 1.5 MiB physically free; roughly 11–13 minute trainer forward/backward; roughly 22 minute rollout generation; historical adapter-transfer bandwidth around 3.63 MB/s; router/indexer frozen for the first experiment. fileciteturn0file0 |
| **Upstream facts** | ScatterMoE avoids full token grouping by gathering through sorted indices and explicitly reuses backward buffers. Prime-RL now has expert LoRA, but its implementation casts base expert operands to BF16 and derives LoRA dtype from the base weight, so that implementation cannot be dropped onto this native block-FP8 trainer. vLLM 0.26 has GLM-5.2 LoRA support and an FP8-capable modular Triton MoE path that implements fused MoE LoRA. PyTorch FSDP2 trades memory for communication with `reshard_after_forward`. citeturn13view0turn9search0turn8search0turn12view2 |
| **Estimates** | The report’s adapter sizes and WAN times, the approximately 7.45% simple PP fill/drain term from \(7/94\), and the low-rank activation-memory calculations below are engineering estimates, not measured future-run performance. fileciteturn0file0 |
| **Hypotheses to benchmark** | A specialized row-mapped Triton LoRA kernel should be the best first production candidate on SM89; saving rank-4/8 A outputs may be cheaper than recomputing them; H200 fused-MoE LoRA dual-stream execution may or may not help GLM-5.2; and true rank-4 expert inference should outperform internally padded rank-16 only after vLLM receives effective-rank awareness. These need measurements before adoption. |

ScatterMoE's most relevant result is architectural, not its headline benchmark. Its `ParallelLinear` sorts/pads **indices** and gathers input tiles from their original locations instead of materializing the entire expert-grouped token matrix; its backward section also says that grouped buffers can be reused once their earlier contents are no longer needed. citeturn13view0 Its reported 38.1% training-throughput improvement and 66.2% relative training memory came from A100 configurations that are not predictive of GLM-5.2 on RTX 4090s. citeturn13view0

Prime-RL is similarly useful as a reference, not as a wholesale transplant. Its current expert-LoRA implementation performs grouped LoRA GEMMs in BF16, while the reviewed implementation also converts the base expert weights to BF16 for that path and initializes LoRA factors from the base-layer dtype. That conflicts with the requirement to preserve native block-FP8 base execution and BF16-only trainable LoRA. fileciteturn6file0 Prime-RL's broader trainer does provide useful patterns around FSDP2, EP/CP, RL orchestration, and optimizer/offload work, but none justify replacing local PP8 or the proven Ada W8A8 kernels. citeturn9search0

**Recommended first system configuration:** all required targets present, router and sparse indexer frozen, uniform **actual rank 4** LoRA, BF16 parameters and stored gradients, FP32 matrix accumulation, existing block-FP8 base expert path unchanged. One correction to the attached report is important: vLLM 0.26 does **not** allow `max_lora_rank=4`; its server choices are 1, 8, 16, 32, 64, 128, 256, 320, and 512. An actual `r=4` adapter is legal provided `max_lora_rank` is at least 4, so the uniform-rank-4 qualification should run vLLM with **`max_lora_rank=8`**, not 4. citeturn8search2 fileciteturn14file0

## Routed-expert LoRA dataflow and gradients

Use one expert-major parameter tensor per projection:

\[
A^{(p)} \in \mathrm{BF16}[E_{\mathrm{local}},r,d_{\mathrm{in}}],
\qquad
B^{(p)} \in \mathrm{BF16}[E_{\mathrm{local}},d_{\mathrm{out}},r]
\]

for \(p\in\{W1,W3,W2\}\). Do not create 256 Python modules or stack per-expert tensors during every forward. This matches the project's planned storage and the general stacked-expert layout used by contemporary expert-LoRA implementations. fileciteturn0file0 Prime-RL likewise represents grouped expert LoRA tensors rather than rebuilding expert lists in the hot path. fileciteturn6file0

Let routed row \(j\) correspond to source row \(s_j\), local expert \(e_j\), and top-8 routing coefficient \(p_j\). Define the ordinary LoRA scale

\[
\lambda=\alpha/r
\]

unless the adapter explicitly uses a different scaling convention. vLLM 0.26's PEFT helper uses \(\alpha/r\) for ordinary LoRA and \(\alpha/\sqrt r\) for rsLoRA, so rank and alpha must always travel together in adapter metadata. fileciteturn13file0

Using row vectors, one generic expert linear is

\[
y_j=F_{\mathrm{FP8}}(x_j,W^{0}_{e_j})
+\lambda
\left(x_j A_{e_j}^{T}\right)B_{e_j}^{T}.
\]

The first term is **exactly the existing frozen W8A8 path**. No LoRA work should cause the base expert matrix to be dequantized to BF16. This is the main reason not to use Prime-RL's grouped-expert class directly. fileciteturn0file0 fileciteturn6file0

For GLM's gated expert:

\[
z_1=xA_1^T,\quad
y_1=F_1(x)+\lambda z_1B_1^T
\]

\[
z_3=xA_3^T,\quad
y_3=F_3(x)+\lambda z_3B_3^T
\]

\[
h=\operatorname{SiLU}(y_1)\odot y_3
\]

\[
z_2=hA_2^T,\quad
o=F_2(h)+\lambda z_2B_2^T.
\]

Then use the **existing expert-combine operation exactly once**:

\[
Y_{s_j}\mathrel{+}=p_j o_j .
\]

vLLM's GLM/DeepSeek-derived implementation confirms that the expert MLP uses SiLU gating, while ScatterMoE's formulation likewise keeps route weighting in the expert-output combination rather than requiring a second full grouped output. fileciteturn16file0 citeturn13view0

The important correctness rule is that **LoRA must enter before the same nonlinearities and routing combine as its base projection**. Adding W1/W3 LoRA after the SiLU/multiply, or adding W2 LoRA after route weighting, is not equivalent.

For a generic LoRA projection with incoming gradient \(g_y\),

\[
g_z=\lambda g_yB
\]

\[
\nabla B=\lambda g_y^Tz
\]

\[
\nabla A=g_z^Tx
\]

\[
g_x^{\mathrm{LoRA}}=g_zA.
\]

All reduction/MMA accumulators for these products should be FP32; the persistent A/B parameters and stored parameter gradients remain BF16. The kernel should write a BF16 gradient only after a tile has accumulated its routed rows in FP32. A full-model persistent FP32 gradient shadow would defeat an important part of the memory design and should not be introduced without a separate memory case. The requirement to preserve BF16 trainable state while accumulating matrix products in FP32 is also consistent with the BF16×BF16→FP32 tensor-core operation available on compute capability 8.9. citeturn9search8turn9search3

For W2, first obtain

\[
g_h =
g_h^{\mathrm{base}}
+
g_h^{\mathrm{LoRA}},
\]

where the base component continues to come from the validated W8A8 `_linear_input_grad()` and

\[
g_h^{\mathrm{LoRA}}=g_{z_2}A_2.
\]

Then

\[
g_{y_3}=g_h\odot \operatorname{SiLU}(y_1),
\]

\[
g_{y_1}=g_h\odot y_3\odot\operatorname{SiLU}'(y_1).
\]

W1 and W3 then independently produce their BF16 LoRA parameter gradients and LoRA input gradients, which are accumulated into the same existing \(g_x\) destination as the two base W8A8 input gradients. There is no reason to allocate separate full-width `dx_base`, `dx_lora_w1`, and `dx_lora_w3` tensors.

**Saved state should be deliberately minimal.** The LoRA autograd object should reference, not copy, the route metadata already required by the expert path: source-row index, local-expert mapping or sorted-index table, expert offsets/counts, and whatever representation the existing combine backward uses for route coefficients. It also needs the values the current expert activation backward already retains or reconstructs, such as \(y_1,y_3\) or the corresponding activation state. The only significant *new* saved values should be the optional low-rank \(z_1,z_3,z_2\).

**Forward scratch reuse should be:**

| Point | Scratch | Lifetime |
|---|---|---|
| W1 A | `[C,r]` | Reuse immediately after B consumes it, except compact save/offload copy |
| W1 B | No `[C,d_ff]` LoRA delta | B writes/accumulates directly into the existing W1 base-output buffer |
| W3 A | Same `[C,r]` scratch | Reused from W1 |
| W3 B | No separate full delta | Accumulate into existing W3 output |
| W2 A | Same `[C,r]` scratch | Reused after gated activation is available |
| W2 B | No separate full output | Accumulate into existing base W2 result before combine |

In backward, `[C,r]` becomes \(g_z\) scratch after the corresponding saved/recomputed \(z\) has been consumed by the B-gradient operation. Full-width LoRA input gradients should be accumulated directly into the base backward destination. This is the practical adaptation of ScatterMoE's explicit observation that its backward grouping buffers can be repurposed once the preceding calculation is complete. citeturn13view0

The **router remains frozen**. Consequently, route probabilities are constants for this backward and there is no \(\nabla p\) path. This differs from ScatterMoE's general backward algorithm, which includes a route-weight gradient, but is correct for the stated GLM experiment. citeturn13view0 fileciteturn0file0

## Activation memory, row mapping, and source-major ordering

The most useful ScatterMoE concept is not “grouped GEMM”; it is **group through indirection instead of grouping the full tensor**. The paper sorts token indices by expert, pads the index structure as necessary, and then lets the kernel load the original token rows according to those indices. It explicitly avoids materializing the padded/expert-sorted embedding tensor in HBM. citeturn13view0

For this trainer, retain the current source-major routed representation and build or reuse a compact mapping such as

\[
q\mapsto
(\text{route\_row}_q,\text{source\_row}_q,e_q),
\]

where \(q\) is in expert-sorted order. The A kernel then loads

\[
x[\text{source\_row}_q]
\]

and

\[
A[e_q]
\]

directly. There is **no** `X_grouped[R,d_model]`.

The A result is tiny enough that its output can be expert-ordered:

\[
Z[q,:]\in[R,r].
\]

The B operation reads the same \(e_q\) and writes its full-width contribution directly to `route_row_q` in the already-existing expert output. This means the data movement becomes:

```text
source/full-width X
      │
      └── indexed read ──> A[e] ──> compact Z[R,r]
                                      │
                                      └─> B[e] ── indexed add ──> existing routed output
```

rather than:

```text
X
 └─> full X_grouped[R,d]
       └─> A
            └─> Z
                 └─> B
                      └─> full second routed output
```

That is the highest-value ScatterMoE idea for the local design. ScatterMoE's own `ParallelLinear` supports scattered-to-grouped, grouped-to-scattered, and scattered-to-scattered operations precisely to eliminate those intermediate grouping copies. citeturn13view0

For weight gradients, expert grouping is still computationally attractive. ScatterMoE reports that directly calculating scattered \(X^TgY\) was slower than grouping followed by grouped weight-gradient computation on its A100 experiments. citeturn13view0 That result does **not** imply this trainer should group its full GLM input. Instead, the weight-gradient program can iterate the expert's sorted route-index range, load \(X\) indirectly, and FP32-accumulate its \(r\times d\) A-gradient tile. If that is too slow, the only object worth explicitly grouping first is a compact low-rank object such as \(Z[R,r]\) or \(g_Z[R,r]\), not \(X[R,d]\).

Top-8 correctness needs one invariant: **the route coefficient is applied at exactly the same point as the frozen base expert path.** If the current combine multiplies expert output by \(p_j\), then LoRA output is added to base output before that combine. Its backward therefore receives \(p_jg_Y\). Do not independently multiply the LoRA branch by \(p_j\) if the existing routed backward has already done so. A route-identity test with unequal top-8 probabilities is mandatory; all-ones or top-1 tests are insufficient.

For the three A-output policies, the new GPU memory per routed row is very small. One BF16 A output is \(2Rr\) bytes; saving all W1/W3/W2 A outputs is

\[
M_Z=6Rr\ {\rm bytes}.
\]

Using local routed-row counts that bracket realistic microbatch cases:

| Local routed rows \(R\) | Rank | One A result | W1 + W3 + W2 |
|---:|---:|---:|---:|
| 2,048 | 4 | 16 KiB | 48 KiB |
| 2,048 | 8 | 32 KiB | 96 KiB |
| 2,048 | 16 | 64 KiB | 192 KiB |
| 4,096 | 4 | 32 KiB | 96 KiB |
| 4,096 | 8 | 64 KiB | 192 KiB |
| 4,096 | 16 | 128 KiB | 384 KiB |
| 8,192 | 4 | 64 KiB | 192 KiB |
| 8,192 | 8 | 128 KiB | 384 KiB |
| 8,192 | 16 | 256 KiB | 768 KiB |
| 16,384 | 4 | 128 KiB | 384 KiB |
| 16,384 | 8 | 256 KiB | 768 KiB |
| 16,384 | 16 | 512 KiB | 1.5 MiB |

These are **estimates**, not measured route counts. With top-8 over node-local EP8, a balanced dispatch gives a useful rule of thumb that each EP rank receives roughly one routed row per original token on average, but real GLM routing skew must come from the trace rather than that assumption.

The important multiplier is not one layer; it is approximately

\[
M_Z
\times
(\text{MoE layers on this PP stage})
\times
(\text{simultaneously outstanding forward microbatches}).
\]

That can make even rank-16 compact state matter near a memory cliff, although it is still dramatically smaller than saving another routed tensor of width \(d_{\rm model}\) or \(d_{\rm expert}\). Given the historical 24 MiB allocation failure with only 1.5 MiB free, allocations of tens of MiB are operationally meaningful here. fileciteturn0file0

The policy comparison is therefore:

| Policy | HBM | Added compute / traffic | Recommendation |
|---|---|---|---|
| **Save A outputs on GPU** | \(6Rr\) bytes/layer/microbatch | None in backward | **First implementation.** Simplest and probably best at r4; determine from live-memory trace. |
| **Recompute A outputs** | Essentially zero LoRA-specific saved Z | Re-run W1/W3/W2 A shrink operations in backward | **Second candidate** if activation peak remains unsafe. Particularly attractive because only low-rank shrink GEMMs are repeated. |
| **Pinned-host Z** | GPU Z freed after D2H | D2H + H2D = \(12Rr\) bytes/layer/microbatch plus synchronization | **Third candidate**, useful only if copies overlap cleanly and recompute is measurably more expensive. |

Host offload should use a **preallocated bounded pinned arena**, indexed by `(microbatch, layer, projection)`. It must never respond to a larger pipeline window by allocating more pinned objects dynamically. Coalesce W1/W3/W2 low-rank data where practical to reduce the cost of many tiny PCIe operations. On backward, use one bounded device prefetch slot. PyTorch's FSDP documentation likewise notes that pinned memory enables asynchronous host/device copies but consumes non-pageable host resources, supporting the need for an explicit bound. citeturn12view3

My expected ordering is **save at rank 4 → recompute if the peak still lacks headroom → bounded Z offload only if recompute loses measurable throughput**. That is a hypothesis to test, not an upstream result.

## RTX 4090 kernel choices

RTX 4090 is Ada, compute capability **8.9**, and CUDA documents BF16 tensor-core support at compute capability 8.9. CUTLASS also explicitly supports the GeForce RTX 40-series/SM89 and provides BF16×BF16→FP32 MMA machinery originating with the SM80 tensor-core ISA. citeturn9search8turn9search1turn9search3 Consequently, there is no need—and no justification—to move this path to Hopper-only DeepGEMM or DeepEP.

The production-kernel search should be narrower than the attached plan's original rank-1/2/4/8/16 exploratory sweep. For the requested future design, ranks **4, 8, and 16** deserve production specialization.

| Rank | Primary RTX 4090 candidate | Secondary candidate | Important issue |
|---:|---|---|---|
| **4** | Custom Triton row-mapped kernel, rank compile-time constant; benchmark a tensor-core path internally covering/padding N to 8 against a small-N SIMT/vector-reduction path | Per-expert cuBLAS/PyTorch reference for large-token experts | BF16 tensor-core MMA naturally has N granularity of 8; rank 4 can waste lanes, so never infer r4 performance from r8. citeturn9search3 |
| **8** | Custom Triton row-mapped BF16 `tl.dot`/tensor-core implementation with FP32 accumulation | SM89 CUTLASS grouped GEMM for selected gradient operations | This is the cleanest low-rank tensor-core shape and should be the first high-performance reference. |
| **16** | Same Triton family with rank-16 specialization | CUTLASS grouped BF16 GEMM / cuBLAS for sufficiently large expert token counts | More compute but generally better GEMM utilization; memory and adapter transport become much worse. |

**The first candidate should be a local Triton kernel, not stock ScatterMoE.** ScatterMoE's kernel is valuable proof that index-directed Triton GEMM can eliminate a full grouping copy, but its published implementation was designed and benchmarked around different expert widths and A100s. citeturn13view0 Its results cannot select an SM89 tile configuration for this workload.

The A forward should specialize \(r\) at compile time and tile along input \(K\). The B forward should specialize \(K=r\) and tile along output width. The B-gradient

\[
dB=g_Y^TZ
\]

is an especially good conventional GEMM shape because one operand is already the compact \(R\times r\) tensor. The A-gradient

\[
dA=g_Z^TX
\]

should use expert-index ranges while gathering X, avoiding a full expert-grouped X copy.

For dA/dB, a CUTLASS grouped implementation is a legitimate SM89 benchmark candidate. CUTLASS documents grouped GEMM support in its pre-Hopper/SM80 family and broad Ada support, so it is not an SM90-only recommendation. citeturn9search1turn9search4 It should nevertheless lose the qualification if preparing its pointer arrays or grouped inputs requires a new large routed-token allocation.

Prime-RL's current `torch._grouped_mm` route is **not** the default recommendation here. Its expert-LoRA code has alignment gating around rank/dimensions and, more importantly, its surrounding expert path casts operands to BF16 rather than preserving the project's block-FP8 base operation. fileciteturn6file0 Rank 4 should therefore not depend on that path; ranks 8 and 16 may include it in a standalone microbenchmark only when the installed PyTorch/CUDA build demonstrably supports the exact SM89 shapes.

Benchmark kernels with the **actual route-count histogram**, not only balanced experts. For each of W1, W3, and W2, preserve route populations such as 0, 1–4, 5–16, 17–64, and larger token counts per local expert. Record separately:

- forward A,
- forward B/add,
- LoRA input gradient,
- dA,
- dB,
- index preparation,
- scratch peak,
- and whole expert-layer time.

A kernel that wins raw GEMM time but creates a large grouping allocation loses. Given the current memory behavior, **peak live bytes are a first-class kernel metric**, not a secondary one. fileciteturn0file0

No SM90-only path—DeepGEMM, DeepEP, or a Hopper-only grouped kernel—should enter the 4090 production shortlist. NVIDIA identifies RTX 40-series as SM89 and H100/H200 as SM90, so those are materially different kernel targets. citeturn9search1

## vLLM serving, mixed ranks, and GLM attention coverage

**vLLM 0.26 verdict: conditionally yes, it has the necessary pieces to serve GLM-5.2 routed-expert LoRA over a supported block-FP8 MoE base, but the exact checkpoint/backend combination still needs an isolated H200 qualification.**

The evidence is unusually strong:

First, vLLM's supported-model table lists `GlmMoeDsaForCausalLM` for GLM-5/5.1/5.2 with both LoRA and PP marked supported. citeturn8search0

Second, the v0.26 Triton experts class directly mixes in `LoRAExpertsMixin` and advertises the static 128-block FP8 weight/dynamic 128-block activation scheme among its supported quantization combinations. When LoRA and quantization interact with gathered expert data, the implementation preserves an unquantized activation for the LoRA shrink and separately quantizes the operand used by the base MoE GEMM. That is exactly the architecture needed for **quantized frozen base + higher-precision LoRA**, rather than converting the base expert to BF16. fileciteturn11file0

Third, `FusedMoEWithLoRA` explicitly checks that the selected modular MoE kernel supports LoRA and directs unquantized/quantized configurations toward supported modular implementations. citeturn7search1

The hard qualification condition is therefore: **inspect the GLM-5.2 runtime quant configuration and stop unless the checkpoint maps to the block-FP8 scheme supported by the chosen modular Triton backend.** “Block FP8” as a general label is not enough to infer that every possible block geometry is supported.

For the first H200 test, force the LoRA dtype to BF16 rather than leaving it at `auto`. vLLM 0.26 accepts `float16` or `bfloat16` explicitly; `auto` adopts the base model's logical dtype. citeturn8search1

The important v0.26 limits are:

| Item | vLLM 0.26 behavior |
|---|---|
| **GLM-5.2 LoRA** | Model architecture is marked supported. citeturn8search0 |
| **FP8 base + MoE LoRA** | Supported by the modular Triton expert implementation for its enumerated FP8 schemes; `TritonExperts` implements `LoRAExpertsMixin`. fileciteturn11file0 |
| **EP** | Fused-MoE LoRA with EP currently asserts that the all-to-all backend is `allgather_reducescatter`; it also rejects MoE sequence parallelism in that EP case. fileciteturn7file0 |
| **EP + fully sharded LoRA** | Explicitly unsupported. citeturn7search1 |
| **Dual CUDA streams** | CUDA-only; incompatible with `fully_sharded_loras`. Fused MoE creates an auxiliary stream/events and can overlap LoRA with the base operation. fileciteturn14file0 citeturn7search1 |
| **Adapter layouts** | v0.26 documents 2D per-expert keys and 3D stacked PEFT-style expert tensors; mixed layout serving has an explicit flag. fileciteturn18file0 |
| **Server max-rank choices** | 1, 8, 16, 32, 64, 128, 256, 320, 512. Rank 4 is not itself a `max_lora_rank` choice. citeturn8search2 |
| **Actual adapter rank** | PEFT `r` may be any positive integer not greater than `max_lora_rank`, so an `r=4` adapter can load with server max rank 8. fileciteturn13file0 |
| **Scaling** | One PEFT `r` and `lora_alpha`; ordinary scaling is \(\alpha/r\), rsLoRA is \(\alpha/\sqrt r\). fileciteturn13file0 |
| **`rank_pattern`** | The v0.26 `PEFTHelper` has no `rank_pattern` field and filters unknown config fields when creating its helper. Thus stock PEFT loading does not provide PEFT per-module rank-pattern semantics. fileciteturn13file0 |

The dual-stream path warrants caution. In the Triton expert code, dual-stream W13 execution creates a separate `zeros_like` LoRA-delta buffer so the base GEMM and LoRA branch can run concurrently before they are added. fileciteturn11file0 That can improve overlap on H200, but it also increases instantaneous inference workspace. Benchmark it only after ordinary fused-MoE LoRA is correct.

### Mixed rank without a rank-16 WAN file

Stock vLLM 0.26 is **not a correct native solution for “dense/shared r16, routed r4” PEFT rank-pattern semantics**. Its PEFT helper carries one adapter `r` and `lora_alpha`; its packed MoE construction takes the first LoRA's rank/scaling and stacks expert tensors under that common rank. fileciteturn13file0 fileciteturn22file0

The correct transport design is:

```text
compact WAN adapter
  ├─ dense/attention/shared tensors: actual rank 16
  ├─ routed expert tensors: actual rank 4
  └─ manifest:
       module family -> {rank, alpha, scaling, layout, shapes, hash}
                    │
                    ▼
            H200-local loader
                    │
      ┌─────────────┴──────────────┐
      ▼                            ▼
dense vLLM buffers             MoE LoRA buffers
rank 16                       rank 4 data
```

Do **not** pad expert A/B tensors to 16 before WAN transfer. The report's estimate is already approximately 4.12 GB for routed rank 4 with dense/shared rank 16, versus 15.90 GB at routed rank 16; at the previously observed 3.63 MB/s, rank 4 is already around 18.9 minutes to one H200 host. fileciteturn0file0 Pre-transfer padding would throw away the primary transport benefit.

There are two implementation levels:

**Minimal correctness bridge:** send compact variable-shape tensors, have an H200-local loader zero the existing rank-16 internal buffer and copy the actual rank-4 expert tensors into its first four rank positions. `FusedMoEWithLoRA.set_lora()` already resets its destination and copies only source-shaped rank slices, so the internal mechanism supports this type of receiver-side padding. fileciteturn7file0 The loader must additionally supply the correct **per-family scaling**; using the global rank-16 PEFT scale for a rank-4 expert would be numerically wrong.

This preserves WAN size but likely **does not preserve rank-4 compute cost**, because the fused runtime buffers/kernel configuration are still sized around the server-wide maximum rank.

**Proper production solution:** give fused MoE LoRA its own effective rank—or a per-layer/per-family rank—so routed expert kernels actually execute rank 4 while dense/shared kernels execute rank 16. The compact manifest becomes authoritative. This is the design to pursue after uniform-rank qualification. It avoids both WAN padding and rank-16 expert compute.

### Attention coverage and `kv_b_proj`

GLM's MLA attention path exposes `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, and `o_proj` when Q-LoRA is configured. vLLM's source instantiates these projections and can internally pack `q_a_proj` and `kv_a_proj_with_mqa` into a fused projection for serving. fileciteturn16file0

The required logical coverage for the submitted model is therefore:

\[
78\times5=390
\]

attention projections,

\[
3\times3=9
\]

dense-MLP projections,

\[
75\times3=225
\]

shared-expert projections if there is one shared expert MLP per MoE layer, and

\[
75\times256\times3=57{,}600
\]

routed-expert projections, totaling

\[
58{,}224
\]

logical base matrices. The loaded model—not this arithmetic—must be authoritative, and startup should fail on any name/count mismatch. fileciteturn0file0

`kv_b_proj` is a trainer-specific correctness trap. In the local MLA absorption path, `.weight` is read directly, so a conventional module wrapper changes ordinary calls to the module but **does not change calculations that bypass `forward()` and consume the base parameter directly**. fileciteturn0file0

Do not solve this by materializing a permanent merged BF16 weight. Instead, make MLA absorption low-rank-aware. Algebraically, for

\[
W_{\mathrm{eff}}=W_0+\lambda BA,
\]

an absorbed product of the form \(UW_{\mathrm{eff}}\) should become

\[
UW_0+\lambda(UB)A,
\]

and a product \(W_{\mathrm{eff}}V\) becomes

\[
W_0V+\lambda B(AV).
\]

Transpose the factors where the local absorption formula uses \(W^T\). That preserves the factorization and does not require a full dense LoRA delta. Key and value slices of `kv_b_proj` need separate parity checks because MLA reshapes/splits that projection. vLLM's own GLM implementation explicitly constructs and consumes `kv_b_proj`, confirming it is a first-class MLA projection rather than an ignorable load-only weight. fileciteturn16file0

The sparse attention indexer's projections should remain frozen in this first experiment, exactly as the report specifies. Adding indexer learning simultaneously with routed experts would confound both numerical validation and memory attribution. fileciteturn0file0

## Pipeline improvements and H200 rollout benchmarking

The PP work should remain incremental because PP8 has already achieved the architecturally important change: expert communication is node-local while pipeline boundaries cross nodes. fileciteturn0file0

**Sparse-index cache boundaries.** Only four of the seven PP boundaries require the cache. Removing the other three means **43% fewer sparse-cache boundary transfers**, but that is not a prediction of a 43% PP speedup: the gain is only the measured time/memory associated with those three sends and receives. Instrument those operations separately. fileciteturn0file0

**Measured-cost partitioning.** Replace equal layer-count splitting with a minimax partition over measured layer costs. Each layer's weight should include actual forward, backward, FSDP all-gather, expert skew, and any boundary-specific side-input work under the real 1F1B execution. The objective is to minimize the maximum stage service time, because steady-state throughput follows the slowest stage. Do not optimize from a single isolated forward profile.

**One bounded side-input slot.** Add exactly one preallocated pinned prefetch slot only if profiling shows the stage waiting for CPU→GPU side-input movement. Key it by microbatch identity rather than assuming FIFO order so that this work does not have to be rewritten if scheduling later changes.

**FSDP resharding stays conservative.** PyTorch FSDP2 documents the exact tradeoff: `reshard_after_forward=True` frees unsharded parameters after forward and re-all-gathers for backward; `False` retains unsharded parameters and avoids that backward all-gather, consuming more memory. citeturn12view2 With a historical run failing on a 24 MiB allocation, this is currently the wrong lever to pull first. Test `False` only after the new expert path demonstrates substantial measured reserve on every rank. PyTorch also permits intermediate integer reshard sizes, which may become a later compromise, but it should not be introduced during initial routed-LoRA qualification. citeturn12view3

**Interleaving remains low priority.** PyTorch's `ScheduleInterleaved1F1B` is explicitly a multi-stage-per-rank schedule, while ordinary 1F1B is a single-stage-per-rank schedule. citeturn12view0 At 94 microbatches and eight stages, the simple fill/drain term is

\[
\frac{8-1}{94}=7.45\%.
\]

Even an impossible schedule that eliminated all of that term would only remove about that component; actual total-step improvement is smaller if compute, FSDP, or other stalls dominate. The local side-input FIFO and current runtime rejection of interleaving are additional reasons to postpone it. fileciteturn0file0

For inference, keep **two independent TP8 GLM replicas**, one per eight-H200 host, and do not turn on inference EP in the baseline. That avoids vLLM's stricter fused-MoE-LoRA EP restrictions while matching the requested deployment. The vLLM 0.26 GLM model is LoRA-capable, and its normal scheduler provides the knobs required for prefix caching, concurrency, and batched-token sweeps. citeturn8search0turn14search3

The benchmark matrix should be:

| Dimension | Values / treatment |
|---|---|
| **Rollout inputs** | Replay actual prompt-length and generated-length distributions from successful GRPO batches. Record p10/p50/p90/p99 and include the longest context cases rather than only fixed synthetic lengths. |
| **Prefix reuse** | Baseline caching off; caching on/cold; caching on/warm across all 16 completions of a GRPO prompt group. vLLM's APC reuses KV blocks for requests sharing a prefix. citeturn14search1 |
| **Replica assignment** | Send a whole prompt/group-of-16 to one replica rather than round-robin its completions across the two H200 hosts; this maximizes same-host prefix reuse. |
| **Sequence concurrency** | Sweep 8, 16, 32, 48, 64 active sequences per replica, stopping at the first memory/thrashing point. |
| **Max batched tokens** | Sweep 4K, 8K, 16K, 32K, 64K; keep only points compatible with the actual model/KV capacity. vLLM exposes this scheduling limit and chooses hardware-dependent defaults, so explicit qualification is preferable. citeturn14search3 |
| **LoRA** | Base only → attention/dense LoRA → routed expert LoRA → complete adapter. |
| **Fused MoE** | Force the known-correct modular Triton FP8+LoRA backend first; then test tuned configurations for the selected rank/token histogram. |
| **Dual stream** | Off baseline, then on; record both speed and extra workspace. Do not combine with `fully_sharded_loras`. fileciteturn14file0 |
| **Both H200 hosts** | Run identical independent tests, then simultaneous load. Report individual and aggregate output tokens/s and rollout wall time. |
| **Transfer/load** | WAN transfer, file verification, adapter unpack/local padding, GPU copy, first request/warmup, and generation are separate timings. |

Record TTFT, prompt tokens/s, output tokens/s, per-sequence latency, p50/p95/p99 completion time, GPU utilization, HBM/KV utilization, prefix-cache hit/reused-token rate, queue time, adapter-load latency, and whole-real-rollout wall time. The actual project acceptance metric remains approximately **10 minutes for rollout generation**, compared with the prior measured 21:58, but adapter WAN transfer and adapter load must remain separate so a fast generator cannot hide an unusable update pipeline. fileciteturn0file0

Fused-MoE tuning should be based on actual prefill and decode token populations at routed experts. vLLM has a tuned-config mechanism in its MoE/LoRA context, so compare untuned and tuned runs for the ranks actually being served. citeturn7search1

**MTP is a guarded final experiment, not a baseline feature.** vLLM 0.26 exposes MTP/speculative-decoding mechanisms, but the project has already observed a failed GLM MTP attempt. citeturn14search7 fileciteturn0file0 Qualification order should be base/no-LoRA MTP correctness on one H200 TP8 replica, then full-LoRA MTP, then the second host. Begin with the smallest speculative depth supported by the selected GLM path. Measure acceptance rate, short-decode correctness, output-token throughput, extra HBM, and net rollout wall time. A positive microbenchmark is insufficient; MTP only survives if the same real rollout batch improves end-to-end generation without changing the established numerical/correctness envelope.

## Validation gates and ranked implementation checklist

Progression should be gated so that a test cannot “mostly work” and silently move to a much more expensive 64-GPU or H200 run.

| Stage | Required work | Progression gate | Immediate stop condition |
|---|---|---|---|
| **One RTX 4090** | Standalone exact-dimension W1/W3/W2 LoRA kernels; r4/r8/r16; save/recompute/host-Z policies; route-count/skew histogram sweep; FP32 reference and BF16 reference comparisons | LoRA-disabled base path is unchanged; all A/B/input gradients finite; output/gradient error is within the existing validated BF16-LoRA numerical envelope; selected kernel has no unbounded/full-width grouping allocation | Any base-path difference with LoRA disabled; wrong expert mapping; nonfinite grad; trainable FP8 LoRA tensor; unexpected compilation/allocation in steady-state |
| **One eight-4090 node** | Real EP8 dispatch/combine, 256-expert ownership, top-8 weighting, source-major ordering, empty and overloaded experts | Route counts conserved exactly; outputs and all gradients match the reference under unequal route probabilities; no second full routed-token tensor appears in memory trace | Lost/duplicated route, incorrect top-8 scaling, expert-ID mismatch, rank-local inconsistency, or new large grouped-copy allocation |
| **All 64 trainers** | PP8/CP8/EP8, strict target registry, `kv_b_proj` absorbed delta, cache-boundary optimization, at least 20 changing trainer-only steps before rollouts | All expected logical targets present; router/indexer remain frozen; finite/nonzero grads for every trained target class; no allocation failures; no step after warmup exceeds the report's 15-minute qualification threshold; parity is no worse than the prior accepted precision envelope fileciteturn0file0 | Missing target; router/indexer gradient; wrong `kv_b_proj`; any CUDA OOM; physical-free-memory cliff that is unexplained by the trace; trainer step >15 min after step 1 |
| **One H200 TP8 host** | vLLM 0.26 isolated image; GLM base parity; compact adapter; dense/attention and expert load/unload/reload; FP8+LoRA backend qualification | Runtime FP8 scheme matches supported modular path; adapter logical coverage/hash exact; unload restores baseline; reload restores identical adapter hash/result; trainer/H200 logprob discrepancy stays inside established baseline envelope | Unsupported quant scheme; silent dropped target; wrong expert layout declaration; wrong mixed-rank scaling; NaN/invalid generation |
| **Both H200 TP8 hosts** | Parallel transfer/load, real rollout benchmark, prefix caching, seq/token sweep, tuned MoE, dual-stream comparison | Both hosts independently reconstruct the same adapter hash; stable repeated batches; aggregate performance improves; target generation wall time is reached or remaining bottleneck is quantitatively isolated | Replica disagreement, load/unload nondeterminism, prefix-cache correctness failure, HBM instability, or host-specific result divergence |
| **MTP** | Only after non-MTP full adapter is accepted | Same correctness envelope and a measured reduction in real rollout wall time | Any short-decode corruption, parity regression, unstable memory, or no net real-batch speedup |

Because this cluster has already failed with only 1.5 MiB physically free, I would add a conservative **engineering guard** during qualification: do not experiment with `reshard_after_forward=False` until every rank has at least **512 MiB of measured additional worst-case reserve relative to the accepted routed-LoRA configuration**, after kernel compilation and several steady-state microbatches. This 512 MiB number is a proposed safety gate, not an upstream requirement or a previously measured threshold. The hard historical evidence is simply that 1.5 MiB was catastrophically insufficient for a 24 MiB request. fileciteturn0file0

The ranked implementation order is:

1. **Freeze and instrument the baseline.** Preserve active Muse, PP8/CP8/node-local EP8, native block-FP8 base forward/input-backward, router/indexer freeze, and record physical free memory around W1/W3/W2. Precompile every candidate production kernel before any live-memory stress test. fileciteturn0file0

2. **Build the strict logical target registry before expert kernels.** Require all five attention projections, all dense/shared MLP projections, and every W1/W3/W2 of every routed expert. Treat the loaded model's names/count as authoritative. Hard-stop on omission rather than allowing PEFT-style “target not found” behavior.

3. **Fix `kv_b_proj` semantics before calling attention coverage complete.** MLA absorption must use the effective \(W_0+\lambda BA\) algebraically; a wrapper around `forward()` is insufficient because local code reads `.weight` directly. fileciteturn0file0

4. **Implement the row-mapped BF16 LoRA primitive while leaving the base expert op untouched.** Reuse the expert-sorted index structure to gather source-major inputs and write B output directly into existing routed buffers. This is the core ScatterMoE adaptation. citeturn13view0

5. **Implement exact W1/W3/W2 backward with direct buffer accumulation.** FP32 MMA/reduction, BF16 persistent gradients, no persistent full FP32 gradient mirror, and no second full routed `dx`.

6. **Qualify A-output memory policies in this order: save → recompute → bounded pinned-host Z.** At rank 4 the compact saves are only \(24R\) bytes across W1/W3/W2, making simple GPU saving the best starting hypothesis.

7. **Autotune SM89 kernels at r4/r8/r16 against real routing histograms.** Prioritize specialized row-mapped Triton; use Ada-capable CUTLASS grouped GEMM as a gradient-operation benchmark and per-expert cuBLAS/PyTorch as the correctness/reference path. Do not import Hopper-only DeepGEMM/DeepEP. citeturn9search1turn9search8

8. **Run the exact EP8 node test before PP64.** This is where source-major ordering, top-8 scaling, expert skew, empty experts, and route conservation should fail cheaply if anything is wrong.

9. **Take the low-risk PP wins before FSDP changes:** suppress the three unnecessary sparse-cache transfers, profile real layer cost, rebalance boundaries, then add one bounded side-input prefetch slot only if profiling justifies it. Keep FSDP resharding unchanged. citeturn12view2

10. **Qualify all 64 trainers with uniform actual rank 4.** Uniform rank makes the first experiment about target coverage and expert correctness rather than capacity tuning. Stop on any memory failure, missing gradient, target mismatch, or post-warmup trainer step above 15 minutes. fileciteturn0file0

11. **Serve that r4 adapter in isolated vLLM 0.26 using `max_lora_rank=8`, explicit BF16 LoRA, TP8, and no inference EP.** Force the modular Triton FP8+LoRA path initially. Validate unload/reload/hash/logprob parity before any performance options. citeturn8search2 fileciteturn11file0

12. **Build compact mixed-rank transport only after uniform-rank serving passes.** Keep dense/shared tensors at r16 and routed experts at r4 in the WAN file, with explicit rank/alpha/scaling/layout metadata. Receiver-side zero padding is acceptable as an intermediate correctness bridge; pre-WAN padding is not.

13. **Add true effective-rank support to inference before claiming a rank-4 expert performance win.** Without this, a compact r4 file can still become rank-16 work after local padding. Treat transport savings and compute savings as separate qualifications.

14. **Optimize both H200 replicas with real rollout distributions.** Preserve whole GRPO prompt groups on a host for prefix reuse; sweep sequence concurrency and batched-token limits; then tune fused MoE LoRA and test dual streams. Prefix caching is particularly relevant when many completions share the same long prompt because vLLM can reuse those prefix KV blocks. citeturn14search1

15. **Only after memory improves, test FSDP resharding changes; only after measured PP idle time justifies it, revisit interleaving.** FSDP `False` explicitly exchanges memory for fewer gathers, and PyTorch interleaved 1F1B requires multiple stages per rank. Neither addresses the current first-order correctness problem. citeturn12view2turn12view0

16. **MTP is last.** Keep it out of the baseline until full expert-LoRA non-MTP serving meets correctness and performance gates. Then retain it only if it reduces *real two-replica rollout wall time*, not merely decode-kernel time. citeturn14search7

The resulting design preserves every part of the system that is already carrying the difficult risk—the block-FP8 base weights, RTX 4090 W8A8 forward/input-backward, PP8/CP8/node-local EP8, frozen router, and adapter-only transfer—while concentrating new complexity in one bounded BF16 low-rank expert primitive. ScatterMoE supplies the right data-movement idea; Prime-RL supplies useful expert-LoRA and RL references but not a compatible base expert implementation; PyTorch validates the conservative PP/FSDP ordering; and vLLM 0.26 is sufficiently close to the required H200 serving architecture to justify an isolated qualification, with mixed-rank semantics being the principal serving feature that still requires project-specific work. citeturn13view0turn9search0turn12view2turn8search0