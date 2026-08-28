# GLM-5.2 MTP backports for vLLM 0.26

Apply the numbered files to a clean vLLM 0.26 source tree after the accepted
files `0001` through `0005`.

`0006` is the narrow local backport of vLLM PR #46908,
`150e73eda5678e80e2eec36642a1234c709275fb`. It unwraps the base-model entry
(`slot_mapping[0]`) in `get_attention_context` and direct MLA calls.

`0007` is the narrow local backport of vLLM PR #47013,
`30030761508a9801f4a8db2a140137d8fefd11ac`. It unwraps the same base-model
entry in hidden-state cache updates and the v0.26 DeepSeek V3.2 NVIDIA MLA
path.

The upstream GLM MTP quality fix, vLLM PR #47448 commit
`d039c171144bdc6e19d14d3dd9b0625f84f16d1c`, is already in vLLM 0.26.

The vLLM PR #53755 FlashMLA update is separate. It only changes the FlashMLA
pin to `0397728d511c4e3d94ea3a01d8dda8654525a611` for no-split decode
workspace allocation. It does not fix a stride overflow.

The CPU tests cover both slot-mapping consumers, TP8/MTP1 at 32K, the target
LoRA contract, router-trace isolation, and patch application. Set
`VLLM_V026_SOURCE_ROOT` to a clean vLLM 0.26 checkout to run source checks and
`git apply --check` for the full series.
