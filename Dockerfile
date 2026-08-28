ARG BASE_IMAGE=prime-rl:glm52-v080-full
FROM ${BASE_IMAGE}
USER root
ADD --checksum=sha256:db0c9410a4c130da8ffeea22f569faca7c85c58c5e183be1ebf758076361a6ec https://github.com/PrimeIntellect-ai/router/releases/download/v0.1.26/vllm_router-0.1.26-cp38-abi3-manylinux_2_28_x86_64.whl /tmp/vllm_router-0.1.26-cp38-abi3-manylinux_2_28_x86_64.whl
RUN VIRTUAL_ENV=/app/.venv uv pip install --no-deps /tmp/vllm_router-0.1.26-cp38-abi3-manylinux_2_28_x86_64.whl
COPY --chmod=644 source/packages/prime-rl-configs/src/prime_rl/configs/shared.py /app/packages/prime-rl-configs/src/prime_rl/configs/shared.py
COPY --chmod=644 source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py /app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py
COPY --chmod=644 source/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py /app/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py
COPY --chmod=644 source/packages/prime-rl-configs/src/prime_rl/configs/rl.py /app/packages/prime-rl-configs/src/prime_rl/configs/rl.py
COPY --chmod=644 source/src/prime_rl/trainer/rl/train.py /app/src/prime_rl/trainer/rl/train.py
COPY --chmod=644 source/src/prime_rl/trainer/rl/token_export.py /app/src/prime_rl/trainer/rl/token_export.py
COPY --chmod=644 source/src/prime_rl/parity_router_trace.py /app/src/prime_rl/parity_router_trace.py
COPY --chmod=644 patches/prime-rl/v0.8.0-glm52/0012-Register-trainer-only-LoRA-runs.patch /tmp/trainer-only-lora.patch
RUN cd /app && if git apply --check --include=src/prime_rl/trainer/runs.py /tmp/trainer-only-lora.patch; then \
        git apply --include=src/prime_rl/trainer/runs.py /tmp/trainer-only-lora.patch; \
    else \
        git apply --reverse --check --include=src/prime_rl/trainer/runs.py /tmp/trainer-only-lora.patch; \
    fi
COPY --chmod=644 source/src/prime_rl/trainer/model.py /app/src/prime_rl/trainer/model.py
COPY --chmod=644 source/src/prime_rl/trainer/parallel_dims.py /app/src/prime_rl/trainer/parallel_dims.py
COPY --chmod=644 source/src/prime_rl/trainer/fp8_checkpoint.py /app/src/prime_rl/trainer/fp8_checkpoint.py
COPY --chmod=644 source/src/prime_rl/trainer/mxfp4.py /app/src/prime_rl/trainer/mxfp4.py
COPY --chmod=644 source/src/prime_rl/trainer/models/layers/fp8_ada.py /app/src/prime_rl/trainer/models/layers/fp8_ada.py
COPY --chmod=644 source/src/prime_rl/trainer/distributed/expert_parallel.py /app/src/prime_rl/trainer/distributed/expert_parallel.py
COPY --chmod=644 source/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py /app/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py
COPY --chmod=644 source/src/prime_rl/trainer/models/glm_moe_dsa/position_metadata_cache.py /app/src/prime_rl/trainer/models/glm_moe_dsa/position_metadata_cache.py
COPY --chmod=644 source/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py /app/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py
COPY --chmod=644 source/src/prime_rl/trainer/models/layers/checkpointing.py /app/src/prime_rl/trainer/models/layers/checkpointing.py
COPY --chmod=644 source/src/prime_rl/trainer/models/layers/lora/multi_linear.py /app/src/prime_rl/trainer/models/layers/lora/multi_linear.py
COPY --chmod=644 source/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py /app/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py
COPY --chmod=644 source/src/prime_rl/trainer/models/layers/moe.py /app/src/prime_rl/trainer/models/layers/moe.py
COPY --chmod=644 patches/prime-rl/build-deepep-fused-score.patch /tmp/deepep-fused-score.patch
RUN cd /app && if git apply --check --include=src/prime_rl/trainer/distributed/deepep.py /tmp/deepep-fused-score.patch; then \
        git apply --include=src/prime_rl/trainer/distributed/deepep.py /tmp/deepep-fused-score.patch; \
    else \
        git apply --reverse --check --include=src/prime_rl/trainer/distributed/deepep.py /tmp/deepep-fused-score.patch; \
    fi
RUN sed -i \
    -e 's/nn.Parameter(self.to_sharded_dtensor(sharded_param))/nn.Parameter(self.to_sharded_dtensor(sharded_param), requires_grad=param.requires_grad)/' \
    -e 's/self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor)$/self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor), requires_grad=self.sharded_param.requires_grad/' \
    /app/.venv/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py
RUN sed -i \
    -e 's/and (is_sm_at_least_100() and is_cuda_version_at_least(12, 8))/and (torch.cuda.get_device_capability() >= (8, 9) and is_cuda_version_at_least(12, 8))/' \
    -e '/IS_ROCM = tl.constexpr(is_ROCM())/a\\    HAS_MXFP8_PTX = tl.constexpr(is_sm_at_least_100() and not is_ROCM())' \
    -e 's/USE_PTX=not IS_ROCM/USE_PTX=HAS_MXFP8_PTX/g' \
    /app/.venv/lib/python3.12/site-packages/torchao/prototype/mx_formats/kernels.py
COPY --chmod=644 source/src/prime_rl/utils/client.py /app/src/prime_rl/utils/client.py
COPY --chmod=644 source/src/prime_rl/utils/monitor/__init__.py /app/src/prime_rl/utils/monitor/__init__.py
COPY --chmod=644 source/src/prime_rl/utils/monitor/swanlab.py /app/src/prime_rl/utils/monitor/swanlab.py
COPY --chmod=644 source/src/prime_rl/orchestrator/orchestrator.py /app/src/prime_rl/orchestrator/orchestrator.py
COPY --chmod=644 source/src/prime_rl/orchestrator/dispatcher.py /app/src/prime_rl/orchestrator/dispatcher.py
COPY --chmod=644 source/src/prime_rl/orchestrator/watcher.py /app/src/prime_rl/orchestrator/watcher.py
COPY --chmod=644 source/src/prime_rl/inference/vllm/worker/filesystem.py /app/src/prime_rl/inference/vllm/worker/filesystem.py
COPY --chmod=644 source/src/prime_rl/inference/vllm/server.py /app/src/prime_rl/inference/vllm/server.py
COPY --chmod=644 source/deps/verifiers/verifiers/v1/harnesses/null/harness.py /app/deps/verifiers/verifiers/v1/harnesses/null/harness.py
USER appuser
