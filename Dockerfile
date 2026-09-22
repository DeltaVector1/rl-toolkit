FROM prime-rl:glm52-decensor
USER root
COPY --chmod=644 prime-rl/src/prime_rl/_compat.py /app/src/prime_rl/_compat.py
COPY --chmod=644 prime-rl/packages/prime-rl-configs/src/prime_rl/configs/shared.py /app/packages/prime-rl-configs/src/prime_rl/configs/shared.py
COPY --chmod=644 prime-rl/packages/prime-rl-configs/src/prime_rl/configs/trainer.py /app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py
COPY --chmod=644 prime-rl/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py /app/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py
COPY --chmod=644 prime-rl/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py /app/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py
COPY --chmod=644 prime-rl/packages/prime-rl-configs/src/prime_rl/configs/rl.py /app/packages/prime-rl-configs/src/prime_rl/configs/rl.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/rl/train.py /app/src/prime_rl/trainer/rl/train.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/rl/data.py /app/src/prime_rl/trainer/rl/data.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/rl/broadcast/filesystem.py /app/src/prime_rl/trainer/rl/broadcast/filesystem.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/model.py /app/src/prime_rl/trainer/model.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/lora.py /app/src/prime_rl/trainer/lora.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/parallel_dims.py /app/src/prime_rl/trainer/parallel_dims.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/utils.py /app/src/prime_rl/trainer/utils.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/weights.py /app/src/prime_rl/trainer/weights.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/fp8_checkpoint.py /app/src/prime_rl/trainer/fp8_checkpoint.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/mxfp4.py /app/src/prime_rl/trainer/mxfp4.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/layers/fp8_ada.py /app/src/prime_rl/trainer/models/layers/fp8_ada.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/distributed/expert_parallel.py /app/src/prime_rl/trainer/distributed/expert_parallel.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/distributed/partial_reshard.py /app/src/prime_rl/trainer/distributed/partial_reshard.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/distributed/pipeline_parallel.py /app/src/prime_rl/trainer/distributed/pipeline_parallel.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/distributed/pp_runner.py /app/src/prime_rl/trainer/distributed/pp_runner.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py /app/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py /app/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py /app/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py /app/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/layers/checkpointing.py /app/src/prime_rl/trainer/models/layers/checkpointing.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/layers/lora/multi_linear.py /app/src/prime_rl/trainer/models/layers/lora/multi_linear.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py /app/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/layers/moe.py /app/src/prime_rl/trainer/models/layers/moe.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/models/layers/norms.py /app/src/prime_rl/trainer/models/layers/norms.py
COPY --chmod=644 prime-rl/src/prime_rl/trainer/runs.py /app/src/prime_rl/trainer/runs.py
RUN sed -i \
    -e 's/nn.Parameter(self.to_sharded_dtensor(sharded_param))/nn.Parameter(self.to_sharded_dtensor(sharded_param), requires_grad=param.requires_grad)/' \
    -e 's/self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor)$/self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor), requires_grad=self.sharded_param.requires_grad/' \
    /app/.venv/lib/python3.12/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param.py
RUN sed -i \
    -e 's/and (is_sm_at_least_100() and is_cuda_version_at_least(12, 8))/and (torch.cuda.get_device_capability() >= (8, 9) and is_cuda_version_at_least(12, 8))/' \
    -e '/IS_ROCM = tl.constexpr(is_ROCM())/a\\    HAS_MXFP8_PTX = tl.constexpr(is_sm_at_least_100() and not is_ROCM())' \
    -e 's/USE_PTX=not IS_ROCM/USE_PTX=HAS_MXFP8_PTX/g' \
    /app/.venv/lib/python3.12/site-packages/torchao/prototype/mx_formats/kernels.py
COPY --chmod=644 prime-rl/src/prime_rl/utils/client.py /app/src/prime_rl/utils/client.py
COPY --chmod=644 prime-rl/src/prime_rl/utils/checkpoint_delta.py /app/src/prime_rl/utils/checkpoint_delta.py
COPY --chmod=644 prime-rl/src/prime_rl/utils/adapter_delta.py /app/src/prime_rl/utils/adapter_delta.py
COPY --chmod=644 prime-rl/src/prime_rl/utils/cp.py /app/src/prime_rl/utils/cp.py
COPY --chmod=644 prime-rl/src/prime_rl/utils/vlm.py /app/src/prime_rl/utils/vlm.py
COPY --chmod=644 prime-rl/src/prime_rl/utils/monitor/__init__.py /app/src/prime_rl/utils/monitor/__init__.py
COPY --chmod=644 prime-rl/src/prime_rl/utils/monitor/swanlab.py /app/src/prime_rl/utils/monitor/swanlab.py
COPY --chmod=644 prime-rl/src/prime_rl/orchestrator/orchestrator.py /app/src/prime_rl/orchestrator/orchestrator.py
COPY --chmod=644 prime-rl/src/prime_rl/orchestrator/algo/__init__.py /app/src/prime_rl/orchestrator/algo/__init__.py
COPY --chmod=644 prime-rl/src/prime_rl/orchestrator/algo/grpo.py /app/src/prime_rl/orchestrator/algo/grpo.py
COPY --chmod=644 prime-rl/src/prime_rl/orchestrator/dispatcher.py /app/src/prime_rl/orchestrator/dispatcher.py
COPY --chmod=644 prime-rl/src/prime_rl/orchestrator/watcher.py /app/src/prime_rl/orchestrator/watcher.py
COPY --chmod=644 prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py /app/src/prime_rl/inference/vllm/worker/filesystem.py
COPY --chmod=644 prime-rl/src/prime_rl/inference/vllm/server.py /app/src/prime_rl/inference/vllm/server.py
COPY --chmod=644 prime-rl/deps/verifiers/verifiers/v1/harnesses/null/harness.py /app/deps/verifiers/verifiers/v1/harnesses/null/harness.py
USER appuser
