import math
import re
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.models.layers.fp8_ada import (
    FP8CheckpointFusedQKVA,
    FP8CheckpointFusedQKVASlice,
    _Linear,
)
from prime_rl.trainer.models.layers.lora import (
    MultiLoRALinear,
    MultiLoRAModule,
    get_lora_num_tokens,
    get_multilora_scaling,
)
from prime_rl.trainer.models.layers.lora.multi_linear import (
    _run_lora_for_loop,
    _run_lora_grouped_mm,
    _SingleLoRA,
)
from prime_rl.trainer.models.layers.lora.multi_moe import (
    MultiLoRAGptOssGroupedExperts,
    MultiLoRAGroupedExperts,
    MultiLoRANonGatedGroupedExperts,
)
from prime_rl.trainer.models.layers.lora.native_fp8_moe import (
    NativeFP8MultiLoRAGroupedExperts,
)
from prime_rl.trainer.models.layers.moe import (
    FP8CheckpointFeedForward,
    GptOssGroupedExperts,
    GroupedExperts,
    NonGatedGroupedExperts,
)
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.utils.logger import get_logger


def strip_lora_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip LoRA from the state dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        new_state_dict[key] = value
    return new_state_dict


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """Get a module by its fully qualified name."""
    parts = module_name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a module by its fully qualified name."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _has_regex_metacharacters(pattern: str) -> bool:
    """Check if a pattern contains regex metacharacters."""
    regex_metachars = {".", "*", "+", "?", "^", "$", "[", "]", "{", "}", "|", "(", ")", "\\"}
    return any(char in pattern for char in regex_metachars)


def _matches_pattern(name: str, pattern: str) -> bool:
    """Check if a name matches a pattern.

    For simple patterns (no regex metacharacters), checks if any component
    in the module path matches the pattern exactly. For regex patterns, uses
    re.search() to match anywhere in the name (mirroring PEFT behavior).

    This handles cases where Linear layers might be nested (e.g.,
    "model.layers.0.q_proj.linear") while still matching standard architectures
    where they're direct children (e.g., "model.layers.0.self_attn.q_proj").
    """
    if _has_regex_metacharacters(pattern):
        return re.search(pattern, name) is not None
    else:
        return pattern in name.split(".")


class MultiLoRAFP8CheckpointFeedForward(MultiLoRAModule):
    """LoRA for an always-on shared expert with native FP8 base weights.

    The frozen checkpoint weights stay in FP8 and use their block scales in
    both forward and activation backward. Only the low-rank matrices are
    trainable, and those matrices stay in BF16.
    """

    _PROJECTIONS = (
        ("w1", "w1_scale_inv", "gate_proj"),
        ("w2", "w2_scale_inv", "down_proj"),
        ("w3", "w3_scale_inv", "up_proj"),
    )

    def __init__(
        self,
        base_layer: FP8CheckpointFeedForward,
        rank: int,
        n_adapters: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        for weight_name, _, _ in self._PROJECTIONS:
            weight = getattr(base_layer, weight_name)
            out_features, in_features = weight.shape
            setattr(
                self,
                f"lora_A_{weight_name}",
                nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.empty(
                                rank,
                                in_features,
                                device=weight.device,
                                dtype=torch.bfloat16,
                            )
                        )
                        for _ in range(n_adapters)
                    ]
                ),
            )
            setattr(
                self,
                f"lora_B_{weight_name}",
                nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.empty(
                                out_features,
                                rank,
                                device=weight.device,
                                dtype=torch.bfloat16,
                            )
                        )
                        for _ in range(n_adapters)
                    ]
                ),
            )
        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        indices = range(self.n_adapters) if index is None else (index,)
        for idx in indices:
            for weight_name, _, _ in self._PROJECTIONS:
                nn.init.kaiming_uniform_(
                    getattr(self, f"lora_A_{weight_name}")[idx],
                    a=math.sqrt(5),
                )
                nn.init.zeros_(getattr(self, f"lora_B_{weight_name}")[idx])

    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        parameters = []
        for weight_name, _, hf_name in self._PROJECTIONS:
            parameters.extend(
                [
                    (f"{hf_name}.lora_A", getattr(self, f"lora_A_{weight_name}")[idx]),
                    (f"{hf_name}.lora_B", getattr(self, f"lora_B_{weight_name}")[idx]),
                ]
            )
        return parameters

    def adapter_target_modules(self) -> tuple[str, ...]:
        return tuple(hf_name for _, _, hf_name in self._PROJECTIONS)

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter_params = 0
        adapted_params = 0
        for weight_name, _, _ in self._PROJECTIONS:
            adapter_params += getattr(self, f"lora_A_{weight_name}")[0].numel()
            adapter_params += getattr(self, f"lora_B_{weight_name}")[0].numel()
            adapted_params += getattr(self.base_layer, weight_name).numel()
        return adapter_params, adapted_params

    def _lora(self, x: torch.Tensor, weight_name: str, fqn: str) -> torch.Tensor:
        lora_x = self.lora_dropout(x)
        lora_A = getattr(self, f"lora_A_{weight_name}")
        lora_B = getattr(self, f"lora_B_{weight_name}")
        if self.n_adapters == 1:
            return self._scaling_factors[0] * _SingleLoRA.apply(
                lora_x, lora_A[0], lora_B[0], fqn
            )

        offsets = self._lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
        assert offsets[-1] == x.shape[0], f"offsets: {offsets}, x.shape: {x.shape}"
        stacked_A = torch.stack(list(lora_A))
        stacked_B = torch.stack(list(lora_B))
        aligned = self.rank % 8 == 0
        aligned = aligned and x.shape[-1] % 8 == 0 and stacked_B.shape[1] % 8 == 0
        if aligned:
            lora_out = _run_lora_grouped_mm(lora_x, stacked_A, stacked_B, offsets)
        else:
            lora_out = _run_lora_for_loop(lora_x, stacked_A, stacked_B, offsets)
        scaling = torch.repeat_interleave(
            self._scaling_factors, self._lora_num_tokens
        ).unsqueeze(-1)
        return scaling * lora_out

    def _base(self, x: torch.Tensor, weight_name: str, scale_name: str) -> torch.Tensor:
        fqn = getattr(
            self.base_layer,
            "_fp8_fqn",
            "<unassigned-fp8-shared-expert>",
        )
        return _Linear.apply(
            x,
            getattr(self.base_layer, weight_name),
            getattr(self.base_layer, scale_name),
            f"{fqn}.{weight_name}",
            (self.base_layer, weight_name, scale_name),
        )

    def _project(self, x: torch.Tensor, weight_name: str, scale_name: str) -> torch.Tensor:
        fqn = getattr(
            self.base_layer,
            "_fp8_fqn",
            "<unassigned-fp8-shared-expert>",
        )
        return self._base(x, weight_name, scale_name) + self._lora(
            x, weight_name, f"{fqn}.{weight_name}.lora"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        gate = self._project(x, "w1", "w1_scale_inv")
        up = self._project(x, "w3", "w3_scale_inv")
        hidden = F.silu(gate) * up
        out = self._project(hidden, "w2", "w2_scale_inv")
        return out.view(*shape[:-1], out.shape[-1])


def _find_target_modules(model: nn.Module, target_patterns: List[str]) -> List[str]:
    """Find all module names that match any of the target patterns.

    Patterns can be simple module names (e.g., "q_proj") or regex patterns
    (e.g., r".*\\.q_proj$"). Simple names match any component in the module path.

    Supports both nn.Linear layers and GroupedExperts (MoE) modules.
    """
    target_modules = []

    for name, module in model.named_modules():
        # Check if module is Linear or one of the supported expert classes
        if not isinstance(
            module,
            (
                nn.Linear,
                FP8CheckpointFusedQKVA,
                FP8CheckpointFusedQKVASlice,
                FP8CheckpointFeedForward,
                GroupedExperts,
                NonGatedGroupedExperts,
                GptOssGroupedExperts,
            ),
        ):
            continue

        for pattern in target_patterns:
            if _matches_pattern(name, pattern):
                target_modules.append(name)
                break

    return target_modules


def _validate_glm_lora_targets(
    model: nn.Module,
    configured_patterns: List[str],
    target_modules: List[str],
) -> None:
    if getattr(getattr(model, "config", None), "model_type", None) != "glm_moe_dsa":
        return

    required_patterns = {
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "shared_expert",
        "experts",
    }
    missing_patterns = required_patterns - set(configured_patterns)
    if missing_patterns:
        raise ValueError(
            "GLM-5.2 LoRA does not cover the full attention and MLP path; "
            f"missing target names: {sorted(missing_patterns)}"
        )

    attention_names = {
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    }
    dense_names = {"gate_proj", "up_proj", "down_proj"}
    expected = set()
    for name, module in model.named_modules():
        final_name = name.rsplit(".", 1)[-1]
        if isinstance(
            module,
            (nn.Linear, FP8CheckpointFusedQKVA, FP8CheckpointFusedQKVASlice),
        ) and final_name in attention_names:
            expected.add(name)
        elif (
            isinstance(module, nn.Linear)
            and final_name in dense_names
            and ".mlp." in f".{name}."
        ):
            expected.add(name)
        elif isinstance(module, FP8CheckpointFeedForward) and final_name == "shared_expert":
            expected.add(name)
        elif isinstance(module, GroupedExperts) and final_name == "experts":
            expected.add(name)

    actual = set(target_modules)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            "GLM-5.2 LoRA target inventory does not match this pipeline stage: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"unexpected={unexpected[:8]} ({len(unexpected)} total)"
        )

    layer_count = sum(
        1
        for _, module in model.named_modules()
        if type(module).__name__ == "GlmMoeDsaDecoderLayer"
    )
    if not layer_count:
        raise ValueError("GLM-5.2 LoRA target validation found no decoder layers")
    get_logger().info(
        "Validated full GLM-5.2 LoRA coverage on this pipeline stage: "
        f"{len(expected)} module containers across {layer_count} decoder layers"
    )


def _should_keep_trainable(param_name: str, modules_to_save_patterns: List[str]) -> bool:
    """Check if a parameter should remain fully trainable.

    Checks both the full parameter name and the parent module name against patterns.
    For example, for param "model.embed_tokens.weight", it checks both:
    - "model.embed_tokens.weight" (full parameter name)
    - "model.embed_tokens" (module name)

    Patterns can be simple module names (e.g., "embed_tokens") or regex patterns.
    """
    for pattern in modules_to_save_patterns:
        if _matches_pattern(param_name, pattern):
            return True

    module_name = param_name.rsplit(".", 1)[0] if "." in param_name else param_name
    for pattern in modules_to_save_patterns:
        if _matches_pattern(module_name, pattern):
            return True

    return False


def freeze_all_except_lora_and_specified(model: nn.Module, config: LoRAConfig) -> None:
    """
    Freeze all parameters except LoRA adapters and specified trainable modules.

    Args:
        model: The model to freeze parameters in
        config: LoRA configuration with modules_to_save patterns
    """
    for name, param in model.named_parameters():
        if any(lora_param in name for lora_param in ["lora_A", "lora_B"]):
            param.requires_grad = True
        elif _should_keep_trainable(name, config.modules_to_save):
            param.requires_grad = True
        else:
            param.requires_grad = False


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> None:
    """
    Apply LoRA to target modules in the model and freeze non-LoRA parameters.

    WARNING: This function modifies requires_grad on parameters. If using FSDP2,
    this MUST be called BEFORE setup_fsdp() to avoid dtensor/sharding issues.

    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
    """
    logger = get_logger()
    from prime_rl.trainer.models import PreTrainedModelPrimeRL

    if isinstance(model, PreTrainedModelPrimeRL):
        get_multi_run_manager().register_adapter_state_dict_converter(type(model).convert_adapter_to_hf)
    n_loras = get_multi_run_manager().max_runs

    from torch.distributed.fsdp import FSDPModule

    if any(isinstance(m, FSDPModule) for m in model.modules()):
        logger.error(
            "Model is already wrapped with FSDP! LoRA must be applied BEFORE FSDP setup to avoid dtensor issues."
        )
        raise RuntimeError("Cannot apply LoRA to FSDP-wrapped model. Apply LoRA before setup_fsdp().")

    logger.debug(f"Applying LoRA to model: {model} for {config.target_modules}")
    target_modules = _find_target_modules(model, config.target_modules)
    logger.debug(
        f"Found {len(target_modules)} target modules for LoRA: {target_modules[:10]} ... {target_modules[-10:]}"
    )

    if not target_modules:
        raise ValueError(f"No LoRA target modules found for patterns {config.target_modules}.")
    _validate_glm_lora_targets(model, config.target_modules, target_modules)

    for module_name in target_modules:
        base_module = _get_module_by_name(model, module_name)

        # Handle Linear layers
        if isinstance(
            base_module,
            (nn.Linear, FP8CheckpointFusedQKVA, FP8CheckpointFusedQKVASlice),
        ):
            lora_module = MultiLoRALinear(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        elif isinstance(base_module, FP8CheckpointFeedForward):
            lora_module = MultiLoRAFP8CheckpointFeedForward(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        # Handle GroupedExperts (MoE)
        elif isinstance(base_module, GroupedExperts):
            if base_module.fp8_checkpoint:
                lora_module = NativeFP8MultiLoRAGroupedExperts(
                    base_layer=base_module,
                    rank=config.rank,
                    n_adapters=n_loras,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
            else:
                lora_module = MultiLoRAGroupedExperts(
                    base_layer=base_module,
                    rank=config.rank,
                    n_adapters=n_loras,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
        # Handle NonGatedGroupedExperts (relu2 experts used by NemotronH's LatentMoE)
        elif isinstance(base_module, NonGatedGroupedExperts):
            lora_module = MultiLoRANonGatedGroupedExperts(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        # Handle GptOssGroupedExperts (gpt-oss fused gate_up + biases)
        elif isinstance(base_module, GptOssGroupedExperts):
            lora_module = MultiLoRAGptOssGroupedExperts(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        else:
            logger.warning(
                f"Module {module_name} is type {type(base_module).__name__}, "
                "expected nn.Linear, FP8CheckpointFeedForward, GroupedExperts, "
                "NonGatedGroupedExperts, or GptOssGroupedExperts. Skipping."
            )
            continue

        lora_module.register_with_runs(get_multi_run_manager(), module_name)
        _set_module_by_name(model, module_name, lora_module)

    if config.initial_adapter_path is not None:
        # MultiRunManager resets a slot when it discovers a new orchestrator
        # run. Loading here would be silently erased by that later reset.
        # The creation hook runs immediately after reset on every rank.
        initial_adapter_path = config.initial_adapter_path

        def load_after_run_reset(adapter_idx: int, _run_id: str) -> None:
            load_initial_lora_adapter(model, initial_adapter_path, adapter_idx)

        get_multi_run_manager().register_creation_hook(load_after_run_reset)

    freeze_all_except_lora_and_specified(model, config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_adapter_params = 0
    lora_adapted_params = 0
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            adapter_params, adapted_params = module.get_lora_param_counts()
            lora_adapter_params += adapter_params
            lora_adapted_params += adapted_params

    fully_trainable = trainable_params - lora_adapter_params
    adapted_or_trainable = lora_adapted_params + fully_trainable

    logger.info(f"LoRA enabled: {lora_adapter_params:,} adapter params adapting {lora_adapted_params:,} base params")
    logger.info(f"LoRA: {fully_trainable:,} fully trainable parameters")
    logger.info(f"LoRA: {adapted_or_trainable:,} adapted or fully trainable out of {total_params:,} parameters")


def load_initial_lora_adapter(model: nn.Module, adapter_path: str, adapter_idx: int = 0) -> None:
    """Load one exported PEFT adapter into a freshly created MultiLoRA slot.

    This restores the policy after a trainer restart without loading the
    frozen base checkpoint again from a distributed checkpoint. Optimizer
    state is not restored.
    """
    from pathlib import Path

    from safetensors.torch import load_file

    path = Path(adapter_path)
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Initial LoRA adapter not found: {path}")

    state_dict = load_file(str(path), device="cpu")
    expected: dict[str, nn.Parameter] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, MultiLoRAModule):
            continue
        if adapter_idx >= module.n_adapters:
            raise ValueError(
                f"Initial LoRA adapter index {adapter_idx} is invalid for "
                f"{module_name} with {module.n_adapters} slots"
            )
        for param_name, param in module.named_parameters_for_adapter(adapter_idx):
            expected[f"{module_name}.{param_name}.weight"] = param

    if hasattr(type(model), "convert_adapter_to_hf"):
        expected = type(model).convert_adapter_to_hf(expected)

    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing:
        raise ValueError(
            "Initial LoRA adapter keys do not match the local pipeline stage: "
            f"missing={missing[:5]} ({len(missing)} total)"
        )
    # Every PP stage loads the same global adapter file but constructs only its
    # own layers. Keys for the other stages are therefore expected here. A
    # missing or misspelled key still fails on the stage that owns that layer.

    with torch.no_grad():
        for key, param in expected.items():
            value = state_dict[key]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(
                    f"Initial LoRA adapter shape mismatch for {key}: "
                    f"checkpoint={tuple(value.shape)}, model={tuple(param.shape)}"
                )
            param.copy_(value.to(device=param.device, dtype=param.dtype))

    get_logger().info(
        f"Loaded {len(expected)} initial LoRA tensors from {path} into adapter slot "
        f"{adapter_idx}; ignored {len(unexpected)} tensors owned by other pipeline stages"
    )


def has_lora_layers(model: nn.Module) -> bool:
    """Check if model has LoRA layers."""
    for module in model.modules():
        if isinstance(module, MultiLoRAModule):
            return True
    return False


def clean_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove LoRA parameters and fix LoRA base layer key names for HF compatibility."""
    clean_state_dict = {}

    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue

        if ".base_layer." in key:
            new_key = key.replace(".base_layer.", ".")
            clean_state_dict[new_key] = value
        else:
            clean_state_dict[key] = value

    return clean_state_dict


def save_lora_config(model: nn.Module, save_path, rank: int, alpha: float, dropout: float) -> None:
    """
    Save LoRA configuration as JSON for adapter portability.

    Args:
        model: Model with LoRA layers to introspect
        save_path: Path object or string pointing to directory where adapter_config.json will be saved
        rank: LoRA rank
        alpha: LoRA alpha scaling parameter
        dropout: LoRA dropout rate
    """
    import json
    from pathlib import Path

    save_path = Path(save_path)

    # Extract actual target modules from the model
    target_modules = set()
    modules_to_save = set()

    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            if hasattr(module, "adapter_target_modules"):
                target_modules.update(module.adapter_target_modules())
            else:
                module_suffix = name.split(".")[-1]
                target_modules.add(module_suffix)

    for name, param in model.named_parameters():
        if param.requires_grad and "lora_A" not in name and "lora_B" not in name:
            module_name = name.rsplit(".", 1)[0].split(".")[-1]
            modules_to_save.add(module_name)

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model.config._name_or_path,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "target_modules": sorted(list(target_modules)),
        "modules_to_save": sorted(list(modules_to_save)) if modules_to_save else None,
    }

    config_path = save_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
