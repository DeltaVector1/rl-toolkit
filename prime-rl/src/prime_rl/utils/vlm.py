"""Vision-language model architecture metadata and component access."""

from dataclasses import dataclass

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig


@dataclass(frozen=True)
class VLMModelInfo:
    """Attribute paths for one VLM family."""

    vision_encoder_attr: str
    language_model_attr: str


VLM_REGISTRY: dict[str, VLMModelInfo] = {
    "qwen3_vl": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_5_moe": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "qwen3_vl_moe": VLMModelInfo(vision_encoder_attr="model.visual", language_model_attr="model.language_model"),
    "gemma4": VLMModelInfo(vision_encoder_attr="model.vision_tower", language_model_attr="model.language_model"),
}

DEFAULT_LAYER_PREFIX = "model.layers."


def get_vision_encoder(model: nn.Module, override: str | None = None) -> nn.Module | None:
    """Return the vision encoder, or None when the model is text-only."""
    if override is not None:
        result = _resolve_attr(model, override)
        if result is None:
            raise ValueError(f"vlm.vision_encoder_attr='{override}' does not resolve on this model")
        return result
    info = _get_model_info(model)
    return _resolve_attr(model, info.vision_encoder_attr) if info is not None else None


def get_language_model(model: nn.Module, override: str | None = None) -> nn.Module:
    """Return the module that owns the transformer layers."""
    if override is not None:
        result = _resolve_attr(model, override)
        if result is None:
            raise ValueError(f"vlm.language_model_attr='{override}' does not resolve on this model")
        return result
    info = _get_model_info(model)
    if info is not None:
        result = _resolve_attr(model, info.language_model_attr)
        if result is not None:
            return result
    return model.model


def is_vlm_architecture(model_config: PretrainedConfig) -> bool:
    """Return whether the config is a registered VLM architecture."""
    return _get_model_info_from_config(model_config) is not None


def get_final_logit_softcapping(model_config: PretrainedConfig) -> float | None:
    """Return text-model final-logit softcapping when it is configured."""
    return getattr(model_config.get_text_config(), "final_logit_softcapping", None)


def get_layer_prefix(model_config: PretrainedConfig, override: str | None = None) -> str:
    """Return the checkpoint prefix for language-model layers."""
    if override is not None:
        return override
    info = _get_model_info_from_config(model_config)
    return info.language_model_attr + ".layers." if info is not None else DEFAULT_LAYER_PREFIX


def _get_model_info(model: nn.Module) -> VLMModelInfo | None:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return VLM_REGISTRY.get(model_type) if model_type else None


def _get_model_info_from_config(model_config: PretrainedConfig) -> VLMModelInfo | None:
    model_type = getattr(model_config, "model_type", None)
    return VLM_REGISTRY.get(model_type) if model_type else None


def _resolve_attr(obj, dotted_path: str):
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj
