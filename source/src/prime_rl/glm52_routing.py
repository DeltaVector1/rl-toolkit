from __future__ import annotations

from collections.abc import Mapping
from typing import Any


GLM_MODEL_TYPE = "glm_moe_dsa"
GLM_ROUTING_REQUIREMENTS = {
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
}

_MISSING = object()


def _get(config: object, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, _MISSING)
    return getattr(config, name, _MISSING)


def glm_routing_errors(
    config: object,
    *,
    router_frozen: bool | None = None,
    correction_bias_present: bool | None = None,
) -> list[str]:
    errors: list[str] = []
    model_type = _get(config, "model_type")
    if model_type is _MISSING:
        errors.append("model_type is missing")
    elif model_type != GLM_MODEL_TYPE:
        errors.append(f"model_type must be {GLM_MODEL_TYPE!r}, got {model_type!r}")

    for name, expected in GLM_ROUTING_REQUIREMENTS.items():
        actual = _get(config, name)
        if actual is _MISSING:
            errors.append(f"{name} is missing; expected {expected!r}")
        elif isinstance(expected, bool) and type(actual) is not bool:
            errors.append(f"{name} must be {expected!r}, got {actual!r}")
        elif actual != expected:
            errors.append(f"{name} must be {expected!r}, got {actual!r}")

    if router_frozen is False:
        errors.append("router_frozen must be true")
    if correction_bias_present is False:
        errors.append("correction bias is missing")
    return errors


def require_glm_routing_config(
    config: object,
    *,
    router_frozen: bool | None = None,
    correction_bias_present: bool | None = None,
    side: str,
) -> None:
    errors = glm_routing_errors(
        config,
        router_frozen=router_frozen,
        correction_bias_present=correction_bias_present,
    )
    if errors:
        raise ValueError(f"{side} GLM routing configuration rejected: " + "; ".join(errors))
