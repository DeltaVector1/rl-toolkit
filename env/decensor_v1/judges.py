from __future__ import annotations

try:
    from verifiers.v1.configs.judge import JudgeConfig
    from verifiers.v1.types import SamplingConfig as JudgeSamplingConfig
except ImportError:
    from verifiers.v1.judge import JudgeConfig, JudgeSamplingConfig


class DecensorJudgeSampling(JudgeSamplingConfig):
    temperature: float = 0.0
    max_tokens: int = 8192


class DecensorJudgeConfig(JudgeConfig):
    model: str = "google/gemma-4-26B-A4B-it"
    base_url: str = "http://198.13.252.41:40002/v1"
    api_key_var: str = "VLLM_API_KEY"
    sampling: DecensorJudgeSampling = DecensorJudgeSampling()
    min_p: float = 0.05
    disable_thinking: bool = True
