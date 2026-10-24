try:
    from refusals_classifier_eval_v1.taskset import RefusalsClassifierEvalTaskset
except ModuleNotFoundError as error:
    if error.name != "verifiers":
        raise
    __all__: list[str] = []
else:
    __all__ = ["RefusalsClassifierEvalTaskset"]
