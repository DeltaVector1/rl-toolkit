import verifiers.v1 as vf

if hasattr(vf, "User"):
    __all__ = ["DecensorTaskset"]
else:
    __all__ = ["DecensorTaskset", "DecensorEnv"]


def __getattr__(name: str):
    if name == "DecensorTaskset":
        from decensor_v1.taskset import DecensorTaskset

        return DecensorTaskset
    if name == "DecensorEnv" and not hasattr(vf, "User"):
        from decensor_v1.environment import DecensorEnv

        return DecensorEnv
    raise AttributeError(name)
