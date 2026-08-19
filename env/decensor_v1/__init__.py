import verifiers.v1 as vf

from decensor_v1.taskset import DecensorTaskset

if hasattr(vf, "User"):
    __all__ = ["DecensorTaskset"]
else:
    from decensor_v1.environment import DecensorEnv

    __all__ = ["DecensorTaskset", "DecensorEnv"]
