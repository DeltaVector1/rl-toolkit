import verifiers.v1 as vf

from decensor_v1.taskset import DecensorTaskset

if hasattr(vf, "User"):
    # The deployed exchange discovers the task-attached user server.
    __all__ = ["DecensorTaskset"]
else:
    # Verifiers d30a3f48 discovers these classes directly through ``__all__``.
    # Its v1 ``eval`` command does not use the legacy load_environment hook.
    from decensor_v1.environment import DecensorEnv

    __all__ = ["DecensorTaskset", "DecensorEnv"]
