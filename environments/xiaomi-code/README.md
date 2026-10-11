# xiaomi-code

Native Verifiers v1 taskset for the published Xiaomi MiMo Code tasks. Adapted from `XiaomiMiMo/MiMo-Agent` commit `467f0a19016f0ac4d63b8d17a1f0da9ba07f232c`, `opensource_code.py` and its base helpers; the MIT license is included in `xiaomi_code/LICENSE.mimoagent`. The root workspace pins Verifiers and the dataset revision. Install with `uv sync` from the repository root.

The standard Verifiers single-agent environment supplies the bash harness and its bash/edit tools. Select the Prime runtime in your evaluation configuration; each task supplies its published container image and working directory. `NEEDS_CONTAINER` requires container execution. The package does not execute task commands on the host.

Some published images install Go at `/usr/local/go/bin/go` without exporting its path. Setup prepends that directory to the sandbox's existing `PATH` only when `go` is otherwise unavailable. Both the agent and verifier receive the repaired environment; an existing selected Go installation takes precedence.

`XiaomiCodeTaskset` loads all published `code.parquet` rows. Set `env.taskset.dataset_path` to an equivalent local parquet, or `env.taskset.revision` to override the pinned Hugging Face revision. Image names are translated strictly through that revision's published image mapping.

Setup captures the image's base commit and rejects reachable history outside its ancestry. Hidden test patches stay in host task data until scoring. Scoring restores both sides of paths touched by the patch, applies it, and runs the original command with its published timeout. Exit code zero earns 1; other exit codes earn 0. Reset, patch and transport failures raise an infrastructure error with `reward/testbed_corrupted` metadata. Test paths are restored afterward without resetting solution files. Unlike upstream’s zero reward plus infrastructure metadata, infrastructure failures raise so Verifiers can distinguish them from incorrect model answers.

This preserves upstream's shared agent/grader filesystem design and image hygiene assumptions. The history assertion does not inspect unreachable objects or search the image for leaked answers. Tests use synthetic repositories and mocked runtimes; live sandbox evaluation is separate.

Test patch headers support the released unquoted paths with any single-component Git prefix, including `ab/`. Unsupported headers, including quoted paths and headerless unified patches, fail when loading instead of leaving test paths outside the reset.
