# xiaomi-code

Native Verifiers v1 taskset for the published Xiaomi MiMo Code tasks. Adapted from `XiaomiMiMo/MiMo-Agent` commit `467f0a19016f0ac4d63b8d17a1f0da9ba07f232c`, `opensource_code.py` and its base helpers; the MIT license is included in `xiaomi_code/LICENSE.mimoagent`. The root workspace pins Verifiers and the dataset revision. Install with `uv sync` from the repository root.

The standard Verifiers single-agent environment supplies the bash harness and its bash/edit tools. Select the Prime runtime in your evaluation configuration; each task supplies its published container image and working directory. `NEEDS_CONTAINER` requires container execution. The package does not execute task commands on the host.

Published Code images are flattened without an image environment. The task supplies Docker's default `PATH` and a Code-only `BASH_ENV` bootstrap to both the agent and verifier. The bootstrap restores the image's login profiles without recursive startup and is rewritten before grading. When the corresponding command is missing, it also restores installed Rust tools under `/usr/local/cargo` (with their Cargo/Rustup homes), Java under `/opt/java/openjdk`, and `okexchaincli` under `$HOME/go/bin`. General tasks are unaffected. Code requests 8 GB memory and a 600-second setup budget; native runtime overrides retain precedence.

`XiaomiCodeTaskset` loads all published `code.parquet` rows. Set `env.taskset.dataset_path` to an equivalent local parquet, or `env.taskset.revision` to override the pinned Hugging Face revision. Image names are translated strictly through that revision's published image mapping.

Every setup captures the image's real base commit, disables Git replacements, detaches at the base, removes other refs except ancestor tags, expires reflogs, and runs strict `git gc --prune=now` with recent-object retention disabled. The base ancestry and staged/working-tree edits are preserved. Setup fails if sanitation fails or the remaining history assertion fails; it also rejects partial/promisor clones, shared object stores/worktrees and packs protected from garbage collection. Production setup does not run `git fsck`.

Setup removes only regenerated `__pycache__`, `.pytest_cache` and `.jest/cache` directories without tracked contents. It skips `.git` and `.build_env`; other build artifacts and residue remain.

Hidden test patches stay in host task data until scoring. Scoring restores both sides of paths touched by the patch, applies it, and runs the original command with its published timeout. Exit code zero earns 1; other exit codes earn 0. Reset, patch and transport failures raise an infrastructure error with `reward/testbed_corrupted` metadata. Test paths are restored afterward without resetting solution files. Traces retain the last 10,000 characters of verifier stdout and 4,000 of stderr. Unlike upstream’s zero reward plus infrastructure metadata, infrastructure failures raise so Verifiers can distinguish them from incorrect model answers.

This preserves upstream's shared agent/grader filesystem design: image profiles and project verifier scripts remain trusted inputs. Sanitation does not search the whole image for leaked answers. See [Code training and exclusions](../../docs/code.md) for remaining known-bad rows. Tests use synthetic repositories and mocked runtimes; live sandbox evaluation is separate.

Test patch headers support released unquoted paths without whitespace, with any single-component Git prefix including `ab/`. Both Git headers and unified `---`/`+++` file headers are read, including mixed patches; `/dev/null` is ignored and hunk payloads are never treated as file headers. Quoted or other unsupported path formats fail when loading instead of leaving test paths outside the reset.
