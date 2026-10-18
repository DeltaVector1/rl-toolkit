# xiaomi-music

Native Verifiers v1 `MusicTaskset` for Xiaomi ABC composition. The taskset and
data helpers are contained in this package. Its direct dependencies are Verifiers
`>=0.3.2.dev206,<=0.3.2.dev208`, `huggingface-hub>=0.24`, and `pyarrow>=18`.
Install it in a fresh environment from this directory:

```sh
uv venv /tmp/xiaomi-music-venv --python 3.12
uv pip install --python /tmp/xiaomi-music-venv/bin/python .
```

No sibling Xiaomi package or repository checkout is required. The repository
workspace separately pins its Verifiers revision.
The exported stock `NullHarness` preserves the text-only prompt by default.
Tasks use a container with `abc2midi`; setup installs Debian `abcmidi` if absent.

The complete upstream scorer and baseline are unchanged. See NOTICE/LICENSE for
attribution and repository `docs/music.md` for scoring and runtime details.
