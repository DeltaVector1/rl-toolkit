# Xiaomi Music

Native Verifiers v1 package `xiaomi-music` exports `MusicTaskset` and the stock
`NullHarness`, so its default is text-only composition with the exact released
prompt messages. It loads all 1,000 released rows from the pinned dataset using
data helpers included in the environment package; use native task selection for
evaluation subsets.

The scorer is the complete, unchanged Apache-2.0 source at XiaomiMiMo/verl commit
`a2ad9f6160b03ff2d47e59832bfb6b289f37c917`, including `ref_full4k.json` (see package
NOTICE/LICENSE). It extracts ABC from the final response, converts to MIDI, applies
the original syntax/bar/blank-line/channel conflict gates, computes all original
features, and divides the rounded 0–100 human-distribution score by 100. It does
not introduce a prompt-adherence judge. A missing `abc2midi` raises an infrastructure
error; invalid compositions retain the original zero scores.

Tasks require a container, default `python:3.11-slim`. Setup installs Debian's
`abcmidi` package if unavailable. The scorer and final response are staged at
scoring time. A prebuilt image with `python3`, `abc2midi`, and `uv` can avoid setup
installation. Runtime/CLI selection is the native Verifiers interface. Reasoning
models may need a large `--sampling.max-tokens` cap for Music; GLM-5.2 was
tested at 80,000, which still did not guarantee an untruncated reply.

Tests cover ABC extraction, all score gates, missing converter handling, native
loading/default harness, and a deterministic MIDI feature-pipeline fixture.
