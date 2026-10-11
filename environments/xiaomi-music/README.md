# xiaomi-music

Native Verifiers v1 `MusicTaskset` for Xiaomi ABC composition.
Install with sibling `xiaomi-common` and the repository's pinned Verifiers.
The exported stock `NullHarness` preserves the text-only prompt by default.
Tasks use a container with `abc2midi`; setup installs Debian `abcmidi` if absent.

The complete upstream scorer and baseline are unchanged. See NOTICE/LICENSE for
attribution and repository `docs/music.md` for scoring and runtime details.
