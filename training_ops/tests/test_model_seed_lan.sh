#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
script=$root/training_ops/runtime/model_seed_lan.sh
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

source_model=$tmp/source-model
target_model=$tmp/target-model
state=$tmp/state
partial=$tmp/partial
fake_bin=$tmp/bin
mkdir -p "$source_model" "$target_model" "$fake_bin"

for i in $(seq 1 81); do
  printf 'shard-%s\n' "$i" >"$source_model/$(printf 'model-%05d-of-00081.safetensors' "$i")"
done
printf '{}\n' >"$source_model/config.json"
printf 'tokenizer\n' >"$source_model/tokenizer.json"
for name in metadata.json generation_config.json tokenizer_config.json tokenizer.json.vocab \
  special_tokens_map.json added_tokens.json chat_template.jinja merges.txt vocab.json \
  preprocessor_config.json model_index.json README.md LICENSE; do
  printf '%s\n' "$name" >"$source_model/$name"
done
mkdir -p "$source_model/.cache"
printf 'not part of the model module\n' >"$source_model/.cache/ignored"

cat >"$fake_bin/rsync" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
if printf '%s\n' "${args[@]}" | grep -Fqx -- '--daemon'; then
  trap 'exit 0' TERM INT
  while :; do sleep 1; done
fi
printf '%s\n' "${args[@]}" >>"$FAKE_RSYNC_LOG"
source=${args[$#-2]}
destination=${args[$#-1]}
if [[ "$source" == */model/ ]]; then
  cp -a "$FAKE_SOURCE"/. "$destination/"
else
  cp "$FAKE_MANIFEST" "$destination"
fi
EOF
chmod +x "$fake_bin/rsync"

export PATH="$fake_bin:$PATH"
export FAKE_RSYNC_LOG=$tmp/rsync.log
export FAKE_SOURCE=$source_model
export FAKE_MANIFEST=$tmp/source-manifest.sha256
export MODEL_SEED_MODEL_DIR=$source_model
export MODEL_SEED_STATE_DIR=$state
export MODEL_SEED_PARTIAL_DIR=$partial

printf 'partial\n' >"$source_model/model-00081-of-00081.safetensors_partial"
if "$script" source-start >"$tmp/range.out" 2>"$tmp/range.err"; then
  echo 'ModelScope range file must be refused' >&2
  exit 1
fi
grep -F 'incomplete or range file' "$tmp/range.err" >/dev/null
rm -f -- "$source_model/model-00081-of-00081.safetensors_partial"

rm -f -- "$source_model/model-00081-of-00081.safetensors"
printf 'replacement\n' >"$source_model/replacement.bin"
if "$script" source-start >"$tmp/count.out" 2>"$tmp/count.err"; then
  echo 'incomplete shard count must be refused' >&2
  exit 1
fi
grep -F 'expected 81 model shards' "$tmp/count.err" >/dev/null
printf 'shard-81\n' >"$source_model/model-00081-of-00081.safetensors"
rm -f -- "$source_model/replacement.bin"

"$script" source-start >"$tmp/start.out"
test "$(wc -l <"$state/manifest.sha256")" -eq 96
awk '{print $2}' "$state/manifest.sha256" | LC_ALL=C sort -c
! grep -F '.cache/' "$state/manifest.sha256"
test "$(stat -c '%a' "$state/rsyncd.conf")" -eq 444
grep -Fx 'hosts allow = 10.92.20.76, 10.92.20.77, 10.92.20.78, 10.92.20.80, 10.92.20.83, 10.92.20.84, 10.92.20.85' "$state/rsyncd.conf" >/dev/null
grep -Fx 'hosts deny = *' "$state/rsyncd.conf" >/dev/null
grep -Fx 'exclude = /*/' "$state/rsyncd.conf" >/dev/null
! grep -F '10.92.20.79' "$state/rsyncd.conf"
cp "$state/manifest.sha256" "$FAKE_MANIFEST"
recorded_pid=$(sed -n '1p' "$state/rsyncd.pid")
chmod u+w "$state/rsyncd.pid"
printf '%s\n' "$$" >"$state/rsyncd.pid"
if "$script" source-stop >"$tmp/wrong-pid.out" 2>"$tmp/wrong-pid.err"; then
  echo 'source-stop must reject an unrelated recorded PID' >&2
  exit 1
fi
grep -F 'not this model seed daemon' "$tmp/wrong-pid.err" >/dev/null
printf '%s\n' "$recorded_pid" >"$state/rsyncd.pid"
chmod 0444 "$state/rsyncd.pid"
"$script" source-stop >"$tmp/stop.out"
test ! -e "$state"

export MODEL_SEED_MODEL_DIR=$target_model
: >"$FAKE_RSYNC_LOG"
printf 'stale\n' >"$target_model/model-00001-of-00081.safetensors_range"
mkdir -p "$partial"
printf 'partial\n' >"$partial/stale"
"$script" target-pull >"$tmp/pull.out"
grep -F -- '--checksum' "$FAKE_RSYNC_LOG" >/dev/null
grep -F -- '--partial' "$FAKE_RSYNC_LOG" >/dev/null
grep -F -- '--append-verify' "$FAKE_RSYNC_LOG" >/dev/null
grep -F -- '--partial-dir='"$partial" "$FAKE_RSYNC_LOG" >/dev/null
grep -F -- '--protect-args' "$FAKE_RSYNC_LOG" >/dev/null
grep -F -- '--timeout=600' "$FAKE_RSYNC_LOG" >/dev/null
grep -F -- '--contimeout=30' "$FAKE_RSYNC_LOG" >/dev/null
! grep -F -- '--delete' "$FAKE_RSYNC_LOG"
test ! -e "$target_model/model-00001-of-00081.safetensors_range"
test ! -e "$partial"
"$script" verify >/dev/null

printf 'model seed LAN checks pass\n'
