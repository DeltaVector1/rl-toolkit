#!/usr/bin/env bash
set -euo pipefail

# Run on the trainer head after the download tunnels are configured.
destination=${DESTINATION:-/data/muse-glimmer/bootstrap}
mkdir -p "$destination"
exec 9>"$destination/.bootstrap.lock"
if ! flock -n 9; then
  echo "another Muse checkpoint bootstrap is already active" >&2
  exit 2
fi
ports=(
  $(seq 18001 2 18047)
  $(seq 18201 18248)
  $(seq 18301 18364)
  $(seq 18401 18464)
  $(seq 18501 18564)
  $(seq 18601 18664)
  $(seq 18701 18764)
  $(seq 18801 18864)
)
port_csv=$(IFS=,; echo "${ports[*]}")
downloader=/data/glm52-rl/deployment/scripts/setup/muse_parallel_download.py

download_file() {
  local filename=$1
  local size=$2
  local output=$3
  local sha256=${4:-}
  local args=("$filename" "$size" "$output" --ports "$port_csv" --chunk-mib 1 --timeout 120)
  [ -n "$sha256" ] && args+=(--sha256 "$sha256")
  python3 "$downloader" "${args[@]}"
}

download_file transformers-5.15.tar 49356800 "$destination/transformers-5.15.tar" cd85d222c3907d880e5a51a467d00d99e0864a6c35e3ee9999a2bcb2b5738651
download_file .gitattributes 1570 "$destination/.gitattributes"
download_file LICENSE 11358 "$destination/LICENSE"
download_file README.md 16892 "$destination/README.md"
download_file USAGE_POLICY.md 5230 "$destination/USAGE_POLICY.md"
download_file chat_template.jinja 9992 "$destination/chat_template.jinja"
download_file config.json 5109 "$destination/config.json"
download_file generation_config.json 202 "$destination/generation_config.json"
download_file processor_config.json 1084 "$destination/processor_config.json"
download_file tokenizer.json 28129897 "$destination/tokenizer.json"
download_file tokenizer_config.json 79936 "$destination/tokenizer_config.json"
download_file model.safetensors.index.json 132674 "$destination/model.safetensors.index.json"
download_file model-00001-of-00002.safetensors 49950112952 "$destination/model-00001-of-00002.safetensors" 8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694
download_file model-00002-of-00002.safetensors 9603322320 "$destination/model-00002-of-00002.safetensors" b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381
