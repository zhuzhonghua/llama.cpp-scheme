#!/usr/bin/env bash
#
# download-model.sh — fetch the development models used by test/*.scm into models/.
#
#   ./download-model.sh          Qwen3-0.6B-Q4_K_M.gguf (~380 MB), used by most tests
#                                and by test/npc-demo.scm
#   ./download-model.sh tiny     stories260K.gguf (1.1 MB), smoke test only
#
# Downloads from huggingface.co and falls back to hf-mirror.com when the direct
# endpoint is unreachable (common from mainland China). Proxy settings are taken
# from the usual environment variables:
#
#   https_proxy=http://127.0.0.1:7890 ./download-model.sh
#
# Existing files of a plausible size are kept; pass --force to re-download.

set -euo pipefail

cd "$(dirname "$0")"

FORCE=0
WHICH=default
for arg in "$@"; do
  case "$arg" in
    tiny|default) WHICH="$arg" ;;
    --force|-f)   FORCE=1 ;;
    -h|--help)    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ "$WHICH" = tiny ]; then
  NAME="stories260K.gguf"
  REMOTE="ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
  MIN_BYTES=1000000
else
  NAME="Qwen3-0.6B-Q4_K_M.gguf"
  REMOTE="unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
  MIN_BYTES=300000000
fi

DEST="models/$NAME"

mkdir -p models

if [ "$FORCE" -eq 0 ] && [ -f "$DEST" ]; then
  SIZE=$(wc -c < "$DEST" | tr -d ' ')
  if [ "$SIZE" -ge "$MIN_BYTES" ]; then
    echo "$DEST already present ($SIZE bytes), skipping (use --force to re-download)"
    exit 0
  fi
  echo "$DEST looks truncated ($SIZE bytes), re-downloading"
fi

fetch() {
  local url="$1"
  echo "fetching $url"
  curl -L --fail --progress-bar --retry 3 --retry-delay 2 -o "$DEST.part" "$url"
}

if ! fetch "https://huggingface.co/$REMOTE"; then
  echo "huggingface.co unreachable, trying hf-mirror.com"
  fetch "https://hf-mirror.com/$REMOTE"
fi

SIZE=$(wc -c < "$DEST.part" | tr -d ' ')
if [ "$SIZE" -lt "$MIN_BYTES" ]; then
  rm -f "$DEST.part"
  echo "download failed: got $SIZE bytes, expected at least $MIN_BYTES" >&2
  exit 1
fi

mv "$DEST.part" "$DEST"
echo "ok: $DEST ($SIZE bytes)"
