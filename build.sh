#!/usr/bin/env bash
#
# build.sh — configure and build the scmllama host executable.
#
#   ./build.sh                     Release build into build/
#   ./build.sh Debug               Debug build into build-debug/
#   ./build.sh Release --clean     remove the build tree first
#
# Extra arguments after the build type are passed to `cmake -S . -B <dir>`.
# Submodules (llama.cpp, dummyscheme) are initialised on first run.
#
# Then:
#   ./download-model.sh            # Qwen3-0.6B Q4_K_M, needed by most tests
#   ./build/scmllama test/hello.scm
#   ./build/scmllama test/npc-demo.scm

set -euo pipefail

cd "$(dirname "$0")"

TYPE="Release"
case "${1:-}" in
  Debug|Release|RelWithDebInfo|MinSizeRel) TYPE="$1"; shift ;;
esac

CLEAN=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    *) ARGS+=("$arg") ;;
  esac
done

if [ ! -f llama.cpp/CMakeLists.txt ] || [ ! -f dummyscheme/CMakeLists.txt ]; then
  echo "initialising submodules"
  git submodule update --init --recursive
fi

DIR="build"
[ "$TYPE" = "Release" ] || DIR="build-$(echo "$TYPE" | tr '[:upper:]' '[:lower:]')"

if [ "$CLEAN" -eq 1 ] && [ -d "$DIR" ]; then
  echo "removing $DIR"
  rm -rf "$DIR"
fi

JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

cmake -S . -B "$DIR" -DCMAKE_BUILD_TYPE="$TYPE" "${ARGS[@]+"${ARGS[@]}"}"
cmake --build "$DIR" -j "$JOBS"

echo
echo "built $DIR/scmllama"
echo "next: ./download-model.sh && ./$DIR/scmllama test/hello.scm"
