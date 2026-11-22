#!/usr/bin/env bash
# usage: vr/server/serve.sh [extra llama-server args]
# Router llama-server with the presets in models.local.ini (if present) or models.ini.
# Env: BUILD (default: <repo>/build-current), MODELS_PRESET, THEROCK_ENV, HOST (0.0.0.0), PORT (8080).
# build-current and build-previous are symlinks to real build dirs (never rename a build dir: its RUNPATH is absolute).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BUILD=${BUILD:-$REPO/build-current}
THEROCK_ENV=${THEROCK_ENV:-}
if [ -n "$THEROCK_ENV" ] && [ -f "$THEROCK_ENV" ]; then
  source "$THEROCK_ENV"
elif [ -f "$HERE/env.local.sh" ]; then
  source "$HERE/env.local.sh"
fi
PRESET=${MODELS_PRESET:-$([ -f "$HERE/models.local.ini" ] && echo "$HERE/models.local.ini" || echo "$HERE/models.ini")}
unset GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
exec "$BUILD/bin/llama-server" \
  --models-preset "$PRESET" --models-max 1 \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" "$@"
