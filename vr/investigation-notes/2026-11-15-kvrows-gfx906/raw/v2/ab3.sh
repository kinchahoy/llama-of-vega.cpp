#!/usr/bin/env bash
# usage: ab3.sh OUT_DIR PER_USER LABEL:BUILD:MODE ... - qwen27b 4-slot server, MODE unified|split
set -u
OUT=$1; PER=$2; shift 2; mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh >/dev/null 2>&1
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
PORT=8199
for spec in "$@"; do
  IFS=: read -r LABEL B MODE <<< "$spec"
  EXTRA="--kv-unified"; [ "$MODE" = split ] && EXTRA="--no-kv-unified"
  "$B/bin/llama-server" --offline --host 127.0.0.1 --port $PORT -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off \
    -b 2048 -ub 2048 -c 262144 -np 4 $EXTRA --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling --no-cache-idle-slots \
    > "$OUT/$LABEL-server.log" 2>&1 &
  SP=$!
  for i in $(seq 300); do curl -sf 127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  grep -E "n_ctx_slot|K/V rows" "$OUT/$LABEL-server.log" | head -3 | sed "s/^/$LABEL log: /"
  if curl -sf 127.0.0.1:$PORT/health >/dev/null; then
    timeout 3000 python3 -I "$(dirname "$0")/srvtest3.py" $PORT $PER 2>&1 | sed "s/^/$LABEL /"
  else
    echo "$LABEL server failed to start"; tail -20 "$OUT/$LABEL-server.log"
  fi
  kill $SP; for i in $(seq 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null; sleep 3
done
echo AB3_DONE
