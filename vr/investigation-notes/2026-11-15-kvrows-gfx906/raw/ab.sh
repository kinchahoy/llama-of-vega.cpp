#!/usr/bin/env bash
# usage: ab.sh LABEL BUILD_DIR OUT_DIR - llama-bench stage, then 4-slot qwen27b server test
set -u
LABEL=$1; BUILD=$2; OUT=$3; mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh >/dev/null 2>&1
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
bash /home/raistlin/infer/vrversion-llama.cpp/vr/scripts/bench_stage.sh "$LABEL" "$BUILD" "$OUT" 2>&1 | tee "$OUT/$LABEL-bench.txt"
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
PORT=8199
"$BUILD/bin/llama-server" --offline --host 127.0.0.1 --port $PORT -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off \
  -b 2048 -ub 2048 -c 262144 -np 4 --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling \
  > "$OUT/$LABEL-server.log" 2>&1 &
SP=$!
for i in $(seq 300); do curl -sf 127.0.0.1:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
grep -E "K/V rows|resolve_fused_ops" "$OUT/$LABEL-server.log" | sed "s/^/$LABEL log: /"
if curl -sf 127.0.0.1:$PORT/health >/dev/null; then
  timeout 2400 python3 -I "$(dirname "$0")/srvtest.py" $PORT 65536 2>&1 | sed "s/^/$LABEL /" | tee "$OUT/$LABEL-srv.txt"
else
  echo "$LABEL server failed to start"; tail -20 "$OUT/$LABEL-server.log"
fi
kill $SP; wait $SP 2>/dev/null
rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | sed "s/^/$LABEL after: /"
