#!/usr/bin/env bash
# usage: abba.sh OUT - batched-bench 4 seqs x 16K unified, kv_rows off/on/on/off, cooldown below 60C before each run
set -u
OUT=$1; mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh >/dev/null 2>&1
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
K=/home/raistlin/infer/vrversion-llama.cpp/.claude/worktrees/agent-a4e1d9d7be78eb377/build-kvrows
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
hot() { rocm-smi --showtemp 2>/dev/null | awk '/junction/ {t=$NF+0; if (t>m) m=t} END {print m+0}'; }
n=0
for mode in off on on off; do
  n=$((n+1))
  while [ "$(hot)" -ge 60 ]; do sleep 10; done
  echo "run$n $mode start junction=$(hot)C"
  if [ $mode = off ]; then export LLAMA_KV_ROWS_DISABLE=1; else unset LLAMA_KV_ROWS_DISABLE; fi
  setsid --wait timeout 1200 $K/bin/llama-batched-bench -m $M -ngl 999 -sm tensor -dev ROCm0,ROCm1 -fa on -kvu -c 66560 -b 2048 -ub 2048 -t 20 -npp 16384 -ntg 128 -npl 4 > "$OUT/run$n-$mode.log" 2>&1
  echo "run$n $mode exit=$? end junction=$(hot)C sclk=$(rocm-smi --showclocks 2>/dev/null | awk '/sclk/ {printf "%s ", $NF}')"
  grep -E "^\| +16384" "$OUT/run$n-$mode.log" | sed "s/^/run$n $mode /"
done
echo ABBA_DONE
