#!/usr/bin/env bash
# usage: prof.sh OUT - rocprofv3 kernel stats, llama-batched-bench 1 user at 32K, unified KV, kv_rows on vs off (same binary)
set -u
OUT=$1; mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh >/dev/null 2>&1
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
K=/home/raistlin/infer/vrversion-llama.cpp/.claude/worktrees/agent-a4e1d9d7be78eb377/build-kvrows
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
ARGS="-m $M -ngl 999 -sm tensor -dev ROCm0,ROCm1 -fa on -kvu -c 33280 -b 2048 -ub 2048 -t 20 -npp 32768 -ntg 128 -npl 1"
for mode in ${MODES:-off on}; do
  if [ $mode = off ]; then export LLAMA_KV_ROWS_DISABLE=1; else unset LLAMA_KV_ROWS_DISABLE; fi
  setsid --wait timeout 900 rocprofv3 --kernel-trace --stats -d "$OUT/$mode" -o run -- $K/bin/llama-batched-bench $ARGS > "$OUT/$mode.log" 2>&1
  echo "$mode exit=$?"
  grep -E "^\|" "$OUT/$mode.log" | tail -2
  grep -E "K/V rows" "$OUT/$mode.log" | tail -1
done
echo PROF_DONE
