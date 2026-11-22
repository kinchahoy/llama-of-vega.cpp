#!/usr/bin/env bash
# usage: bench_multiuser.sh OUT_DIR BUILD_A BUILD_B
# Multi-user unified-KV check: 4 sequences x 16K prompts in one pool, then 4-way decode, on 27B Q8_0 and Flash Next.
# Order A,B (REPS=2: A,B,B,A). Each run waits until both GPU junctions are below COOL_C (default 60).
# Env overrides: REPS, COOL_C, THEROCK_ENV, HF_HUB_CACHE, M27, MFN, ENV_A, ENV_B (e.g. ENV_B=LLAMA_KV_ROWS_DISABLE=1).
set -u
OUT=$1; A=$2; B=$3; mkdir -p "$OUT"
REPS=${REPS:-1}; COOL_C=${COOL_C:-60}
THEROCK_ENV=${THEROCK_ENV:-}
if [ -n "$THEROCK_ENV" ] && [ -f "$THEROCK_ENV" ]; then
  source "$THEROCK_ENV" >/dev/null 2>&1
elif [ -f "$HOME/.config/therock.sh" ]; then
  source "$HOME/.config/therock.sh" >/dev/null 2>&1
elif [ -f "$(cd "$(dirname "$0")/.." && pwd)/server/env.local.sh" ]; then
  source "$(cd "$(dirname "$0")/.." && pwd)/server/env.local.sh" >/dev/null 2>&1
fi
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE LLAMA_KV_ROWS_DISABLE
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
M27=${M27:-$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf}
MFN=${MFN:-$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}
hot() { rocm-smi --showtemp 2>/dev/null | awk '/junction/ {t=$NF+0; if (t>m) m=t} END {print m+0}'; }

order="A B"; [ "$REPS" = 2 ] && order="A B B A"
n=0
for model in 27b fn; do
  if [ $model = 27b ]; then M=$M27; ARGS="-b 2048 -ub 2048 -t 20"; else M=$MFN; ARGS="-b 1024 -ub 512 -t 8"; fi
  for side in $order; do
    n=$((n+1))
    if [ $side = A ]; then BUILD=$A; ENV=${ENV_A:-}; else BUILD=$B; ENV=${ENV_B:-}; fi
    while [ "$(hot)" -ge "$COOL_C" ]; do sleep 10; done
    t0=$(hot)
    # own process group: a GPU program exit can take the caller's group down
    env $ENV setsid --wait timeout --kill-after=5s 900s "$BUILD/bin/llama-batched-bench" -m "$M" -ngl 999 -sm tensor -dev ROCm0,ROCm1 -fa on -kvu \
      -c 66560 $ARGS -npp 16384 -ntg 128 -npl 4 > "$OUT/run$n-$model-$side.log" 2>&1
    rc=$?
    row=$(grep -E '^\| +16384 ' "$OUT/run$n-$model-$side.log" | tail -1)
    printf '%s %s %s rc=%s start=%sC end=%sC %s\n' "$n" "$model" "$side" "$rc" "$t0" "$(hot)" "$row" | tee -a "$OUT/summary.txt"
  done
done

python3 -I - "$OUT/summary.txt" <<'PY'
import sys, collections
res = collections.defaultdict(list)
for line in open(sys.argv[1]):
    f = line.split()
    cols = [c.strip() for c in line.split("|")[1:-1]]
    if len(cols) < 8: continue
    res[(f[1], f[2])].append((float(cols[5]), float(cols[7])))  # S_PP, S_TG
print("model side  PP t/s (4x16K)       TG t/s (4-way aggregate)")
for m in ("27b", "fn"):
    for s in ("A", "B"):
        v = res[(m, s)]
        if v: print("%-5s %-5s %-22s %s" % (m, s, " / ".join("%.1f" % x[0] for x in v), " / ".join("%.2f" % x[1] for x in v)))
PY
