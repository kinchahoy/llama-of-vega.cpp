#!/usr/bin/env bash
# usage: final_bench.sh OUT - llama-bench 27B and Flash Next, base vs final build, alternating, cooldown below 60C
set -u
OUT=$1; mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh >/dev/null 2>&1
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
hot() { rocm-smi --showtemp 2>/dev/null | awk '/junction/ {t=$NF+0; if (t>m) m=t} END {print m+0}'; }
B=/home/raistlin/infer/vrversion-llama.cpp/build-catchup-20261113-200420
K=/home/raistlin/infer/vrversion-llama.cpp/.claude/worktrees/agent-a4e1d9d7be78eb377/build-kvrows
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
MFN=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
for build in base final final base; do
  D=$B; [ $build = final ] && D=$K
  while [ "$(hot)" -ge 60 ]; do sleep 10; done
  setsid --wait timeout 300 $D/bin/llama-bench -m $M27 -ngl 999 -sm tensor -dev ROCm0/ROCm1 -ts 1/1 -fa on -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -r 2 -o json > $OUT/$build-27b-$RANDOM.json 2>/dev/null
  while [ "$(hot)" -ge 60 ]; do sleep 10; done
  setsid --wait timeout 400 $D/bin/llama-bench -m $MFN -ngl 999 -sm tensor -dev ROCm0/ROCm1 -ts 1/1 -fa on -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -t 8 -p 2048 -n 128 -r 2 -o json > $OUT/$build-fn-$RANDOM.json 2>/dev/null
  echo "$build pass done"
done
python3 -I - "$OUT" <<'PY'
import json, glob, sys, collections
res = collections.defaultdict(list)
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    b, m = f.split("/")[-1].split("-")[:2]
    try: rows = json.load(open(f))
    except Exception as e: print("bad", f, e); continue
    for r in rows:
        k = "PP2048" if r["n_prompt"] else "TG128"
        res[(m, k, b)].append(r["avg_ts"])
for m in ("27b", "fn"):
    for k in ("PP2048", "TG128"):
        print(m, k, "base", ["%.2f" % v for v in res[(m, k, "base")]], "final", ["%.2f" % v for v in res[(m, k, "final")]])
PY
echo FINAL_DONE
