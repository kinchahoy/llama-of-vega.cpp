#!/usr/bin/env bash
# usage: sprof.sh OUT MODE(on|off) - rocprofv3 on qwen27b server: 1 user, 24K prompt, 256 tokens MTP2
set -u
OUT=$1; MODE=$2; mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh >/dev/null 2>&1
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
[ "$MODE" = off ] && export LLAMA_KV_ROWS_DISABLE=1
K=/home/raistlin/infer/vrversion-llama.cpp/.claude/worktrees/agent-a4e1d9d7be78eb377/build-kvrows
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
setsid rocprofv3 --kernel-trace --stats -d "$OUT/$MODE" -o run -- $K/bin/llama-server --offline --host 127.0.0.1 --port 8199 -m $M -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off \
  -b 2048 -ub 2048 -c 262144 -np 4 --kv-unified --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling --no-cache-idle-slots > "$OUT/$MODE-server.log" 2>&1 &
for i in $(seq 300); do curl -sf 127.0.0.1:8199/health >/dev/null && break; sleep 1; done
python3 -I - <<'PY'
import glob, json, urllib.request
def post(path, obj):
    r = urllib.request.Request("http://127.0.0.1:8199" + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
R = '/home/raistlin/infer/vrversion-llama.cpp/'
text = "".join(open(f, errors='ignore').read() for f in sorted(glob.glob(R+'src/*.cpp')))
tok = post("/tokenize", {"content": text})["tokens"][:24576]
q = post("/tokenize", {"content": "\n\nQuestion: summarize the code above.\nAnswer:"})["tokens"]
r = post("/completion", {"prompt": tok + q, "id_slot": 0, "n_predict": 256, "temperature": 0, "top_k": 1, "ignore_eos": True, "cache_prompt": False})
t = r["timings"]; print("pp %.1f tg %.1f" % (t["prompt_per_second"], t["predicted_per_second"]))
PY
P=$(pgrep -f "^$K/bin/llama-server"); kill -INT $P; for i in $(seq 60); do pgrep -f "^$K/bin/llama-server" >/dev/null || break; sleep 1; done
sleep 15; echo "SPROF_DONE $MODE"
