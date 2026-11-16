# usage: python3 -I srvtest3.py PORT TOKENS_PER_USER - S1: 1 user decodes at N; S4: 4 users prefilled one at a time, then all 4 decode at once at N each
import glob, json, sys, time, urllib.request, threading
port, per = sys.argv[1], int(sys.argv[2])
def post(path, obj, to=7200):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=to))
R = '/home/raistlin/infer/vrversion-llama.cpp/'
files = sorted(glob.glob(R+'src/*.cpp')) + sorted(glob.glob(R+'tools/server/*.cpp')) + sorted(glob.glob(R+'common/*.cpp')) + sorted(glob.glob(R+'ggml/src/*.c*')) + sorted(glob.glob(R+'ggml/src/ggml-cuda/*.cu*'))
text = ""
for f in files:
    text += "\n// FILE: " + f.split('/')[-1] + "\n" + open(f, errors='ignore').read()
    if len(text) > per * 5 * 3.4: break
tok = post("/tokenize", {"content": text})["tokens"]
assert len(tok) >= per * 5, (len(tok), per * 5)
q = post("/tokenize", {"content": "\n\nQuestion: summarize what the code above does, file by file, in detail.\nAnswer:"})["tokens"]
more = post("/tokenize", {"content": " In detail:"})["tokens"]
def req(slot, prompt, n):
    return post("/completion", {"prompt": prompt, "id_slot": slot, "n_predict": n, "temperature": 0, "top_k": 1, "ignore_eos": True, "cache_prompt": True, "return_tokens": True})
r = req(0, tok[4*per:5*per] + q, 384); t = r["timings"]
print(f"S1 one-user: prompt {t['prompt_n']} tok {t['prompt_per_second']:.1f} t/s, gen {t['predicted_n']} tok {t['predicted_per_second']:.1f} t/s", flush=True)
print("S1 text:", r["content"][:160].replace("\n", " "), flush=True)
prompts = []
for i in range(4):
    p = tok[i*per:(i+1)*per] + q
    r = req(i, p, 1); t = r["timings"]
    prompts.append(p + r["tokens"] + more)
    print(f"S4 prefill slot{i}: prompt {t['prompt_n']} tok {t['prompt_per_second']:.1f} t/s", flush=True)
out = []
def gen(i):
    r = req(i, prompts[i], 384); t = r["timings"]; out.append((i, t["prompt_n"], t["predicted_n"], t["predicted_per_second"]))
th = [threading.Thread(target=gen, args=(i,)) for i in range(4)]
t0 = time.time()
for x in th: x.start()
for x in th: x.join()
w = time.time() - t0
for i, pn, gn, gps in sorted(out):
    print(f"S4 decode slot{i}: reprocessed {pn} tok, gen {gn} tok {gps:.1f} t/s", flush=True)
print(f"S4 decode aggregate: {sum(o[2] for o in out)/w:.1f} t/s over {w:.1f}s", flush=True)
