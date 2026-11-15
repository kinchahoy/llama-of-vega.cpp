# usage: python3 -I srvtest.py PORT FILL_TOKENS - kv_rows A/B: fresh chat, 64K fill, chat beside fill, 3 and 4 concurrent users
import glob, json, sys, time, urllib.request, threading
port, fill = sys.argv[1], int(sys.argv[2])
def post(path, obj, to=7200):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=to))
src = sorted(glob.glob('/home/raistlin/infer/vrversion-llama.cpp/src/*.cpp')) + sorted(glob.glob('/home/raistlin/infer/vrversion-llama.cpp/tools/server/*.cpp'))
text = ""
for f in src:
    text += "\n// FILE: " + f.split('/')[-1] + "\n" + open(f, errors='ignore').read()
    if len(text) > fill * 3.2: break
alltok = post("/tokenize", {"content": text})["tokens"]
assert len(alltok) >= fill, len(alltok)
filltok = alltok[:fill]
def q(s): return post("/tokenize", {"content": s})["tokens"]
# short chats: different 1.2K-token windows from the end of the text, so they share no prefix with the fill
def chat(i): return alltok[-(i+1)*1300:-(i+1)*1300+1200] + q(f"\n\nQuestion {i}: explain what the code above does, step by step.\nAnswer:")
def gen(slot, prompt, n=256, out=None):
    t0 = time.time()
    r = post("/completion", {"prompt": prompt, "id_slot": slot, "n_predict": n, "temperature": 0, "top_k": 1, "ignore_eos": True, "cache_prompt": True})
    t = r["timings"]; res = (slot, t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"], time.time() - t0)
    if out is not None: out.append(res)
    return res
def show(tag, rs):
    for s, pn, pps, gn, gps, w in sorted(rs):
        print(f"{tag} slot{s}: prompt {pn} tok {pps:.1f} t/s, gen {gn} tok {gps:.1f} t/s, wall {w:.1f}s", flush=True)
def conc(tag, jobs):
    out, th = [], []
    t0 = time.time()
    for s, p in jobs:
        th.append(threading.Thread(target=gen, args=(s, p, 256, out))); th[-1].start()
    for x in th: x.join()
    w = time.time() - t0
    show(tag, out)
    print(f"{tag} aggregate gen: {sum(r[3] for r in out)/w:.1f} t/s over {w:.1f}s (prefill included)", flush=True)
gen(1, chat(9), 16)  # warmup
show("A fresh-chat", [gen(1, chat(1))]); show("A fresh-chat-rep", [gen(1, chat(1)[:-5] + q(" Briefly.\nAnswer:"))])
r = gen(0, filltok + q("\n\nQuestion: name three files above.\nAnswer:"), 32); show("B fill", [r])
show("C chat-beside-fill", [gen(1, chat(2))]); show("C chat-beside-fill-rep", [gen(1, chat(3))])
conc("D 3users(fill idle)", [(1, chat(4)), (2, chat(5)), (3, chat(6))])
conc("E 4users(slot0 cont)", [(0, filltok + q("\n\nNow list five functions from the files above.\nAnswer:")), (1, chat(7)), (2, chat(8)), (3, chat(10))])
