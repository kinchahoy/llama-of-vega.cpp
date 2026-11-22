#!/usr/bin/env python3
# Multi-user load test for a running llama-server (router or single model). Stdlib only.
# Builds long prompts from repo text (docs/*.md, src/*.cpp, ggml/src/ggml-cuda/*.cu) plus a task, fires N requests at once,
# and prints per-user prefill/generation t/s, MTP draft acceptance, and aggregate generation t/s.
# usage: python3 -I bench_serve.py --url http://127.0.0.1:8080 --model qwen27b --users 4 --ctx-tokens 50000 --n-predict 384 --temp 0
import argparse, glob, json, os, random, threading, time, urllib.request

TASKS = [
    "Summarize the design of the code and documents above in about ten bullet points, then write a C++ function that parses a comma-separated list of device names (like \"ROCm0,ROCm1\") into a std::vector<std::string>, trimming whitespace and rejecting empty entries.",
    "Explain how the components above fit together and what the main data flow is. Then write a Python function that reads a GGUF-like key/value header from a binary file and returns a dict, with error handling.",
    "Review the code above: list three possible bugs or performance problems with file names, and then write a small CUDA kernel that computes a row-wise softmax over a float matrix, with comments.",
    "Describe the main abstractions above for a new contributor. Then write a C function that implements a fixed-size ring buffer of integers with push, pop and size operations.",
    "Summarize the key ideas above in plain language. Then write a Rust function that merges two sorted vectors of u32 into one sorted vector without duplicates.",
    "Give a short architecture overview of the code above, then write a Bash script that waits until two GPUs report a temperature below 60 C before running a command.",
]


def post(args, path, obj, timeout=300):
    if args.model:
        obj = dict(obj, model=args.model)
    req = urllib.request.Request(args.url + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def build_prompt(args, seed):
    files = sorted(glob.glob(args.repo + "/docs/**/*.md", recursive=True)) + sorted(glob.glob(args.repo + "/src/*.cpp")) + sorted(glob.glob(args.repo + "/ggml/src/ggml-cuda/*.cu"))
    random.Random(seed).shuffle(files)
    want = int(args.ctx_tokens * 3.6)
    parts, n = [], 0
    for f in files:
        with open(f, errors="replace") as fh:
            parts.append("=== FILE: %s ===\n%s\n" % (os.path.relpath(f, args.repo), fh.read()))
        n += len(parts[-1])
        if n > want * 1.5:
            break
    body, task = "".join(parts), TASKS[seed % len(TASKS)]

    def fmt(chars):
        msg = body[:chars] + "\n\n=== END OF FILES ===\n\nTask: " + task
        return post(args, "/apply-template", {"messages": [{"role": "user", "content": msg}]})["prompt"]

    chars = min(want, len(body))
    for _ in range(10):  # rescale char count until token count is within 1% of target
        p = fmt(chars)
        k = len(post(args, "/tokenize", {"content": p})["tokens"])
        if abs(k - args.ctx_tokens) <= args.ctx_tokens * 0.01:
            break
        chars = min(len(body), int(chars * args.ctx_tokens / k))
    return p, k


def run_one(args, prompt, seed, t0, res, i):
    body = {"prompt": prompt, "n_predict": args.n_predict, "stream": True, "cache_prompt": False, "ignore_eos": True,
            "seed": seed, "return_tokens": True, "temperature": args.temp}
    if args.temp == 0:
        body["top_k"] = 1
    if args.model:
        body["model"] = args.model
    req = urllib.request.Request(args.url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    arr, timings, err = [], None, None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if "error" in d:
                    err = d["error"]
                k = len(d.get("tokens") or []) or (1 if d.get("content") else 0)
                if k:
                    arr.append((time.time() - t0, k))
                if d.get("timings"):
                    timings = d["timings"]
    except Exception as e:
        err = repr(e)
    if timings is None and err is None:
        err = "stream ended without timings (server crashed?)"
    res[i] = {"arrivals": arr, "timings": timings, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="", help="router model name, e.g. qwen27b (empty for a single-model server)")
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--ctx-tokens", type=int, default=1500, help="prompt tokens per user")
    ap.add_argument("--n-predict", type=int, default=384)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="selects prompt file order and task per user")
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--json", default="", help="append raw results to this JSONL file")
    args = ap.parse_args()
    args.url = args.url.rstrip("/")

    prompts = [build_prompt(args, args.seed + u) for u in range(args.users)]
    res = [None] * args.users
    t0 = time.time()
    ths = [threading.Thread(target=run_one, args=(args, p, 1000 + args.seed + i, t0, res, i)) for i, (p, _) in enumerate(prompts)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    print("users %d  ctx/user ~%d  n_predict %d  temp %.2f" % (args.users, args.ctx_tokens, args.n_predict, args.temp))
    bad = [(i, r["error"]) for i, r in enumerate(res) if r["timings"] is None]
    if bad:
        for i, e in bad:
            print("  user %d  FAILED after %d streamed tokens: %s" % (i, sum(k for _, k in res[i]["arrivals"]), e))
        raise SystemExit(1)
    tg, acc = [], []
    for i, r in enumerate(res):
        t = r["timings"]
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        a = da / dn if dn else 0.0
        tg.append(t["predicted_per_second"]); acc.append(a)
        print("  user %d  prompt %6d tok  prefill %7.1f t/s (%6.1f s)  gen %3d tok %6.1f t/s  accept %.3f  first token %.1f s" % (
            i, t["prompt_n"], t["prompt_per_second"], t["prompt_ms"] / 1000, t["predicted_n"], t["predicted_per_second"], a, r["arrivals"][0][0]))
    # aggregate over the window where every user is generating (no prefill overlap), and over the whole generation span
    gs = [r["arrivals"][0][0] for r in res]
    ge = [r["arrivals"][-1][0] for r in res]
    w0, w1 = max(gs), min(ge)
    win = sum(k for r in res for (t, k) in r["arrivals"] if w0 < t <= w1)
    total = sum(r["timings"]["predicted_n"] for r in res)
    print("per-user gen t/s  min %.1f  mean %.1f   accept mean %.3f" % (min(tg), sum(tg) / len(tg), sum(acc) / len(acc)))
    if args.users > 1:
        print("aggregate gen t/s  all-generating window %s (%.1f s)   whole span %.1f" % (
            "%.1f" % (win / (w1 - w0)) if w1 > w0 else "n/a", max(0.0, w1 - w0), total / (max(ge) - min(gs))))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps({"args": vars(args), "prompt_tokens": [k for _, k in prompts], "timings": [r["timings"] for r in res]}) + "\n")


if __name__ == "__main__":
    main()
