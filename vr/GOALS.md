# Goals and Intent

Owner-editable statement of *why* this fork exists and how agents should make tradeoffs. Read this before [PROCESS.md](PROCESS.md), [HARDWARE-CARD.md](HARDWARE-CARD.md), or [PERFORMANCE.md](PERFORMANCE.md). Do not treat individual rules in other docs as rigid laws when they conflict with the intent below; instead, flag the conflict or propose a deviation explicitly.

## 1. Mission

Deliver the fastest possible local inference on our two older Radeon GPUs (**MI60 = ROCm0, MI50 = ROCm1**, gfx906, no matrix units) for the strongest open agentic LLMs, while keeping accuracy faithful to the original model and keeping GPU hours and wall-clock time low.

## 2. Target Workload

- **Context length**: Long-context agentic sessions (**50-100K+ tokens**). Both **prompt processing (PP / TTFT)** and **token generation (TG)** matter.
- **Concurrency mix**:
  - **~60%** single-slot long-context latency (TTFT + TG).
  - **~40%** subagent bursts (**2-4 parallel slots**). Multi-slot decode and verify must scale cleanly and must not regress under speculative decoding.
  - **~10%** emerging interest in running a **big/high-quality + small/fast** model pair (e.g. one model per GPU in parallel).
- **Models**:
  - Current equal priorities: **Qwen 3.8 27B Q8_0**, **Qwen 3.8 Flash Next**, **Inkling 2 Mini**.
  - Near-term horizon: **Qwen 4 (27B and MoE)** is expected soon and will likely supplant at least Qwen 3.8 27B. Prefer investments that carry forward to Qwen 4+.

## 3. Priority Hierarchy

1. **Reusable kernel and architecture wins** (long-context GQA attention, GDN/SSM recurrence, MoE routing and grouped GEMM, multi-slot/verify mat-vec, TP/AllReduce, repacked quants).
2. **Qwen 4-relevant wins** (architectures and formats shared with upcoming Qwen 4 dense/MoE models).
3. **Model-specific wins** on current top-3 models. *Code is cheap; generation time is expensive* -- do not skip a real, durable speedup just because it is model-specific.
4. **Architectural placement and sizing** (e.g. GPU-level placement of KV cache vs. draft/MTP head given MI60 vs. MI50 asymmetry, or arch-driven ubatch sizing). This is currently **under-invested** and is *not* banned by the "parameter tuning last" rule.
5. **Parameter tuning (last stage)**: aimless knob-twiddling (e.g. MTP2 vs MTP3, thread count sweeps, `-ts` micro-tweaks) waits until higher-leverage code/architecture work is exhausted and needs owner OK. Diagnostic env checks (like `HIP_FORCE_DEV_KERNARG=1`) and curve measurements needed to design kernels (like 1-16 column verify timing) are **measurements**, not parameter tuning.

## 4. Custom Code vs. Performance (Secondary Goal)

- **Speed wins first; clean up second.** Capture real performance gains first, then refactor to simplify upstream catch-up merges and eventual upstreaming to `llama.cpp` main.
- **Pruning rule of thumb**: Worth trading a tiny gain (e.g. ~0.3% PP) to remove ~200 lines of custom code, or pruning dead/slow fallbacks (like unused kernel paths), but do not sacrifice meaningful speedups solely for LOC. Ask the owner on edge cases.
- **Borrow freely**: Check upstream `llama.cpp`, unsloth's fork, `mixa3607/ML-gfx906`, and Strata for ideas or files before writing custom kernels from scratch.

## 5. Accuracy and Quality

- **Default**: Lossless or FP-order-only changes (verified with unit tests + KLD/PPL checks). Keep **F16 KV cache** (no KV quantization by default) and full expert routing.
- **Flagging rule**: Never introduce quality-affecting or lossy changes silently. Agents *may* propose lossy tradeoffs (e.g. aggressive quants, expert/KV approximations) **only** if explicitly flagged with expected quality impact and offering a **major (>25%) speedup or a critical memory-fit unlock**.

## 6. Hardware, Operations, and Scientific Method

- **Software-only for now**: Do not plan around physical hardware changes (such as moving the MI50 off chipset PCIe lanes) unless the owner reopens it.
- **Server uptime & GPU hours**: Batch GPU validation into concise windows. Keep `llama-model-server` downtime short and batch GPU validation into brief windows (<=15-30 min). Exhaust zero-GPU-hour analysis (saved traces, hardware card math, upstream/external audits) before running the GPUs.
- **Update the model, don't just log failures**: Every failed kernel or surprise measurement must update our quantitative model of these GPUs ([HARDWARE-CARD.md](HARDWARE-CARD.md)) -- bandwidth, occupancy, graph-node vs launch overhead, MI60 vs MI50 asymmetry (HBM streaming is ~equal at ~890-904 GB/s, while MI50 is ~21% slower in PP GEMM and ~20-40% slower in small kernels), and host/PCIe behavior. Do not discard a good route just because one naive implementation lost.

## 7. How Agents Should Propose Deviations

If you see a high-value opportunity that conflicts with a rule in this repo (or an existing assumption on the hardware card):
1. State clearly **which goal or rule** it deviates from.
2. Quantify the **expected upside** (speed, VRAM fit, or maintenance savings) and the **risk** (accuracy, LOC, GPU hours).
3. Ask the owner before spending GPU time or merging the deviation.
