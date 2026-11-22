# Goals and Engineering Priorities

Design goals, target workloads, and tradeoff guidelines for `llama-of-vega.cpp`. Read this before [HARDWARE.md](HARDWARE.md), [PERFORMANCE.md](PERFORMANCE.md), or [MAINTENANCE.md](MAINTENANCE.md). Treat individual rules in other docs as guidelines subordinate to these goals; flag conflicts or high-value tradeoffs explicitly.

## 1. Mission

Deliver fast, high-fidelity local inference on dual legacy AMD **gfx906** GPUs (**MI60 = ROCm0, MI50 = ROCm1**, 32 GiB HBM2 each, no matrix units) for strong open agentic LLMs, while keeping model accuracy faithful and keeping profiling/benchmark GPU time lean.

## 2. Target Workload

- **Context length**: Long-context agentic sessions (**50-100K+ tokens**). Both **prompt processing (PP / TTFT)** and **token generation (TG)** matter.
- **Concurrency**:
  - Primary: single-slot long-context latency (TTFT + TG) and **2-4 concurrent parallel slots** (subagent bursts; multi-slot decode and verify must scale cleanly and must not regress under speculative decoding).
  - Secondary: optional dual-model serving (e.g. one large high-quality model and one fast smaller model, or one model per GPU).
- **Models**:
  - Current equal priorities: **Qwen3.8-27B Q8_0**, **Qwen3.8 Flash Next**, and **Inkling 2 Mini**.
  - Near-term horizon: upcoming **Qwen 4 (27B and MoE)** models. Prefer investments that carry forward to Qwen 4+.

## 3. Priority Hierarchy

1. **Reusable kernel and architecture wins** (long-context GQA/sparse attention, GDN/SSM recurrence, MoE routing and grouped GEMM, multi-slot/verify mat-vec, TP/AllReduce, repacked quants).
2. **Qwen 4-relevant wins** (architectures and formats shared with upcoming Qwen 4 dense/MoE models).
3. **Model-specific wins** on current target models -- accept targeted model-specific kernels when they yield real, durable speedups.
4. **Architectural placement and sizing** (e.g. GPU-level placement of KV cache vs. draft/MTP head given MI60 vs. MI50 asymmetry, or arch-driven ubatch sizing). This is a first-class architectural lever, not knob tuning.
5. **Parameter tuning (last stage)**: low-leverage parameter sweeps (e.g. draft depth, thread counts, `-ts` micro-tweaks) come last, after higher-leverage kernel and architectural work. Diagnostic env checks (like `HIP_FORCE_DEV_KERNARG=1`) and curve measurements needed to design kernels (like 1-16 column verify timing) are **measurements**, not parameter tuning.

## 4. Custom Code vs. Performance (Secondary Goal)

- **Speed wins first; clean up second.** Capture real performance gains first, then refactor to simplify upstream catch-up merges and eventual upstreaming to `llama.cpp` main.
- **Pruning rule of thumb**: Worth trading a tiny gain (e.g. ~0.3% PP) to remove ~200 lines of custom code, or pruning dead/slow fallbacks, without sacrificing meaningful speedups solely for line count. Weigh edge cases against ongoing upstream-merge maintenance cost.
- **Borrow freely**: Check upstream `llama.cpp`, `unslothai/llama.cpp`, `mixa3607/ML-gfx906` / `mx-llama.cpp`, `thejacer/slop.cpp`, `alex4300/llama.cpp-gfx906-opt`, `milpster/gfx906-llama-cpp`, `iacopPBK/gfx906-quartz`, and `Niko1221/Strata` for ideas or reference kernels before building from scratch.

## 5. Accuracy and Quality

- **Default**: Lossless or FP-order-only changes (verified with CPU-reference unit tests + KLD/PPL checks). Keep **F16 KV cache** (no KV quantization by default) and full expert routing.
- **Flagging rule**: Never introduce quality-affecting or lossy changes silently. Propose lossy tradeoffs (e.g. aggressive quants, expert/KV approximations) **only** when explicitly flagged with expected quality impact and offering a **major (>25%) speedup or a critical memory-fit unlock**.

## 6. Hardware Scope & Scientific Method

- **Software-only scope**: Assume fixed PCIe topology (no direct P2P, mixed CPU/chipset lanes); do not plan around physical hardware changes.
- **Lean GPU validation**: Batch GPU validation into concise windows (<=15-30 min) and exhaust offline analysis (saved traces, hardware-card math, upstream/external audits) before running GPU benchmarks.
- **Update the model, don't just log failures**: Every failed kernel or surprise measurement must update our quantitative GPU model ([HARDWARE.md](HARDWARE.md)) -- bandwidth, occupancy, graph-node vs launch overhead, MI60 vs MI50 asymmetry (HBM streaming is ~equal at ~891-904 GB/s, while MI50 is ~21% slower in PP GEMM and ~20-40% slower in small kernels), and host/PCIe behavior. Do not discard a good route just because one naive implementation lost.

## 7. Evaluating Tradeoffs and Deviations

When an optimization conflicts with a guideline above (or an assumption on the hardware card):
1. State clearly **which goal or assumption** it deviates from.
2. Quantify the **expected upside** (speed, VRAM fit, or maintenance savings) and the **risk** (accuracy, LOC, GPU validation cost).
3. Document the decision explicitly before merging.
