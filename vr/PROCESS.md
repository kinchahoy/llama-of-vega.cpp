# Process: how this fork finds performance

Read [GOALS.md](GOALS.md) first for the owner's intent, target workload (50-100K+ PP+TG, 1-4 parallel slots), and deviation rules.

The owner chose this process on 2026-11-22. It replaces "profile, try things, benchmark a lot". The old way cost wall clock, GPU power, and two wrong cost models that stood for 22 and 23 days (see [Why this process](#why-this-process)). Constants live in [HARDWARE-CARD.md](HARDWARE-CARD.md); numbers, routes and the worklist live in [PERFORMANCE.md](PERFORMANCE.md).

## Order of work

The owner's rule (2026-11-22, refined in [GOALS.md](GOALS.md)): kernel and architecture changes come before aimless parameter tuning, with a preference for reusable wins that carry forward to upcoming models (e.g. Qwen 4).

- First: reusable kernel and code changes (long-context attention, GDN/SSM, MoE, multi-slot mat-vec, TP/AllReduce, repacked quants), then model-specific kernel/code wins.
- Alongside architecture work (not banned): architectural GPU-level placement and sizing decisions (e.g. which GPU holds KV cache vs. draft/MTP head given MI60 vs. MI50 asymmetry, or arch-driven ubatch sizing).
- Always permitted as measurements (not tuning): fast diagnostic env checks (e.g. `HIP_FORCE_DEV_KERNARG=1`) and hardware/kernel cost curves (e.g. `llama-bench -p 1..16` for multi-slot/verify kernel design).
- Last, and only after the owner agrees that kernel-level work is tapped out and gives permission: parameter tuning (MTP2 vs MTP3 depth sweeps, thread counts, `-ts` micro-tweaks, serving presets used as a speed lever).
- A parameter change that is needed for correctness, or to run a required comparison (for example the settings that AGENTS.md or the final validation pass fix), is not tuning.
- Worklist rows carry a `kind` tag for this. Parameter rows are in their own table at the end of the [worklist](PERFORMANCE.md#parameter-tuning---last-stage-needs-owner-permission).

## The four stages

### Stage 1: learn the hardware once

A short set of synthetic microbenchmarks (no model loads, seconds each) gives a small table of constants and curves: the [hardware card](HARDWARE-CARD.md). Rerun only after a ROCm update or for a genuinely new hard question. Rule: a hardware question that a microbench can answer in seconds is not answered by an end-to-end run.

### Stage 2: predict from data on disk

For each kernel in a saved trace, predict its cost from its launch geometry (blocks, threads per block, registers, shared memory, all recorded in the trace) and the bytes it must touch. Kernels measured well above the prediction are badly laid out and fixable. Kernels at the prediction are at their limit for that layout. Tool: `vr/scripts/kernel_budget.py` (see [Tools](#tools)). Rule: read the geometry before naming a mechanism.

### Stage 3: design on paper, in batches

Every change carries a predicted saving computed from the card. Under 1% predicted: do not build it. Variants are chosen by the model, not explored on the GPU. A parameter that really needs tuning is tuned in a standalone kernel sandbox (compiles in seconds), not by a full rebuild and a model load. Changes are grouped in batches. The worklist in [PERFORMANCE.md](PERFORMANCE.md#worklist) is ordered by predicted gain; its `priced by` column says whether the estimate follows from the card, from one trace, or is a guess.

### Stage 4: one confirmation per batch

- Unit tests for the touched ops (`test-backend-ops`, and the CPU backend for new ops).
- One short trace comparing measured with predicted, per kernel. A mismatch above about 20% is investigated and updates the card: that is the learning step.
- One alternated end-to-end run per affected model (A B B A, start below 60 C).
- Budget: at most one build and about ten GPU-minutes per batch.

## After an upstream merge

Diff the graph and kernel geometry without the GPU. Then run the one end-to-end comparison that AGENTS.md requires (rebuild and compare PP and TG with the pre-merge commit using identical `llama-bench` settings, and confirm the HIP host-staged AllReduce path) and nothing more, unless the model predicts a change. Details and safety items: [how-to-catchup.md](how-to-catchup.md).

## Pre-registration template

Write these five lines before any experiment and keep them in the record ([record format](PERFORMANCE-RECORDS.md#record-format)):

1. Mechanism: what the change does and why it should be cheaper.
2. Predicted gain in ms per token, from the card (show the arithmetic).
3. The cheap check that would falsify it (a microbench, a trace column or a unit test, not an end-to-end run).
4. Best-case bound: the most any implementation could gain.
5. What result would change the card.

## Tools

- `python3 vr/scripts/kernel_budget.py TRACE_DIR N_TOKENS` turns one `rocprofv3` kernel trace into a ranked work list. It splits the kernels into three classes from the launch geometry in the trace: at the floor (only fewer launches help), one block and above the floor (apply the one-block rule), and many blocks above the floor (real work; check bytes against the mat-vec bandwidth on the card).
  - **Skeptical caveats (2026-11-22)**: (1) `kernel_budget.py` is hardcoded to 1-slot depth-0 TG (`-p 0 -n N`) and misses long-context (50-100K), multi-slot (2-4 slots), and PP bottlenecks. (2) Dropping the first 1/3 of launches and dividing by `n_tok * 2 / 3` distorts per-token counts whenever warmup/load differs from decode. (3) Subtracting the 4.2 us empty-kernel floor from non-trivial one-block kernels (`topk_moe`, `k_argsort`) over-estimates recoverable savings. (4) Grouping by kernel name and averaging across both GPUs merges distinct shapes and hides MI50 (ROCm1) small-kernel lag. (5) Using 2.2 us/launch over-prices fusion under captured HIP graphs (~1.22 us/node microbench, 1.5-1.9 us saved in ABBA).
  - First run, 2026-11-22, on the Flash Next trace: it listed five one-block kernels above the floor that were in no hypothesis list (`topk_moe_wave_cuda<512, 10>` 1 block of 32 threads at 14.1 us, `lightning_indexer_kernel_vec`, `k_argsort_f32_i32`, `rope_multi`, `top_k_mask_kernel`; 0.92 ms raw excess over empty floor, realistic savings lower). `vr/scripts/token_budget.py` makes the older step-by-step budget ([TOKEN-BUDGET.md](TOKEN-BUDGET.md)).
- Kernel sandbox: standalone microbenchmarks that compile in seconds and need no model. Examples: [single_block_bench.cpp](investigation-notes/2026-11-22-one-block-rule/single_block_bench.cpp) (one-block cost, threads per block, phases, two-kernel form) and [launch_floor_bench.cpp](investigation-notes/2026-11-21-inkling2-mini/launch_floor_bench.cpp) (`clang++ -x hip --offload-arch=gfx906 -O3 -std=c++17 ... -pthread`). Tune block size and similar parameters here.
- Scripts for runs are in [investigation-notes/2026-11-21-inkling2-mini](investigation-notes/2026-11-21-inkling2-mini/).
  1. Kernel profile: `kernel_profile.sh` (one GPU) or the profile part of `ppl_and_focus_profiles.sh` (two GPUs). It runs `rocprofv3 --kernel-trace` on `llama-bench -p 0 -n 64` and prints calls and time per kernel from the sqlite file. Skip the first third of the launches (load and warmup).
  2. Rank by two lists: kernels with many calls at the 4.2 us floor (fusion candidates) and kernels well above the floor (work; check against the roofline first).
  3. A fused kernel that shows 10 us or more per call at one token is badly shaped, not busy. Fix it before adding new ones (see pitfalls).
  4. A/B: same binary with the kill switch off, on, on, off, or two builds with `ab_two_builds.sh`. Start below 60 C and alternate A B B A.
  5. Quality: greedy text for a short and a 3K-token prompt, then `llama-perplexity` on and off; for an attention change also `--kl-divergence` against the old path (`kld_banded.sh`).
  6. `perf` on the host is blocked here (`perf_event_paranoid=4`). The gaps between kernels in the trace tell the same thing: zero gaps mean the GPU queue is the limit (B: trace gaps were zero at p99 on Inkling 2 Mini).

## How to add a fused op

Pattern: commit `85bb8a0f7` (SCONV_RES) is the smallest full example. Places to touch: `ggml.h` enum and builder, `ggml.c` (name, symbol, builder, both `GGML_OP_COUNT` asserts), `ggml-cpu.c` + `ops.cpp` + `ops.h` (CPU reference), a new `ggml-cuda/*.cu` + `.cuh`, `ggml-cuda.cu` (dispatch, supports_op), `ggml-backend-meta.cpp` (whitelist), `tests/test-backend-ops.cpp`, the model file with an env kill switch read once.

Rules that came out of the work:

- Cut along the weight matmuls. Tried: a weight matmul inside a one-block kernel did not work (TG 40.5 to 11.7 t/s, A; one block does not stream the weights, reasoned; [notes](old-ignore/2026-11-21/loop-ctrl-fused/README.md)). A different shape (a block per output slice) is not tried.
- A reduction over a row of a few thousand floats must use a wide block (1024 threads) and warp reductions, and reduce all needed sums in one pass. The first `res_mix_f32` used 256 threads and one tree reduction per sum: 26.7 us per call. After the rewrite TG went from 43.57 to 47.73 t/s (A). Above about 5000 floats per block use the two-launch form from the [one-block rule](HARDWARE-CARD.md#the-one-block-rule).
- An output that feeds several consumers should be laid out so that each slice is contiguous. Then the consumers take views and no copy kernel runs.
- Keep the old graph behind the kill switch. It is the reference for the A/B and for quality checks.
- Test on the CPU backend too (`--device none -ngl 0`, a few tokens). It checks `get_rows` indices; this found the Inkling crash.
- A new op for attention must be compared with a logit-level check (KLD), not just greedy text.

### What a fused kernel should do on these cards

- No matrix units: a dense bias or mask operand has no advantage. Compute small terms inside the kernel (the Inkling 2 relative bias is a table lookup per key in the tile kernel now; this removed a gather, two permute copies and an F16 copy).
- Formats: the repacked path keeps 4-bit types as a nibble plane plus a scale plane. A new 4-bit format is a layout variant of MXFP4 / IQ4_NL, not a new kernel family (NVFP4: two F16 scales per 32 values).
- The per-token graph is replayed as a captured graph. A change that makes node properties differ between tokens (for example a changing pointer or shape) can turn this off for that graph; check with `GGML_CUDA_GRAPH_DEBUG=1 -v` for "warmup reset".

## Why this process

Facts from [the history dataset](investigation-notes/2026-11-22-process-retro/history-retro.md), 35 landed wins and 62 detours over the life of the fork (the dataset's own gaps apply; pre-2026-10-29 history is squashed).

- Wins by origin: profile 18 of 35 (15 of those landed on the first try), differential 4, bug 4, fork-port 4, upstream-port 2, unknown 3.
- Detours by kind: process 14, measurement-error 11, wrong-model 10, no-theory 8, naive-impl 7, unknown 6, analogy 2, noise 2, bound-ignored 2.
- Two cost-model errors stood for a long time: the host-gap thesis for 23 days and the 4-5 us launch floor (the profiler floor taken as the app cost) for 22 days. They consumed other work: N2a, N2b and parts of TB5 on the first, ceilings of P1, T1, TB2, TB3, L2 and W64 on the second.
- 40 of 62 detours were caught within a day. The long ones were beliefs and merge items.
- 10 of 35 wins have grade A, and 7 of those are from one day (2026-11-21).
- 27B plain TG went 30.16 to 36.8 in the week of 2026-10-12, and 36.82 to about 37.2 from 2026-10-29 to 2026-11-21.
- The one-block rule was measured after the one-block loop controller (R-IN8) was built; its arithmetic (about 190 us against a saving of 24 us) existed before the build.

Results of the full launch-floor work of 2026-11-21 (the first use of the cost model): [PERFORMANCE-RECORDS.md](PERFORMANCE-RECORDS.md#2026-11-21-inkling-2-mini-crash-fix-fused-ops-nvfp4-baseline).
