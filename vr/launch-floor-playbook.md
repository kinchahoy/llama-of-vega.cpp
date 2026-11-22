# Launch floor playbook (gfx906)

How the 2026-11-21 Inkling 2 Mini speedup was found, and how to measure and fuse launches on any model. Numbers are from two gfx906 cards (ROCm0 MI60, ROCm1 MI50). Results: [PERFORMANCE-RECORDS.md](PERFORMANCE-RECORDS.md#2026-11-21-inkling-2-mini-crash-fix-fused-ops-nvfp4-baseline). Facts with grades: [What we know](PERFORMANCE.md#what-we-know).

## Cost model

Token time = launches x launch cost + real kernel work.

- Launch cost in the app is about 2.2 us per kernel (A: each launch removed by a kill switch saved 1.5 to 1.9 us in three ABBA tests; B: a bare HIP loop costs 2.0 us on the MI60 and 2.6 us on the MI50). Captured graphs (`GGML_HIP_GRAPHS`) are used for Inkling 2 Mini token generation (B: debug log) and are in these numbers.
- In a microbench the bare cost did not change with grid size up to 64 blocks, threads per block or argument count (B, microbench). Kernels with real reductions are longer: the first `res_mix_f32` took 26.7 us per call.
- Kernels on one stream run one after another. More streams gave 1.45x launch throughput from one host thread and 1.9x with a host thread per stream (B, microbench); each join costs an event wait.
- `rocprofv3` adds about 1.2 us to every kernel and shows a floor of 4.2 us (B). Use it for counts and for ranking, not for absolute time.
- Real work is bound by memory bandwidth (about 800 GB/s reached by the repacked mat-vec, B; other values are in [Numbers that disagree](PERFORMANCE.md#numbers-that-disagree)). Work near the roofline is not removed by fusing launches.

So: one launch per token removed is worth about 2.2 us per GPU (C, derived). One launch per layer pass removed is worth 2.2 us x passes.

| Model | Launches per token per GPU | Launch floor (C) | Token time | Floor share (C) |
| --- | ---: | ---: | ---: | ---: |
| Inkling 2 Mini, start of day | 7,219 | 15.9 ms | 27.5 ms | 58% |
| Inkling 2 Mini, after the fused ops (IQ4_NL, 47.7 t/s) | 4,128 | 9.1 ms | 21.0 ms | 43% |
| Flash Next UD-IQ3_XXS, 2 GPUs | 2,811 | 6.2 ms | 19.8 ms | 31% |
| 27B Q8_0, 2 GPUs | 1,645 | 3.6 ms | 26.9 ms | 13% |

## The one-block rule

Measured 2026-11-22 (B, microbench; numbers in [PERFORMANCE.md](PERFORMANCE.md#what-we-know)): one thread block gets about 1/30 of the GPU. A kernel that must reduce over a whole row runs as one block, so its cost is about 2.9 us + 0.46 us per 1000 floats. Use this before writing or rewriting a fused kernel:

- Count the floats one block touches. Under about 5000, one block of 1024 threads is fine. Above it, split: reductions in one launch (many blocks, one partial each), the output in a second launch (many blocks, each reads the partials). Two launches cost about 5.4 us, flat.
- Use 1024 threads per block for a one-block kernel. 256 threads cost twice as much for the same row.
- Do not look for the cost in barriers (0.08 us each). Count dependent phases (1.6 us each) and floats per block.
- A kernel that streams weights through one block is the worst case (the one-block loop controller: 600K floats).
- Before a rewrite, compute what the rule predicts for the current kernel and for the new form. If the current cost already matches the rule, only a different block layout can help.

## How to measure

Start here: `python3 vr/scripts/kernel_budget.py TRACE_DIR N_TOKENS` turns one `rocprofv3` kernel trace into a ranked work list. It splits the kernels into three classes from the launch geometry in the trace: at the floor (only fewer launches help), one block and above the floor (apply the one-block rule), and many blocks above the floor (real work; check bytes against 800 GB/s). First run, 2026-11-22, on the Flash Next trace: it listed five one-block kernels above the floor that were in no hypothesis list (`topk_moe_wave_cuda<512, 10>` 1 block of 32 threads at 14.1 us, `lightning_indexer_kernel_vec`, `k_argsort_f32_i32`, `rope_multi`, `top_k_mask_kernel`; 0.92 ms excess per token per GPU in total, B, one trace).

Scripts are in [investigation-notes/2026-11-21-inkling2-mini](investigation-notes/2026-11-21-inkling2-mini/).

1. Kernel profile: `kernel_profile.sh` (one GPU) or the profile part of `ppl_and_focus_profiles.sh` (two GPUs). It runs `rocprofv3 --kernel-trace` on `llama-bench -p 0 -n 64` and prints calls and time per kernel from the sqlite file. Skip the first third of the launches (load and warmup).
2. Rank by two lists: kernels with many calls at the 4.2 us floor (fusion candidates) and kernels well above the floor (work; check against the roofline first).
3. A fused kernel that shows 10 us or more per call at one token is badly shaped, not busy. Fix it before adding new ones (see pitfalls).
4. Launch microbench: `launch_floor_bench.cpp` (`clang++ -x hip --offload-arch=gfx906 -O3 -std=c++17 ... -pthread`). Rerun after a ROCm update.
5. A/B: same binary with the kill switch off, on, on, off, or two builds with `ab_two_builds.sh`. Start below 60C and alternate A B B A.
6. Quality: greedy text for a short and a 3K-token prompt, then `llama-perplexity` on and off; for an attention change also `--kl-divergence` against the old path (`kld_banded.sh`).
7. `perf` on the host is blocked here (`perf_event_paranoid=4`). The gaps between kernels in the trace tell the same thing: zero gaps mean the GPU queue is the limit (B: trace gaps were zero at p99 on Inkling 2 Mini).

## How to add a fused op

Pattern: commit `85bb8a0f7` (SCONV_RES) is the smallest full example. Places to touch: `ggml.h` enum and builder, `ggml.c` (name, symbol, builder, both `GGML_OP_COUNT` asserts), `ggml-cpu.c` + `ops.cpp` + `ops.h` (CPU reference), a new `ggml-cuda/*.cu` + `.cuh`, `ggml-cuda.cu` (dispatch, supports_op), `ggml-backend-meta.cpp` (whitelist), `tests/test-backend-ops.cpp`, the model file with an env kill switch read once.

Rules that came out of the work:

- Cut along the weight matmuls. Tried: a weight matmul inside a one-block kernel did not work (TG 40.5 to 11.7 t/s, A; one block does not stream the weights, reasoned; [notes](old-ignore/2026-11-21/loop-ctrl-fused/README.md)). A different shape (a block per output slice) is not tried.
- A reduction over a row of a few thousand floats must use a wide block (1024 threads) and warp reductions, and reduce all needed sums in one pass. The first `res_mix_f32` used 256 threads and one tree reduction per sum: 26.7 us per call. After the rewrite TG went from 43.57 to 47.73 t/s (A).
- An output that feeds several consumers should be laid out so that each slice is contiguous. Then the consumers take views and no copy kernel runs.
- Keep the old graph behind the kill switch. It is the reference for the A/B and for quality checks.
- Test on the CPU backend too (`--device none -ngl 0`, a few tokens). It checks `get_rows` indices; this found the Inkling crash.
- A new op for attention must be compared with a logit-level check (KLD), not just greedy text.

## What a fused kernel should do on these cards

- No matrix units: a dense bias or mask operand has no advantage. Compute small terms inside the kernel (the Inkling 2 relative bias is a table lookup per key in the tile kernel now; this removed a gather, two permute copies and an F16 copy).
- Formats: the repacked path keeps 4-bit types as a nibble plane plus a scale plane. A new 4-bit format is a layout variant of MXFP4 / IQ4_NL, not a new kernel family (NVFP4: two F16 scales per 32 values).
- The per-token graph is replayed as a captured graph. A change that makes node properties differ between tokens (for example a changing pointer or shape) can turn this off for that graph; check with `GGML_CUDA_GRAPH_DEBUG=1 -v` for "warmup reset".

## Profile: 27B Q8_0 (2 GPUs, tensor split)

Profile 2026-11-21 (`build-banded-20261121/validation/ppl/prof-27b`), per token, both GPUs summed, traced: 3,290 launches, both GPUs 96-99% busy (B, one trace). 69% of traced kernel time is weight mat-vec.

| Kernel | Calls | us per call | ms per token | Share |
| --- | ---: | ---: | ---: | ---: |
| `mul_mat_vec_q8_0_repacked<8, 8, false, 64, false>` | 761 | 30.5 | 23.2 | 40% |
| `mul_mat_vec_q8_0_repacked<8, 8, false, 64, true>` (fused FFN) | 132 | 127.4 | 16.8 | 29% |
| `ggml_cuda_ar_kernel` | 264 | 20.0 | 5.3 | 9% |
| `add_rms_norm_mul_f32<1024>` | 264 | 11.3 | 3.0 | 5% |
| `quantize_q8_1<32>` | 266 | 4.5 | 1.2 | 2% |
| about ten kinds of tiny kernels of the recurrent layers, 99 calls each | about 1,000 | 4.5 | 4.5 | 8% |

## Profile: Flash Next UD-IQ3_XXS (2 GPUs, tensor split)

Profile 2026-11-21 (`.../ppl/prof-fn`), per token, both GPUs summed, traced: 5,621 launches; GPUs 83-88% busy (B, one trace).

| Kernel | Calls | us per call | ms per token | Share |
| --- | ---: | ---: | ---: | ---: |
| `mul_mat_vec_rp` (experts) | 510 + 92 | 15.4 | 9.3 | 21% |
| `ggml_cuda_ar_kernel` | 197 | 25.8 | 5.1 | 11% |
| `quantize_q8_1<32>` | 598 | 4.6 | 2.7 | 6% |
| `mul_mat_vec_f<float, float, 1, 128>` | 344 | 6.9 | 2.4 | 5% |
| `dsv4_hc_mix_down_f32<1, 5>` | 197 | 11.6 | 2.3 | 5% |
| `dsv4_hc_mix_up_f32<2, 1>` | 197 | 10.7 | 2.1 | 5% |
| `mul_mat_vec_q` (generic, two kinds) | 53 + 43 | 34 | 3.3 | 7% |
| `unary_op_kernel`, `k_bin_bcast`, `scale_f32`, `dsv4_hc_mix_prep`, `dsv4_hc_post` | 370 + 274 + 227 + 197 + 197 | 4.2 | 5.3 | 12% |

## Ideas and estimates

The per-model opportunity lists are in the Hypotheses section of [PERFORMANCE.md](PERFORMANCE.md#hypotheses) (one copy, with grades and cheapest tests).
