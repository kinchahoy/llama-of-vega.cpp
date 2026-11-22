# Launch floor playbook (gfx906)

How the 2026-11-21 Inkling 2 Mini speedup was found, and how to do the same for 27B Q8_0 and Flash Next. Numbers are from two gfx906 cards (ROCm0 MI60, ROCm1 MI50). Results: [PERFORMANCE.md](PERFORMANCE.md#2026-11-21-inkling-2-mini-crash-fix-fused-ops-nvfp4-baseline).

## Cost model

Token time = launches x launch cost + real kernel work.

- Launch cost in the app is about 2.2 us per kernel (measured: each launch removed by a kill switch saved 1.5 to 1.9 us; a bare HIP loop costs 2.0 us on the MI60 and 2.6 us on the MI50). Captured graphs (`GGML_HIP_GRAPHS`) are already used for token generation and are in these numbers.
- The cost does not depend on grid size up to 64 blocks, on threads per block, or on the number of arguments. A kernel that does almost nothing costs the same as one that reads a few thousand floats.
- Kernels on one stream run one after another. More streams give 1.45x at most from one host thread, so this is not worth the sync cost.
- `rocprofv3` adds about 1.2 us to every kernel and shows a floor of 4.2 us. Use it for counts and for ranking, not for absolute time.
- Real work is bound by memory bandwidth (about 800 GB/s reached by the repacked mat-vec). Work at the roofline cannot be fused away.

So: one launch per token removed is worth about 2.2 us per GPU. One launch per layer pass removed is worth 2.2 us x passes.

| Model | Launches per token per GPU | Launch floor | Token time | Floor share |
| --- | ---: | ---: | ---: | ---: |
| Inkling 2 Mini, start of day | 7,219 | 15.9 ms | 27.5 ms | 58% |
| Inkling 2 Mini, after the fused ops (IQ4_NL, 47.7 t/s) | 4,128 | 9.1 ms | 21.0 ms | 43% |
| Flash Next UD-IQ3_XXS, 2 GPUs | 2,811 | 6.2 ms | 19.8 ms | 31% |
| 27B Q8_0, 2 GPUs | 1,645 | 3.6 ms | 26.9 ms | 13% |

## How to measure

Scripts are in [investigation-notes/2026-11-21-inkling2-mini](investigation-notes/2026-11-21-inkling2-mini/).

1. Kernel profile: `kernel_profile.sh` (one GPU) or the profile part of `ppl_and_focus_profiles.sh` (two GPUs). It runs `rocprofv3 --kernel-trace` on `llama-bench -p 0 -n 64` and prints calls and time per kernel from the sqlite file. Skip the first third of the launches (load and warmup).
2. Rank by two lists: kernels with many calls at the 4.2 us floor (fusion candidates) and kernels well above the floor (work; check against the roofline first).
3. A fused kernel that shows 10 us or more per call at one token is badly shaped, not busy. Fix it before adding new ones (see pitfalls).
4. Launch microbench: `launch_floor_bench.cpp` (`clang++ -x hip --offload-arch=gfx906 -O3 -std=c++17 ... -pthread`). Rerun after a ROCm update.
5. A/B: same binary with the kill switch off, on, on, off, or two builds with `ab_two_builds.sh`. Start below 60C.
6. Quality: greedy text for a short and a 3K-token prompt, then `llama-perplexity` on and off; for an attention change also `--kl-divergence` against the old path (`kld_banded.sh`).
7. `perf` on the host is blocked here (`perf_event_paranoid=4`). The gaps between kernels in the trace tell the same thing: zero gaps mean the GPU is the limit.

## How to add a fused op

Pattern: commit `85bb8a0f7` (SCONV_RES) is the smallest full example. Places to touch: `ggml.h` enum and builder, `ggml.c` (name, symbol, builder, both `GGML_OP_COUNT` asserts), `ggml-cpu.c` + `ops.cpp` + `ops.h` (CPU reference), a new `ggml-cuda/*.cu` + `.cuh`, `ggml-cuda.cu` (dispatch, supports_op), `ggml-backend-meta.cpp` (whitelist), `tests/test-backend-ops.cpp`, the model file with an env kill switch read once.

Rules that came out of the work:

- Cut along the weight matmuls. Never put a weight matmul into a one-block kernel: one block cannot stream the weights (tried: TG 40.5 to 11.7 t/s, [notes](old-ignore/2026-11-21/loop-ctrl-fused/README.md)).
- A reduction over a row of a few thousand floats must use a wide block (1024 threads) and warp reductions, and reduce all needed sums in one pass. The first `res_mix_f32` used 256 threads and one tree reduction per sum: 26.7 us per call. After the rewrite TG went from 43.6 to 47.7 t/s.
- An output that feeds several consumers should be laid out so that each slice is contiguous. Then the consumers take views and no copy kernel runs.
- Keep the old graph behind the kill switch. It is the reference for the A/B and for quality checks.
- Test on the CPU backend too (`--device none -ngl 0`, a few tokens). It checks `get_rows` indices; this found the Inkling crash.
- A new op for attention must be compared with a logit-level check (KLD), not only greedy text.

## What a fused kernel should do on these cards

- No matrix units: a dense bias or mask operand has no advantage. Compute small terms inside the kernel (the Inkling 2 relative bias is a table lookup per key in the tile kernel now; this removed a gather, two permute copies and an F16 copy).
- Formats: the repacked path keeps 4-bit types as a nibble plane plus a scale plane. A new 4-bit format is a layout variant of MXFP4 / IQ4_NL, not a new kernel family (NVFP4: two F16 scales per 32 values).
- The per-token graph is replayed as a captured graph. A change that makes node properties differ between tokens (for example a changing pointer or shape) turns this off for that graph; check with `GGML_CUDA_GRAPH_DEBUG=1 -v` for "warmup reset".

## Inkling 2 Mini: left to do

First: the banded bias for token generation (small ubatches still build the dense bias; TG falls from 45.8 to 36.4 t/s between an empty and a 4K context). See the handoff. Then the launch count: 86 launches per layer pass at the end of the day, 48 passes per token. Each launch per pass is worth about 0.1 ms per token (0.5%).

| Item | Launches per pass | Est. TG |
| --- | ---: | ---: |
| Take views of the mix output and the loop messages without `ggml_cont` (lay the RES_MIX output out as [n_embd, n_tokens, n_q]) | 5 | 2.5% |
| Loop controllers Q/K/V as two batched kernels (a block per gate value, then a block per output slice) | 8 on looped passes | 4% |
| One kernel for clip Q, clip K, clip R, norm V and the tau scale (packed output) | 4 | 2% |
| Short conv reads (history, current) without the concat; state write as one in-place kernel | 4-6 | 2.5% |
| Fold input norms into the producer; MoE glue (router weights, activation, combine) | 10 | 5% |
| One mat-vec launch over the weights that share an input (`wq`, `attn_rel`, `attn_sink_proj`, `wqkv_gate`) | 4 | 2.5%, hardest |

About 20% in total; the floor share cannot go below the mat-vec count.

## 27B Q8_0 (2 GPUs, tensor split)

Profile 2026-11-21 (`build-banded-20261121/validation/ppl/prof-27b`), per token, both GPUs summed, traced: 3,290 launches, both GPUs 96-99% busy.

| Kernel | Calls | us per call | ms per token | Share |
| --- | ---: | ---: | ---: | ---: |
| `mul_mat_vec_q8_0_repacked<8, 8, false, 64, false>` | 761 | 30.5 | 23.2 | 40% |
| `mul_mat_vec_q8_0_repacked<8, 8, false, 64, true>` (fused FFN) | 132 | 127.4 | 16.8 | 29% |
| `ggml_cuda_ar_kernel` | 264 | 20.0 | 5.3 | 9% |
| `add_rms_norm_mul_f32<1024>` | 264 | 11.3 | 3.0 | 5% |
| `quantize_q8_1<32>` | 266 | 4.5 | 1.2 | 2% |
| about ten kinds of tiny kernels of the recurrent layers, 99 calls each | about 1,000 | 4.5 | 4.5 | 8% |

27B is work-bound: 69% is weight mat-vec, and 28.6 GB of weights at about 800 GB/s per GPU is 17.9 ms of the 26.9 ms token. The reachable part is small. In order:

1. `add_rms_norm_mul_f32<1024>` costs 11.3 us where a plain row norm costs 4.5 us. Same symptom as the first `res_mix_f32`. Est. 0.9 ms per token per GPU, about 3% TG. Small job; `test-backend-ops` covers it.
2. The recurrent-layer chain (`concat_dim0_transposed`, `ssm_conv_f32`, `k_bin_bcast`, unary, gated unary, two `get_rows`, `cpy_scalar`, `gated_delta_net_cuda`): about 500 launches per GPU per token, 1.1 ms, about 4% if most are fused. `SCONV_RES` is the same shape of chain as `ssm_conv` + concat here.
3. AllReduce (9%) is hardware (idea A4 in the ledger).
4. Mat-vec is at about 90% of the roofline (idea Q3 in the ledger). Only fewer bits per weight moves this.

Honest total: 5 to 8% TG from kernels.

## Flash Next UD-IQ3_XXS (2 GPUs, tensor split)

Profile 2026-11-21 (`.../ppl/prof-fn`), per token, both GPUs summed, traced: 5,621 launches; GPUs only 83-88% busy (the rest is host and AllReduce waits).

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

Flash Next is the most launch-bound focus model (floor about 31% of the token). In order:

1. `dsv4_hc_mix_down` / `_up` cost 11 us per call. Rewrite them like `res_mix_f32` (wide block, one pass, warp reductions). Est. 1.3 ms per GPU per token, about 6% TG. Tests exist (`-o DSV4_HC_MIX`).
2. The generic `mul_mat_vec_q` calls (34 us): find which tensors and quant type are outside the repack list and add that type as a layout variant. Est. 4 to 8%.
3. `quantize_q8_1`: 299 launches per GPU per token, the largest count. Check the hit rate of the q8_1 cache; let the fused kernel before a mat-vec (`dsv4_hc_post`, the norm) also write the q8_1 form. Est. 3%.
4. The elementwise chains around the hyper-connections (unary, bcast, scale, prep, post): dump the graph of one layer, group the runs between mat-vecs, fuse each run. About 400 launches per GPU per token, est. 4%.
5. The 12-17% idle time: see the host-gap notes (`mx-tg-hostgap`) and AllReduce (A4).

Honest total: 15 to 20% TG from kernels.

## Cleverer ideas, not tried

- A small "program" kernel: one generic kernel that runs a short list of elementwise steps (add, mul, scale, tanh, sigmoid, clamp) over a small tensor in one launch. The backend would fuse any run of small elementwise nodes at graph time, for every model, with no new op per chain. Needs use counts so that only values not read outside the run are skipped.
- Write the q8_1 form of an activation in the kernel that produces it. Removes one launch per mat-vec input on every model.
- One mat-vec launch over several weights with the same input (grid spans the matrices; the result must be bit-identical to separate calls, which makes it easy to test).
- A host thread per stream gave 1.9x launch throughput in the microbench. Two lanes for independent branches of a layer could use this, but each join costs an event wait.
