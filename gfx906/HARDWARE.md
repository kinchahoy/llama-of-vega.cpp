# Hardware Reference (gfx906)

Single source of measured hardware constants for the two gfx906 GPUs (**ROCm0 = MI60**, 64 CUs, 225 W, CPU PCIe lanes; **ROCm1 = MI50**, 60 CUs, 178 W, chipset PCIe lanes) plus cross-GPU portability rules. Other documents link here instead of repeating constants. Calibration microbenchmarks live in [gfx906/bench/](bench/q1_occupancy.cpp).

Rerun the microbenches only after a ROCm, driver, or hardware change, or for a genuinely new hard question. A mismatch above ~20% between a card prediction and a real kernel trace updates the card (see [Checks against real kernels](#checks-against-real-kernels)).

## Measured constants

Grades: **A** alternated A/B on a real model, **B** single measurement or microbench, **C** derived estimate, **D** assertion. Historical benchmark records: [BENCHMARK-TIMELINE.md](BENCHMARK-TIMELINE.md).

| Quantity | Value | Device | Grade | Date | Source / Method | Used for |
| --- | --- | --- | --- | --- | --- | --- |
| Graph-replayed launch cost (app) | 1.22 us/node pure launch floor; 1.5-1.9 us saved per removed launch (incl. trivial work); earlier 2.2 us over-prices graph decode | two gfx906 | A / B | 2026-11-21..22 | 3 ABBA kill-switch tests on Inkling 2 Mini + 1000-node HIP graph microbench (source not yet saved; rerun before relying on 1.22 us) | Value of a removed launch in captured HIP graphs |
| Bare launch cost, HIP loop | 2.0 us (MI60), 2.6 us (MI50) | MI60 / MI50 | B | 2026-11-21 | [bench/launch_floor.cpp](bench/launch_floor.cpp) | Bare eager launch floor; MI50 +0.6 us small-kernel lag (`K2`) |
| Profiler (`rocprofv3`) overhead | +1.2 us/kernel; shows 4.2 us floor | two gfx906 | B | 2026-11-21 | `rocprofv3 --kernel-trace` vs bare loop | Convert trace floor to app time |
| Bare launch cost vs geometry | Flat up to 64 blocks, any threads/block or arg count (trivial kernel) | MI60 | B | 2026-11-21 | [bench/launch_floor.cpp](bench/launch_floor.cpp) | Small trivial kernels sit at floor |
| Multi-stream launch throughput | 1.45x (1 host thread), 1.9x (1 host thread/stream) | MI60 | B | 2026-11-21 | [bench/launch_floor.cpp](bench/launch_floor.cpp) | Branch parallelism (each join costs an event wait) |
| One thread block bandwidth | ~1/30 of GPU (26-30 GB/s vs ~891 GB/s card peak) | MI60 | B | 2026-11-22 | [bench/one_block.cpp](bench/one_block.cpp) | The one-block rule |
| One-block cost, elementwise | 2.1 us + 0.26 us per 1,000 floats touched (1024 threads) | MI60 | B | 2026-11-22 | [bench/one_block.cpp](bench/one_block.cpp) | Single-block elementwise bound |
| One-block cost, row reduce + 2nd pass | 2.9 us + 0.46 us per 1,000 floats touched (1024 threads) | MI60 | B | 2026-11-22 | [bench/one_block.cpp](bench/one_block.cpp) | Single-block row-reduction bound |
| Multi-block equivalent work | 2.3-2.8 us across tested sizes (up to 72 x 1024 threads) | MI60 | B | 2026-11-22 | [bench/one_block.cpp](bench/one_block.cpp) | Target cost when splitting a one-block kernel |
| Threads per block (one block, 4608 floats) | 64 thr: 18.1 us; 256 thr: 6.30 us; 1024 thr: 3.39 us | MI60 | B | 2026-11-22 | [bench/one_block.cpp](bench/one_block.cpp) | Always use 1024 threads for wide one-block rows |
| Barrier vs dependent reduce-then-use phase | Barrier ~0.08 us; dependent phase ~1.6 us | MI60 | B | 2026-11-22 | [bench/one_block.cpp](bench/one_block.cpp) | Count dependent phases, not `__syncthreads()` |
| Two-kernel reduction (partials + read) | ~5.4 us total up to 36,864 floats; wins over 1-block above ~5,000 floats | MI60 | B | 2026-11-22 | [bench/one_block.cpp](bench/one_block.cpp) | Split form for row-wide reductions |
| Peak HBM2 streaming | 891 GB/s (MI60), 904 GB/s (MI50) = 87-88% of 1024 GB/s spec | MI60 / MI50 | B | 2026-11-22 | [bench/q2_stream.cpp](bench/q2_stream.cpp) | Streaming roofline (dev1 is NOT bandwidth-slower) |
| Repacked Q8_0 mat-vec bandwidth | ~800 GB/s effective (28.6 GB -> 17.9 ms of 26.9 ms 27B token; older denominators 650-715 GB/s) | per card | B | 2026-11-21 | Trace + Q3 microbench | Roofline for decode work kernels |
| Wave dispatch rate / round boundary | ~30 waves/us (~33 ns/wave); round boundary ~2 us | MI60 | B | 2026-11-22 | [bench/q1_occupancy.cpp](bench/q1_occupancy.cpp) | Tiny-block high-grid dispatch penalty |
| Host-staged DMA duplex (`hipMemcpyAsync` + sync) | 17 us + P / 3.46 GB/s (>=160 KB); flat 25-41 us (<=40 KB; 20 KB fastest at 24.8 us) | both | B | 2026-11-22 | [bench/q3_exchange.cpp](bench/q3_exchange.cpp) | Prefill DMA AllReduce floor (NOT kernel-polled decode AR) |
| Idle stream sync vs kernel + sync | `hipStreamSynchronize` idle: 0.17 us; trivial kernel + sync: 9.7 us (MI60), 11.7 us (MI50) | both | B | 2026-11-22 | [bench/q3_exchange.cpp](bench/q3_exchange.cpp) | Host sync round-trip cost |
| Thermal drift | 15-30% slowdown when hot (98-104 C -> 938 MHz); start A/B below 60 C, alternate A B B A | both | B | 2026-11-21 | Thermal logs & 4x16K false regression | Mandatory validity rule for all A/B runs |
| Run-to-run TG noise | 27B: +-0.3 t/s (0.8%) cool; Flash Next: +-1 t/s (2%, PP +-18 t/s); Inkling: KLD shifts 0.0049 on ubatch change alone | both | B | 2026-10..11 | Alternated sweeps | Minimum detectable A/B signal |

## Skeptical corrections (2026-11-22)

1. **Graph-replayed launch cost vs 2.2 us**: Decode on all three models runs inside captured HIP graphs. Pure graph-node launch floor measured 1.22 us/node (B, unsaved scratch microbench), and Inkling ABBA kill-switch tests saved 1.5-1.9 us per removed launch (which includes trivial kernel work). Pricing fusion at 2.2 us/launch over-estimates graph-replayed decode savings by ~1.2x-1.8x. Use **1.22 us** (pure launch floor) to **1.5-1.7 us** (launch + tiny work) for graph-replayed decode.
2. **AllReduce floor category error**: Q3 measured explicit `hipMemcpyAsync` DMA transfers plus host `hipStreamSynchronize` (25-41 us below 40 KB). Decode `ggml_cuda_ar_kernel` does *not* use `hipMemcpyAsync` or host syncs -- it polls mapped pinned host memory directly from the GPU (which is why it already runs in 20 us traced on 27B vs 40 us for Q3 DMA at 2.5-10 KB). Q3 therefore bounds **prefill DMA staging**, not kernel-polled decode AllReduce (`AR-LL` is not ruled out by Q3, though its decode ceiling remains modest at ~0.5-0.6 ms/token).
3. **MI60 (ROCm0) vs MI50 (ROCm1) asymmetry reconciled**:
   - **HBM2 streaming (decode weights / KV reads)**: equal (~891 GB/s MI60 vs 904 GB/s MI50; 27B mat-vec is only 1.1% slower on MI50).
   - **Compute-bound PP GEMM**: MI50 is ~21% slower (60 CUs / 178 W cap vs 64 CUs / 225 W).
   - **Small-kernel dispatch**: MI50 is +0.6 us slower bare (2.6 vs 2.0 us) and 20-40% slower on tiny decode kernels (suspected chipset PCIe lanes; test with `K2`).
   - **Placement implication**: Bandwidth-heavy work (KV cache, decode-heavy weights) runs fine on MI50; compute-heavy PP or long chains of tiny serial kernels suffer on MI50.

## Calibrations (2026-11-22)

All three calibrations are grade B microbenchmarks (~8 GPU-minutes total). Sources are in `gfx906/bench/`.

### Q1: Occupancy (blocks resident per CU)

Source: [bench/q1_occupancy.cpp](bench/q1_occupancy.cpp). Each block spins on `wall_clock64` (25 MHz) for D us while N float accumulators stay live (so time measures residency rounds, independent of SIMD ALU sharing). Swept G=16..4096 plus an atomic peak-residency counter, `hipFuncGetAttributes`, `hipOccupancyMaxActiveBlocksPerMultiprocessor`, and `rocprofv3` VGPR/LDS columns. Variant `Tt_Nn_Lk_Mm`: T threads, N live accumulators, k KB static LDS, M = 2nd arg of `__launch_bounds__(T, M)`.

| Variant | VGPR (rocprof) | numRegs (HIP attr) | LDS B | Scratch B | T | API occ | Counter b/CU (D=3ms) | Counter (D=300us) | Sweep R/CU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T64_N8_L0K_M1 | 16 | 15 | 0 | 0 | 64 | 32 | 25.33 | 19.39 | [12,16) |
| T64_N32_L0K_M1 | 36 | 35 | 0 | 0 | 64 | 28 | 28.00 | 25.16 | [12,16) |
| T64_N64_L0K_M1 | 100 | 98 | 0 | 0 | 64 | 8 | 8.00 | 8.00 | [8,12) |
| T64_N100_L0K_M1 | 152 | 152 | 0 | 0 | 64 | 4 | 4.00 | 4.00 | [4,6) |
| T64_N140_L0K_M1 | 212 | 212 | 0 | 0 | 64 | 4 | 4.00 | 4.00 | [4,6) |
| T64_N200_L0K_M1 | 256 | 256 | 0 | 0 | 64 | 4 | 4.00 | 4.00 | [4,6) |
| T128_N8_L0K_M1 | 16 | 15 | 0 | 0 | 128 | 16 | 12.56 | 12.22 | [4,6) |
| T128_N32_L0K_M1 | 36 | 35 | 0 | 0 | 128 | 14 | 14.00 | 14.00 | [8,12) |
| T128_N64_L0K_M1 | 100 | 98 | 0 | 0 | 128 | 4 | 4.00 | 4.00 | [4,6) |
| T128_N100_L0K_M1 | 152 | 152 | 0 | 0 | 128 | 2 | 2.00 | 2.00 | [2,3) |
| T128_N140_L0K_M1 | 212 | 212 | 0 | 0 | 128 | 2 | 2.00 | 2.00 | [2,3) |
| T128_N200_L0K_M1 | 256 | 256 | 0 | 0 | 128 | 2 | 2.00 | 2.00 | [2,3) |
| T256_N8_L0K_M1 | 16 | 15 | 0 | 0 | 256 | 8 | 7.08 | 7.95 | [3,4) |
| T256_N32_L0K_M1 | 36 | 35 | 0 | 0 | 256 | 7 | 7.00 | 7.00 | [3,4) |
| T256_N64_L0K_M1 | 100 | 98 | 0 | 0 | 256 | 2 | 2.00 | 2.00 | [2,3) |
| T256_N100_L0K_M1 | 152 | 152 | 0 | 0 | 256 | 1 | 1.00 | 1.00 | [1,2) |
| T256_N140_L0K_M1 | 212 | 212 | 0 | 0 | 256 | 1 | 1.00 | 1.00 | [1,2) |
| T256_N200_L0K_M1 | 256 | 256 | 0 | 0 | 256 | 1 | 1.00 | 1.00 | [1,2) |
| T1024_N8_L0K_M1 | 16 | 15 | 0 | 0 | 1024 | 2 | 2.00 | 2.00 | [1,2) |
| T1024_N32_L0K_M1 | 36 | 36 | 0 | 0 | 1024 | 1 | 1.00 | 1.00 | [1,2) |
| T1024_N64_L0K_M1 | 64 | 64 | 0 | 12 | 1024 | 1 | 1.00 | 1.00 | [1,2) |
| T1024_N100_L0K_M1 | 64 | 64 | 0 | 240 | 1024 | 1 | 1.00 | 1.00 | [1,2) |
| T1024_N140_L0K_M1 | 64 | 64 | 0 | 448 | 1024 | 1 | 1.00 | 1.00 | [1,2) |
| T1024_N200_L0K_M1 | 64 | 64 | 0 | 792 | 1024 | 1 | 1.00 | 1.00 | [0.25,1) |
| T64_N8_L4K_M1 | 52 | 16 | 4096 | 0 | 64 | 16 | 16.00 | 16.00 | [16,24) |
| T64_N8_L16K_M1 | 132 | 16 | 16384 | 0 | 64 | 4 | 4.00 | 4.00 | [4,6) |
| T64_N8_L32K_M1 | 132 | 16 | 32768 | 0 | 64 | 2 | 2.00 | 2.00 | [2,3) |
| T64_N8_L48K_M1 | 132 | 16 | 49152 | 0 | 64 | 1 | 1.00 | 1.00 | [1,2) |
| T256_N8_L4K_M1 | 16 | 16 | 4096 | 0 | 256 | 8 | 8.00 | 8.00 | [4,6) |
| T256_N8_L16K_M1 | 52 | 16 | 16384 | 0 | 256 | 4 | 4.00 | 4.00 | [4,6) |
| T256_N8_L32K_M1 | 88 | 16 | 32768 | 0 | 256 | 2 | 2.00 | 2.00 | [2,3) |
| T256_N8_L48K_M1 | 132 | 16 | 49152 | 0 | 256 | 1 | 1.00 | 1.00 | [1,2) |
| T256_N200_L0K_M2 | 128 | 128 | 0 | 464 | 256 | 2 | 2.00 | 2.00 | [2,3) |
| T256_N200_L0K_M4 | 64 | 64 | 0 | 792 | 256 | 4 | 4.00 | 4.00 | [1,2) |
| T256_N200_L0K_M8 | 32 | 32 | 0 | 808 | 256 | 8 | 8.00 | 8.00 | [2,3) |
| T128_N200_L0K_M2 | 128 | 128 | 0 | 464 | 128 | 4 | 4.00 | 4.00 | [4,6) |
| T128_N200_L0K_M4 | 64 | 64 | 0 | 792 | 128 | 8 | 8.00 | 8.00 | [3,4) |
| T128_N200_L0K_M8 | 32 | 32 | 0 | 808 | 128 | 16 | 16.00 | 16.00 | [2,3) |
| T64_N200_L0K_M2 | 128 | 128 | 0 | 464 | 64 | 8 | 8.00 | 8.00 | [8,12) |
| T64_N200_L0K_M4 | 64 | 64 | 0 | 792 | 64 | 16 | 16.00 | 16.00 | [6,8) |
| T64_N200_L0K_M8 | 32 | 32 | 0 | 808 | 64 | 32 | 32.00 | 32.00 | [4,6) |

**Exact occupancy formula (matches HIP API and hardware counter everywhere not dispatch-limited)**:
- VGPR pool per SIMD = 256, allocation granule = 4 (`vgpr_alloc = ceil(V / 4) * 4`).
- `waves_per_simd = min(8, floor(256 / vgpr_alloc))` -- hardware cap is **8 waves/SIMD (32 waves/CU)**, not 10. Thresholds: `<=32 -> 8`, `<=36 -> 7`, `<=40 -> 6`, `<=48 -> 5`, `<=64 -> 4`, `<=84 -> 3`, `<=128 -> 2`, `>128 -> 1`.
- `waves_per_block = ceil(T / 64)`.
- `blocks_per_cu = min(floor(4 * waves_per_simd / waves_per_block), floor(65536 / LDS_bytes))`.
- Whole-GPU resident blocks `R = blocks_per_cu * CU_count` (64 on MI60, 60 on MI50). For grid `G > R`, time scales cleanly as `ceil(G / R) * D` with ~2 us per round boundary.
- Dispatch rate ceiling: ~30 waves/us (~33 ns/wave), which limits tiny-block high-grid cases (T64 N8 G=4096 takes 1.55 ms).
- **`__launch_bounds__(T, M)` rule**: on HIP, `M` means **min waves per SIMD**, not blocks per CU. It caps VGPRs at `floor(256 / M)` (`M=2/4/8 -> 128/64/32 VGPR`). With `T=1024`, even `M=1` caps VGPRs at 64 (16 waves/CU = 4/SIMD) and spills if live state exceeds 64. Spills hurt sharply (464/792/808 B scratch added +12/+25/+43% wall time even in a spin loop).
- Note: `rocprofv3` reports inflated VGPR counts on LDS variants (52/132 vs 16 from `hipFuncGetAttributes`); `scripts/kernel_budget.py` implements the exact formula above directly on trace columns.

### Q2: Streaming efficiency by geometry

Source: [bench/q2_stream.cpp](bench/q2_stream.cpp). Reads an int8 buffer of S bytes split into contiguous `S/G` slices across G blocks using unrolled-4 `uint4` (16 B) loads. `chunk` = coalesced (`j, j+T, ...`); `run` = per-thread contiguous run. Modes: 0 add words, 1 int8->float FMA (16 FMA / 16 B), 2 F16->float, 3 `v_dot4` (sdot4).

Peak: **891 GB/s** on MI60 (S=256 MB, G=8192, T=1024, 301.4 us) and **904 GB/s** on MI50 (G=8192, T=512, 297 us) = 87-88% of 1024 GB/s spec. Dev1 tracks within 5% of dev0 across geometries.

GB/s on MI60, S = 32 MB, coalesced `chunk` pattern (rows = G blocks, cols = T threads):

| G \ T | 64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 319 | 549 | 636 | 605 | 591 |
| 128 | 498 | 580 | 555 | 543 | 586 |
| 256 | 371 | 410 | 394 | 591 | 428 |
| 512 | 315 | 323 | 571 | 589 | 637 |
| 1024 | 341 | 580 | 546 | 723 | 820 |
| 2048 | 513 | 707 | 700 | 753 | 853 |
| 8192 | 757 | 757 | 840 | 693 | 359 |

- **S = 256 MB (`chunk`)**: only `(8192, 1024)` (891 GB/s) and `(8192, 512)` (872 GB/s) exceed 90% of peak (`>=802 GB/s`); `(2048, 1024)` reaches 792 GB/s (89%). Nothing with `G <= 512` exceeds 83% (741 GB/s).
- **S = 3.5 MB**: best is 5.01 us at `(512, 128)` (732 GB/s, L2-resident on repeats; expect >=6.1 us cold). Deterministic unexplained slowdown to 82 GB/s (44 us) whenever `G * T == 131072` at 3.5 MB -- avoid those exact geometries. Huge grids `(8192, 1024)` collapse to 91 us (dispatch cost).
- **Access pattern & compute**: uncoalesced `run` pattern loses 2-10x at small `G * T` and never beats `chunk`. At peak geometry, FMA (889), `v_dot4` (891), and F16 (892) match pure loads (891); at low occupancy `(1024, 128)`, 16 FMA/16 B drops to 317 vs 523 GB/s (`v_dot4`/add).
- **Practical rule**: reaching >=90% HBM bandwidth requires >=~32 MB per launch with coalesced 16 B loads, `G >= 2048 x T=1024` (or `8192 x 256`), 4-16 KB per block.

### Q3: Cross-GPU host-staged exchange (DMA)

Source: [bench/q3_exchange.cpp](bench/q3_exchange.cpp). Pinned `hipHostMalloc` staging buffers, one stream per device, single host thread (`hipSetDevice` pair = 0.10 us). Median of 200 repeats (20 at 16 MB), microseconds:

| Payload P | D2H dev0 | H2D dev1 | D2H dev1 | H2D dev0 | One-way 0->h->1 | Duplex | Kernel+duplex+kernel | Duplex pageable | `hipMemcpyPeer` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 KB | 11.4 | 14.7 | 13.6 | 12.0 | 37.0 | 39.9 | 105 | 94.7 | 73.4 |
| 10 KB | 13.2 | 16.7 | 14.4 | 13.9 | 37.9 | 41.3 | 75.7 | 91.0 | 55.8 |
| 20 KB | 9.5 | 12.2 | 10.4 | 11.0 | 30.0 | 24.8 | 67.3 | 51.2 | 44.3 |
| 40 KB | 13.3 | 14.9 | 13.2 | 14.3 | 33.1 | 30.6 | 73.3 | 67.4 | 49.7 |
| 160 KB | 30.6 | 32.0 | 30.3 | 32.3 | 67.2 | 65.1 | 108.5 | 142.5 | 88.9 |
| 1 MB | 154.3 | 155.8 | 154.1 | 162.5 | 347.6 | 320.5 | 372.0 | 710.4 | 368.1 |
| 16 MB | 2354.7 | 2357.5 | 2357.1 | 2472.0 | 4859.6 | 4868.7 | 5105.5 | 9832.5 | 5022.1 |

- **Fit (>=160 KB)**: D2H ~7.15 GB/s both cards; H2D 7.14 GB/s (dev1) / 6.8 GB/s (dev0); duplex = **17 us + P / 3.46 GB/s** (two sequential 7 GB/s hops; both directions overlap with zero slowdown vs one-way).
- **Small sizes (<=40 KB)**: flat and non-monotone (20 KB fastest at 24.8 us duplex vs 40-41 us at 2.5-10 KB; likely driver copy-path switch around 16 KB).
- **Memory & peer API**: pinned memory is 2x faster than pageable. `hipDeviceCanAccessPeer == 0` in both directions, yet `hipMemcpyPeer*` succeeds by host-staging internally at the same one-way speed (no advantage over manual duplex staging).
- **Scope**: bounds DMA-staged prefill exchanges; decode uses GPU-kernel-polled mapped host memory (see [Skeptical corrections](#skeptical-corrections-2026-11-22)).

## Rules derived from the card

### The one-block rule

One thread block accesses ~1/30 of the GPU (~26-30 GB/s). A row-reduction kernel running as one block costs `~2.9 us + 0.46 us / 1,000 floats`:
- Under ~5,000 floats per block: one block of **1024 threads** is fine (256 threads costs ~2x more).
- Above ~5,000 floats per block: split into two launches (partial sums across many blocks, then final pass across many blocks reading partials) = **~5.4 us flat** up to 36,864 floats.
- Never stream weights through one block (the one-block Inkling loop controller streamed 2.4 MB / 600K floats -> ~630 us).
- Count dependent reduce-then-use phases (~1.6 us), not barriers (~0.08 us).
- Always read launch geometry (`blocks`, `VGPR`, `LDS`, `scratch`) from the trace before naming a mechanism.

### Value of a removed launch

In a captured HIP graph, removing one launch per token saves **1.22 us** (pure graph-node floor) up to **1.5-1.9 us** (including trivial kernel work, ABBA). Check both bounds before building small fusions; skip anything predicting <1% total speedup.

## Checks against real kernels

Pre-rule checks (grade C prediction vs traced us; remember `rocprofv3` adds ~1.2 us/kernel):

| Kernel | Card prediction | Traced |
| --- | --- | --- |
| 27B `add_rms_norm_mul_f32` (no Q8_1, 5120 floats, 3 inputs) | ~8.7 us | 8.2 us |
| Inkling `res_mix_f32` (~11.5K floats read twice) | ~12 us | 16.2 us |
| One-block loop controller (~600K floats) | 160-280 us | ~630 us |
| `flash_attn_ext_vec` @ 4K ctx (if 1 block/head, 2.1 MB) | ~70 us | 85.8 us |

Post-rule checks (2026-11-22):

| Kernel | Prediction | Result | Lesson |
| --- | --- | --- | --- |
| Two-launch `res_mix_f32` | ~5.4 us (~7.8 us traced) | 17.0 -> 9.5 us traced | Split form won big (+2.2% Inkling TG); block size still untuned (`BA-5`) |
| Two-launch `add_rms_norm_mul_f32` | ~7.8 us traced | 11.37 -> 9.05 us traced | +1.6% 27B TG (grade A); block size still untuned (`BA-5`) |
| `flash_attn_ext_vec` @ 4K ctx | ~70 us vs 85.8 us | Wrong mechanism: kernel already split keys over blocks; bottleneck was **>128 VGPRs -> 2 b/CU** | Fixed by `__launch_bounds__(128, 4)` (`efb85a1a5`, +16-21% Inkling 4K/16K TG). Always check VGPR/occupancy first |

## Cross-GPU portability (gfx906 vs newer GPUs)

Reference matrix for judging external PRs and optimizations (grade D spec background unless linked to [PERFORMANCE.md](PERFORMANCE.md)):

| Feature | MI50 / MI60 (gfx906) | Radeon R9700 (gfx1201) | RTX 3090 (sm_86) | RTX 5090 (sm_120) |
| --- | --- | --- | --- | --- |
| Memory | 32 GB HBM2, ~1 TB/s | 32 GB GDDR6, ~640 GB/s | 24 GB GDDR6X, ~936 GB/s | 32 GB GDDR7, ~1.8 TB/s |
| Matrix units | None (vector ALU + `v_dot4`) | WMMA (FP16/BF16/INT8/FP8) | Tensor cores (FP16/BF16/INT8/INT4) | Tensor cores (+FP8/FP6/FP4) |
| Last-level cache | 4 MB L2, 64 KB LDS/CU | ~64 MB Infinity Cache | 6 MB L2 | ~96 MB L2 |
| Execution width | wave64 (DPP / `ds_swizzle`) | wave32 (wave64 opt) | warp32 | warp32 |
| Async copy | None (stage via VGPRs) | RDNA4 async loads | `cp.async` | TMA bulk async |
| Interconnect (this host) | No P2P (MI50 on chipset lanes); host-staged AR | PCIe P2P | NVLink / P2P | PCIe P2P |

**What carries over to gfx906**:
- **YES**: Weight repack layouts, wide coalesced loads, fewer memory passes, kernel fusion, HIP graph capture, word-op rewrites of emulated byte-SIMD (`__vsub4` / `__vcmpne4` cost ~6 word ops on GCN -> S2 word ops gave +5.1% FN TG), software-dequantized repacked NVFP4 (layout variant of IQ4_NL), short speculative drafts (MTP2-3, DFlash5), tiled split-K small-M SGEMM, long-context KV tile skipping and GQA head grouping.
- **NO**: Tensor-core / MFMA / WMMA GEMM libraries (Marlin, CUTLASS, FA2/FA3, AITER), native FP8/FP4 arithmetic, large-cache persistence assumptions (only 4 MB L2), deep speculative verify trees, P2P custom AllReduce (no P2P on this board), warp32-hardcoded reductions (note HIP >= 7 `__ballot_sync(0xFFFFFFFF, p)` silently drops lanes 32-63), or quantized KV cache (excluded by the F16 KV policy in [GOALS.md](GOALS.md)).
- **RTX 3090 note**: `#if defined(GGML_USE_HIP) && defined(__gfx906__)` code compiles out cleanly on CUDA; generic fork wins (MTP replay, DFlash taps, split KV/rollback fixes, tied head / shared-expert TP split, host n-gram placement) still apply.

## Calibration microbenchmarks

Standalone HIP sources in `gfx906/bench/` (no model loads, compile in seconds):

```bash
# Optional: source your ROCm / TheRock environment if not on PATH
# source "${THEROCK_ENV:-/path/to/therock.sh}"
clang++ -x hip --offload-arch=gfx906 -O3 -std=c++17 gfx906/bench/<file>.cpp -pthread -o /tmp/bench_bin
```

| File | Measures |
| --- | --- |
| [bench/q1_occupancy.cpp](bench/q1_occupancy.cpp) | VGPR / LDS / `__launch_bounds__` residency, dispatch rate, round boundaries |
| [bench/q2_stream.cpp](bench/q2_stream.cpp) | HBM2 streaming GB/s across buffer sizes, `(G, T)` grids, chunk vs run, FMA/dot4/F16 |
| [bench/q3_exchange.cpp](bench/q3_exchange.cpp) | Host-staged D2H/H2D/duplex DMA latency and bandwidth, pinned vs pageable, peer copy |
| [bench/one_block.cpp](bench/one_block.cpp) | One-block cost vs float count, threads/block, barriers vs phases, two-kernel reduction |
| [bench/launch_floor.cpp](bench/launch_floor.cpp) | Bare HIP launch cost on MI60/MI50, geometry invariance, multi-stream launch throughput |
