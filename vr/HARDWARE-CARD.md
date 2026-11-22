# Hardware card (gfx906)

The single source of hardware constants for the two gfx906 cards (ROCm0 MI60, ROCm1 MI50). Other documents link here and do not repeat the values. Cross-GPU comparison and background: [HARDWARE-GUIDE.md](HARDWARE-GUIDE.md) (it links back here for measured constants). How the card is used: [PROCESS.md](PROCESS.md).

## What the card is and when to rerun it

The card is a small table of constants and curves from synthetic microbenchmarks: no model loads, seconds each. Every predicted saving in the [worklist](PERFORMANCE.md#worklist) is computed from it. Rerun it only after a ROCm update, a driver or hardware change, or for a genuinely new hard question. A mismatch above about 20% between a prediction and a trace is not an error to hide: it updates the card (see [Checks of the card against real kernels](#checks-of-the-card-against-real-kernels)).

## Constants

Grades: A alternated A/B on a real model, B single measurement or microbench, C estimate. Sources are under [investigation-notes](investigation-notes/).

| Quantity | Value | Device | Grade | Date | How measured | Used for |
| --- | --- | --- | --- | --- | --- | --- |
| Launch cost in the app | about 2.2 us per kernel; 1.5 to 1.9 us saved per launch removed | two gfx906 | A | 2026-11-21 | three ABBA tests with kill switches on Inkling 2 Mini ([record](PERFORMANCE-RECORDS.md#2026-11-21-inkling-2-mini-late-nvfp4-repacked-experts-banded-prefill-attention-faster-mix-kernels)) | value of a removed launch: launches x 2.2 us |
| Bare launch cost, HIP loop | 2.0 us | MI60 | B | 2026-11-21 | [launch_floor_bench.cpp](investigation-notes/2026-11-21-inkling2-mini/launch_floor_bench.cpp) | launch floor |
| Bare launch cost, HIP loop | 2.6 us | MI50 | B | 2026-11-21 | same | ROCm1 small-kernel lag (K2 in the worklist) |
| Cost per node in a replayed graph | 1.22 us | not recorded | B | 2026-11-22 | scratch microbench, 1000 chained nodes in a captured graph; the source file is not saved under vr yet, so rerun before relying on it | what a graph saves per node against a plain launch |
| Profiler cost | adds about 1.2 us per kernel; shows a 4.2 us floor | two gfx906 | B | 2026-11-21 | `rocprofv3 --kernel-trace` against the bare loop | convert trace time to app time |
| Launch cost against geometry | unchanged with grid size up to 64 blocks, threads per block or argument count (kernels with real reductions are longer) | MI60 | B | 2026-11-21 | [launch_floor_bench.cpp](investigation-notes/2026-11-21-inkling2-mini/launch_floor_bench.cpp) | a trivial kernel costs the floor at any small geometry |
| More streams | 1.45x launch throughput from one host thread; 1.9x with a host thread per stream | MI60 | B | 2026-11-21 | same | branch parallelism (each join costs an event wait) |
| One thread block | behaves like about 1/30 of the GPU (26 to 30 GB/s against about 800 GB/s for the card) | MI60 | B | 2026-11-22 | [single_block_bench.cpp](investigation-notes/2026-11-22-one-block-rule/single_block_bench.cpp) | the one-block rule |
| One-block cost, elementwise | about 2.1 us + 0.26 us per 1000 floats touched (1024 threads) | MI60 | B | 2026-11-22 | same | cost formula 1 |
| One-block cost, row reduction and second pass | about 2.9 us + 0.46 us per 1000 floats | MI60 | B | 2026-11-22 | same | cost formula 2 |
| Same work over many blocks | 2.3 to 2.8 us at any size tested (up to 72 blocks of 1024 threads) | MI60 | B | 2026-11-22 | same | what a split buys |
| Threads per block, one block | 4608 floats: 64 threads 18.1 us, 256 threads 6.30 us, 1024 threads 3.39 us | MI60 | B | 2026-11-22 | same | block size choice |
| Barrier | about 0.08 us each | MI60 | B | 2026-11-22 | same | barriers are not the cost |
| Dependent reduce-then-use phase | about 1.6 us each | MI60 | B | 2026-11-22 | same | count phases, not barriers |
| Two-kernel reduction (partial sums, then many blocks read the partials) | about 5.4 us for both launches up to 36864 floats; cheaper than one block above about 5000 floats | MI60 | B | 2026-11-22 | same | split form of a row-wide fused op |
| Repacked mat-vec bandwidth | about 800 GB/s | per card | B | 2026-11-21 | 28.6 GB at 800 GB/s is 17.9 ms of the 26.9 ms 27B token; other values and their denominators are in [Numbers that disagree](PERFORMANCE.md#numbers-that-disagree) | roofline for work kernels |
| Thermal drift | benchmarks drift 15 to 30% when the GPUs are hot; GPUs reached 98 to 104 C and 938 MHz in long sessions (noted 2026-11-15); start A/B below 60 C and alternate A B B A | both | B | 2026-11-21 | temperature logs; the 4x16K false regression ([records](PERFORMANCE-RECORDS.md#2026-11-21-catch-up-to-official-46eb1742d)) | validity rule for any end-to-end run |
| TG noise, 27B Q8_0 | about +-0.3 t/s (0.8%), alternated and cool | two gfx906 | B | 2026-10 | older notes, not repeated after the drift finding | smallest effect an A/B can show |
| TG noise, Flash Next | about 1 t/s (2%); one-repeat baselines ranged 33.3 to 40.1 t/s and PP varied up to about 18 t/s | two gfx906 | B | 2026-10-29 | A1 sweep ([record](PERFORMANCE-RECORDS.md#2026-10-29-a1-config-sweep-tg)) | use r5 alternating or a trace |
| TG noise, Inkling 2 Mini | not measured as t/s noise (gap); KLD moves by 0.0049 when only the ubatch size changes | ROCm0 | B | 2026-11-21 | 2 chunks | measure when an Inkling A/B needs it |

## Calibrations

Three open questions that the card must answer for the occupancy scan and the exchange rule. They are filled from the 2026-11-22 microbenchmarks.

### Occupancy (blocks resident per compute unit)

Measured 2026-11-22: results below.
Grade B (microbench, 2026-11-22, about 8 GPU-minutes for all three). Sources and raw tables: [investigation-notes/2026-11-22-hardware-card](investigation-notes/2026-11-22-hardware-card/) (`q1.cpp`, `run.sh`, `RESULTS.md`).

Method: each block spins on wall_clock64 (25 MHz) for D us while N float accumulators stay live. Duration is then independent of how many waves share a SIMD (a pure FMA loop does NOT work: one wave already saturates a SIMD, so time scales with total waves, not rounds).
Sweep G = 16..4096 (D=300us, median of 5 x 10 launches), plus an exact residency counter: each block atomicAdd's a global counter at start and atomicSub at end; atomicMax of it = true resident blocks (G=4096).
Also hipFuncGetAttributes, hipOccupancyMaxActiveBlocksPerMultiprocessor, and rocprofv3 kernel-trace VGPR/LDS (q1.vgpr).
Variant name Tt_Nn_Lk_Mm: t threads, n accumulators, k KB static LDS, m = 2nd arg of __launch_bounds__(T,m) (always present; m=1 baseline).

Table (dev0, 64 CU). Counter = measured peak resident blocks per CU. Sweep range = R/CU between last G with one round and first G with time >1.5x.

| variant | VGPR rocprof | numRegs (hipFuncAttr) | LDS B | scratch B | T | API occ | counter blocks/CU (D=3000us) | counter (D=300us) | sweep R/CU range |
|---|---|---|---|---|---|---|---|---|---|
| T64_N8_L0K_M1 | 16 | 15 | 0 | 0 | 64 | 32 | 25.33/CU | 19.39/CU | [12,16) |
| T64_N32_L0K_M1 | 36 | 35 | 0 | 0 | 64 | 28 | 28.00/CU | 25.16/CU | [12,16) |
| T64_N64_L0K_M1 | 100 | 98 | 0 | 0 | 64 | 8 | 8.00/CU | 8.00/CU | [8,12) |
| T64_N100_L0K_M1 | 152 | 152 | 0 | 0 | 64 | 4 | 4.00/CU | 4.00/CU | [4,6) |
| T64_N140_L0K_M1 | 212 | 212 | 0 | 0 | 64 | 4 | 4.00/CU | 4.00/CU | [4,6) |
| T64_N200_L0K_M1 | 256 | 256 | 0 | 0 | 64 | 4 | 4.00/CU | 4.00/CU | [4,6) |
| T128_N8_L0K_M1 | 16 | 15 | 0 | 0 | 128 | 16 | 12.56/CU | 12.22/CU | [4,6) |
| T128_N32_L0K_M1 | 36 | 35 | 0 | 0 | 128 | 14 | 14.00/CU | 14.00/CU | [8,12) |
| T128_N64_L0K_M1 | 100 | 98 | 0 | 0 | 128 | 4 | 4.00/CU | 4.00/CU | [4,6) |
| T128_N100_L0K_M1 | 152 | 152 | 0 | 0 | 128 | 2 | 2.00/CU | 2.00/CU | [2,3) |
| T128_N140_L0K_M1 | 212 | 212 | 0 | 0 | 128 | 2 | 2.00/CU | 2.00/CU | [2,3) |
| T128_N200_L0K_M1 | 256 | 256 | 0 | 0 | 128 | 2 | 2.00/CU | 2.00/CU | [2,3) |
| T256_N8_L0K_M1 | 16 | 15 | 0 | 0 | 256 | 8 | 7.08/CU | 7.95/CU | [3,4) |
| T256_N32_L0K_M1 | 36 | 35 | 0 | 0 | 256 | 7 | 7.00/CU | 7.00/CU | [3,4) |
| T256_N64_L0K_M1 | 100 | 98 | 0 | 0 | 256 | 2 | 2.00/CU | 2.00/CU | [2,3) |
| T256_N100_L0K_M1 | 152 | 152 | 0 | 0 | 256 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T256_N140_L0K_M1 | 212 | 212 | 0 | 0 | 256 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T256_N200_L0K_M1 | 256 | 256 | 0 | 0 | 256 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T1024_N8_L0K_M1 | 16 | 15 | 0 | 0 | 1024 | 2 | 2.00/CU | 2.00/CU | [1,2) |
| T1024_N32_L0K_M1 | 36 | 36 | 0 | 0 | 1024 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T1024_N64_L0K_M1 | 64 | 64 | 0 | 12 | 1024 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T1024_N100_L0K_M1 | 64 | 64 | 0 | 240 | 1024 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T1024_N140_L0K_M1 | 64 | 64 | 0 | 448 | 1024 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T1024_N200_L0K_M1 | 64 | 64 | 0 | 792 | 1024 | 1 | 1.00/CU | 1.00/CU | [0.25,1) |
| T64_N8_L4K_M1 | 52 | 16 | 4096 | 0 | 64 | 16 | 16.00/CU | 16.00/CU | [16,24) |
| T64_N8_L16K_M1 | 132 | 16 | 16384 | 0 | 64 | 4 | 4.00/CU | 4.00/CU | [4,6) |
| T64_N8_L32K_M1 | 132 | 16 | 32768 | 0 | 64 | 2 | 2.00/CU | 2.00/CU | [2,3) |
| T64_N8_L48K_M1 | 132 | 16 | 49152 | 0 | 64 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T256_N8_L4K_M1 | 16 | 16 | 4096 | 0 | 256 | 8 | 8.00/CU | 8.00/CU | [4,6) |
| T256_N8_L16K_M1 | 52 | 16 | 16384 | 0 | 256 | 4 | 4.00/CU | 4.00/CU | [4,6) |
| T256_N8_L32K_M1 | 88 | 16 | 32768 | 0 | 256 | 2 | 2.00/CU | 2.00/CU | [2,3) |
| T256_N8_L48K_M1 | 132 | 16 | 49152 | 0 | 256 | 1 | 1.00/CU | 1.00/CU | [1,2) |
| T256_N200_L0K_M2 | 128 | 128 | 0 | 464 | 256 | 2 | 2.00/CU | 2.00/CU | [2,3) |
| T256_N200_L0K_M4 | 64 | 64 | 0 | 792 | 256 | 4 | 4.00/CU | 4.00/CU | [1,2) |
| T256_N200_L0K_M8 | 32 | 32 | 0 | 808 | 256 | 8 | 8.00/CU | 8.00/CU | [2,3) |
| T128_N200_L0K_M2 | 128 | 128 | 0 | 464 | 128 | 4 | 4.00/CU | 4.00/CU | [4,6) |
| T128_N200_L0K_M4 | 64 | 64 | 0 | 792 | 128 | 8 | 8.00/CU | 8.00/CU | [3,4) |
| T128_N200_L0K_M8 | 32 | 32 | 0 | 808 | 128 | 16 | 16.00/CU | 16.00/CU | [2,3) |
| T64_N200_L0K_M2 | 128 | 128 | 0 | 464 | 64 | 8 | 8.00/CU | 8.00/CU | [8,12) |
| T64_N200_L0K_M4 | 64 | 64 | 0 | 792 | 64 | 16 | 16.00/CU | 16.00/CU | [6,8) |
| T64_N200_L0K_M8 | 32 | 32 | 0 | 808 | 64 | 32 | 32.00/CU | 32.00/CU | [4,6) |

#### Fit (matches the API occupancy exactly wherever the counter is not dispatch-limited)
- VGPR pool per SIMD: 256. Allocation granule 4 (35 -> 36, 98 -> 100). Waves per SIMD = min(8, floor(256 / VGPR)). The cap is 8 per SIMD (32 per CU), not 10: T64_N200_M8 (32 VGPR) reaches exactly 32 blocks/CU = 2048 resident; 36 VGPR gives exactly 28/CU (7 waves per SIMD).
- LDS limit per CU: 64 KB (48 KB -> 1 block, 32 KB -> 2, 16 KB -> 4, 4 KB -> 16).
- Waves per block = ceil(T/64). Blocks per CU = min( floor(4*min(8,floor(256/V)) / ceil(T/64)), floor(65536/L) ). Checks: T256 V=36 -> 7 (measured 7.00); T128 V=36 -> 14 (14.00); T64 V=100 -> 8 (8.00); T1024 V<=32 -> 2 (2.00).
- Tiny-block cases (T64 15 VGPR, API 32/CU) do not reach R in the counter (19-25 per CU at D=300us, 25 at D=3000us): the dispatcher launches about 30 waves/us (T64 ~23 blocks/us, T256 ~8.6 blocks/us, T1024 ~1.8 blocks/us). Time vs G is distorted there too (T64 N8 G=4096: 1.55 ms, not 5 rounds x 0.3 ms). Cost rule: about 33 ns per wave dispatched, however many fit.
- A grid larger than R: time = ceil(G/R) x D, verified exactly (T256 V=256 R=64: 304 us at G<=64, 604 at G=128, 905 at G=192, 1206 at G=256, D=300us). A round boundary costs about 2 us.
- Whole GPU resident blocks = 64 (dev0) or 60 (dev1) x blocks per CU. VGPR to waves/SIMD: <=32 -> 8, <=36 -> 7, <=40 -> 6, <=48 -> 5, <=64 -> 4, <=84 -> 3, <=128 -> 2, >128 -> 1.
- LDS rows: rocprof reports odd VGPR/SGPR for the LDS variants (52 or 132 VGPR, 112 SGPR) while hipFuncGetAttributes says 16 VGPR. Not resolved. The 64 KB conclusion does not depend on it: T64 at 48 KB and 32 KB can only be explained by LDS (132 VGPR would allow 4 blocks/CU).

#### __launch_bounds__
- With T=1024 the baseline __launch_bounds__(1024,1) already caps VGPR at 64 (16 waves in a CU = 4 per SIMD) and spills when more is live: N=64 12 B scratch, N=100 240 B, N=200 792 B (one block takes 565 us instead of 304 for N=200, D=300). A launch with 1024 threads can never use more than 64 VGPRs.
- The 2nd argument is "min waves per SIMD", not min blocks per CU: M=2/4/8 gives <=128/64/32 VGPR at every T (T64, T128, T256 all show 128/64/32). It raises resident blocks to the API/counter value (T256: 2/4/8; T128: 4/8/16; T64: 8/16/32 per CU) but the live state spills: 464/792/808 B scratch, one-block time 338/378/430 us (T64, D=300) = +12/25/43%.
- Practical rule: resident blocks per CU = min(floor(4*min(8,floor(256/V))/ceil(T/64)), floor(65536/L)); GPU = x64 (dev0) or x60 (dev1). __launch_bounds__(T,M) only caps registers at 256/M, it never adds residency beyond that cap and it spills when the live set does not fit. HIP's M means waves per SIMD.

### Streaming efficiency (bytes per second by geometry)

Measured 2026-11-22: results below.
Grade B (microbench, 2026-11-22, about 8 GPU-minutes for all three). Sources and raw tables: [investigation-notes/2026-11-22-hardware-card](investigation-notes/2026-11-22-hardware-card/) (`q2.cpp`, `run.sh`, `RESULTS.md`).

Method: int8 buffer of S bytes, G blocks each own a contiguous S/G slice, 16-byte uint4 loads (unroll 4), per-block sum written once, wall time per launch over about 50 ms x median of 5, one sync per batch, GB/s = S/time (decimal GB).
Pattern "chunk": thread j reads chunks j, j+T, j+2T.. (coalesced, mat-vec style). Pattern "run": thread j reads its own contiguous slice/T bytes.
Modes: 0 = add 32-bit words (memory only), 1 = int8 -> float, fma with broadcast weights (16 fma per 16 B), 3 = v_dot4 (sdot4), 2 = F16 -> float add. Full tables for all S, both patterns, both devices are in q2_dev0.out / q2_dev1.out.

Peak: dev0 891 GB/s (S=256 MB, G=8192, T=1024, 301.4 us; repeat 890.6-890.7, min/max 301.3/301.6 us); dev1 904 GB/s (G=8192, T=512, 297 us). That is 87-88% of the 1024 GB/s HBM2 spec.

GB/s, dev0, S = 32 MB, chunk pattern (rows G, columns T):

| G \ T | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| 64 | 319 | 549 | 636 | 605 | 591 |
| 128 | 498 | 580 | 555 | 543 | 586 |
| 256 | 371 | 410 | 394 | 591 | 428 |
| 512 | 315 | 323 | 571 | 589 | 637 |
| 1024 | 341 | 580 | 546 | 723 | 820 |
| 2048 | 513 | 707 | 700 | 753 | 853 |
| 8192 | 757 | 757 | 840 | 693 | 359 |

(S=32 MB: G=8192 gives a 4 KB slice = 256 chunks, so T=1024 leaves 3/4 of the threads idle: 359.) Repeat check (3 runs): 819.3-819.7, 853.0-853.3, 841.5, 359; all within 0.5%. Rows with 394-430 (G=256) are real, not noise.

GB/s, dev0, S = 256 MB, chunk (rows G, columns T 64..1024): G=1024: 241 523 658 706 713; G=2048: 337 554 698 746 792; G=8192: 636 753 793 872 891. G=64..512 sit at 349-741.
Which (G,T) reach 90% of peak (>= 802 GB/s): S=32 MB: (1024,1024) 820, (2048,1024) 853, (8192,256) 841. S=256 MB: only (8192,1024) 891 and (8192,512) 872 (and 2048x1024 792 = 89%). Nothing with G<=512 gets above 83% (741 GB/s).
"run" pattern (each thread own contiguous run): 2-10x slower at small G*T (S=256 MB G=64,T=64: 174 vs 349; T=512: 45 vs 736 GB/s), because lanes of a wave then touch addresses slice/T apart (uncoalesced 16-byte accesses). It only converges to the chunk pattern when G*T is huge (G=8192: 555-695 vs 793-891) and never beats it, except by tie for 3.5 MB.
S = 3.5 MB (3670016 B): best 5.01 us at G=512, T=128 (732 GB/s; repeats 5.00-5.05 us). Others: G=256 T=256 708, G=64 T=1024 705, G=1024 T=64 606. Compare 2.0 us + S/891 GB/s = 6.1 us: measured is FASTER, because the same 3.5 MB buffer is re-read each launch and fits in the 4 MB L2 (effective 2.0 + S/1.22 TB/s fits). For cold data expect 6.1 us or more. A set of (G,T) with G*T = 131072 (128x1024, 256x512, 512x256, 1024x128) shows a deterministic slowdown to 82 GB/s (44 us) at S=3.5 MB for both patterns: not explained (it did not happen at 32 or 256 MB); avoid those exact geometries at this size until understood.
Larger G*T at S=3.5 MB gets worse (G=8192 T=1024: 40 GB/s, 91 us) since every block pays launch/dispatch ~33 ns per wave.
(a) compute (modes 1 and 3) and (b) F16 (mode 2): at the best geometry the peak is unchanged: S=256 MB, G=8192 T=1024: 889 (fma) / 891 (dot4) / 892 (F16) vs 891; S=32 MB (1024,1024): 807/820/815 vs 820. With fewer threads the fma version loses: 256 MB (1024,128): fma 317 vs dot4 523 vs add 523; (1024,256): 469/674/658. So 16 fma per 16 B does bind when occupancy is low; with enough resident waves memory binds. F16: no cost.
(d) dev1, 256 MB: peak 904 GB/s (G=8192, T=512), table in q2_dev1.out is within 5% of dev0 elsewhere. Dev1 does not stream slower.

Practical rule: streaming needs about 2M threads x 16 B loads, i.e. >= ~32 MB per launch, to reach 90% of peak (~800 GB/s of the 891): use >= 2048 blocks of 1024 threads (or 8192 blocks of 256) with 4-16 KB per block (1-2 loads of 16 B per thread, slices up to 32 KB are fine). With G<=512 you get at most 83%. For small buffers (3.5 MB) the best is ~5 us from L2 at 512 blocks x 128 threads (7 KB per block); cold data would be 6.1 us; add the 2-3 us launch/graph node cost on top. Use the chunk (coalesced) pattern always: the run pattern loses up to 10x.

### Cross-GPU exchange (host staged)

Measured 2026-11-22: results below.
Grade B (microbench, 2026-11-22, about 8 GPU-minutes for all three). Sources and raw tables: [investigation-notes/2026-11-22-hardware-card](investigation-notes/2026-11-22-hardware-card/) (`q3.cpp`, `run.sh`, `RESULTS.md`).

Method: pinned (hipHostMalloc) host buffers, one stream per device, both devices in one process, a SINGLE host thread calling hipSetDevice before each call (a set-device pair costs 0.10 us, so two threads were not needed). Median of 200 repeats (20 for 16 MB), 5 warm-ups. Run 3 times (q3.out, q3_rep1.out, q3_rep2.out).
Duplex = D2H on both devices issued together, sync both, then H2D on both (each from the other device's buffer), sync both. One-way = D2H dev0, sync, H2D dev1, sync. Round (col 7) = write kernel on both devices, sync, duplex, read kernel on both, sync, so it includes 2 extra syncs and 2 launches.
Times in microseconds, medians of run 1 (other runs within 3% unless noted; outlier max in brackets for 2.5 KB):

| P | (1) D2H dev0 | (2) H2D dev1 | D2H dev1 | H2D dev0 | (3) one-way 0->host->1 | (4) duplex | (5) kernel+duplex+kernel | (6) duplex pageable | (8) hipMemcpyPeer |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 KB | 11.4 | 14.7 | 13.6 | 12.0 | 37.0 | 39.9 | 105 (85-218; 89-108 in other runs) | 94.7 | 73.4 |
| 10 KB | 13.2 | 16.7 | 14.4 | 13.9 | 37.9 | 41.3 (39.5 / 54.9 in other runs) | 75.7 (107 in one run) | 91.0 | 55.8 |
| 20 KB | 9.5 | 12.2 | 10.4 | 11.0 | 30.0 | 24.8 (26.7 / 24.9) | 67.3 | 51.2 | 44.3 |
| 40 KB | 13.3 | 14.9 | 13.2 | 14.3 | 33.1 | 30.6 | 73.3 | 67.4 | 49.7 |
| 160 KB | 30.6 | 32.0 | 30.3 | 32.3 | 67.2 | 65.1 | 108.5 | 142.5 | 88.9 |
| 1 MB | 154.3 | 155.8 | 154.1 | 162.5 | 347.6 | 320.5 | 372.0 | 710.4 | 368.1 |
| 16 MB | 2354.7 | 2357.5 | 2357.1 | 2472.0 | 4859.6 | 4868.7 | 5105.5 | 9832.5 | 5022.1 |

Fitted constants (from the 1 MB and 16 MB points, slope = payload/time):
- D2H dev0: 7.15 GB/s, intercept 7.6 us. D2H dev1: 7.15 GB/s, 7.4 us. H2D dev1: 7.14 GB/s, 9 us. H2D dev0: 6.8 GB/s, 8 us (H2D on dev0 is about 5% slower at 16 MB).
- Duplex: 3.46 GB/s per direction (i.e. two sequential 7 GB/s hops; both devices run in parallel without slowing each other: duplex 16 MB 4869 us = 2 x 2430), intercept about 17 us from 160 KB (65.1 us) and 1 MB (320 us).
- Small sizes are flat and non-monotone, repeatably in 3 runs: single D2H or H2D costs 9.5-17 us from 2.5 KB to 40 KB, and 20 KB is the fastest (9.5 / 12.2 us single, 24.8 us duplex) while 2.5 KB and 10 KB are slower (11-14 us, duplex 40 us). Looks like the driver switches copy path at about 16 KB. The one-way exchange costs the same as the duplex (30-38 us small, 4860 us at 16 MB): the two hops are sequential, the two directions overlap perfectly.
- (7) hipStreamSynchronize on an idle stream: 0.17 us (dev0, dev1 0.17; min..max 0.16..0.32, occasional 2.4 us). One trivial kernel + sync: 9.7 us on dev0 (9.4-13.3), 11.7 us on dev1. So launch+sync round trip is about 9.7-11.7 us and is hidden inside the 20 KB duplex only partly: each end of a real round pays it.
- Pageable memory (6): 2.2x slower at 1 MB (710 vs 320) and 2.0x at 16 MB, and 51-95 us at small sizes vs 25-41.
- (8) hipDeviceCanAccessPeer is 0 in both directions, yet hipMemcpyPeer / hipMemcpyPeerAsync / hipMemcpyAsync(DeviceToDevice) all return success and work: they are staged through the host by the runtime. hipMemcpyPeer: 55.8 us at 10 KB, 368 us at 1 MB, 5.02 ms at 16 MB = 3.34 GB/s; the async forms + sync are faster at small size (38.0 us at 10 KB, 345.7 us at 1 MB, 4.91 ms). That equals the manual one-way staging (37.9 us at 10 KB, 347.6 at 1 MB, 4860 at 16 MB), so there is nothing to gain from the runtime peer copy and it is one-directional only.

Practical rule: a duplex exchange of P bytes between the two GPUs costs about 17 us + P / 3.46 GB/s (>= 160 KB), and a flat 25-41 us below 40 KB. Floors: 10 KB duplex = 41 us (39.5-55); 20 KB duplex = 25 us (24.8-27); 40 KB = 31 us; 2.5 KB = 40 us. A one-way exchange costs the same. If it must be preceded by a producing kernel and followed by a consuming kernel with syncs, add about 27-45 us (full round 67-76 us for 10-40 KB, 85-108 us at 2.5 KB). Use pinned memory (2x faster); do not expect peer copies to help.
Surprises: the duplex does not cost more than the one-way; 20 KB is faster than 2.5 KB and 10 KB; sync alone is nearly free (0.17 us) while a kernel+sync costs 10-12 us; canAccessPeer=0 but peer copies still work (via the host).

## What the three calibrations change (2026-11-22)

- Occupancy: blocks resident per compute unit = min(floor(4 x min(8, floor(256 / VGPRs)) / ceil(threads / 64)), floor(65536 / LDS bytes)). The second argument of `__launch_bounds__` is waves per SIMD, not blocks. This is the rule behind the `flash_attn_ext_vec` fix (more than 128 VGPRs gave 2 blocks per compute unit). A grid above the resident count costs one more round each time. The occupancy scan of all kernels (worklist BA-4) can now be done from the trace columns with no GPU time.
- Streaming: the peak is 891 GB/s on the MI60 and 904 GB/s on the MI50, but a small buffer does not reach it. 3.5 MB costs 5.0 us at best (512 blocks of 128 threads); grids of 512 blocks or fewer reach at most 83% of the peak; 90% needs at least 32 MB per launch. Read with coalesced 16-byte chunks across the threads of a block: one contiguous run per thread is up to 10 times slower. This prices the Flash Next `dsv4_hc_mix_down` / `_up` route (TB2 / F2b): they stream about 3.5 MB and cost about 10 us untraced, against 5.0 us measured for that size, so about 5 us per call each, 98 calls per token per GPU, about 1.0 ms per token per GPU (C, from the card; the four variants tried on 2026-11-22 did not change the block layout).
- Cross-GPU exchange: a host-staged duplex exchange via `hipMemcpyAsync` + host sync costs about 17 us + bytes / 3.46 GB/s, with a floor of 25 to 41 us below 40 KB (10 KB: 41 us, 20 KB: 25 us); a kernel plus a sync is 9.7 us (MI60) and 11.7 us (MI50). **Correction (2026-11-22)**: earlier text concluded `ggml_cuda_ar_kernel` (20.0 us on 27B, 25.8 us on Flash Next in traces) already sat at this floor; see [Skeptical corrections (2026-11-22)](#skeptical-corrections-2026-11-22) below.
- Open on the card: at 3.5 MB, geometries with blocks x threads = 131072 drop to 82 GB/s (not explained); a 20 KB exchange is faster than 2.5 KB and 10 KB (repeatable, not explained); the 1.22 us per node in a replayed graph has no saved source.

## Skeptical corrections (2026-11-22)

1. **Graph-replayed launch cost vs. 2.2 us pricing**: Decode on all three models runs inside captured HIP graphs. The card records 1.22 us per node in a replayed graph (source not saved yet) and 1.5 to 1.9 us saved per removed launch in Inkling ABBA kill-switch tests (which includes kernel work removed, not just launch overhead). Using 2.2 us per removed launch over-prices graph-replayed fusion by ~1.2x to 1.8x. Use **1.5 to 1.8 us** (upper bound) or **1.22 us** (pure launch floor) for graph-replayed decode until the graph-node microbench is saved and re-verified on both GPUs.
2. **AllReduce floor category error**: Q3 measured explicit `hipMemcpyAsync` DMA copies plus host `hipStreamSynchronize` (25 to 41 us below 40 KB). Decode `ggml_cuda_ar_kernel` does *not* use `hipMemcpyAsync` or host syncs -- it uses GPU-kernel-polled mapped pinned host memory (which is why it already runs in 20 us traced vs. 40 us for Q3 DMA at 2.5-10 KB). Therefore Q3 bounds **prefill DMA staging**, not kernel-polled decode AllReduce (`AR-LL` is not ruled out by Q3, though its decode ceiling remains modest at ~0.5-0.6 ms/token).
3. **MI60 (ROCm0) vs. MI50 (ROCm1) asymmetry reconciled**:
   - **HBM2 streaming (bandwidth-bound decode / KV reads)**: equal (~891 GB/s MI60 vs. 904 GB/s MI50; 27B mat-vec only 1.1% slower on MI50).
   - **Compute-bound PP GEMM**: MI50 is ~21% slower (60 CUs / 178 W cap vs. 64 CUs / 225 W).
   - **Small-kernel dispatch**: MI50 is 0.6 us slower bare (2.6 vs. 2.0 us) and 20-40% slower on tiny decode kernels (suspected chipset PCIe lanes; test with `K2`).
   - Placement implication: putting bandwidth-heavy work (KV cache, decode-heavy draft) on MI50 is cheap; putting compute-heavy PP or long chains of tiny serial kernels on MI50 hurts.

## Rules derived from the card

### The one-block rule

One thread block gets about 1/30 of the GPU. A kernel that must reduce over a whole row runs as one block, so its cost is about 2.9 us + 0.46 us per 1000 floats (constants above). Use this before writing or rewriting a fused kernel:

- Count the floats one block touches. Under about 5000, one block of 1024 threads is fine. Above it, split: reductions in one launch (many blocks, one partial each), the output in a second launch (many blocks, each reads the partials). Two launches cost about 5.4 us, flat.
- Use 1024 threads per block for a one-block kernel. 256 threads cost twice as much for the same row.
- Do not look for the cost in barriers (0.08 us each). Count dependent phases (1.6 us each) and floats per block.
- A kernel that streams weights through one block is the worst case (the one-block loop controller: 600K floats).
- Before a rewrite, compute what the rule predicts for the current kernel and for the new form. If the current cost already matches the rule, only a different block layout can help.
- The rule predicts cost, not mechanism. Read the launch geometry in the trace (blocks, registers, shared memory) before naming a mechanism: see the `flash_attn_ext_vec` check below.

### Value of a removed launch

One launch removed per token in a captured HIP graph is worth roughly **1.22 us** pure launch floor (B, unsaved microbench) up to **1.5 to 1.9 us** total including trivial kernel work (A, Inkling ABBA). Earlier estimates used 2.2 us (C, derived), which over-prices graph-replayed fusion; check both bounds before building small (<2%) fusions. Work near the roofline is not removed by fusing launches.

## Checks of the card against real kernels

Predictions against one trace each (grade C for the prediction; the trace adds about 1.2 us per kernel). Four were measured before the rule existed:

| Kernel | Prediction | Traced |
| --- | --- | --- |
| 27B `add_rms_norm_mul_f32` without the Q8_1 output, 5120 floats and three inputs | about 8.7 us | 8.2 us |
| Inkling `res_mix_f32`, about 11.5K floats read twice | about 12 us | 16.2 us |
| One-block loop controller, about 600K floats | 160 to 280 us | about 630 us (measured) |
| `flash_attn_ext_vec` at 4K context if it runs one block per head, about 2.1 MB per head | about 70 us | 85.8 us |

Three results from 2026-11-22, after the card existed:

| Kernel | Prediction | Result | What it taught |
| --- | --- | --- | --- |
| Two-launch `res_mix_f32` | rule: about 5.4 us for the pair (about 7.8 us in a trace) | trace: 17.0 to 9.5 us | the split form is much faster; the pair measured above the prediction |
| Two-launch `add_rms_norm_mul_f32` | about 7.8 us in a trace | trace: 11.37 to 9.05 us; block size not tuned | above the prediction; block-size tuning is on the worklist |
| `flash_attn_ext_vec` at 4K context | about 70 us against 85.8 us traced: the cost prediction was right | the stated mechanism (one block per head) was wrong: the kernel already split the keys over blocks; the limit was occupancy (more than 128 VGPRs, 2 blocks per compute unit), fixed by launch bounds (128, 4) on HIP, commit `efb85a1a5` | read the launch geometry and registers from the trace before naming a mechanism; the occupancy constants are needed |
