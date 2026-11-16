# Flash Attention over K/V rows (kv_rows) for gfx906

2026-11-15. With `--kv-unified` and several slots, attention used to scan every cell in the shared pool (dense mask). Upstream's `ggml_flash_attn_ext_rows` (#29510) reads only each sequence's own cells, but ran only on NVIDIA tensor cores, so gfx906 logged "Flash Attention over K/V rows not supported". This change adds kv_rows to the HIP tile kernel, uses it only where it saves work, and fixes a multi-slot MTP rollback abort found while testing.

In short: with 4 users in one pool, 27B prompt processing is 25% faster, decode is unchanged, and a new user's prompt no longer slows down as the pool fills. Single-user speed is unchanged. Flash Next is neutral.

## Change (commit range 518d6233d..7ddd49693)

- `fattn-tile.cuh`: a `use_kv_rows` kernel template parameter. The plain kernels compile to the same code as before (same VGPR count, no new spills). In kv_rows kernels a tile whose rows form one run (rows are sorted, so `rows[last] - rows[first] == nbatch_fa - 1`) is read directly; only mixed tiles gather per row.
- `fattn.cu`: on AMD, kv_rows uses the tile kernel for F16 K/V; other types are unsupported (the probe disables kv_rows, the dense path stays). NVIDIA unchanged.
- `llama-kv-cache.cpp` `get_n_kv_rows`: return 0 (dense path for this ubatch) unless rows save work. The dense path reads each cell once for all slices. Prefill ubatches (more than 8 tokens per slice) use rows when they cover fewer cells than the pool (saves compute); decode and draft verify use rows only if `n_slices * n_rows < n_kv` (saves reads). A single sequence alone therefore stays on the dense path.
- `ggml-backend-meta.cpp`: assert kv_rows (src[5]) is mirrored under `-sm tensor`.
- `llama-memory-hybrid.cpp`: with recurrent rollback (`n_rs_seq > 0`), split ubatches sequentially also for unified KV, as `llama-memory-recurrent.cpp` does. A non-consecutive ubatch such as {0,1,3} made `find_slot` swap the cell of seq 2, which lost its rollback planes; the next draft rejection aborted in `common_context_seq_rm` ("refusing rollback seq 2 ... limit=0").

## Results (2x gfx906, -sm tensor, F16 KV)

Multi-user, `vr/scripts/bench_multiuser.sh` REPS=2, same binary, A = kv_rows on, B = `LLAMA_KV_ROWS_DISABLE=1`, each run started below 60C (`raw/v6/`):

| Model | kv_rows | PP t/s (4x16K) | 4-way TG t/s |
| --- | --- | ---: | ---: |
| Qwen3.8-27B Q8_0 | on | 577.8 / 577.5 | 80.86 / 80.93 |
| Qwen3.8-27B Q8_0 | off | 462.3 / 462.2 | 81.13 / 81.04 |
| Flash Next UD-IQ3_XXS | on | 470.5 / 473.1 | 84.17 / 83.90 |
| Flash Next UD-IQ3_XXS | off | 472.2 / 474.2 | 84.24 / 84.11 |

Single user, llama-bench, base `build-catchup-20261113-200420` vs 80212cad5 (kernels as final; 7ddd49693 only adds the dense choice for single sequences), base/new/new/base, cooled (`raw/v6/single-user-llama-bench.out`):

| Model | Test | base | new | change |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-27B Q8_0 | PP2048 | 703.3 / 700.6 | 699.7 / 699.1 | -0.4% |
| Qwen3.8-27B Q8_0 | TG128 | 37.29 / 37.19 | 37.17 / 37.27 | 0.0% |
| Flash Next (q8_0 KV) | PP2048 | 654.0 / 655.6 | 652.6 / 651.6 | -0.4% |
| Flash Next (q8_0 KV) | TG128 | 48.35 / 49.09 | 48.92 / 49.09 | +0.6% |

qwen27b server preset (4 slots, 262144 unified, MTP2, `--no-cache-idle-slots`), 4 users with 48K prompts prefilled one after another, prefill t/s per user: base 430 / 277 / 203 / 161 (cool GPUs); kv_rows 358 / 354 / 353 / 353 (hot GPUs, throttled; same-binary off on hot GPUs: 353 / 193 / 132).

Server 4-slot MTP2 at 16K per user, prefill one by one, then all 4 continued at once (`raw/v6/server-mtp-4slot.out`): one user PP 553.9, TG 59.8; 4 users 15.7-16.3 t/s each, 60.9 aggregate; 0 rollback refusals. Earlier builds: 2 more clean runs with the rollback fix; before it, 1 abort in 3 runs.

Correctness: `test-backend-ops -o FLASH_ATTN_EXT` vs CPU, full 4372/4372 on ROCm0, kv_rows cases (`-p n_cells`) 50/50 on both GPUs (the 10 Q8_0 kv_rows cases are unsupported by design).

## What did not work, and hints

- Attempt 1 (518d6233d, Sonnet): per-row index loads in every tile, and a vec-kernel gather. Long single sequences lost 20% PP and 33% TG at 64K; the vec change cost Flash Next q8_0 TG about 4.5% even with kv_rows off.
- Attempt 2 (a17ab0d66, contiguous-tile reads) fixed the per-row cost, but the gather code in the same kernel pushed every tile variant to the 128-VGPR cap with 57-97 spilled VGPRs (base: 74-92 VGPRs, 0-5 spills). That cost 1.5% single-user PP and made the "kv_rows off" numbers of that build too low, so its "+16% 4-way decode" was not real. Check register use with `llvm-readelf --notes` on the gfx906 code object of `fattn-tile-instance-dkq256-dv256.cu.o` (unbundle `.hip_fatbin` with `clang-offload-bundler`).
- kv_rows cannot speed up decode when all users of the pool decode together: the dense path reads the pool once and shares each K/V tile across all queries. Hence the work rule in `get_n_kv_rows`.
- The kv_rows kernels still spill (62-95 VGPRs). Reducing that (e.g. load the tile's row indices once into shared memory, or 32-bit row offsets) should raise the prefill gain further.
- A quantized KV cache keeps the dense path; a kv_rows path for it needs a tile kernel that reads quantized rows.
- The GPUs throttle under sustained load (junction 98-104C, sclk 938 MHz): use cooled, alternating A/B runs.
- Idle-slot caching (`--cache-idle-slots`, default on) moves idle slots to host RAM and clears them from the unified pool when another task starts.
