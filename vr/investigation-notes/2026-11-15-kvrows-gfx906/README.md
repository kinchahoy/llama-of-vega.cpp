# Flash Attention over K/V rows (kv_rows) for gfx906

2026-11-15. With `--kv-unified` and several slots, each sequence's attention used to scan every cell in the shared pool (dense mask). Upstream's `ggml_flash_attn_ext_rows` (#29510) reads only the sequence's own cells, but ran only on NVIDIA tensor cores, so gfx906 logged "Flash Attention over K/V rows not supported". This change lets the HIP tile kernel read kv_rows, and fixes a multi-slot MTP rollback abort found while testing.

In short: for 4 users in one unified pool, prompt processing is 54% faster and 4-way decode 16% faster; a new user's prompt no longer slows down as the pool fills. Single-user speed is unchanged within about 1%.

## Change (41+/17- in 4 files)

- `fattn-tile.cuh`: K/V tile loads take an optional row list. kv_rows are sorted, so a tile whose rows form one run (`rows[last] - rows[first] == nbatch_fa - 1`) is read directly from `rows[first]`; only mixed tiles gather per row.
- `fattn.cu`: on AMD, kv_rows uses the tile kernel when K and V are F16; otherwise unsupported (the probe then disables kv_rows and the dense path stays). NVIDIA is unchanged.
- `ggml-backend-meta.cpp`: assert that kv_rows (src[5]) is mirrored under `-sm tensor`.
- `llama-memory-hybrid.cpp`: with recurrent rollback (`n_rs_seq > 0`), split ubatches sequentially also for unified KV, as `llama-memory-recurrent.cpp` already does. A non-consecutive ubatch such as seqs {0,1,3} made `find_slot` swap the cell of seq 2, which then lost its rollback planes; the next draft rejection aborted in `common_context_seq_rm` ("refusing rollback seq 2 ... limit=0").

## Results (2x gfx906, -sm tensor, F16 KV, 27B Q8_0)

Thermally matched A/B, `llama-batched-bench -kvu -npp 16384 -ntg 128 -npl 4` (4 sequences in one pool, plain decode), same binary with `LLAMA_KV_ROWS_DISABLE=1` for off, order off/on/on/off, each run started below 60C junction:

| | off | on |
| --- | ---: | ---: |
| PP (t/s) | 366.7 / 365.6 | 565.1 / 565.9 |
| 4-way TG aggregate (t/s) | 54.4 / 54.3 | 63.0 / 63.1 |

qwen27b server preset (4 slots, 262144 unified, MTP2, `--no-cache-idle-slots`), 4 users with 48K prompts prefilled one after another:

| Prefill t/s, users 1-4 | |
| --- | --- |
| base build (16:40, cool) | 430 / 277 / 203 / 161 |
| kv_rows off, same binary (hot) | 353 / 193 / 132 / - |
| kv_rows on (hot) | 358 / 354 / 353 / 353 |

Single user: server 24K under rocprofv3, off vs on: PP 471.1 vs 468.9, TG 40.9 vs 40.4; flash attention kernel time +1.2%. At 48K (hot GPUs) on vs off: PP 371 vs 357, TG 31.4 vs 31.8.

llama-bench (kv_rows unused there), base `build-catchup-20261113-200420` vs final, alternating base/final/final/base, each run started below 60C, `-r 2`:

| Model | Test | base | final | change |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-27B Q8_0 | PP2048 | 698.2 / 703.6 | 689.4 / 691.3 | -1.5% |
| Qwen3.8-27B Q8_0 | TG128 | 37.29 / 37.41 | 37.03 / 36.71 | -1.3% |
| Flash Next UD-IQ3_XXS (q8_0 KV) | PP2048 | 653.6 / 650.1 | 651.6 / 652.2 | 0.0% |
| Flash Next UD-IQ3_XXS (q8_0 KV) | TG128 | 49.56 / 48.12 | 48.64 / 49.63 | +0.6% |

The 27B cost is the optional row list in the tile loads (repeatable across two sessions). Making the row path a template parameter of the load functions could remove it.

Crash fix: 4-slot MTP2 server, 4 x 16K prefilled, then all 4 continued at once: two runs, 0 rollback refusals, all requests finished (54.3 t/s aggregate). Before the fix this aborted once in three similar runs.

## Notes for later

- The GPUs throttle under sustained load (junction 98-104C, sclk 938 MHz). Runs late in a long session are slower; use matched, cooled, alternating runs for any A/B.
- Idle-slot caching (`--cache-idle-slots`, default on) moves idle slots to host RAM and clears them from the unified pool when another task starts, so in production the pool mostly holds active users.
- A quantized KV cache keeps the dense path. A kv_rows path for it would need the vec kernel (the first port's vec gather cost Flash Next q8_0 TG about 4.5% even with kv_rows off) or a tile kernel that reads quantized rows.
- Attempt 1 (raw/, commit 518d6233d): per-row index loads in every tile; long single sequences lost 20% PP and 33% TG at 64K.
- Decode with 4 x 48K on the MTP server (35.3 vs 42.3 t/s aggregate) was measured on throttled GPUs against a cool baseline and is not a valid comparison.
