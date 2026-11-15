# Flash Attention over K/V rows (kv_rows) for gfx906

2026-11-15. Commit 518d6233d adds kv_rows reads to the HIP vec and tile FA kernels, so unified-KV servers on gfx906 can attend only to each sequence's own cells. Base: build-catchup-20261113-200420 (same source minus this commit). Harness: raw/ab.sh, raw/srvtest.py.

In short: correct, but slower. It does not help a short chat beside a long one, and it makes a long single conversation much slower. Not for deployment as is.

## Correctness

test-backend-ops FLASH_ATTN_EXT vs CPU: kv_rows cases (`-p n_cells`) 60/60 on ROCm0 and ROCm1; full FLASH_ATTN_EXT 4382/4382 on ROCm0. Under `-sm tensor` the server no longer logs "Flash Attention over K/V rows not supported"; the Meta mirrored-src[5] assert holds.

## Performance (2x gfx906, tensor parallel)

llama-bench (kv_rows unused there), alternating runs: 27B TG128 base 37.29 / 37.27, kv_rows 37.15 / 37.03. First pass: 27B PP2048 703.5 vs 676.7, FN PP2048 654.5 vs 645.9, FN TG128 48.1 vs 47.35 (the first-pass 27B TG 37.35 vs 32.70 did not reproduce).

qwen27b preset (4 slots, 262144 unified, MTP2, 2048/2048), one run each:

| Stage | base | kv_rows |
| --- | ---: | ---: |
| A short chat TG, fresh | 61.0 / 62.0 | 60.6 / 61.1 |
| B 64K fill PP | 409.0 | 328.2 |
| B TG right after 64K fill | 37.3 | 25.0 |
| C short chat TG beside 64K slot | 59.9 / 54.5 | 56.0 / 52.9 |
| D 3 users aggregate TG | 39.2 | 38.8 |
| E 4 users, slot 0 re-prefills 64K, aggregate | 5.2 | 4.2 |

Stage E did not reuse slot 0's cache (full 65549-token re-prefill, likely hybrid-model checkpoint limits), so it mostly measures users stalled behind a prefill.

## Why, and hints for next time

- The tile kernel now reads each K/V row index from global memory (`rows[i]`) before the row load. On long single sequences this dependent load costs 20% PP and 33% TG at 64K.
- The base already loses little beside a long neighbour (C vs A: 2-12%): KV_max trims the masked tail, and attention is a small part of a decode step at 64K. The ceiling for this feature on this workload is small.
- Next try: in the tile kernel, check per FA tile if rows are contiguous (rows[last] - rows[first] == n - 1) and use the plain contiguous load from rows[first]; gather only for mixed tiles. Also stage the tile's row indices in shared memory once. This should make B equal to base; then remeasure C with chats that start after a long fill.
- A cheaper alternative (no kernel gather): skip fully masked 256-cell blocks in the existing kernels and keep sequences in large chunks in find_slot.
