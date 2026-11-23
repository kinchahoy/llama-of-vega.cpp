# Performance Live Status & Worklist

Read [GOALS.md](GOALS.md) first (target: long-context 50-100K+ PP+TG, 1-4 parallel slots, equal priority on 27B / Flash Next / Inkling 2 Mini + Qwen 4 readiness). Hardware constants live in [HARDWARE.md](HARDWARE.md); chronological history and raw benchmark logs live in [BENCHMARK-TIMELINE.md](BENCHMARK-TIMELINE.md); build/merge rules live in [MAINTENANCE.md](MAINTENANCE.md).

## Current numbers (2026-11-22 validated baseline)

Alternated ABBA (`llama-bench -r 2`, cool-start <50 C) comparing `build-inkling-final-20261121` (`0ba4be7f1`) vs `build-final-20261122` (`ae32e5e58`, two-launch `res_mix`/`add_rms_norm_mul` + `flash_attn_ext_vec` `(128,4)` launch bounds). Note: `build-current` (`build-dspark-20261122`, `3a8d768df`) adds opt-in DSpark drafting on top of `build-final-20261122` (plain paths unchanged; DSpark at default settings currently 33.7 vs 37.7 t/s short prompt, 37.5 vs 36.6 long prompt). Full history: [BENCHMARK-TIMELINE.md](BENCHMARK-TIMELINE.md).

| Model | Config | PP2048 (prev -> curr) | TG128 (prev -> curr) | Grade |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B Q8_0 | 2 GPUs, `-sm tensor`, ub 512 | 614.7 -> 614.0 | 37.35 -> **37.96** (+1.6%) | A (4 pairs) |
| Flash Next UD-IQ3_XXS | 2 GPUs, `-sm tensor`, ub 512 | 655.9 -> 651.0 | 50.28 -> **50.46** (noise) | A |
| Inkling 2 Mini NVFP4 + Q8_0 | ROCm0, ub 512 | 836.0 -> 835.7 | 45.94 -> **46.97** (+2.2%) | A |
| Inkling 2 Mini IQ4_NL | ROCm0, ub 512 | 880.2 -> 878.6 | 47.79 -> **48.98** (+2.5%) | B (1 run) |
| Inkling 2 Mini NVFP4 @ 4K ctx | ROCm0, `llama-batched-bench -npl 1` | 746-819 (noisy) | 38.19 -> **43.12** (+12.9%) | A |
| Inkling 2 Mini NVFP4 @ 16K ctx | ROCm0, `llama-batched-bench -npl 1` | 657.1 -> 658.6 | 33.50 -> **40.53** (+21.0%) | B (1 run) |

Reference serving/speculative baselines (see [BENCHMARK-TIMELINE.md](BENCHMARK-TIMELINE.md)): 27B 4x16K unified KV (`kv_rows` on): 577.8 PP / 80.9 4-way TG; 27B MTP2 1-slot: ~62-65 t/s (40.6 ms/step); 27B DFlash5 short prompt: ~69-71 t/s; Flash Next MTP3 @ 100K: ~62.7 t/s (49.5 ms/step).

## 2026-11-22 skeptical review & BA-4 audit findings

Zero-GPU-hour audits of saved `rocprofv3` traces (with calibrated Q1 occupancy in [scripts/kernel_budget.py](scripts/kernel_budget.py)), kernel source, and external repos revealed much higher-leverage targets for the 50-100K+ 1-4 slot + Qwen 4 workload than the old short-context `Batch 1`:

1. **QSA attention reads the full KV on gfx906 (corrected 2026-11-22)**: Flash Next / Qwen 4 QSA picks top 512 blocks of 4 cells per token (`qwen4exp.cpp:909`) and passes a dense `-INFINITY` mask over all `n_kv` (`:946`). Upstream sparse gather (`use_sparse`) is MMA-only, so `flash_attn_tile` computes every KV tile. 94-98% of *keys* are masked at 32K-100K, but a whole-tile skip only helps if selected blocks cluster: with scattered selection, skippable 32-wide tiles are only ~2% (32K) / ~27% (100K) in ub512 prefill and ~36% / ~72% of 64-wide tiles in decode. Index gather (upstream PR #30370, GCN tile path) touches only selected cells regardless of clustering. SWA caches are already `n_swa + n_ubatch` cells, so SWA has little to skip.
2. **`G1` padded `ncols2=8` for 6:1 and 12:1 GQA (~45 LOC)**: 27B (6:1 GQA) currently falls back to `ncols2=2`, reading the F16 KV cache 3x per token (and even worse across 4 parallel slots). Padded `ncols2=8` reads KV once for all 6 Q heads and packs 4 slots into 1 KV pass: **-1.7..2.0 ms/tok @ 32K** and **-5.5..6.5 ms/tok @ 100K** on both Qwen3.8-27B and upcoming Qwen 4 27B.
3. **Inkling PP4K+ SWA ring-wrap dense-bias fallback (~965 ms = 20% of PP4096)**: When Inkling's SWA-1024 ring wraps at PP >= 4K, `get_swa()->get_n_kv_pos_contiguous()` (`src/models/inkling2.cpp:354`) returns 0, causing 40/48 attention passes to fall back from banded `flash_attn_tile` to dense attention with full relative-bias materialization (`cpy_scalar` 520 ms + `cpy_scalar_contiguous` 212 ms + `k_bin_bcast` 97 ms + `k_get_rows` 42 ms + 93 ms slower FA = ~965 ms, 20% of PP4096). Handling ring-wrap in the banded tile path removes the cliff.
4. **BA-4 register spill hit (`flash_attn_tile<128,128,64>`)**: Spills **1,040-1,072 B/thread** to scratch memory, costing **655 ms (13.5% of PP4096)** and **158 ms (6.1% of PP2048)** on Inkling (and affects `D=128` tile prefill broadly).
5. **`MU1(b)` 9..16 token multi-slot & verify cliff**: Above `MMQ_RP_Q8_MMV_MAX_TOKENS = 8`, dense repacked mat-vec drops from `nc` to `w32`, and MoE drops narrow MMV + fused GLU straight to `w32` (wasting **94-97% of lanes** on 1-2 tokens/expert). Explains why 4-slot MTP2 server TG collapses from 81 -> 61 t/s. Look at `alex4300`-style wide-GEMV with LDS activation staging for 2-16 columns.
6. **`allreduce.cu` dynamic block cap (~5 LOC)**: Launches fixed grids where 5-6 idle blocks spam `3x __threadfence_system()` + PCIe polls on every single AllReduce (128x/tok on 27B, 96x/tok on FN). Capping blocks to actual work removes redundant cross-PCIe fence traffic.
7. **Flash Next BA-4 & trace hits**:
   - `lightning_indexer_kernel_vec` spills **136 B/thread** at 8K-100K context.
   - `dsv4_hc_mix_down_f32` uses **100 VGPRs** (`2 b/CU`, `8 w/CU`), limiting residency; `dsv4_hc_mix_up_f32` has a **160/128** block tail round.
   - Generic `mul_mat_vec_q` in Flash Next decodes **`26 IQ2_S + 21 IQ3_XXS + 3 Q2_0`** tensors per token (**1.75 / 1.83 ms/tok** + **~0.5 ms 3:2 split AR skew** between GPUs).
   - **Solved "12-17% idle share"**: anchor-segmented `prof-fn` exposes a **3.04 ms pre-graph host token-start gap (`TB5`)** right before every decode token graph (vs 0.19 ms on 27B).
8. **Inkling PP small-M rocBLAS under-fill**: Small-M BF16/F32 loop-controller GEMMs fall to rocBLAS on 1-16 CUs, burning **248 ms (9.6% of PP2048)** on 8 CUs and **244 ms (5.0% of PP4096)** on 1-16 CUs.
9. **Tiny surgical bugs found in code audit**:
   - `fattn-vec.cuh`: missing `any_weight` zero-skip on V under `V_DOT2_F32_F16_AVAILABLE`.
   - `qwen4exp_conv_windows`: fires **9 launches/layer** under MTP verify.
   - `P4` (`SGEMM_F32`): stride/shape bounds, 8192 scratch cap, and anchored tensor-name guards hardened on 2026-11-22; Q8-reference KLD still pending.
   - `F2` (`DSV4_HC_MIX`): wave32 `warp_size` and cross-tensor shape guards hardened in `ggml-cuda.cu` on 2026-11-22.
   - `unsloth #241`: missing empty-graph guard.
10. **Qwen 4 convergence & external prior art**: Qwen 4 converges on **3:1 GDN + QSA**, **6:1 GQA on 27B**, and **hyper-connections (HC) on both 27B and MoE** -- making QSA gather (`QSA-GATHER`), `G1`, `TB3` (GDN), `MU1(b)`, and HC fixes directly reusable. Check upstream PR `#30370` (GCN sparse FA) and `milpster` `__vsub4` word-op improvements before writing custom code.
11. **Cost-model & tooling fixes**: Upgraded [scripts/kernel_budget.py](scripts/kernel_budget.py) fixes the old 1/3-drop denominator bug (giving exact steady-state counts: 27B **1,595**, FN **2,747**, Inkling **4,275/4,056**), groups by full launch geometry, flags spills/low-occupancy/MI60-vs-MI50 asymmetry, and prices graph-replayed launch removal at **1.22-1.7 us** instead of 2.2 us.

### External comparison: `iacopPBK/gfx906-quartz` (2026-11-22, code read only)

Standalone C++/HIP engine (~11.7K LOC, no tests): Qwen3.8-27B Q8_0 only, exactly 2 gfx906 with PCIe P2P (`devices.h:30-31` refuses to start otherwise), 1 request at a time, MTP or DSpine draft always on, F16 KV. Its own claims (grade B, 2x MI50 with P2P, short prompts): **777 PP2048** (631 @ 16K) vs our ~590-614; MTP greedy **93 t/s** decode vs our MTP2 ~62-65. No long-context decode or multi-slot numbers. Peer-write AllReduce needs P2P: on this host `hipDeviceCanAccessPeer` is 0 both ways, also with `HSA_FORCE_FINE_GRAIN_PCIE=1` (checked 2026-11-22), so quartz cannot run here and a fair A/B needs a P2P box. Reusable ideas are linked from `G1`, `MU1(b)`, `TB3`, `SP1` below; also `pf_gemm` (`prefill.h:23`, int8 GEMM with fused gate+up+SiLU epilogue) for PP.

## Method & testing

Compact version of the four-stage model-first workflow (order of work follows [GOALS.md](GOALS.md): reusable kernels & Qwen 4 first -> model-specific kernels -> architectural placement -> parameter tuning last):

1. **Learn the hardware once**: synthetic microbenchmarks in `gfx906/bench/` populate [HARDWARE.md](HARDWARE.md). Never spend an end-to-end run answering a question a microbench answers in seconds.
2. **Predict from traces on disk**: run `python3 gfx906/scripts/kernel_budget.py TRACE_DIR` on saved `rocprofv3` SQLite traces. Read geometry (`blocks`, `VGPR`, `LDS`, `scratch`, per-GPU skew) before naming a mechanism.
3. **Design on paper in batches**: compute predicted ms/tok from [HARDWARE.md](HARDWARE.md). Skip changes predicting **<1%**. Tune kernel parameters in standalone sandboxes, not full model builds.
4. **One confirmation per batch** (<=1 build, <=10-15 GPU-min): `test-backend-ops` (plus `--device none` CPU check for new ops), one short trace comparing measured vs predicted (>20% mismatch updates [HARDWARE.md](HARDWARE.md)), and one cool-start alternated ABBA run per affected model.

**Pre-registration template (5 lines before any experiment)**:
1. *Mechanism*: what changes and why it is cheaper.
2. *Predicted gain*: ms/tok (or % PP) from [HARDWARE.md](HARDWARE.md) arithmetic.
3. *Cheap falsifier*: microbench, trace column, or unit test (not end-to-end).
4. *Best-case bound*: upper bound for any implementation.
5. *Card update trigger*: what result would revise [HARDWARE.md](HARDWARE.md).

**Evidence grades & status words**: **A** alternated/repeated A/B on real model; **B** single measurement, microbench, or trace; **C** estimate/derivation; **D** assertion. Never write "proven", "always", or "cannot" below grade A. Status words: `Worked`, `Did not work` (specific impl only; name untested angle), `Not actually tested`, `Hypothesis`, `Retired`.

**Testing & quality rules**:
- Always confirm the change actually fired (trace kernel counts or debug counter).
- **Thermal rule**: GPUs throttle 15-30% hot (98-104 C). Start every A/B run **below 60 C** and alternate **A B B A**. Keep 50/50 `-ts` for A/B comparisons.
- **Speculative**: use [scripts/spec_accept.sh](scripts/spec_accept.sh) (5-prompt `llama-server`) and compare **ms/verify step**, never single-prompt `llama-cli` TG (a 1-ULP rounding change flips greedy text and acceptance by 10-18%).
- **Multi-user**: use [scripts/bench_multiuser.sh](scripts/bench_multiuser.sh) (`4x16K` unified KV + 4-way TG).
- **Quality**: greedy text equality is only a smoke test. Run `llama-perplexity --kl-divergence` (use `-b 8 -ub 8` for small-batch/decode changes; mandatory for attention changes). Flash Next (~4.5e-3) and Inkling (~4.9e-3) have a MoE routing near-tie KLD floor, so always pair KLD with CPU-ref `test-backend-ops`.
- **Profiler caveats**: `rocprofv3` adds ~1.2 us/kernel (4.2 us floor) and slows TG by 7-45% (PP 2-4%); use traces for kernel counts, geometry, spills, and ranking. Host `perf` is blocked (`perf_event_paranoid=4`); use host timestamps around graph compute or trace gaps.

## Where each model's token goes

Exact anchor-segmented steady-state decode counts (fixing the old 1/3-drop distortion of 1,645 / 2,811) plus 2026-11-22 BA-4 findings:

| Model | Steady kernels/tok/GPU | AR/tok | GPU busy | Key bottlenecks (decode & prefill) |
| --- | ---: | ---: | --- | --- |
| **Qwen3.8-27B Q8_0** (2 GPUs, TP) | **1,595** | 128 | 96-99% | Depth 0 TG: repacked mat-vec ~69% of traced kernel time (~800 GB/s, near roof), AR ~9% (idle blocks spam fences), post-AR norm (now 2-launch), GDN ~13 tiny kernels/layer (3.2 ms). **At 32K-100K**: 6:1 GQA reads KV 3x (`G1` fixes); **4-slot MTP2**: hits >8-col `w32` cliff (`MU1(b)`). PP: repacked GEMM 76% (MI50 +21% slower), exposed AR 5.4%, attention grows to 32% at 32K. |
| **Flash Next UD-IQ3_XXS** (2 GPUs, TP) | **2,747** | 96 | 83-88% | **3.04 ms pre-graph host token-start gap (`TB5`)** explains the 12-17% idle share. Active work: routed experts 21%, generic MMVQ (`26 IQ2_S + 21 IQ3_XXS + 3 Q2_0`, 1.75/1.83 ms + ~0.5 ms 3:2 AR skew), AR 11%, HC mix (`down` 100 VGPR = 2 b/CU; `up` 160/128 tail), QSA indexer (`lightning_indexer_kernel_vec` 136 B/thr spill) & 12-launch radix top-k, ~300 `quantize_q8_1` + glue chains. **At 32K-100K**: QSA FA reads all KV tiles (no sparse gather on gfx906, `QSA-GATHER`). PP: stock `mul_mat_q` gate/up 23%, HC 15%, QSA FA grows to 23% at 32K. |
| **Inkling 2 Mini** (ROCm0, 1 GPU) | **4,275** (NVFP4) / **4,056** (IQ4_NL) | 0 | 98% | Short-ctx TG: launch floor (~5.2-7.3 ms at 1.22-1.7 us/node) dominates; 4K-16K decode FA fixed by `(128,4)` (`efb85a1a5`). **PP4K+**: SWA ring-wrap falls back to dense attention + bias materialization (**965 ms = 20% PP4096**); `flash_attn_tile<128,128,64>` spills **1,040-1,072 B/thr** (**655 ms = 13.5% PP4096**, 158 ms = 6.1% PP2048); small-M rocBLAS under-fill (**248 ms = 9.6% PP2048**, 244 ms = 5.0% PP4096). DSpark verify step costs 2.1-2.2 plain tokens (needs trace). |

*Reconciled historical discrepancies*: Older docs listed 27B bandwidth as 650-715 GB/s vs ~800 GB/s (different byte/time denominators; Q2 peak is 891/904 GB/s), 27B/FN kernel counts as 1,645/2,811 (artifact of old `kernel_budget.py` dropping 1/3 of launches and dividing by `n_tok*2/3`), and launch floor as 4.2 us (profiler) vs 2.2 us (bare) vs **1.22-1.7 us** (HIP graph).

## Prioritized worklist

Ordered for [GOALS.md](GOALS.md): 50-100K+ PP+TG, 1-4 parallel slots, and Qwen 4 reusability. Estimates are grade C unless noted; launch savings priced at **1.22-1.7 us/launch** (HIP graph).

### Tier 1: High-impact reusable, long-context, multi-slot & cheap surgical wins

| Id | Target | Change & mechanism | Est. gain (C) | Cheapest verification |
| --- | --- | --- | --- | --- |
| **QSA-GATHER** (done) | FN, Qwen 4 (32K-100K PP+TG) | Port upstream PR #30370 (GCN sparse FA: per-query-tile index list from `flash_attn_mask_to_sparse_indices`, gathered in `fattn-tile.cuh`). Replaces the old all-`-INFINITY` tile-skip idea (premise wrong for scattered block selection, see finding 1) | Decode touches 2,051 of N cells; prefill ub512 <=16.4K union cells. QSA FA ~23% of FN PP @ 32K -> ~+10-20% PP @ 32K (C), more at 100K; PR reports micro-qwen4exp pp32768 +38.7% (B, other HW) | PR's `test-backend-ops -o FLASH_ATTN_EXT` cases, KLD, FN 32K PP+TG ABBA; cheap falsifier: dump `indexer_top_k` at 32K and count empty tiles. **Worked 2026-11-22** (`0b360810b`, kill switch `GGML_CUDA_SPARSE_FA=0`): sparse lists reuse the `kv_rows` gather; FN PP512 @ 32K **473.2 -> 568.1 (+20.1%, A, ABBA)**, TG64 @ 32K 41.26 -> 41.83 (noise); KLD vs dense 0.0095 (FP-order floor: `-ub 256` vs 512 gives 0.0111); 27B / Inkling neutral. Next: TG and PP at 64K-100K, MTP verify |
| **G1** | 27B, Qwen 4 27B (32K-100K, 1-4 slots) | Padded `ncols2=8` in `fattn-tile`/`vec` for 6:1 & 12:1 GQA (~45 LOC); 4 slots in 1 KV pass. Reference: quartz `flash9.h` (K tile in LDS once, all 6 Q heads per lane, F16 P for `v_dot2` PV, split + merge pass) | -1.7..2.0 ms @ 32K, **-5.5..6.5 ms @ 100K**; big 4-slot TG gain | `test-backend-ops`, KLD, 27B 32K/64K + 4-slot ABBA |
| **FA-SPILL** | Inkling, D=128 PP | Eliminate 1,040-1,072 B/thr register spill in `flash_attn_tile<128,128,64>` | Up to 655 ms (**13.5% PP4096**), 158 ms (**6.1% PP2048**) | Check `scratch_size=0` in `kernel_budget.py`, PP2K/4K ABBA, KLD |
| **IK-SWA** | Inkling PP4K+ & cleanup | Handle SWA ring-wrap in `fattn-tile.cuh` `use_banded` path; then prune 1,306-LOC `fattn-banded.cu` | Up to 965 ms (**20% PP4096**) + -1,306 LOC maintenance | Inkling PP4096 ABBA + banded KLD |
| **MU1(b) + SP3** (dense done) | All models (2-4 slots, MTP/DFlash/DSpark) | First measure `llama-bench -p 1..16 -n 0` cliff (`SP3`); extend dense `nc` MMV past 8 cols & add MoE wide-GEMV w/ LDS staging (`alex4300` style) to stop `w32` 94-97% lane waste. Alt reference: quartz `qz_gemv_core<T,S>` (`qwen35_cycle.hip:161-250`, T=1..12 rows in 1 weight pass, register-tiled `acc[T][S]`, SGPR buffer descriptors; check VGPRs vs `Q3` losses) | Closes 81 -> 61 t/s 4-slot MTP gap (**+20-30% 4-slot TG**); helps DSpark 4-tok verify | `SP3` 1..16 col curve (no code), then kernel sandbox + 4-slot ABBA. **2026-11-22**: SP3 measured a 2.1x (27B) / 2.5x (FN) jump from 8 to 9 tokens. Dense fix (`ce504ae78`, `GGML_CUDA_RP_NC_SPLIT=0` off): 9..16 tokens run the narrow kernel twice; 27B 12-token batch 95.5 -> 66.9 ms, 4-user MTP2 serving 75.7 -> 98.4 t/s total (+30%, A); KLD 0.0004. Open: MoE side (grouped kernels hold 8 tokens), true 12/16-column kernel |
| **SRV-PF** | All models, 2-4 slots with long prompts | Server prefill/decode interleave: at 4 x 50K (2026-11-22) prefills ran one at a time and decoding users dropped to 0.7-1.9 t/s for minutes (last first token at 640 s). Measure step time of a mixed batch (decode rows + 2048-token prefill chunk); try a smaller prefill chunk while other slots decode, or a decode-priority batch split | Per-user TG during others' prefill from ~1 t/s toward ~10-15 t/s; TTFT fairness across slots | `bench_serve.py --users 2 --ctx-tokens 60000` with per-step timings from server log. **2026-11-22 (B)**: opt-in `LLAMA_SERVER_PREFILL_CAP` (branch `srv-prefill-cap`, not merged), 2 x 60K: cap off user 0 gen 1.9 t/s, user 1 TTFT 344 s; cap 512: 1.8 t/s; cap 128: 4.0 t/s, TTFT 379 s, prefill 285 -> 243 t/s. Steps did get shorter (2044 -> ~512 tokens) without the expected decode gain, so the stall is not only chunk size; trace MTP verify vs prefill interleave next |
| **TB5** | Flash Next TG | Fix **3.04 ms pre-graph host token-start gap** (device-side QSA k-pool / n-gram ids or host timestamp root-cause) | Up to **~12-15% FN TG** (3.04 ms of ~20-26 ms) | Host timestamps around graph compute / input prep (no GPU rebuild) |
| **TB3** | 27B, FN, Qwen 4 TG | Fused GDN decode step (conv + l2 norms + gates + GDN + gated norm: ~13 -> 2-3 launches/layer). Reference: quartz `qz_dn_rows` (one phase per head, recurrent state committed after verify) | 27B: 480 launches x 1.22-1.7 us = **0.6-0.8 ms**; FN: **0.4-0.6 ms** (+ Qwen 4) | Check upstream fused GDN; dump 1 layer trace, sandbox microbench |
| **AR-CAP + SURGICAL** | All models (tiny LOC / bugs) | (1) `allreduce.cu` dynamic block cap (~5 LOC, stops 5-6 idle blocks spamming `3x __threadfence_system()`); (2) `fattn-vec.cuh` `any_weight` zero-skip on V under `V_DOT2_F32_F16`; (3) `qwen4exp_conv_windows` 9 launches/layer under MTP; (4) `P4` Q8-ref KLD (guards done 2026-11-22); (5) `unsloth #241` empty-graph guard | Low-risk efficiency + correctness; AR-CAP cuts redundant PCIe fence storms on 96-128 AR/tok | Unit tests, `gfx906-test-tp`, trace check, KLD |

### Tier 2: Targeted BA-4, PP & speculative wins

| Id | Target | Change & mechanism | Est. gain (C) | Cheapest verification |
| --- | --- | --- | --- | --- |
| **FN-BA4** | Flash Next, Qwen 4 MoE | (a) Fix 136 B/thr spill in `lightning_indexer_kernel_vec` @ 8K-100K; (b) cut `dsv4_hc_mix_down` VGPRs 100 -> <=64 (`2 -> 4 b/CU`) & fix `up` 160/128 tail; (c) repack/balance generic MMVQ (`26 IQ2_S + 21 IQ3_XXS + 3 Q2_0`, 1.75/1.83 ms + ~0.5 ms 3:2 AR skew; check `milpster` `__vsub4`) | (a) long-ctx QSA speed; (b) ~0.5-1.0 ms; (c) ~0.8-1.5 ms FN TG | `kernel_budget.py` scratch/VGPR verify + sandbox + ABBA |
| **IK-SGEMM** | Inkling PP | Route small-M BF16/F32 loop-controller GEMMs off under-filled rocBLAS (1-16 CUs) to tiled split-K kernel | Up to 248 ms (**9.6% PP2048**), 244 ms (**5.0% PP4096**) | Trace CU grid check + `test-backend-ops` + PP ABBA |
| **FN-A** | Flash Next PP | Weight-stationary grouped MoE PP GEMM (8-16 q8_1 tokens in LDS vs 78 GB/s stock `mul_mat_q`) | **+12-14% FN PP** | Single-layer microbench vs `mul_mat_q` + KLD |
| **IK2..IK6** | Inkling TG | Zero-copy views for `RES_MIX`/loop msgs (`IK2`, 5 l/pass), fold norms/MoE glue (`IK3`, 10 l/pass), 2-kernel loop controllers (`IK4`), conv/clip fusions (`IK5`/`IK6`) | At 1.22-1.7 us/launch: `IK2` ~0.3-0.4 ms (1.5-2%), `IK3` ~0.6-0.8 ms (3-4%) | Build behind kill switch, trace + ABBA |
| **SP1 / SP2** | 27B & FN speculative | `SP1`: split-aware GPU top-k/greedy sampling under `-sm tensor` (avoid 1 MB logit readback/row; quartz does device top-64 per row + device verify, `engine.h:45-53`); `SP2`: FR-Spec reduced-vocab draft head | `SP1`: 1.5-3 ms/MTP2 step (4-7%); `SP2`: 1.5-2.5 ms/step (3-5%) | Host timestamps around sampler (`SP1`); vocab coverage stats (`SP2`) |
| **K2** | Both GPUs (diagnostic) | Test `HIP_FORCE_DEV_KERNARG=1` (VRAM kernargs) to isolate MI50 +0.6 us small-kernel lag | 0-1.1 ms (27B), 0-2.0 ms (FN) | Env-only alternating TG + trace (zero code) |
| **DSpark / IK8** | Inkling | Trace 1 DSpark verify step (4-tok verify + BF16 Markov head; 2.1-2.2x plain cost) and Inkling server vs bench gap (`IK8`, 37.1 vs 45.8 t/s) | Unlock DSpark >1.0x speedup; explain 8.7 t/s server gap | `rocprofv3` + `kernel_budget.py` on both |

### Tier 3: Lower ceiling, longer horizon & housekeeping

- **Prefill & decode micro-wins**: `N3` (two-half PP AR pipelining, +3-5% PP); `P1/T1` (`quantize_q8_1` / `scale_f32` elimination, ~0.4-0.7 ms at graph floor); `TB1 route 2` (fused single-block radix QSA top-k); `TB4` (fold MoE weighted reduce + shared-expert add); `BA-5` (block-size tuning for two-launch `res_mix`/`add_rms_norm_mul` in sandbox); `AR-LL` (single-trip kernel-polled LL AllReduce, <=0.5-0.6 ms ceiling); `Q3 route 2` (contiguous-block Q8_0 MMV + LDS re-reduce, <=1.9 ms ceiling); `SP4` (stop charging deferred DFlash replay to TG); `L-DF10` (trace 10 ms DFlash host gap); `P2`/`W64`/`S1`/`S4`/`M2`/`TB6`.
- **Memory fit & cleanup**: `IK9` (decode raw UE4M3 NVFP4 scales in kernel, -1.3 GiB VRAM); `U3` (unsloth shared MTP heads, -1.8 GiB); `D2`/`E1` (QSA/MoE cache under `-sm tensor`); `IK11` (log expert near-tie flips behind 0.005 ubatch KLD); `test-backend-ops` repack allocation mode; `ggml-cuda.cu` glue extraction.

### Architectural placement & deferred parameter tuning

- **Architectural placement (alongside kernel work per [GOALS.md](GOALS.md))**: `IK10` (1 Inkling instance per GPU for 2-4 slots or dual-model serving; NVFP4 uses ~19 of 32 GiB); asymmetric KV/draft placement leveraging MI60 (faster PP/tiny kernels) vs MI50 (equal HBM2 decode streaming).
- **Deferred parameter tuning (last stage)**: `MU1(a)` (slot-adaptive draft depth), `A2` (env default checks), `A1-FN` (`-ts` tweaks), `L-T16` (`-t 16`).

## Routes that lost (summary of all 28)

A lost implementation does not close a route. Historical measurements stay in [BENCHMARK-TIMELINE.md](BENCHMARK-TIMELINE.md). Counts: **10 OPEN, 9 NARROWED, 6 NOT TESTED, 2 CLOSED BY BOUND, 1 SUPERSEDED BY A WIN**.

| Id | Route | Status | Best-case bound | What lost & untested angle |
| --- | --- | --- | --- | --- |
| L-CMOED | MTP draft experts on CPU | OPEN | +17% TG (32K MTP2) | Full `-cmoed` dropped 43.8 -> 34.9; partial CPU offload / overlap untested |
| R-IN8 | One-block Inkling loop controller | OPEN | 1.3-1.8 ms (6-9%, graph-priced) | 1 block streamed 2.4 MB weights (40.5 -> 11.7 TG); 2-kernel split (`IK4`) untested |
| TB1 | Bitonic top-k for QSA indexer | OPEN | ~0.5 ms launches (graph-priced) | 1-row bitonic 2-10x slower than radix; fused single-block radix untested |
| M1 | Upstream `LLAMA_DFLASH_DEV=1` | OPEN | ~2 ms / 55 ms step (3.6%) | Mirrored meta tensor copied 200 MiB/step (128 vs 55 ms); ROCm0 direct view untested |
| L-Q81 | Multi-row planar Q8_1 in norm | OPEN | ~0.2 ms (27B, graph-priced) | Moving Q8_1 into norm shifted cost; fuse into producer epilogue untested |
| L-UB4K | `-b/-ub 4096` for 27B PP | OPEN | ~0.3% PP | -11.4% PP4096, cause unprofiled; ceiling too small to chase |
| R-IN5 | Unsloth scalar banded kernels alone | OPEN (won via tile) | Realized (+13-47% PP) | Scalar/per-query kernels halved PP (no matrix units); gfx906 tile path won |
| L-SML | `-sm layer` on Flash Next | OPEN | none | -20% TG (single GPU active at a time); multi-req pipeline untested |
| L-O12 | Oct 12 five quick items | OPEN | none | Quick reverts (`p-min`, `nc` unroll/fusion, MTP catchup, mirrored head); untested cleanly |
| L-NGM | `ngram-mod` speculation fallback | CLOSED (no gain) | none | Matched plain speed; dropped |
| Q3 | Faster 27B Q8_0 mat-vec | NARROWED | <=1.9 ms (7%) | Extra loads/rows/k-split all lost (VGPR/occ); contiguous-block + LDS re-reduce untested |
| A1-27B | Uneven `-ts` for 27B TG | NARROWED | ~1.1 ms lag (4%) | All uneven splits slowed TG (lag is in replicated small kernels); `K2`/fusions target cause |
| L-CPY | Replace HIP 2D memcpy in `cpy` | NARROWED | ~4% MTP3 kernel time | Neutral (hot rect copies were load-time upload); tensor-split staging remains (`F1`) |
| N2a | Parallel lane `hipGraphLaunch` | NARROWED | 27B 1-4% idle | Runtime lock contention, neutral; 27B already 96-99% busy; FN gap is `TB5` |
| N2b | Async staging of small F32 inputs | NARROWED | ~0 ms (delta ~0) | No gain on either model; old premise was profiler overhead |
| A7 | Higher occupancy 64x64 repacked GEMM | NARROWED | none (76% 27B PP) | Spilled 15-60 VGPRs, -24..57%; spill-free or larger `TN=3..4` tiles untested |
| L-D0 | MTP sidecar on ROCm0 (`-ts .46/.54`) | NARROWED | none | Step equal (49.9 vs 49.5 ms); earlier +4% was acceptance noise |
| L-MDR | `LLAMA_MTP_DEV_ROWS=1` | NARROWED | none | Step identical, disables deferred prefill (+18% TTFT) |
| L-NIGHT | ROCm nightly 27B slowdown check | NARROWED | none (regression check) | Step equal (40.28 vs 40.27 ms); CLI drop was greedy acceptance flip |
| SP1a | Enable backend sampling `-sm tensor` | NOT TESTED | 1.5-3 ms / MTP2 step | Aborted on PARTIAL tensor readback; split-aware GPU top-k (`SP1`) untested |
| N3 | Two-half PP AllReduce pipelining | NOT TESTED | +5.7% PP (5.4% exposed AR) | Nothing built yet |
| T1 | Fewer small launches in FN decode | NOT TESTED | ~0.5-0.7 ms (graph-priced) | Unfinished WIP crashed / wrong output; bisect one fusion at a time (`P1`) |
| F1 | Move TP host traffic off blits | NOT TESTED | ~4% MTP3 kernel time | Premise check showed 15-19% was model load upload |
| Q1 | Fold AR + add + norm + Q8_1 | NOT TESTED | <=0.8 ms (3.0%) | Unrolled lost; split blocked by aliasing; fold into AR kernel not built |
| R-IN9 | Dense bias chain slows Inkling @ 4K | NOT TESTED | 0.26 ms (0.9%) | Trace proved 93% growth was `flash_attn_ext_vec` VGPR occupancy (retired) |
| N1 | Keep both DFlash draft graphs | CLOSED BY BOUND | 0.7 ms / 80 ms (0.9%) | Premise check: draft already replays (67 replays); below 4.8% spread |
| L-MGC | Content-keyed meta draft graph | CLOSED BY BOUND | 0.7 ms / 80 ms (0.9%) | Broke acceptance (2.95 -> 1.1); superseded by N1 single-device replay |
| 27B-ARN | Micro-rewrite `add_rms_norm_mul` | SUPERSEDED BY WIN | 0.94 ms (3.5%) | 3 micro-variants neutral/slower; 2-launch split (`R-IN10`, `ae32e5e58`) won (+1.6% 27B TG) |

## Retired claims (do not revive)

| Retired claim | Current reality |
| --- | --- |
| Per-token host gap limits 27B & Inkling ("4.3 ms host gap") | 27B is 96-99% busy and Inkling 98% busy in steady state; old gaps were profiler cost. (**Exception**: Flash Next *does* have a real 3.04 ms pre-graph host token-start gap -> `TB5`.) |
| Every removed launch saves 4-5 us (or 2.2 us in HIP graphs) | 4.2 us is the `rocprofv3` floor; 2.0-2.6 us is bare eager launch. Captured HIP graphs cost **1.22 us/node** (pure floor) to **1.5-1.9 us** saved in ABBA. |
| Q3 DMA floor (25-41 us) proves decode `ggml_cuda_ar_kernel` is at floor | Category error: Q3 measures `hipMemcpyAsync` + host sync (bounds PP DMA), whereas decode AR polls mapped host memory from the GPU (20-26 us traced). |
| Old `Batch 1` `BA-1`/`BA-2` savings (0.49 + 0.41 = 0.92 ms) | Subtracted the 4.2 us empty-kernel profiler floor from non-trivial work (`topk_moe`, `k_argsort`) and used the distorted 1/3-drop token denominator. |
| Old `kernel_budget.py` 1/3-drop counts (1,645 on 27B, 2,811 on FN) | Inflated by load/warmup spillover and `n_tok*2/3` denominator; exact anchor-segmented steady counts are **1,595** (27B) and **2,747** (FN). |
| `flash_attn_ext_vec` slowed at 4K because it ran 1 block/head | Wrong mechanism: kernel already split keys over blocks; bottleneck was >128 VGPRs (`2 b/CU`), fixed by `__launch_bounds__(128, 4)` (`efb85a1a5`). |
| "MI60 is 10-20% faster everywhere" / uneven `-ts` helps 27B TG | HBM2 streaming is equal (891 vs 904 GB/s); MI50 is ~21% slower in PP GEMM and +0.6 us on small kernels. Every uneven `-ts` slowed 27B TG. |
| "FP4 / NVFP4 loses speed or precision on gfx906" | NVFP4 is Inkling's native checkpoint format; repacked as IQ4_NL nibble + split F16 scales it hits 836 PP / 47.0 TG (near IQ4_NL 879 / 49.0). |
| "F2 and P4 are validated default-on safe" | Opinion (grade D): `P4`/`F2` guards hardened on 2026-11-22, Q8-ref KLD for `P4` still pending. |
| "Byte-identical greedy text or 2-chunk KLD ~0.005 proves parity" | Both Flash Next (~4.5e-3) and Inkling (~4.9e-3) have MoE near-tie KLD floors; always pair KLD with CPU-reference unit tests. |
| Quantized KV (`-ctk q8_0`), removed env knobs, peer-write AR, `A4` = 4-7 ms | KV stays F16 per [GOALS.md](GOALS.md); ~30 sweep knobs and `tp-allreduce.cu` are archived/removed; `A4` (PCIe relocation) is out of software-only scope (~2-2.5 ms ceiling). |
