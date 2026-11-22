# gfx906 Benchmark Timeline (2026-08-12 to 2026-11-22)

Single lossless, append-only historical benchmark record for this gfx906 `llama.cpp` fork on two 32 GiB AMD gfx906 GPUs. Consolidates all historical benchmark data, profiles, microbenchmarks, multi-user serving sweeps, quality/KLD checks, and optimization wins.

Live current numbers and active worklist stay in [PERFORMANCE.md](PERFORMANCE.md). Measured hardware constants stay in [HARDWARE.md](HARDWARE.md).

## Standard Hardware, Build, and Test Baseline

### Host Hardware

| Component | Details |
| --- | --- |
| ROCm0 | AMD Radeon Instinct MI60 32 GiB (gfx906), 64 CUs, power cap 210 W (max 225 W), CPU root port PCIe Gen3 x8 |
| ROCm1 | AMD MI50 32 GiB (gfx906, Radeon Pro VII VBIOS), 60 CUs, power cap 170 W (max 178 W), PCH root port PCIe Gen4 x4 behind DMI |
| Interconnect | No direct P2P (`hipDeviceCanAccessPeer=0`); host-staged pinned AllReduce ~7 GB/s per direction per GPU (3.46 GB/s duplex) |
| Host CPU & RAM | 20 threads (hybrid P/E cores), 96 GiB RAM |
| Toolchain | TheRock ROCm (`rocm-sdk`), 10.2.0a20261012 (clang `6da73151`) through 2026-11-05, 10.3.0a20261105 (clang `36c1564f` / Clang 24.0.0) from 2026-11-06 onward |

### Standard Build and Run Settings

- Build flags (required since 2026-10-12 / confirmed 2026-10-29; omitting `max-ilp`/`-O3` costs ~2-3% on 27B):
  `-DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DLLAMA_OPENSSL=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release -DHIP_COMPILER=clang -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"`
- Standard 2-GPU TP settings (records note only what differs): both GPUs (`-dev ROCm0/ROCm1`), `-ngl 999 -sm tensor -ts 1/1 -fa on`, no env knobs.
- **Qwen3.8-27B Q8_0**: `-b 2048 -ub 2048 -t 20` (`-ub 512` in 2026-11-21/22 cross-model screens), F16 KV, `llama-bench` PP2048/TG128.
- **Qwen3.8 Flash Next UD-IQ3_XXS**: `-b 1024 -ub 512`, F16 KV (policy since 2026-11-06: F16 KV cache only; records before 2026-11-06 and the 2026-11-06 8K rows used Q8_0 KV).
- **Inkling 2 Mini**: single GPU ROCm0 (`-dev ROCm0`), `-ub 512` (or `-ub 2048` where noted), F16 KV.
- **Evidence grades**:
  - **A**: alternated (e.g. A/B/B/A) or repeated A/B on the real model, cooled below 60C (50C on 2026-11-22), with numbers.
  - **B**: single or one-order measurement, microbenchmark, or comparison against an older baseline.
  - **C**: estimate or arithmetic derivation.
  - **D**: assertion without a saved measurement record.

## Append Format

Add each new benchmark run as a dated subsection at the top of Section 2 (newest first), and append major milestones to the Section 1 tables. Existing records are not edited.

```markdown
### YYYY-MM-DD short title

Purpose in one sentence. Kind: regression screen | A/B experiment | profile.

- Source: `<commit>`; baseline `<commit or build_commit>`.
- Model: `<quant>` rev `<12 chars>`; draft `<file>` if any.
- Settings: standard 27B | standard Flash Next | standard Inkling, plus what differs.
- Workload: standard, or what differs; repeats and order (e.g. r3 alternating, below 50C).
- Predicted (from the card): predicted saving in ms per token with arithmetic.
- Measured: measured saving per kernel (trace) and end to end, with grade.
- Prediction error and card update: error in percent; "none" if inside ~20%; otherwise constant updated in HARDWARE.md.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | MTP2 | 514.0 | 60.7 | - | 2.6/3 | sha `ab12...` |
| candidate | MTP2 | 516.2 | 61.0 | +0.5% | 2.6/3 | same |

Verdict: one sentence, with grade. Not covered: what was not run.
```

---

## Section 1: Chronological Progression Summary (2026-08-12 -> 2026-11-22)

### 1.1 Master Landed Wins Index (W01-W37 and Config/Check Milestones K1x-K7x)

Compiled from `gfx906/investigation-notes/2026-11-22-process-retro/history-retro.md` plus `R-IN10` (2026-11-22). Pre-2026-10-29 fork commits were squashed into `f5031e739`, so earlier landing dates come from dated logs.

| ID | Name | Model(s) | Landed | Commit / Build | Measured Effect (Grade) | Origin | Mechanism |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W01 | I-5 host-staged HIP AllReduce default (upstream PR 27825 + HIP guard removal) | 27B (all TP) | 2026-09-10 (note 2026-09-29) | `ee5b990d1` + guard removal | 27B @16K: 325.49 / 18.68 -> 394.17 / 25.89 (+21.10% PP, +38.57% TG, B); AGENTS.md 2026-09-29 note: excluding HIP cost -21% PP2048, -38% TG256 (D) | upstream-port | Removed stale HIP overlap-only guard so generic host-staged BF16 pipeline runs |
| W02 | L-PD parallel lane dispatch | 27B, Flash Next | <= 2026-09-10 (rechecked 2026-10-29) | squashed (`f5031e739`) | Off costs ~1.5 t/s TG on both models (2026-10-29 A1 sweep: 27B 37.37->35.83, FN 37.27-37.99->35.82, B) | fork-port (mx-llama) | Both GPUs dispatched concurrently |
| W03 | R-FLAG compiler flags `max-ilp` and `-O3` | 27B | <= 2026-10-12 (rechecked 2026-10-29) | `build-ilp` | Without them: 27B PP 684->669 (-2.2%), TG 36.8->35.7 (-3.0%) (B) | config | `-mllvm -amdgpu-sched-strategy=max-ilp` + `-O3` |
| W04 | L-GR whole-token HIP graph replay (capture-safe AR via device counters) | 27B, FN, Inkling | 2026-10-12 | squashed (`f5031e739`) | 27B TG64 30.2 -> 34.77 (CLI 29.9 -> 34.4, text identical, B) | profile | Device-side AR counters + `prepare`/`launch_rank` enable graph capture without P2P |
| W05 | Fused `add + rms_norm + mul` with Q8_1 producer | 27B | 2026-10-12 | squashed (`f5031e739`) | Part of 27B TG 34.8 -> 36.2, CLI text identical (B) | profile | Single kernel at post-AR add+norm site filling Q8_1 activation cache |
| W06 | Q8_0 repacked mat-vec rows per workgroup = 8 (was 16) | 27B | 2026-10-12 | squashed (`f5031e739`) | 27B TG 36.2 -> 36.8 (B) | differential (16/8/4 sweep) | Better occupancy/register balance on gfx906 |
| W07 | Duplex copy-engine AllReduce (separate H2D stream) | 27B | 2026-10-12 | squashed (`f5031e739`) | 27B PP 599 -> 647 (+8%, B) | profile | H2D and D2H PCIe transfers overlap |
| W08 | L-AR AllReduce slab overlap in prefill (`GGML_CUDA_AR_SLABS=4`) | 27B | 2026-10-12 | squashed (`f5031e739`) | 27B PP2048 651.1 (0) / 670.7 (2) / 682.0 (4) / 680.1 (8) (+4.7% PP, TG neutral, B) | profile | Transfer of slab i overlaps GEMM of slab i+1 |
| W09 | Tiled-transpose dim-0 concat for GDN conv input | 27B | 2026-10-12 | squashed (`f5031e739`) | Part of PP bundle 551.8 -> 649.2 (own share D) | profile | Removes `concat_non_cont` (71-75 ms/ubatch) |
| W10 | Narrow Q8_0 mat-vec (2-8 cols): planar activations, 16-lane rows, LDS-staged activations | 27B (verify) | 2026-10-12 | squashed (`f5031e739`) | pp8 57.0 -> 45.9 ms, pp6 48.7 -> 41.4 ms, pp4 40.6 -> 37.2 ms; LDS 6-col -3%; df5 CLI 37.8 -> 38.4 (B) | profile | Weight-reuse narrow-batch mat-vec for verify widths |
| W11 | Draft repack, MTP repack, lazy draft-prompt replay | 27B | 2026-10-12 | squashed (`f5031e739`) | df5 n512 TG 42.9 -> 45.9; mtp3 n512 PP 512 -> 549.3, TG 52.9 -> 54.2 (B) | profile | Draft/MTP weights on repacked layout; replay deferred to first draft |
| W12 | Tap accumulation accepts M-RoPE text pos + tail deferral | 27B (DFlash) | 2026-10-12 | squashed (`f5031e739`) | df5 n64 PP 433.8 -> 547.8, L unchanged 3.71 (B) | bug | Qwen3.5 `n_pos=4` prevented tap accumulation; prompt gather 130->21 ms, tgtsync 206->0 ms |
| W13 | Single-GPU DFlash draft, private repacked head, async lag-one replay | 27B | 2026-10-12 | squashed (`f5031e739`) | 5-prompt mean TG 47.1 -> 55.5 vs MTP2 52.3, PP unchanged (B) | profile | Draft off TP path; private repacked head 4.3 -> 1.6 ms/step |
| W14 | L-MTP MTP2 + `--spec-draft-backend-sampling` default | 27B, FN | 2026-10-12 (rechecked 2026-10-29) | flags | Oct 12 n4000: 56.7-60.3 vs plain 31.7-32.3; Oct 29: 27B +68% (60.7 vs 36.2), FN +33% (49.6 vs 37.2) (B) | differential | Depth/mode sweep winner |
| W15 | Three-slot fixes: 2049-row persistent taps + per-seq DFlash prompt state | 27B | 2026-10-12 | `a39e959d9` + fix | Startup assert and multi-slot draft-cache errors fixed; 3 slots clean (D) | bug | Rounded reservation size and per-seq `in_prompt` tracking |
| W16 | Duplex-AR race fix (`src_read_done` event) | 27B | 2026-10-12 | squashed (`f5031e739`) | Fixed NaN MTP drafts / 0% DFlash acceptance on 1 MB+ non-slab ubatches (D) | bug | Event wait between stage-1 D2H and pool BF16 buffer reuse |
| W17 | R-HEAD row-first MMQ traversal restored on official-head port | Flash Next | 2026-10-29 | `418f905a4` port | Unrepaired port PP -28.3% (247.3 vs 345.1); repaired 351.30 / 39.51 vs 347.97 / 36.77, same output (B) | bug | Restored row-tile inner grid dimension for gfx906 non-stream-K MMQ |
| W18 | L-IQ2 gfx906 two-row IQ2_S one-token expert path | Flash Next | 2026-10-29 | `fe34c2bf` lib | 396.35 / 37.20 -> 398.80 / 38.08 (+2.37% TG, one pair, B) | unknown | Two even rows per wave in IQ2_S MMVQ |
| W19 | L-GM grouped MoE for 2-8 token verify (default on) | Flash Next | 2026-11-06 (exp 2026-10-25) | `298882544` | Oct 25 MTP2 47.65 -> 50.45 (+5.9% TG); Nov 6 100K MTP2 48.9 -> 51.8, MTP3 49.1 -> 52.8, byte-identical (B) | unknown | Decode each IQ2_S/IQ4_NL expert weight once for up to 8 verify tokens |
| W20 | S2 cheaper IQ sign apply (`iq_apply_signs`) | Flash Next | 2026-11-06 | `1785a4513` | r5 alternating: PP 541.45 -> 557.97 (+3.05%), TG 41.17 -> 43.27 (+5.11%), bit-exact (A) | fork-port (Strata idea) | `(g ^ m) + (m & 0x01010101)` replaces emulated `__vcmpne4`/`__vsub4` |
| W21 | L-RPK repack Q5_0 (native) and IQ4_XS (via IQ4_NL) | Flash Next (big quants) | 2026-11-06 | `ebd76ced1`, `3e19b6adc` | UD-Q3_K_XL @8K +2.3% TG (~+6% PP); UD-IQ4_XS `-ncmoe 3` +9% PP / +3.5% TG; PPL unchanged (B) | unknown | Map Q5_0/IQ4_XS onto repacked kernels |
| W22 | F2 fused hyper-connection mix op (`GGML_OP_DSV4_HC_MIX`) | Flash Next | 2026-11-06 | `0920eb0ac` | TG128 @8K 40.8 -> 46.0 (+12%); MTP3 100K step 52.18 -> 49.48 ms (-5.2%), TG 59.4 -> 62.7; KLD 4.4e-3 (B) | profile review | 7 launches -> 3 per HC mix for 1-8 tokens |
| W23 | P4 tiled split-K F32 GEMM for small-M (`hc_*`, `ssm_alpha/beta`) | Flash Next | 2026-11-06 | `52bcba210`, `9fb19f750`, `da11f5b99` | PP2048 @8K 508.9 -> 563.8 (+11%), PP512 +12%, TG unchanged; KLD 4.7-5.1e-3 (B) | profile review | Replaces slow rocBLAS SGEMM on small-M GCN shapes |
| W24 | K1 one `set_rows` for recurrent conv snapshots + vector F32 `set_rows` | 27B | 2026-11-06 | `cadb56aa2` | 27B MTP2 spec_accept 40.38 -> 38.95 ms/step (-3.5%), mean TG 62.4 -> 64.7, same acceptance 0.777 (B) | profile review (round 2) | One permuted view store instead of 3 per layer + float4 `set_rows` |
| W25 | B1 one-wave MoE routing kernel `topk_moe_wave_cuda<512,10>` + HIP ballot fix | Flash Next | 2026-11-10 | `6d8f0c1c7`, `457d4470b` | 28.8 -> 14.9 us/call; TG128 @8K 45.71 -> 47.53 (+4.0%, 4x r5 alternating, bit-exact, A) | profile review (round 2) | One wave per row, rolled k-loop, DPP min-reduce |
| W26 | Multi-slot MTP rollback for Flash Next (remove `n_rs_seq=0` clamp) | Flash Next | 2026-11-11 | `fbc067862` | 2 slots 1 user TG 32.5 -> 69.1 / 67.2 (+110%), ms/step 82.9 -> 44.3 / 45.7 (B) | differential | Uses plane-indexed snapshot path instead of full-state checkpoints; `seq_cp`/`find_slot` fixes |
| W27 | `kv_rows` flash attention on gfx906 tile kernel + sequential rollback split | 27B, FN | 2026-11-15 | `80212cad5`, `7ddd49693`, `3bd5e1909` | 27B 4x16K PP 462.3 -> 577.8 (+25.0%), 4-way TG neutral, single-user neutral (-0.4%/0.0%); FN neutral (A) | upstream-port (#29510) | Separate `use_kv_rows` tile kernels keep plain path spill-free; `get_n_kv_rows` work gate |
| W28 | U1 / R-CU1 device MoE id counting sort (unsloth #264) | Flash Next | 2026-11-21 | `175153274`, `400989a0d` | PP2048 @`-ub 2048` 877.4 -> 895.5 (+2.1%), @`-ub 512` 0.0% (inside noise, B) | fork-port (unsloth #264) | Device-side counting sort for >= 512 tokens; OOB id guard for meta expert slices |
| W29 | R-IN7a `CLIP_ROWS` fused op | Inkling 2 Mini | 2026-11-21 | `3253927e4` | IQ4_NL TG128 36.07 -> 39.15 (+8.5%), PP 565.3 -> 567.6 (A) | profile | Replaces 9-kernel q/k/r norm clip (3 per layer) with 1 kernel |
| W30 | R-IN7b `RES_MIX` fused op | Inkling 2 Mini | 2026-11-21 | `b6c927fd1` | IQ4_NL PP2048 568.8 -> 751.6 (+32%), TG 39.54 -> 40.45 (+2.3%) (A) | profile | Fuses concat chain, rms norm, 2 matmuls, softmax of block residual mix |
| W31 | R-IN7c `LOOP_GATE` + `RMS_NORM_CAT` fused ops | Inkling 2 Mini | 2026-11-21 | `f17889023` | IQ4_NL PP 754.5 -> 772.9 (+2.4%), TG 40.47 -> 41.43 (+2.4%) (A) | profile | Fuses elementwise parts of loop controllers (12 kernels -> 5); keeps matmuls parallel |
| W32 | R-IN7d `SCONV_RES` fused op | Inkling 2 Mini | 2026-11-21 | `85bb8a0f7` | IQ4_NL PP 774.0 -> 777.7, TG 41.56 -> 43.26 (+4.1%) (A) | profile | Fuses 4-tap short conv, dynamic taps, residual, and scale |
| W33 | R-IN4 banded relative-bias attention in tile kernel (unsloth #386 + gfx906 path) | Inkling 2 Mini | 2026-11-21 | `06aa25f25`, `d37d86dd7`, `2162c74dd` | IQ4_NL PP2048 ub 512: 775.4 -> 879.6 (+13.4%), ub 2048: 772.9 -> 1135.7 (+47%); 16K compute buf 8665 -> 3170 MiB (A) | fork-port (unsloth #386) | Banded bias lookup inside gfx906 `flash_attn_tile` for prefill |
| W34 | R-IN3 wide-block `res_mix_f32` / `rms_norm_cat_f32` | Inkling 2 Mini | 2026-11-21 | `a99a7930b` | IQ4_NL TG128 43.57 -> 47.73 (+9.5%) (A) | profile | 1024-thread blocks, warp reductions, single pass (was 26.7 / 11 us/call) |
| W35 | R-IN6 NVFP4 MoE experts on repacked path | Inkling 2 Mini | 2026-11-21 | `0ba4be7f1` | NVFP4 PP2048 637.8 -> 737.3 (+15.6%), TG128 37.9 -> 45.5 (+20%); +1,305 MiB VRAM (A) | differential | Nibble-plane layout variant of IQ4_NL with 2 F16 scales per 32 values |
| W36 | IK1 `flash_attn_ext_vec` launch bounds `(128, 4)` on HIP | Inkling 2 Mini | 2026-11-22 | `efb85a1a5` | NVFP4 TG @4K: 38.19 -> 43.12 (+12.9%, A); @16K: 33.50 -> 40.53 (+21%, B) (commit msg standalone: 38.3->41.4 / 33.6->39.0) | profile + card | VGPR cap raised occupancy from 2 blocks/CU |
| W37 | R-IN10 two-launch `res_mix` and `add_rms_norm_mul` at small row counts | 27B, Inkling 2 Mini | 2026-11-22 | `ae32e5e58` | 27B TG128 37.35 -> 37.96 (+1.6%, A, 4 pairs <50C); Inkling NVFP4 TG128 45.94 -> 46.97 (+2.2%, A), IQ4_NL 47.79 -> 48.98 (+2.5%, B) | profile + card (one-block rule) | Partials in launch 1 (many blocks), output in launch 2; `res_mix` 17.0->9.5 us, `add_rms_norm_mul` 11.37->9.05 us |

Config / Check / Aggregate rows (Type K from `history-retro.md`):
- **K1x (R-IN1)**: All 2026-11-21 Inkling work combined: IQ4_NL PP2048 564.2 -> 878.6 (+55.7%), TG128 36.28 -> 47.75 (+31.6%, A); NVFP4 490.8/30.5 (merge-only, B) -> 836.3/45.80 (+70.4%/+50.2%).
- **K2x (R-IN2)**: 2026-11-21 Inkling work left 27B (+0.2% PP/+0.2% TG) and Flash Next (+0.4% PP/+0.6% TG) unchanged (A).
- **K3x (R-100K)**: 2026-11-06 Flash Next UD-IQ3_XXS MTP3 Q8_0 sidecar at 100K: 52.2/52.5 TG (+26% vs plain, CLI, B).
- **K4x (R-F2P4)**: 2026-11-06 Flash Next MTP3 100K (62.7 t/s) matched 27B MTP2 (62.5 t/s) on 5-prompt server suite (B).
- **K5x (R-32K)**: 2026-11-05 27B speculative mode by depth: DFlash5 67.4/67.7 at 2K, MTP2 42.7 vs 39.9 at 32K (single/double CLI runs, B; 2K winner noisy).
- **K6x (R-CU2-4)**: Catch-up merges `4b77dee05` (11-05), `9d25e0014` (11-13), `edec462cc` (11-18), `46eb1742d` (11-21), `e6d624b9a` (11-21): no regressions (B/A).
- **K7x (A3)**: 2026-11-05 sysfs read confirmed ROCm1 already at max 178 W cap (ROCm0 225 W) (B).

---

### 1.2 Model Progression Tables (Chronological, Oldest -> Newest)

#### A. Qwen3.8-27B Q8_0 (and Aug 2026 Qwen3.6-27B predecessor)

| Date | Commit / Build | Milestone / Change | Exact Config / Workload | PP t/s | TG t/s | Delta / Note |
| --- | --- | --- | --- | ---: | ---: | --- |
| 2026-08-12 | `b8efb9510` (10275) | Qwen3.6-27B Q8_0 production reference | 2 GPU TP, d16384, b/ub 2048, PP2048/TG1024, r1 | 282.98 | 19.82 | Baseline (score 48.12) |
| 2026-08-13 | `b8efb9510-dirty` (10275) | Exact BF16 two-slab Q8 TP overlap (C3) | Same (Qwen3.6-27B Q8_0, d16384) | 318.78 | 19.63 | +12.65% PP vs prod |
| 2026-08-13 | `b8efb9510-dirty` (10275) | Qwen3.6-27B UD-Q8_K_XL BF16 overlap | Same (33.31 GiB vs 27.04 GiB) | 314.85 | 15.77 | -19.6% TG vs Q8_0 |
| 2026-08-14 | `46b95d97e-dirty` (10279) | First Qwen3.8-27B Q8_0 run + MTP2 | d16384, b/ub 2048, PP2048/TG1024; CLI MTP2 n1024 | 319.48 (MTP2 243.9) | 18.40 (MTP2 29.7) | First Qwen3.8 baseline |
| 2026-09-01 | `0eadefebd` + `63ed8cd1a` | Upstream+MX merge CLI check | CLI `flake.nix`, c16384, b2048, BF16 overlap | 320.0 | 11.3 | Post-merge CLI |
| 2026-09-10 | `434ddbbc0` (clean head) | Clean upstream llama.cpp baseline | d16384, b/ub 4096, PP2048/TG256, r1 | 196.97 | 20.13 | Upstream reference |
| 2026-09-10 | `c0df68229` + `434ddbbc0` | Merged MX + upstream (fallback AR) | Same (d16384, b/ub 4096, r1) | 325.49 | 18.68 | Stale HIP guard hid AR |
| 2026-09-10 | `c0df68229` + guard fix | W01: HIP overlap-only guard removed | Same (d16384, b/ub 4096, r1) | 394.17 | 25.89 | +21.1% PP, +38.6% TG |
| 2026-09-10 | later Sep 10 ref | Later Sep 10 reference checkpoint | Same (d16384, b/ub 4096) | 414.24 | 30.28 | Cited in Oct 1 kit |
| 2026-10-01 | `38b4da7ba` / `8535054eb` | Upstream `e358d591` port checkpoint | d16384, b/ub 4096, PP2048/TG256, r1 | 424.57 | 28.86 | Pre-merge ref 431.13 / 27.81 |
| 2026-10-11 | `5d6e2c412` (11547) | Head merge w/ gfx906 GDN | d0, b/ub 2048, PP2048/TG256, r1 | 506.89 | 30.01 | Pre-merge `before`: 499.66 / 31.54 |
| 2026-10-12 | `8d73dc74e` (`build-ilp`) | Pre-optimization baseline (1-GPU & TP) | 1-GPU ROCm0 PP512/TG128; 2-GPU TP d0 PP2048/TG128 r3 | 437.2 (1-GPU) / 551.8 | 22.76 (1-GPU) / 30.16 | Baseline for Oct 12 push |
| 2026-10-12 | `build-ilp` + W04 | Whole-token HIP graph capture-safe AR | d0, llama-bench TG64 | - | 34.77 | +4.6 t/s TG (+15%) |
| 2026-10-12 | `build-ilp` + W05-W07,W09 | Fused add+norm+mul+Q8_1, ROWS=8, duplex AR, concat | d0, PP2048/TG128 r3 | 649.2 | 36.8 | +18% PP, +22% TG vs 551.8/30.16 |
| 2026-10-12 | `build-ilp` + W08 | Slab-overlapped prefill AR (`AR_SLABS=4`) | d0, PP2048/TG128 r2 | 682.0 | 36.8 | +4.7% PP (651.1->682.0) |
| 2026-10-12 | `build-ilp` + W10-W14 | Speculative push (MTP2 / DFlash5) | CLI `flake.nix` n4000; 5-prompt mean n512 | 553-558 (CLI) | 60.3-61.6 (MTP2 n4000); 55.5 (DF5 5-pr) | Single-GPU DFlash beat MTP2 52.3 |
| 2026-10-25 | `766f7c5e0` (11776) | CLI 100K capacity check (6K used) | CLI `flake.nix` n4000, c100096 | 549.7 / 545.2 / 543.7 | 56.8 (MTP2) / 60.0 (MTP3) / 59.9 (DF7) | Single runs |
| 2026-10-29 | `36a0e2ffc` (U) vs `f5031e739` (D) | Official upstream vs streamlined rebuild D | d0, PP2048/TG128 r2; CLI n512 MTP2 | 448.60 -> 685.34 | 32.77 -> 36.67 (MTP2 60.7) | +53% PP, +12% TG vs official U |
| 2026-11-05 | `bc0ce9691` (`4b77dee05` merge) | Catch-up + 2K/32K spec sweep | d0 PP2048/TG128; CLI 2K / 32K | 696.88 (bench); 541.2 (32K) | 37.90 (bench); 67.6 DF5 (2K); 42.7 MTP2 (32K) | Plain 32K: 29.4 TG |
| 2026-11-06 | `cadb56aa2` | W24 (K1): single `set_rows` conv snapshots | Server 5-prompt `spec_accept.sh` MTP2 | - | 62.4 -> 64.7 (40.38 -> 38.95 ms/step) | -3.5% step time, acc 0.777 |
| 2026-11-10 | `0f7ce1b04` (`aa75ff323` merge) | Token budget unprofiled baselines | d0 / d32K / 1-GPU@0, PP2048 & TG64/32 | 705.3 (d0) / 457.8 (32K) | 37.45 (d0) / 33.48 (32K) / 23.42 (1-GPU) | MTP2 62.1 (40.6 ms), DF5 69.3 (54.9 ms) |
| 2026-11-11 | `dc9c4e367` / `fbc067862` | Multi-user 4-slot 262K unified serving | 4 slots c262144 unified, MTP2; 200K fill | 705.5 (bench); 219 (200K) | 37.7 (bench); 65.6 (1-usr); 41.2 (4-usr agg) | 200K long-fill TG 28.3 |
| 2026-11-15 | `7ddd49693` (`build-kvrows-20261116`) | W27: `kv_rows` FA on gfx906 tile kernel | `bench_multiuser.sh` 4x16K (A); bench d0 | 462.3 -> 577.8 (4x16K); 699.4 (d0) | 81.1 -> 80.9 (4-way); 37.22 (d0) | +25.0% 4x16K PP (Grade A) |
| 2026-11-18 | `d7ea110cb` (`edec462cc` merge) | Catch-up screen | d0 PP2048/TG128; MTP2 5-pr; 4x16K | 712.6 (d0); 572.9 (4x16K) | 37.93 (d0); 66.2 (MTP2 38.6 ms); 81.4 (4x16K) | +1.1% MTP2 vs 11-16 |
| 2026-11-21 | `698f466dd` / `400989a0d` | Catch-up `46eb1742d` + knob cleanup | d0 ub 2048; d0 ub 512 | 698.9 (ub2048) / 614.5 (ub512) | 37.23 (ub2048) / 37.21 (ub512); 63.9 (MTP2) | Neutral vs 11-18 |
| 2026-11-22 | `ae32e5e58` (`build-final-20261122`) | W37 (R-IN10): two-launch `add_rms_norm_mul` | d0, `-ub 512`, 4 alternated pairs <50C (A) | 614.7 -> 614.0 (-0.1%) | 37.35 -> 37.96 (+1.6%) | First 27B TG kernel win in 3 wks (Grade A) |
| 2026-11-22 | `3a8d768df` (`build-dspark-20261122`) | DSpark port screen (neutral on 27B) | d0 `-ub 512`; MTP2 5-prompt | 596-615 (d0) | 37.97 (d0); 62.1 (MTP2, 40.7 ms/step, acc 0.78) | No regression |

#### B. Qwen3.8 Flash Next (UD-IQ3_XXS primary, plus UD-Q3_K_XL / UD-IQ4_XS)

| Date | Commit / Build | Milestone / Change | Exact Config / Workload | PP t/s | TG t/s | Delta / Note |
| --- | --- | --- | --- | ---: | ---: | --- |
| 2026-09-01 | `0eadefebd` + `63ed8cd1a` | First Flash Next run (layer split) | UD-IQ3_XXS, `-sm layer`, d4096, b2048 | 223.37 (PP2048) | 23.30 (TG128) / 23.44 (TG512) | CLI 223.7 / 23.2 |
| 2026-10-25 | `344be0a60` overlay | First 100K TP run (Q8_0 KV, rev `c8b5954a`) | CLI 4K prompt, c100096, b/ub 2048, Q8 KV | 552.3 | 36.7 | Short 2K CLI: 434.7 / 38.0 |
| 2026-10-25 | `b11929-fdc8e5b65` | First Flash Next MTP2 (Q8_0 sidecar) | CLI 4K prompt, c100096, b/ub 1024/512, Q8 KV | 396.9 -> 368.5 | 37.3 -> 42.9 | +15% TG from MTP2 |
| 2026-10-25 | `flashnext-fast` on `fdc8e5b65` | W19 exp: grouped MoE verify (IQ2_S/IQ4_NL) | CLI 2K prompt, c100096, MTP2 `-devd ROCm1` | 330.3 -> 372.9 | 47.65 -> 50.45 | +5.9% TG, identical text |
| 2026-10-29 | `418f905a4` port | W17: row-first MMQ traversal fix; W18: IQ2_S 2-row | CLI MTP2 (W17); CLI 4K plain (W18) | 247.3 -> 351.3 (W17); 398.8 (W18) | 36.77 -> 39.51 (W17); 37.20 -> 38.08 (W18) | Fixed -28% PP port regression |
| 2026-10-29 | `36a0e2ffc` (U) vs `f5031e739` (D) | Official U vs rebuild D (Q8_0 KV, old upload) | llama-bench b/ub 1024/512 r2/r5; CLI MTP2 | 303.0 -> 527.6 (r5 520.1) | 39.66 -> 39.09 (CLI MTP2 49.6) | +74% PP vs official U |
| 2026-11-06 | `1785a4513` | W20 (S2): cheaper IQ sign apply (old upload) | llama-bench b/ub 1024/512 r5 alternating (A) | 541.45 -> 557.97 | 41.17 -> 43.27 | +3.05% PP, +5.11% TG (Grade A) |
| 2026-11-06 | `ebd76ced1` / `3e19b6adc` | W21: Q5_0/IQ4_XS repack + new upload `d9620119` | @8K Q8 KV: Q3_K_XL (+Q8 table); Q3_K_XL 32K MTP2 F16 KV | 470.4 -> 541.0 (533.1 Q8 tbl); 390.5 (MTP2) | 39.23 -> 40.12 (40.54 Q8 tbl); 51.5 (MTP2 32K) | IQ4_XS `-ncmoe 3`: 398.4/29.56 -> 434.4/30.58 |
| 2026-11-06 | `298882544` (`build-fnspeed`) | W19 default: grouped MoE + MTP3 @100K F16 KV | CLI 2K, c100352, `-ts 0.52,0.48`, `-devd ROCm1` | 407.8 -> 438.9 | 41.7 (plain) -> 52.2 / 52.5 (MTP3) | +26% vs plain, +8% vs MTP3 off |
| 2026-11-06 | `0920eb0ac` / `52bcba210` | W22 (F2) HC mix + W23 (P4) small-M F32 GEMM | bench @8K; 5-prompt server MTP3 100K | 508.9 -> 563.8 (+11% P4) | 40.8 -> 46.0 (+12% F2); MTP3 59.4 -> 62.7 (49.48 ms) | Matches 27B MTP2 (62.5) |
| 2026-11-10 | `0f7ce1b04` (`aa75ff323` merge) | Token budget unprofiled baselines (F16 KV) | bench b/ub 1024/512: d0, d8K, d32K | 591.4 (d0) / 441.6 (32K) | 43.68 (d8K) / 41.32 (d32K) | 2908 kernels/token |
| 2026-11-10 | `6d8f0c1c7` (`build-play-b1`) | W25 (B1): one-wave `topk_moe_wave_cuda<512,10>` | bench TG128 @d8192, 4x r5 alternating (A) | - | 45.71 -> 47.53 (+4.0%) | 28.8 -> 14.9 us/call (Grade A) |
| 2026-11-11 | `fbc067862` (`build-mslot-ring`) | W26: multi-slot MTP rollback + ub384 90K preset | 2-slot c32K/c90K MTP3; 4-slot c131K no-draft; bench d0 | 609.1 (bench); 470 (srv) | 50.5 (bench); 32.5 -> 69.1/67.2 (2-slot 1-usr); 64.7 (90K) | 2x30K concurrent & 81K fill verified |
| 2026-11-13 | `3423a1c0f` (`9d25e0014` merge) | Upstream v1.0.0 GCN MMQ retune + MoE cache probe | bench d0 ub512/ub2048; d16K f16 vs q8_0 KV | 753 (ub512) / 652 (ub2048); 414 (16K) | 39.8 (ub512) / 40.4 (ub2048); 36.6 f16 vs 34.2 q8 KV | `-ncmoe 4` + 2 GiB cache: 39.0 short / 34.8 long |
| 2026-11-15 | `7ddd49693` (`build-kvrows-20261116`) | W27: `kv_rows` check | 4x16K F16 KV (A); single-user q8 KV | 471.8 (4x16K); 652.1 (d0) | 84.0 (4-way); 49.0 (d0) | Neutral on FN |
| 2026-11-18 | `d7ea110cb` (`edec462cc` merge) | Catch-up (stream-k MMQ #30022) | d0 b/ub 1024/512; MTP3 5-pr; 4x16K | 664.0 -> 672.4 (+1.3%); 478.7 (4x16K) | 50.70 (d0); 59.6 (MTP3 49.3 ms); 84.5 (4x16K) | Acc shifted 0.682 -> 0.660 |
| 2026-11-21 | `400989a0d` (`build-u1sort-20261121`) | Catch-up `46eb1742d` + W28 (unsloth #264 sort) | d0 ub512 / ub2048; 4x16K | 657.8 (ub512) / 895.5 (ub2048); 486.9 (4x16K) | 50.55 (d0); 59.9 (MTP3); 85.98 (4x16K) | +2.1% PP at ub2048 |
| 2026-11-22 | `ae32e5e58` (`build-final-20261122`) | Final 2026-11-22 screen (neutral on FN) | d0 ub 512, ABBA <50C (A) | 655.9 -> 651.0 (-0.8%) | 50.28 -> 50.46 (+0.4%) | Inside noise |
| 2026-11-22 | `3a8d768df` (`build-dspark-20261122`) | DSpark port screen (neutral on FN) | d0 ub 512; MTP3 5-prompt | 640.5-660.2 | 49.8-50.5 (d0); 57.8 (MTP3, acc 0.64) | No regression |

#### C. Inkling 2 Mini (ROCm0 single GPU, IQ4_NL / Q8_0 / NVFP4+Q8_0)

| Date | Commit / Build | Milestone / Change | Exact Config / Workload | PP t/s | TG t/s | Delta / Note |
| --- | --- | --- | --- | ---: | ---: | --- |
| 2026-11-21 | `400989a0d` (`build-u1sort`) | Baseline before Inkling ops (no NVFP4 yet) | ROCm0 ub 512: IQ4_NL / Q8_0 | 564.2 (IQ4_NL) / 566.1 (Q8_0) | 36.28 (IQ4_NL) / 34.47 (Q8_0) | 7,219 launches/token |
| 2026-11-21 | `4c5738580` (`build-catchup-e6d6`) | Upstream `e6d624b9a` merge enables NVFP4 | ROCm0 ub 512 (ub 2048): NVFP4+Q8_0 | 490.8 (509.7 @ub2048) | 30.5 | Generic NVFP4 kernels |
| 2026-11-21 | `3253927e4` | W29: `CLIP_ROWS` fused op | IQ4_NL ub 512 ABBA (A) | 565.3 -> 567.6 | 36.07 -> 39.15 | +8.5% TG |
| 2026-11-21 | `b6c927fd1` | W30: `RES_MIX` fused op | IQ4_NL ub 512 ABBA (A) | 568.8 -> 751.6 | 39.54 -> 40.45 | +32% PP, +2.3% TG |
| 2026-11-21 | `f17889023` | W31: `LOOP_GATE` + `RMS_NORM_CAT` | IQ4_NL ub 512 ABBA (A) | 754.5 -> 772.9 | 40.47 -> 41.43 | +2.4% PP, +2.4% TG |
| 2026-11-21 | `85bb8a0f7` (`build-sconvfuse`) | W32: `SCONV_RES` + image crash fix `ddca7dd9b` | ub 512 ABBA (A): IQ4_NL / Q8_0 / NVFP4 | 775.1 / 772.7 / 641.0 | 43.35 / 40.33 / 35.43 | IQ4_NL +37.4% PP, +19.0% TG |
| 2026-11-21 | `a99a7930b` | W34: wide-block `res_mix` / `rms_norm_cat` | IQ4_NL ub 512 ABBA (A) | 771.1 -> 776.9 | 43.57 -> 47.73 | +9.5% TG |
| 2026-11-21 | `2162c74dd` | W33: banded bias in gfx906 tile kernel | IQ4_NL ub 512 (ub 2048) ABBA (A) | 775.4 -> 879.6 (772.9 -> 1135.7) | 47.7 -> 47.7 | +13.4% PP ub512, +47% ub2048 |
| 2026-11-21 | `0ba4be7f1` (`build-inkling-final`) | W35: NVFP4 repacked experts | ub 512 ABBA (A): NVFP4 / IQ4_NL / Q8_0 | 836.3 / 878.6 / 880.1 | 45.80 / 47.75 / 44.41 | NVFP4 +70.4% PP, +50.2% TG vs merge-only; 4,128 launches/tok |
| 2026-11-22 | `efb85a1a5` + `ae32e5e58` (`build-final-20261122`) | W36: `flash_attn_ext_vec` `(128,4)` + W37: two-launch `res_mix` | ROCm0 ub 512: NVFP4 / IQ4_NL; NVFP4 @4K & @16K ctx | 835.7 (NVFP4) / 878.6 (IQ4_NL) | 46.97 (+2.2% NVFP4) / 48.98 (+2.5% IQ4_NL); @4K 38.19 -> 43.12 (+12.9%); @16K 33.50 -> 40.53 (+21%) | Fixed long-ctx attention occupancy |
| 2026-11-22 | `3a8d768df` (`build-dspark-20261122`) | Opt-in DSpark drafting (`#32218` port) | `llama-server` NVFP4 ROCm0 greedy 200 tok (short / 3255-tok) | 838.2 (bench plain) | Plain 46.9 (bench); server off->on: 37.65->33.65 (short), 36.55->37.50 (3255-tok) | Opt-in; step ~2.1-2.2 plain tok |

#### D. Gemma 4 31B-it Q8_0 and Muse-Glimmer-30B Q8_0

| Date | Commit / Build | Model & Quant | Exact Config / Workload | PP t/s | TG t/s | Note |
| --- | --- | --- | --- | ---: | ---: | --- |
| 2026-09-10 | `9b06d8c7a` (10900) `02-rocm-host-allreduce` | Muse-Glimmer-30B Q8_0 (`faa5b025c584`) | 2 GPU TP, d16384, b/ub 4096, t4, PP2048/TG256, r1 | 375.06 | 29.43 | Host-staged AR only |
| 2026-09-10 | `fd04166db` (10902) `03-mxxm-repack-moe` | Muse-Glimmer-30B Q8_0 (`faa5b025c584`) | Same | 450.57 | 29.55 | +20.1% PP from repack |
| 2026-09-10 | `0abd145ec` (10903) `04-vr-bench-harness` | Muse-Glimmer-30B Q8_0 (`faa5b025c584`) | Same | 453.13 | 27.20 | Harness checkpoint |
| 2026-09-10 | `fd04166db` (10902) `03-mxxm-repack-moe` | Gemma 4 31B-it Q8_0 (`c1ac76e99d55`) | 2 GPU TP, d16384, b/ub 4096, t4, PP2048/TG256, r1 | 318.76 | 22.50 | First Gemma 4 TP record |
| 2026-09-10 | `0abd145ec` (10903) `04-vr-bench-harness` | Gemma 4 31B-it Q8_0 (`c1ac76e99d55`) | Same | 322.71 | 21.18 | Harness checkpoint |
| 2026-11-10 | `0f7ce1b04` (`build-exp-spec`) | Gemma 4 31B-it Q8_0 (`c1ac76e99d55`) | 2 GPU TP, d0, b/ub 2048, PP2048 / TG64 (unprofiled) | 602.45 +- 1.40 | 32.68 +- 0.15 | Token budget baseline (1463 kernels/tok, 120 AR) |
| 2026-11-10 | `gfx906/server/serve.sh` router smoke | Gemma 4 31B-it Q8_0 (router preset `gemma4`) | `llama-server` router mode, 96-token reply (B) | - | 32 | Router smoke: 27B 51 / 27B-DF 61 / FN 48 / G4 32 |

---

## Section 2: Detailed Dated Records (Newest First, 2026-11-22 -> 2026-08-12)

### 2026-11-22 Inkling 2 DSpark block drafting (opt-in, not a speed gain yet)

Block drafter for Inkling 2 Mini ported from open upstream PR ggml-org#32218 (`0a9105de4`, applied with no conflicts; one fork comment restored by hand). Nothing changes unless `--spec-type draft-dspark` is passed.

- Source: `3a8d768df`; baseline `build-final-20261122` (`ae32e5e58`). Build: `build-dspark-20261122`.
- Model: Inkling 2 Mini NVFP4+Q8_0 (`Inkling-2-Mini-NVFP4-Q8_0.gguf`); draft `unsloth/Inkling-2-Mini-GGUF` `dspark-Inkling-2-Mini-Q8_0.gguf` (402 MB, HF cache).
- Settings: publisher's settings `--spec-type draft-dspark --spec-draft-n-max 3 -ngld 99` (no parameter tuning). `llama-server`, ROCm0, greedy, 200 tokens, order off/on/on/off, GPU <= 40C (Grade A for order; one prompt each).

| Prompt | TG off | TG on | Accepted / drafted | Mean accepted length |
| --- | ---: | ---: | ---: | ---: |
| short | 37.5, 37.8 | 33.5, 33.8 | 93 / 310 (0.30) | 1.89 |
| 3255 tokens | 36.6, 36.5 | 37.6, 37.4 | 109 / 265 (0.41) | 2.22 |

- Step cost derivation: speculative step takes `1.89 / 33.65 = 56 ms` (short) and `2.22 / 37.5 = 59 ms` (long) vs 26.6 and 27.4 ms for a plain token (~2.1-2.2 plain tokens per step for 1.9-2.2 returned). Break-even needs step < 1.9 plain tokens at this acceptance. Conjecture: 4-token verify batch leaves single-token paths (narrow mat-vec, one-token MoE fusion, fused ops), and draft Markov head is BF16.
- Lossless check (greedy on vs off): equal for 200 tokens on the 3255-token prompt; short prompt differs repeatably from byte 40 ("earlier techniques" vs "earlier forms"), where off logprobs are a near tie (`' forms' -1.23`, `' techniques' -1.42`, consistent with 0.005 KLD from batching alone).
- Cross-model regression checks on `build-dspark-20261122` vs `build-final-20261122` (order old, new, new, old; Grade A/B):
  - 27B Q8_0 plain: TG 38.03, 37.93, 38.00, 38.07; PP 598.7, 614.6, 596.9, 596.2.
  - Flash Next UD-IQ3_XXS plain: TG 49.98, 49.79, 50.49, 50.35; PP 640.5, 660.2, 658.9, 656.2.
  - Inkling NVFP4 plain (`llama-bench`): PP2048 838.2, TG128 46.9.
  - Speculative modes (`spec_accept.sh` defaults, Grade B): 27B MTP2 mean TG 62.1, acceptance 0.78, 40.7 ms/step; Flash Next MTP3 mean TG 57.8, acceptance 0.64.
- Open issue: `test-backend-sampler` `test_greedy` aborts on Inkling with `get_sub_batch: failed to add embedding at index 0 (error -2)` (not checked without port; passes on 27B).

### 2026-11-22 Decode attention occupancy, two-launch reductions, and hardware card calibration

Inkling 2 Mini retains speed at long context (4K TG 38.2 -> 43.1 t/s, 16K 33.5 -> 40.5 t/s), 27B gets its first kernel gain in three weeks (+1.6% TG), and gfx906 hardware constants are calibrated via synthetic microbenchmarks.

- Source: `efb85a1a5` (`flash_attn_ext_vec` launch bounds `(128, 4)` on HIP), `ae32e5e58` (two-launch `res_mix` and `add_rms_norm_mul` at small row counts). Old build `build-inkling-final-20261121` (`0ba4be7f1`), new build `build-final-20261122` (`ae32e5e58`).
- Workload: `llama-bench -r 2`, order old/new/new/old, each run started below 50C (archived origin: `build-final-20261122/validation/final/`, `investigation-notes/2026-11-22-hardware-card/final_validation_3.sh`).

| Model | Config | PP2048 old | PP2048 new | TG128 old | TG128 new | Grade |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 27B Q8_0 | 2 GPUs, `-sm tensor`, ub 512, two rounds (4 runs each) | 614.7 | 614.0 (-0.1%) | 37.35 | 37.96 (+1.6%) | A |
| Flash Next UD-IQ3_XXS | 2 GPUs, `-sm tensor`, ub 512 | 655.9 | 651.0 (-0.8%) | 50.28 | 50.46 (+0.4%) | A |
| Inkling 2 Mini NVFP4 + Q8_0 | ROCm0, ub 512 | 836.0 | 835.7 | 45.94 | 46.97 (+2.2%) | A |
| Inkling 2 Mini IQ4_NL | ROCm0, ub 512 (1 run each) | 880.2 | 878.6 | 47.79 | 48.98 (+2.5%) | B |

- 27B TG individual runs: old `37.54, 37.22, 37.41, 37.24`; new `38.02, 37.90, 38.06, 37.87` (an earlier hot run gave +2.9%; cool alternated gives +1.6%).
- Inkling 2 Mini NVFP4 generation at depth (`llama-batched-bench -npl 1 -ub 2048`):
  - 4096-token context: TG old `38.25, 38.13` (mean 38.19) -> new `43.19, 43.05` (mean 43.12, **+12.9%**, Grade A, alternated). PP ranged 746 to 819 t/s unrelated to build. (Note: standalone commit `efb85a1a5` msg earlier recorded 38.3 -> 41.4 at 4K and 33.6 -> 39.0 at 16K before `ae32e5e58`.)
  - 16384-token context: PP old 657.1 -> new 658.6; TG old 33.50 -> new 40.53 (**+21.0%**, Grade B, 1 run each).
- Predicted (from card) vs measured:
  - `flash_attn_ext_vec` at 4K context: predicted ~70 us/call from one-block rule, traced 85.8 us (was 15.6 us at 16 tokens; 93% of context slowdown). Mechanism differed: kernel already split keys over blocks, but used >128 VGPRs (only 2 blocks/CU, ~3 rounds). Fixed by `__launch_bounds__(128, 4)` on HIP.
  - Two-launch `res_mix_f32`: predicted ~5.4 us pair (~7.8 us traced), traced 17.0 -> 9.5 us.
  - Two-launch `add_rms_norm_mul_f32`: predicted ~7.8 us traced, traced 11.37 -> 9.05 us (block size not tuned).
- Correctness & serving checks:
  - `test-backend-ops` on ROCm0: MUL_MAT 2727/2727, MUL_MAT_ID 1162/1162, FLASH_ATTN_EXT 4420/4420, FLASH_ATTN_EXT_BANDED 71/71, RMS_NORM 121/121, ADD_RMS_NORM_MUL 28/28, RES_MIX 9/9, LOOP_GATE 3/3, RMS_NORM_CAT 6/6, CLIP_ROWS 9/9, SCONV_RES 12/12, DSV4_HC_MIX 12/12.
  - Greedy text old vs new (48 tokens): 27B and Flash Next byte-identical; Inkling NVFP4 equal for 88/213 bytes (moves with rounding). CPU check ends without assert.
  - `llama-server` NVFP4 + mmproj: text TG 37.9, image request PP 461.2, TG 36.7, stays up.

#### Hardware Card Synthetic Microbenchmark Calibration (2026-11-22, Grade B, ~8 GPU-min)

Archived origin: `investigation-notes/2026-11-22-hardware-card/RESULTS.md` (`q1.cpp`, `q2.cpp`, `q3.cpp`, `single_block_bench.cpp`, `launch_floor_bench.cpp`).

- **Launch floor & graph-node constants**:
  - App launch cost: ~2.2 us/kernel (1.5-1.9 us saved per removed launch in 3 Inkling ABBA kill-switch tests, Grade A, 2026-11-21).
  - Bare HIP loop launch: 2.0 us (MI60 ROCm0), 2.6 us (MI50 ROCm1); `rocprofv3` adds ~1.2 us/kernel (4.2 us floor).
  - Captured HIP graph replayed node: 1.22 us/node (1000 chained nodes, Grade B, 2026-11-22).
  - Multi-stream launch throughput on MI60: 1.45x (1 host thread), 1.9x (1 thread/stream).
- **One-block rule (MI60)**:
  - One thread block gets ~1/30 of GPU (26-30 GB/s vs ~800 GB/s full card).
  - Elementwise 1-block (1024 thr): `2.1 us + 0.26 us per 1000 floats`.
  - Row reduction + 2nd pass 1-block: `2.9 us + 0.46 us per 1000 floats`.
  - Many blocks (up to 72x1024): 2.3-2.8 us flat.
  - 4608 floats in 1 block: 64 thr = 18.1 us, 256 thr = 6.30 us, 1024 thr = 3.39 us.
  - Barrier cost: ~0.08 us each; dependent reduce-then-use phase: ~1.6 us each.
  - Two-kernel split reduction: ~5.4 us total up to 36,864 floats (beats 1-block above ~5,000 floats).
- **Q1 Occupancy (dev0 MI60 64 CU)**:
  - Formula: `blocks_per_CU = min(floor(4 * min(8, floor(256 / VGPR)) / ceil(T / 64)), floor(65536 / LDS_bytes))`. VGPR granule 4; max 8 waves/SIMD (32 waves/CU), NOT 10. LDS cap 64 KB/CU. `__launch_bounds__(T, M)` sets min waves/SIMD (caps VGPR at `256/M`: M=2/4/8 -> 128/64/32 VGPR; T=1024 always caps at 64 VGPR). Dispatch rate ~30 waves/us (~33 ns/wave); round boundary ~2 us.
- **Q2 Streaming efficiency (int8 buffer, coalesced "chunk" uint4x4 loads, median of 5 x ~50 ms)**:
  - Peak: dev0 MI60 **891 GB/s** (S=256 MB, G=8192, T=1024, 301.4 us); dev1 MI50 **904 GB/s** (S=256 MB, G=8192, T=512, 297 us) = 87-88% of 1024 GB/s HBM2 spec.
  - S=32 MB chunk (dev0, GB/s): G=64 -> 319..636; G=512 -> 315..637; G=1024,T=1024 -> 820; G=2048,T=1024 -> 853; G=8192,T=256 -> 840.
  - S=256 MB chunk (dev0, GB/s): G=1024 -> 241..713; G=2048 -> 337..792; G=8192 -> 636/753/793/872/891 (T=64..1024).
  - S=3.5 MB (3,670,016 B, L2-resident): best **5.01 us** (732 GB/s) at G=512, T=128; cold estimate >= 6.1 us. Unexplained deterministic drop to 82 GB/s (44 us) when `G*T == 131072`.
  - Compute/F16 at peak (256 MB, 8192x1024): add 891, fma 889, sdot4 891, F16 892 GB/s. Uncoalesced "run" pattern is 2-10x slower.
- **Q3 Cross-GPU host-staged exchange (pinned `hipHostMalloc`, median of 200 reps, us)**:

| Payload | D2H dev0 | H2D dev1 | D2H dev1 | H2D dev0 | One-way 0->h->1 | Duplex | Kernel+duplex+kernel | Duplex pageable | `hipMemcpyPeer` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 KB | 11.4 | 14.7 | 13.6 | 12.0 | 37.0 | 39.9 | 105 (85-218) | 94.7 | 73.4 |
| 10 KB | 13.2 | 16.7 | 14.4 | 13.9 | 37.9 | 41.3 | 75.7 | 91.0 | 55.8 |
| 20 KB | 9.5 | 12.2 | 10.4 | 11.0 | 30.0 | 24.8 | 67.3 | 51.2 | 44.3 |
| 40 KB | 13.3 | 14.9 | 13.2 | 14.3 | 33.1 | 30.6 | 73.3 | 67.4 | 49.7 |
| 160 KB | 30.6 | 32.0 | 30.3 | 32.3 | 67.2 | 65.1 | 108.5 | 142.5 | 88.9 |
| 1 MB | 154.3 | 155.8 | 154.1 | 162.5 | 347.6 | 320.5 | 372.0 | 710.4 | 368.1 |
| 16 MB | 2354.7 | 2357.5 | 2357.1 | 2472.0 | 4859.6 | 4868.7 | 5105.5 | 9832.5 | 5022.1 |

- Q3 fit: D2H dev0/dev1 7.15 GB/s, H2D dev1 7.14 GB/s, H2D dev0 6.8 GB/s; duplex `17 us + P / 3.46 GB/s` (>= 160 KB), flat 25-41 us below 40 KB (20 KB is repeatably faster than 2.5/10 KB). Idle `hipStreamSynchronize` 0.17 us; trivial kernel+sync 9.7 us (dev0) / 11.7 us (dev1). Note (2026-11-22 skeptical correction): Q3 bounds prefill DMA staging, NOT kernel-polled decode `ggml_cuda_ar_kernel` (which runs in 20 us traced via mapped pinned memory without `hipMemcpyAsync` or host sync).

### 2026-11-22 Kernel micro-variants (tried, not kept)

Archived origins: `old-ignore/2026-11-22/hc-mix-variants/README.md`, `old-ignore/2026-11-22/add-rms-norm-variants/README.md`.

1. **Flash Next `dsv4_hc_mix_down` / `_up` variants (TB2 / F2b, Grade B trace + A/B inside noise)** - 98 calls/token/GPU (~3.5 MB Q8_0 weights/call), `rocprofv3` over 9,408 calls:

| Variant | `up` us/call | `down` us/call | Notes |
| --- | ---: | ---: | --- |
| baseline (2 runs) | 11.07 / 11.09 | 11.55 | - |
| U1 (`up` 256-thread blocks) | 11.17 | 11.45 | Not occupancy-bound |
| U2 (weight loads before quantize barrier) | 13.93 | 12.42 | Slower |
| U3 (U1 + U2) | 12.69 | 12.41 | Slower |
| D1 (`down` `warp_reduce_sum` replacing serial 64-step LDS loop) | 10.92 | 10.83 | -0.7 us/call |

- D1 end-to-end A/B on Flash Next (old, new, new, old): TG128 `50.17, 50.63, 50.56, 50.24`; PP2048 `647.2, 625.8, 625.6, 629.2`. Greedy matches ~38/48 tokens. Dropped: inside 1 t/s noise and changes sum order.

2. **27B `add_rms_norm_mul_f32<1024>` one-block micro-variants (27B-ARN, Grade B traces, patches not kept; superseded same day by two-launch `ae32e5e58`)**:
- Baseline: 11.3 us/call (without Q8_1 output: 8.2 us).
- Registers 512x10: 14.8 us.
- `float4` per thread 256x8, branch-free loads: 11.0 to 13.5 us.
- Same + reciprocal replacing divide: 10.7 to 11.5 us.

### 2026-11-21 (late) Inkling 2 Mini: NVFP4 repacked experts, banded prefill attention, wide-block mix kernels

Exact NVFP4 file reaches 836 t/s PP and 45.8 t/s TG on ROCm0 (up from 490.8 / 30.5 earlier the same day), and long prompts use 5.5 GB less compute VRAM. 27B and Flash Next unchanged.

- Source: `06aa25f25` + `d37d86dd7` + `2162c74dd` (banded relative-bias attention, unsloth #386 port + gfx906 tile-kernel path, default on for prefill), `a99a7930b` (wide-block `res_mix`/`rms_norm_cat`), `0ba4be7f1` (NVFP4 experts on repacked path). Old build `build-u1sort-20261121` (`400989a0d`), new build `build-inkling-final-20261121` (`0ba4be7f1`).
- Workload: `llama-bench -r 2`, order old/new/new/old, below 60C (Grade A; archived origin: `build-inkling-final-20261121/validation/final/`, `investigation-notes/2026-11-21-inkling2-mini/final_validation_2.sh`).

| Model | Config | PP2048 old | PP2048 new | TG128 old | TG128 new |
| --- | --- | ---: | ---: | ---: | ---: |
| 27B Q8_0 | 2 GPUs, `-sm tensor`, ub 512 | 613.5 | 614.5 (+0.2%) | 37.14 | 37.21 (+0.2%) |
| Flash Next UD-IQ3_XXS | 2 GPUs, `-sm tensor`, ub 512 | 655.0 | 657.7 (+0.4%) | 49.90 | 50.18 (+0.6%) |
| Inkling 2 Mini IQ4_NL | ROCm0, ub 512 | 564.2 | 878.6 (+55.7%) | 36.28 | 47.75 (+31.6%) |
| Inkling 2 Mini Q8_0 | ROCm0, ub 512 (1 run each) | 566.1 | 880.1 (+55.5%) | 34.47 | 44.41 (+28.8%) |
| Inkling 2 Mini NVFP4 + Q8_0 | ROCm0, ub 512 | 490.8 (a) | 836.3 (+70.4%) | 30.5 (a) | 45.80 (+50.2%) |

(a) `build-catchup-20261121-e6d6` (merge-only, Grade B; old build lacks NVFP4). PP2048 at `-ub 2048` on new build: IQ4_NL 1137.7 (old 578.1), Q8_0 1137.3 (old 575.9), NVFP4 1081.1 (merge-only 509.7).

Per-change isolated A/B (same binary or two builds, alternated, Grade A):

| Change | File | PP2048 (off / on) | TG128 (off / on) |
| --- | --- | ---: | ---: |
| Wide-block `res_mix_f32` / `rms_norm_cat_f32` (were 26.7 / 11 us/call) | IQ4_NL | 771.1 / 776.9 | 43.57 / 47.73 (+9.5%) |
| Banded bias, unsloth scalar kernels only (first port, rejected as default) | IQ4_NL | 761 / 383 (-50%) | 47.3 / 47.5 |
| Banded bias in gfx906 tile kernel, ub 512 | IQ4_NL | 775.4 / 879.6 (+13.4%) | 47.7 / 47.7 |
| Banded bias in gfx906 tile kernel, ub 2048 | IQ4_NL | 772.9 / 1135.7 (+47%) | - |
| NVFP4 experts repacked (`GGML_CUDA_REPACK_NVFP4=0` / default), ub 512 | NVFP4 | 637.8 / 737.3 (+15.6%) | 37.9 / 45.5 (+20%) |

- Banded attention at 16K context (`llama-batched-bench -npp 16384 -ub 2048`): PP 576.5 -> 686.9, TG 34.3 -> 34.6, compute buffer **8,665 -> 3,170 MiB** (-5,495 MiB). Decode stayed on dense bias (TG 36.4 at 4K vs 45.8 empty; fixed next day by `efb85a1a5`).
- Quality (`llama-perplexity -c 4096 --chunks 2`, NVFP4, KLD vs first run): identical config twice -> KLD 0 (deterministic); dense ub 512 vs dense ub 2048 -> mean KLD 0.0049, same top token 97.8%; banded vs dense -> KLD 0.0055 (ub 512), 0.0060 (ub 2048), 0.0054 (ub 96), same top 97.0-97.6%; NVFP4 repacked vs generic -> KLD 0.0054, same top 97.2%, PPL ratio `0.9997 +- 0.0019`. PPL over 8 chunks (dense / banded): IQ4_NL 2.8307 / 2.8305 (ub 512), 2.8288 / 2.8283 (ub 2048); NVFP4 2.8210 / 2.8261; Q8_0 dense 2.8268.
- Unit tests (`test-backend-ops` on ROCm0): MUL_MAT 2727/2727, MUL_MAT_ID 1162/1162, FLASH_ATTN_EXT 4420/4420, FLASH_ATTN_EXT_BANDED 71/71, CLIP_ROWS 9/9, RES_MIX 9/9, LOOP_GATE 3/3, RMS_NORM_CAT 6/6, SCONV_RES 12/12. (Known pre-existing failure: generic NVFP4/Q2_K/Q3_K MUL_MAT_ID with 1 token and 2 used experts.)
- NVFP4 repack VRAM: ROCm0 weights 16,320 MiB generic -> 17,625 MiB repacked (+1,305 MiB for 2 F16 scales per 32 values).
- Server & multi-user on ROCm0 (`build-inkling-final-20261121`):
  - `llama-server` NVFP4 + mmproj Q8_0 `-c 16384`: text TG 37.1 (was 30.4), image request PP 460.0 (was 389.5), TG 35.0, stays up.
  - `llama-batched-bench -npp 4096 -npl 1,2,4 -ub 2048`:
    - NVFP4 new: PP `991 / 746 / 559`, aggregate TG `36.4 / 51.0 / 87.7`.
    - IQ4_NL new: PP `1032 / 768 / 570`, aggregate TG `37.5 / 53.5 / 97.1`.
    - IQ4_NL old (`build-u1sort-20261121`): PP `507 / 429 / 378`, aggregate TG `30.6 / 44.7 / 79.2`.
  - CPU check (`--device none -ngl 0`) passes without assert.
- Launch counts: Inkling 2 Mini dropped from **7,219** launches/token at start of 2026-11-21 to **4,128** after fused ops; steady-state TG is 98% GPU-busy, captured HIP graphs active.

### 2026-11-21 (early) Inkling 2 Mini: image crash fix, fused ops, NVFP4 baseline

Fixes Inkling image crash (`s_copy_main`/`s_write` missing in ring recurrent input class), adds four fused ops (+37% PP / +19% TG on IQ4_NL), and merges upstream `e6d624b9a` (`#32669`/`#32704`) to enable native NVFP4.

- Source: upstream `e6d624b9a` merged (`4c5738580`), crash fix `ddca7dd9b`, fused ops `3253927e4`, `b6c927fd1`, `f17889023`, `85bb8a0f7`. Old build `build-u1sort-20261121` (`400989a0d`), new build `build-sconvfuse-20261121` (`85bb8a0f7`).
- Workload: `llama-bench -r 2`, order old/new/new/old, below 60C (Grade A; archived origin: `build-sconvfuse-20261121/validation/final/`, `investigation-notes/2026-11-21-inkling2-mini/final_validation.sh`).

| Model | Config | PP2048 old | PP2048 new | TG128 old | TG128 new |
| --- | --- | ---: | ---: | ---: | ---: |
| 27B Q8_0 | 2 GPUs, `-sm tensor`, ub 512 | 614.4 | 611.0 (-0.6%) | 37.09 | 37.24 (+0.4%) |
| Flash Next UD-IQ3_XXS | 2 GPUs, `-sm tensor`, ub 512 | 658.2 | 655.9 (-0.3%) | 50.02 | 50.48 (+0.9%) |
| Inkling 2 Mini IQ4_NL | ROCm0, ub 512 | 564.2 | 775.1 (+37.4%) | 36.42 | 43.35 (+19.0%) |
| Inkling 2 Mini Q8_0 | ROCm0, ub 512 (1 run each) | 564.3 | 772.7 (+36.9%) | 34.30 | 40.33 (+17.6%) |
| Inkling 2 Mini NVFP4 + Q8_0 | ROCm0, ub 512 | 490.8 (a) | 641.0 (+30.6%) | 30.5 (a) | 35.43 (+16.2%) |

(a) `build-catchup-20261121-e6d6` (merge-only, no fused ops, same evening, not alternated). PP2048 at `-ub 2048` on new build: IQ4_NL 774.2 (old 578.1), NVFP4 662.1 (merge-only 509.7).

Per fused op (Inkling 2 Mini IQ4_NL on ROCm0, same binary kill-switch off/on, mean of 2, order off/on/on/off, Grade A):

| Op | Replaces | PP2048 (off / on) | TG128 (off / on) |
| --- | --- | ---: | ---: |
| `CLIP_ROWS` (`3253927e4`) | 9-kernel q/k/r norm clip, 3 per layer | 565.3 / 567.6 | 36.07 / 39.15 (+8.5%) |
| `RES_MIX` (`b6c927fd1`) | concat chain, rms norm, 2 matmuls, softmax of block residual mix | 568.8 / 751.6 (+32%) | 39.54 / 40.45 (+2.3%) |
| `LOOP_GATE` + `RMS_NORM_CAT` (`f17889023`) | elementwise parts of loop controllers (12 kernels -> 5) | 754.5 / 772.9 (+2.4%) | 40.47 / 41.43 (+2.4%) |
| `SCONV_RES` (`85bb8a0f7`) | conv, dynamic taps, residual and scale of short convs | 774.0 / 777.7 | 41.56 / 43.26 (+4.1%) |

- Tried, did not work (`GGML_OP_LOOP_CTRL`, archived origin: `old-ignore/2026-11-21/loop-ctrl-fused/`): fusing the whole loop controller including its three F32 matmuls into one 256-thread block per token (1-8 tokens) dropped Inkling IQ4_NL TG128 from **40.5 -> 11.7 t/s** (PP unchanged 737.5 off / 751.5 on; `test-backend-ops -o LOOP_CTRL` 2/2 passed). One block streamed ~2.4 MB weights (~630 us/controller vs ~24 us launch saving).
- NVFP4 file (`Inkling-2-Mini-NVFP4-Q8_0.gguf`, 17.57 GB, 18.8 GB VRAM at `-c 4096`): requantized from `Q8_K_XL` with `blk.([0-9]|1[0-9]).ffn_(gate_up|down)_exps=nvfp4` and `Q8_0`. On merge-only build, `rocprof` showed generic NVFP4 vs Q8_0: `mul_mat_vec_q<NVFP4>` 8.1 vs 4.6 ms/token; `mul_mat_q<NVFP4>` 2.87 vs 1.72 s per 2048-token prompt. Converting 144 BF16 loop-controller tensors to F32 had zero effect (PP 488.5 vs 490.9, TG 30.5 vs 30.65).
- Server & multi-user (`build-sconvfuse-20261121`, NVFP4, ROCm0): `llama-server` with mmproj Q8_0 `-c 16384`: text TG 30.4, image request (2750 prompt tokens) PP 389.5, TG 28.5, stays up. `llama-batched-bench -npp 4096 -npl 1,2,4 -ub 2048`: PP `551 / 475 / 414`, aggregate TG `29.7 / 43.7 / 68.6`.
- Checks: greedy text matches 35-48 tokens with switches off/on; `test-backend-ops` on ROCm0 passed MUL_MAT 2727/2727, MUL_MAT_ID 1152/1152, CLIP_ROWS 9/9, RES_MIX 9/9, LOOP_GATE 3/3, RMS_NORM_CAT 6/6, SCONV_RES 12/12; CPU check `llama-completion --device none -ngl 0 -n 6` passed without `get_rows` assert.

### 2026-11-21 Catch-up to official 46eb1742d, knob cleanup, and unsloth #264 MoE sort

Merge `296a9e2c0` + build fixes, peer-AR archival (`2ec03a2e3`) and removal of ~30 sweep-only env knobs (`698f466dd`), plus unsloth #264 device MoE id counting sort (`175153274`, meta OOB id fix `400989a0d`).

- Builds: `build-merge-20261121` (merge only), `build-cleanup-20261121` (`698f466dd`), `build-u1sort-20261121` (`400989a0d`), each against `build-catchup-20261118-134459`. Cooled below 60C, order base/new/new/base, `llama-bench -r 3`.

| Model | Test | base (`11-18`) | merge | cleanup (`698f466dd`) | change (cleanup) |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B Q8_0 | PP2048 (`-b/-ub 2048`) | 700.2 / 700.1 | 698.8 | 698.9 | -0.2% |
| Qwen3.8-27B Q8_0 | TG128 | 37.33 / 37.35 | 37.31 | 37.23 | -0.3% |
| Qwen3.8-27B Q8_0 | MTP2 server 5-prompt mean | 63.46 t/s, 40.31 ms/step, acc 0.797 | - | 63.92 t/s, 40.02 ms/step, acc 0.797 | ms/step -0.7% |
| Flash Next UD-IQ3_XXS | PP2048 (`-b 1024 -ub 512`) | 656.7 / 652.3 | 656.6 | 657.8 | +0.9% |
| Flash Next UD-IQ3_XXS | TG128 | 50.17 / 50.33 | 50.55 | 50.55 | +0.4% |
| Flash Next UD-IQ3_XXS | MTP3 server 5-prompt mean | 59.44 t/s, 49.49 ms/step, acc 0.660 | - | 59.04 t/s, 49.81 ms/step, acc 0.660 | ms/step +0.6% (base spread 1.9%) |
| 27B, 4 users x 16K | PP / 4-way TG | 577.2 / 84.26 | - | 572.8 / 82.24 | -0.8% / -2.4% (1 pair, thermal; see repeat) |
| Flash Next, 4 users x 16K | PP / 4-way TG | 475.6 / 84.66 | - | 486.9 / 85.98 | +2.4% / +1.6% |

- Unsloth #264 (`GGML_CUDA_MOE_COUNTING_SORT=0` vs default on `build-u1sort-20261121`), Flash Next:
  - `-ub 512`: PP2048 657.3 vs 657.0 (0.0%), TG128 50.31 vs 50.32.
  - `-b 2048 -ub 2048`: PP2048 877.4 vs 895.5 (**+2.1%**, both on runs above both off runs, inside 18 t/s Flash Next PP noise).
  - MTP3 vs cleanup: 59.57 vs 59.91 t/s, 49.32 vs 49.08 ms/step, acc 0.660 both; greedy Flash Next text identical.
- 27B 4x16K thermal repeat (A/B/B/A, base vs `build-u1sort-20261121`): PP `576.3 / 568.0` vs `566.3 / 571.8` (-0.5%); 4-way TG `85.57 / 82.27` vs `82.27 / 82.46` (-1.8%). Only first cold run (49C vs 56-59C) was faster; second base matched new (82.27). Confirmed thermal artifact.
- Checks: TP check pass at n=1 and n=2048; 27B greedy identical; `test-backend-ops` MUL_MAT_ID (1152 cases), TOP_K, SET_ROWS, GATED_DELTA_NET, GET_ROWS (352 cases) pass on ROCm0. Core delta vs upstream `46eb1742d`: +15,276/-1,039 in 121 files (was +16,426/-995 in 124 against `edec462cc`; #264 added +715).

### 2026-11-18 Catch-up to official edec462cc and SP1a TP backend sampling test

- Source: merge `992b4580f`, final `d7ea110cb`; build `build-catchup-20261118-134459` vs `build-kvrows-20261116`. Cooled below 60C, order base/new/new/base.

| Model | Test | base (`11-16`) | new (`11-18`) | change |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-27B Q8_0 | PP2048 (`llama-bench -r 2`) | 715.2 / 709.1 | 713.3 / 711.9 | 0.0% |
| Qwen3.8-27B Q8_0 | TG128 | 37.91 / 37.89 | 37.97 / 37.89 | +0.1% |
| Qwen3.8-27B Q8_0 | MTP2 server 5-prompt mean | 65.40 t/s, 39.11 ms/step, acc 0.797 | 66.11-66.26 t/s, 38.59-38.68 ms/step, acc 0.797 | +1.1% |
| Flash Next UD-IQ3_XXS | PP2048 (`-b 1024 -ub 512`) | 663.7 / 664.3 | 671.9 / 672.9 | +1.3% |
| Flash Next UD-IQ3_XXS | TG128 | 50.50 / 50.73 | 51.16 / 50.24 | 0.0% |
| Flash Next UD-IQ3_XXS | MTP3 server 5-prompt mean | 59.17 / 61.00 t/s, 50.87 / 49.31 ms/step, acc 0.682 | 59.14 / 60.05 t/s, 49.68 / 48.94 ms/step, acc 0.660 | ms/step -1.3% |
| 27B, 4 users x 16K (`batched-bench`) | PP / 4-way TG | 577.9 / 82.6 | 572.9 / 81.4 | -0.9% / noise |
| Flash Next, 4 users x 16K | PP / 4-way TG | 474.1 / 83.5 | 478.7 / 84.5 | +1.0% / +1.2% |

- Checks: TP check pass n=1 and n=2048; 27B greedy identical. Flash Next greedy text shifted (acceptance 0.682 -> 0.660), likely from upstream GCN stream-k MMQ (#30022) altering IQ prefill sum order (not verified by KLD).
- **SP1a (`LLAMA_TP_BACKEND_SAMPLING=1`)** (archived origin: `old-ignore/2026-11-18/sp1-tp-backend-sampling/`): lifting `llama_context::set_sampler` refusal under `-sm tensor` on 27B MTP2 aborted at load twice: (1) `GGML_ASSERT((nbytes & 0xF) == 0)` on tiny PARTIAL sampler tensor (fixed and kept in `6f21fe163`); (2) `ggml-backend-meta.cpp:3055` `get_tensor_async` lacks PARTIAL readback. Switch removed; no speed measured.
- **pi adapter check** (`harness-hub Qwen3.8-27B-pi-adapted` F16 LoRA on 27B Q8_0): `--lora` aborts under `-sm tensor` (no split rule for `lora_a`/`lora_b`); `-sm layer` requests 47 GB compute buffer on ROCm0 and fails.

### 2026-11-15 Flash Attention over K/V rows (`kv_rows`) for gfx906

Ports upstream `ggml_flash_attn_ext_rows` (#29510) to the gfx906 HIP tile kernel (F16 K/V only), adds a work gate in `get_n_kv_rows` so single-sequence/all-slot-decode batches stay on the dense path, and fixes a multi-slot MTP rollback abort (`3bd5e1909`, sequential ubatch split when `n_rs_seq > 0`). Archived origin: `investigation-notes/2026-11-15-kvrows-gfx906/`.

- Source: `518d6233d..7ddd49693` (final kernels `80212cad5`, work gate `7ddd49693`); build `build-kvrows-20261116` vs `build-catchup-20261113-200420`.
- Multi-user A/B (`gfx906/scripts/bench_multiuser.sh`, REPS=2, same binary: on vs `LLAMA_KV_ROWS_DISABLE=1`, cooled below 60C, Grade A):

| Model | `kv_rows` | PP t/s (4x16K) | 4-way TG t/s | Change |
| --- | --- | ---: | ---: | --- |
| Qwen3.8-27B Q8_0 | on | 577.8 / 577.5 | 80.86 / 80.93 | **+25.0% PP**, TG -0.2% |
| Qwen3.8-27B Q8_0 | off | 462.3 / 462.2 | 81.13 / 81.04 | baseline |
| Flash Next UD-IQ3_XXS | on | 470.5 / 473.1 | 84.17 / 83.90 | -0.3% PP, -0.2% TG (neutral) |
| Flash Next UD-IQ3_XXS | off | 472.2 / 474.2 | 84.24 / 84.11 | baseline |

- Single-user A/B (`llama-bench`, base `build-catchup-20261113-200420` vs `80212cad5`, order base/new/new/base, cooled, Grade A):

| Model | Test | base | new | change |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-27B Q8_0 | PP2048 | 703.3 / 700.6 | 699.7 / 699.1 | -0.4% |
| Qwen3.8-27B Q8_0 | TG128 | 37.29 / 37.19 | 37.17 / 37.27 | 0.0% |
| Flash Next (`q8_0` KV) | PP2048 | 654.0 / 655.6 | 652.6 / 651.6 | -0.4% |
| Flash Next (`q8_0` KV) | TG128 | 48.35 / 49.09 | 48.92 / 49.09 | +0.6% |

- 4-slot 48K server sequential-prefill scaling (`qwen27b` preset, 4 slots, 262144 unified, MTP2, `--no-cache-idle-slots`), PP t/s per user (1st -> 4th):
  - Base (cool GPUs): `430 / 277 / 203 / 161` (degrades 2.7x as pool fills).
  - `kv_rows` on (hot throttled GPUs): `358 / 354 / 353 / 353` (flat across all 4 users).
  - Same binary `kv_rows` off (hot GPUs): `353 / 193 / 132`.
- 4-slot 16K MTP2 server check: 1-user PP 553.9, TG 59.8; 4 concurrent users 15.7-16.3 t/s each, **60.9 aggregate**, 0 rollback refusals (before `3bd5e1909`: 1 abort in 3 runs).
- Unit tests: `test-backend-ops -o FLASH_ATTN_EXT` vs CPU passed 4372/4372 on ROCm0; `kv_rows` (`-p n_cells`) 50/50 on both GPUs (10 Q8_0 cases unsupported by design).
- Failed earlier attempts (same day): Attempt 1 (`518d6233d`, per-row index loads + vec gather) lost 20% PP and 33% TG at 64K single-sequence and -4.5% FN q8_0 TG even with `kv_rows` off. Attempt 2 (`a17ab0d66`, contiguous tiles inside shared kernel) pushed every tile variant to 128-VGPR cap with 57-97 spills (vs 74-92 VGPRs, 0-5 spills base), losing 1.5% single-user PP and producing a false "+16% 4-way decode". Solved in `80212cad5` via separate `use_kv_rows` template instantiations (kv_rows kernels still spill 62-95 VGPRs, future headroom). Thermal throttling observed at junction 98-104C, sclk 938 MHz.

### 2026-11-13 Upstream v1.0.0 (`9d25e0014`) merge and Flash Next MoE cache / `-ncmoe` sweep

Merge `3423a1c0f` (upstream `9d25e0014`, v1.0.0, including GCN MMQ retune and upstream MoE GPU cache). Build `build-catchup-20261113-200420`. One run per row, no MTP, `-fa on`, `-ts 0.52/0.48`, host experts pinned (`-lm none`), Grade B (archived origin: `investigation-notes/2026-11-13-flashnext-moe-cache/`).

Model weight footprint (GiB; 50.7 GiB PLE table on host; F16 KV ~30 KB/token: 90K = 2.6 GiB, 200K = 5.9 GiB, 262K = 7.5 GiB):
- UD-IQ3_XXS: routed experts 46.8, other GPU weights 4.2, per-layer experts 0.72-1.27 GiB.
- UD-Q3_K_XL: routed experts 52.3, other 4.2, per-layer 0.72-1.37 GiB.
- UD-IQ4_XS: routed experts 56.2, other 4.2, per-layer 0.72-1.37 GiB.

`llama-bench` PP2048 / TG128 on Flash Next UD-IQ3_XXS:

| Config | ub | PP2048 t/s | TG128 t/s | Note |
| --- | ---: | ---: | ---: | --- |
| all GPU | 512 | 753 | 39.8 | GCN MMQ retune boosted ub 512 PP (+8.9%) |
| all GPU | 2048 | 652 | 40.4 | ub 2048 slower than ub 512 on new MMQ table |
| `-ncmoe 2` (MI60 layers) | 512 | 600 (-20%) | 36.3 (-9%) | ~10.5 GiB/s non-overlapped expert upload |
| `-ncmoe 4` (MI60 layers) | 512 | 490 (-35%) | 33.1 (-17%) | - |
| `-ncmoe 4` | 2048 | 575 (-12% vs ub2048) | 33.7 | ~3%/layer PP cost at ub 2048 |
| layers 44-47 on host (MI50 layers) | 512 | 443 (-41%) | 33.7 | Worse than MI60 layers (DMI link) |
| all GPU, `-ts 0.58/0.42` | 512 | 721 (-4%) | 40.1 | - |
| f16 KV at depth 16K | 512 | 414 | 36.6 | - |
| q8_0 KV at depth 16K | 512 | 412 | 34.2 (-7%) | Halves KV VRAM, -7% TG at 16K |

MoE GPU cache (`llama-completion`, 256 tokens on short prompt or 2464-token prompt + 128 tokens):

| Config | Short TG t/s | Long PP t/s | Long TG t/s |
| --- | ---: | ---: | ---: |
| IQ3_XXS all GPU | 39.4 | 456 | 38.95 |
| IQ3_XXS `-ncmoe 4`, no cache | 34.2 | 329 | 19.1 |
| IQ3_XXS `-ncmoe 4`, 1 GiB cache | 33.6 | - | - |
| IQ3_XXS `-ncmoe 4`, 2 GiB cache | 39.0 | 331 | 34.8 |
| IQ3_XXS `-ncmoe 4`, 2 GiB cache, `--moe-cache-cpu-share 0.5` | 32.3 | - | - |
| IQ3_XXS `-ncmoe 12`, 4 GiB cache | 33.4 | - | - |
| IQ4_XS `-ncmoe 9`, no cache (`llama-bench` ub 512 / 2048) | 30.7 (TG128) | 311 / 492 (PP2048) | - |
| IQ4_XS `-ncmoe 12`, 4 GiB cache | 32.1 | 198 | 27.6 |

Verdict: MoE cache helps decode only when sized >= 50% of host expert bytes (1 GiB thrashes on 4 layers; 2 GiB recovers 39.0 short / 34.8 long vs 39.4 / 38.95 all-GPU), does not help PP, and `--moe-cache-cpu-share` hurts on this host.

### 2026-11-11 Multi-slot MTP rollback fix and multi-user serving capacity sweeps

Three linked studies on 2026-11-11 (archived origins: `investigation-notes/2026-11-11-multislot-rollback/`, `2026-11-11-multiuser-serving/`, `2026-11-11-unified-kv-unverified/`).

#### A. Multi-slot MTP rollback fix for Flash Next (`fbc067862`)

Root cause of 2-slot Flash Next MTP slowdown (63.8 -> 32.3 t/s): NOT unified KV, but a `llama-context.cpp` clamp setting `n_rs_seq = 0` for `LLM_ARCH_QWEN4EXP` whenever `n_seq_max > 1`, forcing full-state checkpoints (~35 ms/step extra). Removed clamp so multi-slot Flash Next uses plane-indexed snapshots (plus `seq_cp` and `find_slot` fixes in `llama-memory-recurrent.cpp`).

- Builds: `build-mslot-ring` (`fbc067862`) vs `build-play-b1` (`dc9c4e367`). 2x gfx906 TP, F16 KV. Workload: `spec_accept.py` 5 prompts x 256 greedy tokens (1 user) + `conc.py` N simultaneous requests (aggregate, prefill included). Flash Next: 32K unified, `-ts 0.52/0.48`, `-b 1024 -ub 512`, MTP3 ROCm1. Qwen 27B: MTP2, `-b/-ub 2048`, 32K (np1) or 262144 (np4) unified. Grade B (one run each unless noted).

| Model | Slots | Build | 1-user TG t/s | ms/step | Accept | PP (server) | N-user aggregate t/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Flash Next MTP3 | 1 | old (`play-b1`) | 67.2 | 45.6 | 0.709 | 481 | - |
| Flash Next MTP3 | 1 | new (`mslot-ring`) | 66.6 | 45.9 | 0.709 | 478 | - |
| Flash Next MTP3 | 2 | old (checkpoints) | 32.5 | 82.9 | 0.749 | 486 | 2 users: 22.0 |
| Flash Next MTP3 | 2 | new | 69.1, 67.2 (2 runs) | 44.3, 45.7 | 0.709 | 485, 478 | 2 users: 36.5 |
| Flash Next MTP3 | 4 | new | 66.7 | 46.0 | 0.709 | 476 | 4 users: 31.0 (31.6/33.9 GB peak) |
| Qwen 27B MTP2 | 1 | old | 57.7 | 43.7 | 0.777 | 539 | - |
| Qwen 27B MTP2 | 1 | new | 57.3 | 44.0 | 0.777 | 523 | - |
| Qwen 27B MTP2 | 4 | old | 58.9 | 42.8 | 0.777 | 528 | 4 users: 36.1 |
| Qwen 27B MTP2 | 4 | new | 58.8 | 42.9 | 0.777 | 515 | 4 users: 35.9 |

- Plain `llama-bench` check (r3, no draft):
  - Qwen 27B Q8_0 (`-ts 1/1 -b/-ub 2048`): old 702.1 PP / 36.0 TG -> new 705.5 PP / 37.7 TG.
  - Flash Next UD-IQ3_XXS (`-ts 0.52/0.48 -b 1024 -ub 512`): old 606.9 PP / 49.1 TG -> new 609.1 PP / 50.5 TG.
- Correctness: new np2 MTP3 matches old np1 ring token-for-token across all 4 serial cases; 11/11 repeat np2 cases identical; 27B new=old on all np1/np4 cases. Pre-existing anomaly noted on `p2.txt`: all MTP3 builds (old ring, checkpoints, new) emit token 262 as 3rd output token (outside plain top-5).

#### B. Multi-user serving capacity, long-fill, and ubatch sweeps

Pre-fix (`build-play-b1`, `dc9c4e367`, Grade B, 1 run each):

| Config | 1-user TG | ms/step | N-user agg | Long fill (PP / TG / output) | Peak GPU0 / GPU1 (GB of 34.3) |
| --- | ---: | ---: | ---: | --- | --- |
| Qwen 27B MTP2, 4 slots, 262144 unified | 65.6 (61.7 rerun) | 38.4 | 4: 41.2 | 200K: 219 / 28.3 / correct | 33.0 / 33.0 |
| Qwen 27B no draft, 4 slots, 262144 | 34.7 | 28.8 | 4: 43.9 | - | 28.4 / 28.5 |
| FN MTP3, 1 slot, 32K (old preset) | 63.8 | 48.1 | 2 (queued): 34.6 | - | 30.6 / 33.0 |
| FN MTP3, 1 slot, 100352 (old preset) | - | - | - | 90K: 327 / 2.4 (24s draft catch-up) / correct | - |
| FN no draft, 4 slots, 100352, ts 0.52 | 50.2 | 19.8 | 4: 41.2 | - | 33.0 / 30.9 |
| FN no draft, 4 slots, 131072, ts 0.47 (preset) | 48.3 | 20.7 | 4: 40.7 | 120K: 290 / 28.6 / correct | 31.7 / 33.7 |
| FN no draft, 4 slots, 163840, ts 0.49 | 48.4 | 20.6 | 4: 41.2 | 150K: 258 / 27.5 / correct | 31.7 / 34.2 |

Isolating run on `build-play-b1` (32K, 1024/512, ts 0.52/0.48, draft ROCm1): 1-slot MTP3 (`n_rs_seq=3`) 63.8 TG (48.1 ms/step, acc 0.71); 2-slot unified MTP3 (`n_rs_seq=0`) 32.3 TG (83.4 ms/step, acc 0.75); 2-slot unified no-draft 49.9 TG (20.0 ms/step).

Post-fix Flash Next 2-slot MTP3 context sweep (`build-mslot-ring`, `-np 2 --kv-unified`, 1024/512, draft ROCm1):

| Pool | Split | 1-user TG (ms/step) | 2-user agg | Long fill (tokens, PP / TG) | ROCm1 free at peak |
| --- | --- | ---: | ---: | --- | ---: |
| 131072 / 114688 / 98304 | 0.52 | - | - | Load failed on ROCm1 (short 3317 / 224 / 135 MiB) | - |
| 98304 | 0.56 | - | - | Loaded; 50 MiB runtime alloc abort on 1st request | - |
| 81920 | 0.52 / 0.55 | 67.6 / 68.9 (45.3 / 44.4) | 36.1 / 36.4 | 73757, 342 / 36.6 | 0.03 GB |
| 73728 | 0.52 | 65.6 (46.8) | 35.4 | 66384, 347 / 42.4 | 0.16 GB |
| 65536 | 0.52 | 67.6 (45.3) | 36.1 | 59011, 359 / 44.1 | 0.37 GB |

Flash Next ubatch sweep (`llama-bench` PP2048, no draft, `build-mslot-ring`):
- `-ub 256`: d0 `534.5 +- 0.9`, d32768 `388.8 +- 11.8`.
- `-ub 384`: d0 `599.6 +- 2.3`, d32768 `430.2 +- 13.3` (matches ub 512 while saving compute VRAM).
- `-ub 448`: d0 `534.2 +- 2.0`, d32768 `405.6 +- 0.7` (non-monotonic tile sizes).
- `-ub 512`: d0 `598.2 +- 3.3`, d32768 `434.0 +- 6.9`.

Flash Next 2-slot MTP3 server ubatch/pool sweep (draft ROCm1, split 0.52/0.48):

| ubatch / pool | Short-prompt PP | 1-user TG (ms/step) | 2-user agg | Long fill (tokens, PP / TG) | 2 x 30K concurrent (fresh server) |
| --- | ---: | ---: | ---: | --- | --- |
| 256 / 98304 | 383-420 | 63.8 (48.2) | 34.2 | 88502, 294 / 35.5 | not run |
| 256 / 81920 | 393-442 | 66.4 (46.1) | 35.0 | 73757, 310 / 37.0 | not run |
| 384 / 98304 | - | - | - | Server died during first short requests | - |
| 384 / 94208 | 422-482 | 65.5 (46.7) | 36.0 | 84816, 315 / 35.2 | Both answered; PP 418/295, TG 1.0/41.7 |
| 384 / 90112 (preset) | 463-481 | 64.7 (47.6) | 34.8 | 81129, 322 / 33.6 | Both answered; PP 377/289, TG 1.0/42.2 |
| 384 / 86016 | 446-498 | 66.5 (46.0) | 35.0 | 77443, 323 / 36.4 | not run clean |

#### C. Exploratory unified-KV verification (`build-play-b1`, build 12328, `dc9c4e367`, Clang 24.0.0)

- Corrected `llama-bench` 27B Q8_0 (r3, `-dev ROCm0/ROCm1 -ts 1/1 -b/-ub 2048`, F16 KV): PP2048 **706.904 +- 0.116 t/s**, TG128 **37.779 +- 0.159 t/s**. (An earlier comma-syntax sweep ran single-GPU and was marked INVALID.)
- `spec_accept.py` 5-prompt sequential screen (1 run each, Grade B):
  - Qwen 27B MTP2 non-unified (16384 / 1 slot): mean TG 64.20, 39.23 ms/step, acc 0.777; long-prompt PP p1/p2/p3 = 652/620/582.
  - Qwen 27B MTP2 unified (274432 / 4 slots): mean TG 63.44, 39.81 ms/step, acc 0.777; long-prompt PP p1/p2/p3 = 593/598/577 (-9.0/-3.5/-0.9%).
  - Qwen 27B MTP2 non-unified control (274432 / 4 slots): mean TG 66.48, 38.48 ms/step, acc 0.795.
  - Flash Next MTP3 reference (100352 / 1 slot, ROCm1 draft, 52/48): mean TG 63.72, 48.15 ms/step, acc 0.709; PP p1/p2/p3 = 469/488/454.
  - Flash Next MTP3 unified pre-fix (32768 / 4 slots, ROCm0 draft, 46/54): mean TG 31.25, 84.82 ms/step, acc 0.742; PP p1/p2/p3 = 448/500/469.
- Greedy argmax check: Qwen 768/768 top-1; Flash Next 758/768 top-1 + 10 suppressed EOG rows (`ignore_eos=true`).

### 2026-11-10 Upstream `aa75ff323` merge (`0f7ce1b04`), Token Budget, B1 MoE Routing, Router Smoke, and Experiments

Major profiling and experimentation day on merged tree `0f7ce1b04` (upstream `aa75ff323`). Archived origins: `gfx906/TOKEN-BUDGET.md`, `gfx906/token-budget-columns.csv`, `gfx906/README.md`, `old-ignore/2026-11-10/{play-b1,exp-spec,exp-q3,exp-p3,exp-q1,play-a7,play-tb1}/`.

#### A. B1: One-wave MoE routing kernel `topk_moe_wave_cuda<512,10>` (LANDED, Grade A)

- Source: `6d8f0c1c7` (gfx906 512-expert top-10, rows <= 8, rolled k-loop, DPP min-reduce) + `457d4470b` (HIP 7 `__ballot_sync` wave64 mask fix: `test-backend-ops -o TOPK_MOE` 980/980 pass x3). Build `build-play-b1` vs `build-exp-spec`.
- Microbench (warm caches, 1 row): old 17.5 us, new 8.8 us, empty kernel 1.9 us.
- Decode trace (`rocprofv3` TG64 @d8192): `topk_moe_cuda<512>` **28.84 us/call** (7648 calls; 0 scratch, 116 VGPRs) -> unrolled v1 25.0 us -> rolled-k `topk_moe_wave_cuda<512,10>` **14.89 us/call** (6256 calls). Cold instruction fetch of single-wave kernel mattered strongly.
- Balanced A/B (`llama-bench`, Flash Next UD-IQ3_XXS TG128 @d8192, `-r 5`, `-ts 1/1`, strictly alternating, 60s idle, 4 rounds, Grade A):
  - Baseline `build-exp-spec`: `45.68, 45.74, 45.68, 45.74` -> mean **45.71 +- 0.04 t/s**.
  - Candidate `build-play-b1`: `47.64, 47.53, 47.29, 47.65` -> mean **47.53 +- 0.17 t/s** (**+4.0%**, ~0.9 ms/token).
  - Earlier unbalanced rounds: A mean 45.03 +- 1.42 (6 runs), B unrolled v1 mean 45.62 (3 runs), B final mean 46.93 +- 0.65 (+4.2%).
- MTP3 `llama-cli` (`-c 16384`, 256 tok greedy): A `48.3 / 49.8 t/s` -> B `50.6 / 50.5 / 50.5 t/s`, text identical. Greedy `llama-completion` 200 tok byte-identical (971 bytes). Bitwise tie/NaN checks passed.

#### B. Token Budget Baselines (2026-11-10, tree `0f7ce1b04`, F16 KV, Grade B)

Builds: `build-exp-q3` (`246887e7bebf / d67316b02bbe`) for 27B@0 TG; `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) for all other columns. Note: `rocprofv3` adds ~1.2 us/kernel (4.2 us floor vs ~2.2 us app cost) and inflates host token-start gaps (~1 ms/lane vs ~19 us unprofiled).

1. **Decode (TG) unprofiled + profiled critical-path breakdown (ms/token)**:

| Metric / Step | 27B@0 | 27B@32K | FN@8K | G4@0 | FN@32K | 27B 1-GPU@0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model & settings | 27B Q8_0 `4ca72078`, 2048/2048, tg64 | 27B Q8_0, d32K, tg32 | FN IQ3_XXS `d9620119`+Q8 tbl, 1024/512, d8K, tg64 | Gemma 4 31B Q8_0 `c1ac76e9`, 2048/2048, tg64 | FN IQ3_XXS, d32K, tg64 | 27B Q8_0 ROCm0 only, 2048/2048, tg64 |
| **Unprofiled TG t/s (ms/tok)** | **37.45 +- 0.21 (26.70)** | **33.48 +- 0.97 (29.87)** | **43.68 +- 0.60 (22.90)** | **32.68 +- 0.15 (30.60)** | **41.32 +- 0.97 (24.20)** | **23.42 +- 0.01 (42.70)** |
| Profiled ms/tok (all / steady / bench) | 29.44 / 28.58 / 29.55 | 34.29 / 32.88 / 35.12 | 27.18 / 25.63 / 28.39 | 33.08 / 32.84 / 33.09 | 27.93 / 27.23 / 28.64 | 43.99 / 43.99 / 44.02 |
| Kernels / AllReduces per token | 1595 / 128 | 1594 / 128 | 2908 / 96 | 1463 / 120 | 2870 / 96 | 1594 / 0 |
| Token start host gap (mostly profiler) | 1.02 (3%) | 1.20 (3%) | 2.09 (8%) | 1.07 (3%) | 2.51 (9%) | 0.35 (1%) |
| Warm-up excess spread over tokens | 0.86 (3%) | 1.41 (4%) | 1.55 (6%) | 0.24 (1%) | 0.70 (3%) | - |
| Residual add+norm / HC combine+mix | 1.39 (5%) | 1.47 (4%) | 3.88 (14%) | 1.66 (5%) | 3.86 (14%) | 0.63 (1%) |
| Per-layer n-gram embd (FN) | - | - | 0.25 (1%) | - | 0.22 (1%) | - |
| Mixer input projections (mat-vec) | 4.60 (16%) | 4.74 (14%) | 2.69 (10%) | 4.51 (14%) | 2.69 (10%) | 8.07 (18%) |
| Attention q/k norm, rope, KV write | 0.66 (2%) | 0.67 (2%) | 0.50 (2%) | 1.67 (5%) | 0.49 (2%) | 0.59 (1%) |
| Attention: flash attention | 0.45 (2%) | 3.34 (10%) | 0.65 (2%) | 1.96 (6%) | 1.46 (5%) | 0.38 (1%) |
| Attention: QSA indexer (FN) | - | - | 2.45 (9%) | - | 2.99 (11%) | - |
| Recurrent: conv, GDN, gates, state | 3.23 (11%) | 3.24 (9%) | 2.32 (9%) | - | 2.29 (8%) | 3.46 (8%) |
| Mixer output projection (mat-vec) | 1.57 (5%) | 1.62 (5%) | 0.76 (3%) | 2.40 (7%) | 0.76 (3%) | 3.05 (7%) |
| AllReduce after mixer (exchange) | 0.86 (3%) | 0.91 (3%) | 0.51 (2%) | 0.71 (2%) | 0.52 (2%) | - |
| FFN dense gate/up (+ act) | 8.19 (28%) | 8.78 (26%) | - | 10.15 (31%) | - | 16.12 (37%) |
| FFN dense down | 4.28 (15%) | 4.57 (13%) | - | 5.24 (16%) | - | 8.41 (19%) |
| MoE router+top-k / experts / shared / red | - | - | 1.82 / 2.58 / 1.41 / 1.03 | - | 1.86 / 2.61 / 1.42 / 0.87 | - |
| AllReduce after FFN (exchange) | 0.75 (3%) | 0.77 (2%) | 0.50 (2%) | 1.18 (4%) | 0.50 (2%) | - |
| Output LM head (+ norm) | 0.89 (3%) | 0.96 (3%) | 0.60 (2%) | 1.03 (3%) | 0.60 (2%) | 1.76 (4%) |
| q8_1 quantize of mat-vec inputs | 0.63 (2%) | 0.62 (2%) | 1.50 (6%) | 1.18 (4%) | 1.49 (5%) | 1.11 (3%) |

- Per-GPU steady TG summary (ROCm0 / ROCm1, ms/token):
  - 27B@0: mat-vec 19.30 / 19.51 (433 calls), FA 0.36 / 0.45, AR exch 1.40 / 1.57, AR wait 1.89 / 0.00, start gap 0.25 / 1.02, busy 28.30 / 27.50.
  - 27B@32K: mat-vec 20.70 / 19.57, FA 3.08 / 3.34, AR exch 1.43 / 1.65, AR wait 1.85 / 0.84, start gap 0.32 / 1.20, busy 32.49 / 31.56.
  - FN@8K: mat-vec 7.86 / 8.25 (592 calls), FA 0.67 / 0.65, AR exch 0.87 / 0.99, AR wait 3.22 / 0.00, start gap 0.90 / 2.09, busy 24.71 / 23.51.
  - G4@0: mat-vec 22.95 / 23.26 (360 calls), FA 1.54 / 1.96, AR exch 1.68 / 1.86, AR wait 2.20 / 0.00, start gap 0.25 / 1.07, busy 32.56 / 31.72.
  - FN@32K: mat-vec 7.89 / 8.28, FA 1.45 / 1.46, AR exch 0.89 / 1.00, AR wait 3.25 / 0.01, start gap 1.22 / 2.50, busy 25.99 / 24.70.
  - 27B 1-GPU@0 (ROCm0): mat-vec 37.40 (433 calls, ~767 GB/s), FA 0.38, start gap 0.35, busy 43.59. TP speedup: 42.70 -> 26.70 ms = **1.60x** unprofiled.
- Speculative step breakdowns (`TOKEN-BUDGET.md`, measured step/accept + estimated split):
  - 27B MTP2 (`build-exp-spec`, 2026-11-10): step **40.6 ms** (40.81/40.34), **2.55 tok/step**, acc 0.777 -> 15.9 ms/tok (62.8 t/s, server mean 62.1); est split ~31 ms verify(3) + ~4.5 ms 2x draft + ~5 ms host sampling/rollback.
  - 27B DFlash5 (`build-exp-spec`, 2026-11-10): step **54.9 ms** (55.03/54.79; 48-49 ms short prompts, 60-67 ms ~1.75K prompts), **3.88 tok/step**, acc 0.581 -> 14.2 ms/tok (70.7 t/s, server mean 69.3); est split ~35 ms verify(6) + ~20 ms ROCm0 draft+taps+host.
  - FN MTP3 (`52bcba210`, 2026-11-06, 100K, ROCm1 sidecar): step **49.5 ms**, ~3.10 tok/step, acc 0.721 -> 15.9 ms/tok (62.7 t/s; ROCm0 idle 22% ~11 ms); est split ~32 ms verify(4) + ~7.5 ms 3x draft + ~10 ms host.

2. **Prefill (PP) unprofiled + profiled critical-path breakdown (ms per 1K prompt tokens)**:

| Metric / Step | 27B@0 | 27B@32K | FN@0 | G4@0 | FN@32K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Unprofiled PP2048 t/s (ms/1K)** | **705.3 +- 1.4 (1417.8)** | **457.8 +- 4.2 (2184.4)** | **591.4 +- 11.9 (1690.9)** | **602.5 +- 1.4 (1659.9)** | **441.6 +- 4.3 (2264.6)** |
| Profiled ms/1K (all / traced / t/s) | 1421.4 / 1406.5 / 703.5 | 2280.9 / 2194.2 / 438.4 | 1660.3 / 1608.9 / 602.3 | 1668.1 / 1646.5 / 599.5 | 2315.7 / 2256.9 / 431.8 |
| Ubatches; kernels / ARs per ubatch | 1; 3510 / 128 | 1; 3414 / 128 | 4; 4314 / 96 | 1; 2963 / 120 | 4; 4454 / 96 |
| Run start + ubatch start + upload | 14.9 (1.1%) | 86.8 (3.8%) | 187.6 (11.3%) | 21.7 (1.3%) | 212.5 (9.1%) |
| Residual add+norm / HC pre+post | 17.8 (1.2%) | 18.4 (0.8%) | 256.4 (15.4%) | 25.7 (1.5%) | 265.5 (11.5%) |
| Mixer GEMM (attn + GDN + out) | 342.6 (24.1%) | 359.4 (15.8%) | 235.3 (14.2%) | 377.8 (22.7%) | 239.1 (10.3%) |
| Flash attention (batched) | 27.1 (1.9%) | 725.5 (31.8%) | 20.1 (1.2%) | 204.3 (12.2%) | 532.9 (23.0%) |
| QSA indexer + key pooling (FN) | - | - | 22.3 (1.3%) | - | 73.4 (3.2%) |
| Recurrent chunked GDN + conv/gates | 142.0 (10.0%) | 149.6 (6.6%) | 101.5 (6.1%) | - | 101.9 (4.4%) |
| FFN dense GEMM (gate/up + act + down) | 755.7 (53.1%) | 779.9 (34.3%) | - | 901.6 (54.1%) | - |
| MoE (router + ids + routed + shared + red) | - | - | 682.4 (41.1%) | - | 722.9 (31.2%) |
| AllReduce exposed (mixer + FFN) | 77.2 (5.4%) | 104.0 (4.5%) | 84.7 (5.1%) | 65.0 (3.9%) | 86.3 (3.7%) |
| AllReduce overlapped (info, not in total) | 276.0 | 275.7 | 84.3 | 287.4 | 82.4 |
| q8_1 quantize of GEMM inputs | 36.3 (2.6%) | 38.2 (1.7%) | 36.9 (2.2%) | 30.9 (1.8%) | 39.0 (1.7%) |

- Per-GPU PP summary (ROCm0 / ROCm1, ms per 1K tokens):
  - 27B@0: GEMM 899.6 / 1088.4 (+21% slower on MI50), FA 23.9 / 27.1, AR exposed 200.6 / 70.9, AR overlapped 157.1 / 276.0, busy 1312.2 / 1398.1.
  - 27B@32K: GEMM 1043.8 / 1111.1, FA 716.6 / 713.7, AR exposed 148.9 / 137.9, busy 2122.8 / 2176.1.
  - FN@0: GEMM 915.2 / 997.5, FA 18.5 / 20.1, AR exposed 119.3 / 83.3, start gaps 165.3 / 165.6, busy 1313.4 / 1390.5.
  - G4@0: GEMM 1083.9 / 1266.1, FA 170.3 / 204.3, AR exposed 178.5 / 58.3, busy 1529.6 / 1637.5.
  - FN@32K: GEMM 987.9 / 1020.0, FA 437.9 / 532.9, AR exposed 114.5 / 87.6, busy 1868.7 / 1991.7.

#### C. Router-mode server smoke test (`gfx906/server/serve.sh`, 2026-11-10, Grade B)

One 96-token reply per preset switching on the fly via `gfx906/server/models.ini`: `qwen27b` (MTP2) **51 t/s**, `qwen27b-dflash` (DFlash5) **61 t/s**, `flashnext` (MTP3, 100K) **48 t/s**, `gemma4` **32 t/s**.

#### D. `exp-spec`: Merge validation & upstream speculative device-row paths (2026-11-10)

- Source `0f7ce1b04` (`build-exp-spec`) vs `build-catchup-20261105-213622`; 27B Q8_0 (`4ca720788d1e`), DFlash2 Q8_0 (`2d9571f8ce46`), `spec_accept.sh` 5 prompts x 256 greedy tokens (host shared with 2 builds; compare ms/step within session, Grade B).
- Share-view check (`-sm layer -ngl 56`, 48 tok): all 4 texts (`SHARE_VIEW_INPUTS` 1/0 on both builds) identical `3ce39e8a822d`. Plain greedy `-sm tensor` 192 tok identical `1ecfab2fe93d`.

| Variant | TG t/s (r1 / r2) | ms/step (r1 / r2) | Change ms/step | Accept | Prompt ms (mean or p1) | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline (`11-05`) MTP2 | 57.45 / 59.70 | 45.23 / 43.56 | - | 0.816 | 1789 / 1732 (mean) | - |
| candidate (`0f7ce1b04`) MTP2 | 61.74 / 62.43 | 40.81 / 40.34 | -8.6% | 0.777 | 1690 / 1707 (p1: 2791/2852) | differs p2,p3 |
| candidate MTP2 `LLAMA_MTP_DEV_ROWS=1` | 61.80 / 62.37 | 40.76 / 40.39 | +0.0% | 0.777 | p1: 3343 / 3331 (+18% TTFT) | same |
| baseline (`11-05`) DFlash5 | 65.81 / 65.62 | 58.90 / 59.06 | - | 0.595 | 1682 / 1694 (mean) | - |
| candidate (`0f7ce1b04`) DFlash5 | 69.15 / 69.36 | 55.03 / 54.79 | -6.8% | 0.581 | 1654 / 1661 (p1: 2918) | differs p2,p3 |
| candidate DFlash5 `LLAMA_DFLASH_DEV=1` (M1) | 29.77 | 128.07 | +133% (2.3x slower) | 0.581 | p1: 3059 | same |

- Verdict: merge passes; `LLAMA_MTP_DEV_ROWS=1` disables deferred prefill (+18% TTFT, 0 step gain); `LLAMA_DFLASH_DEV=1` copies a 200 MiB mirrored meta tensor to ROCm0 every draft decode.

#### E. `exp-q3`: 27B Q8_0 mat-vec streaming ceiling microbenchmark (2026-11-10, Grade B)

Standalone `mvbench.hip` (16 layers = 3.24 GB, 108 mat-vecs, 5 passes) on `0f7ce1b04` (`build-exp-q3`):

| Test / Variant | ROCm0 (MI60) GB/s | ROCm1 (MI50) GB/s | Note |
| --- | ---: | ---: | --- |
| Flat 2 GiB, 1 KB contiguous/wave | 905 | 937 | Peak flat stream |
| Flat 2 GiB, grid-stride uint4x4 / D2D memcpy | 764 / 828 | 753 / 744 | - |
| Output head 5120x124160: stream 1 KB/wave / row stream / real ROWS=8 | 903 / 797 / 775 | 931 / 788 / 761 | Row access pattern sets ceiling |
| Token sequence: best stream (1 KB/wave) | 795-800 | 761-767 | Real per-matrix ceiling |
| Token sequence: real kernel `ROWS=8` (default) / `ROWS=4` | 715-718 / 720 | 682-689 / 685 | 89-90% of best stream |
| Variants: U=2 / 3 / 4 / 5 loads in flight (bit-exact) | 710 / 687 / 668 / 590 | 675 / 644 / 618 / 544 | Higher VGPRs lose |
| Variants: U=4 4 rows/blk / 16 rows/blk; k-split WPR=2/4/8 | 709 / 209 / 697 max | 666 / 204 / 631 max | All slower |

- `llama-bench` TG128 r3 alternating `GGML_RP_MMV_ROWS` 8/4/8/4: `37.45 +- 0.21, 37.60 +- 0.19, 37.44 +- 0.20, 37.52 +- 0.20` (+0.3% inside noise; kept 8).
- Profile (`rocprofv3`, tg64, 33.8 t/s = 29.55 ms/tok): Q8_0 mat-vec 433 calls 19.33 / 19.54 ms (65%), `ggml_cuda_ar_kernel` 128 calls 3.40 (26.6 us) / 1.59 ms (12.4 us), `add_rms_norm_mul_f32` 128 calls 1.39 (10.9 us) / 1.40 ms, ~600 small kernels 2.6 / 3.0 ms, idle 1.07 / 1.96 ms, 1596 launches/tok.

#### F. `exp-p3`: Late-rank attribution, parallel graph launch, and cooled `-ts` sweep (2026-11-10)

- Attribution: ROCm0 3.90 ms vs ROCm1 1.64 ms in `ggml_cuda_ar_kernel`: 1.2 ms tail was profiler artifact (`hipGraphLaunch` ~993 us under `rocprofv3` vs 19/18 us unprofiled); 1.1 ms steady skew is ROCm1 running small kernels ~0.6 us longer (4.8 vs 4.2 us, 0.74 ms/tok; mat-vec only 1.1% slower: 19.90 vs 19.67 ms).
- Parallel token-graph launch: under profiler skew 993 -> 177 us (31.49 -> 30.19 ms/tok); unprofiled A/B r3 TG128 37.10 +- 0.25 vs 36.97 +- 0.04, PP2048 704.3 vs 700.5 -> reverted.
- Cooled `-ts` sweep (`build-catchup-20261110-151625`, `llama-bench -r 3`, GPUs <55C edge before each run, Grade A/B):

| `-ts` | PP2048 t/s | TG128 t/s | TG vs 1/1 | PP vs 1/1 |
| --- | ---: | ---: | ---: | ---: |
| 1/1 (3 runs) | 701.8 / 699.6 / 699.9 (+- 0.5-1.7) | 37.20 / 37.25 / 37.32 (+- 0.06-0.13) | - | - |
| 1.03/1 | 709.6 +- 0.9 | 36.84 +- 0.08 | -1.2% | +1.3% |
| 1.05/1 | 709.5 +- 1.4 | 36.72 +- 0.20 | -1.4% | +1.3% |
| 1.1/1 | 714.0 +- 1.7 | 36.36 +- 0.11 | -2.4% | +2.0% |
| 1.15/1 | 716.9 +- 0.7 | 36.03 +- 0.17 | -3.3% | +2.4% |
| 1/1.03 | 675.7 +- 1.0 | 36.01 +- 0.04 | -3.4% | -3.5% |

- Note: an uncooled back-to-back sweep dropped to 31.7 TG by config 5 (thermal drift; invalidated). Keep `-ts 1/1`.

#### G. `exp-q1`, `play-a7`, and `play-tb1` negative microbenchmark/kernel experiments (2026-11-10)

1. **`exp-q1` (post-AR add+rms_norm+mul+q8_1 fusion on 27B TG, Grade B)**:
   - Discovered option (a) was already fused (`add_rms_norm_mul_f32<1024>` 130 calls: 10.6 / 10.9 us = 1.37 / 1.42 ms/tok; with `GGML_CUDA_Q8_1_PRODUCER=0`: 7.1 + 4.2 = 11.3 us).
   - B1 (unrolled epilogue, row in registers): slower at **12.6 / 13.2 us** (profiled 31.00 vs 31.49 ms/tok). B2 (5-block column split): never taken due to in-place `sum == a` / `dst == a|b` aliasing. Greedy 64 tok identical (`65cbbfedfad1`, `e0aff3500f30`). Reverted.
2. **`play-a7` (repacked Q8_0 prefill GEMM occupancy variants, standalone `rpgemm.cpp`, n=2048, k=5120, 3 rounds, Grade B)**:
   - VAR0 (baseline 64x128, 256 thr, 126 VGPR, 29.7 KB LDS, 2 waves/SIMD, ~26 TOPS ROCm0 / 22.7 TOPS ROCm1) vs VAR1 (64x64, 256 thr, 84 VGPR + 15 spills, 3 waves) vs VAR2 (64x64, 512 thr, 64 VGPR + 60 spills, 4 waves). All 20 shapes NMSE=0 bit-identical.
   - ROCm0 (ms): m=17408 `14.02 / 18.47 / 22.2`, m=8704 `6.81 / 9.10 / 10.84`, m=5120 `4.14 / 5.47 / 6.54`.
   - ROCm1 (ms): m=17408 `16.1 / 21.25 / 25.3`, m=8704 `7.93 / 10.55 / 12.42`, m=5120 `4.85 / 6.35 / 7.51`.
   - Verdict: VAR1 24-31% slower, VAR2 37-57% slower (VALU/LDS-issue bound + spills). Dropped.
3. **`play-tb1` (single-row bitonic top-k for QSA indexer, `test-backend-ops perf -o TOP_K -b ROCm0`, 1 row, k=512, 677/677 pass, Grade B)**:
   - Radix (12 launches) vs bitonic (`276834fdb` path) in us: ncols=2112 `71.3 vs 130.9`; 4096 `78.7 vs 168.1`; 8192 `74.3 vs 396.0`; 8256 `71.1 vs 624.9`; 12288 `87.5 vs 743.4`; 16384 `87.3 vs 882.6`.
   - Verdict: bitonic loses 1.8x-12x at every QSA width; dropped (test cases kept).

### 2026-11-06 K1: Single `set_rows` for recurrent conv snapshots + vector F32 `set_rows`

- Source: `cadb56aa2` (landed from round 2 profile review). Switches: `GGML_K1_STATE_STORE=0`, `GGML_K1_SET_ROWS_VEC=0`.
- Mechanism: one permuted view `set_rows` instead of 3 conv snapshot stores per layer, removes redundant `cont` before SSM snapshot store, adds `float4` vector F32 `set_rows` (K/V cache writes were already fused into `ROPE+VIEW+SET_ROWS`; the 193 `k_set_rows`/token in the profile were recurrent snapshots).
- Result (27B Q8_0 MTP2, 5-prompt `spec_accept.sh`, one pass each way, Grade B): step time **40.38 -> 38.95 ms/step (-3.5%)**, mean TG **62.4 -> 64.7 t/s**, same acceptance **0.777**, replies byte-identical. TP check pass at n=1/2048; `SET_ROWS` tests pass. Flash Next not separately measured.

### 2026-11-06 F2, P4, ROCm toolchain A/B, and acceptance-aware speculative runs

Lands first two profile-review kernels (F2 fused hyper-connection mix `0920eb0ac`, P4 tiled split-K small-M F32 GEMM `52bcba210`, dispatch guards `9fb19f750`/`da11f5b99`), checks suspected 27B MTP2 toolchain regression, and establishes 5-prompt `spec_accept.sh` ms/step methodology.

- Source: `52bcba210` (F2 `0920eb0ac` + P4); baseline `a27e4931e`. Toolchain A/B: same source built with ROCm 10.2.0a20261012 (clang `6da73151`) vs ROCm 10.3.0a20261105 (clang `36c1564f`, TheRock venv since 2026-11-06 00:33).
- Models: 27B Q8_0 rev `4ca720788d1e`; Flash Next UD-IQ3_XXS rev `d96201194f10` + Q8_0 n-gram table + Q8_0 MTP sidecar (`-devd ROCm1`, ctx 100352, `-ts 0.52,0.48`, `--spec-draft-n-ubatch 64`).
- Load-excluded profiles (window after last gap > 20 ms; `rocprofv3` slows TG 7-24%): Flash Next TG ~3,280 launches/token/GPU at ~8 us each, 86-89% busy; MTP3 at 100K leaves ROCm0 22% idle; Flash Next PP is 33% repacked GEMM, 26% stock `mul_mat_q` IQ experts, 19% rocBLAS SGEMM; 27B TG is 70% Q8_0 mat-vec at 94% busy.

| Build / Arm | Mode | PP t/s | TG t/s | ms/step | Accept | Change |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| F2 off | Flash Next `llama-bench` @8K | 514 | 40.8 | - | - | - |
| F2 on | Flash Next `llama-bench` @8K | 510 | 46.0 | - | - | **+12% TG** (Grade B) |
| P4 off (with F2) | Flash Next `llama-bench` @8K | 508.9 | 41.8 (tg32) | - | - | - |
| P4 on (with F2) | Flash Next `llama-bench` @8K | 563.8 | 41.9 (tg32) | - | - | **+11% PP** (+12% PP512, Grade B) |
| F2 off, new clang | Flash Next MTP3 100K (`spec_accept.sh`) | - | 59.39 | 52.18 | 0.720 | - |
| F2 on, new clang | Flash Next MTP3 100K (`spec_accept.sh`) | - | 62.72 | 49.48 | 0.721 | **-5.2% step** |
| F2 on, old clang (10.2.0) | Flash Next MTP3 100K (`spec_accept.sh`) | - | 60.76 | 51.36 | 0.728 | +3.8% step vs new clang |
| F2 on, draft ROCm0 `-ts 0.46,0.54` | Flash Next MTP3 100K (`spec_accept.sh`) | - | 63.18 | 49.94 | 0.737 | +0.9% step (draft device neutral) |
| old clang (10.2.0) | 27B MTP2 (`spec_accept.sh`) | - | 64.45 | 40.28 | 0.816 | - |
| new clang (10.3.0) | 27B MTP2 (`spec_accept.sh`) | - | 62.54 | 40.27 | 0.777 | **same step** (TG diff was acceptance) |
| old / new clang | 27B `llama-bench` r2 | 712.1 / 705.2 | 37.77 / 37.71 | - | - | -1.0% PP, -0.2% TG |

- Per-prompt `spec_accept.sh` details (`accbench/results.txt`, 5 prompts p1..p5 x 256 greedy tokens):
  - `q27-old`: p1 63.5/0.847/41.82ms/pp634; p2 71.0/0.949/40.38ms/pp617; p3 64.3/0.809/40.08ms/pp585; p4 59.4/0.701/39.78ms/pp77; p5 64.1/0.775/39.36ms/pp97 -> mean 64.45 / 0.816 / 40.28 ms.
  - `q27-new`: p1 63.9/0.861/42.03ms/pp596; p2 68.2/0.906/40.65ms/pp616; p3 56.4/0.641/40.01ms/pp581; p4 60.0/0.701/39.37ms/pp75; p5 64.3/0.775/39.27ms/pp101 -> mean 62.54 / 0.777 / 40.27 ms.
  - `fn-new` (F2 on): p1 53.6/0.663/54.64ms/pp372; p2 72.0/0.881/49.90ms/pp453; p3 68.7/0.814/48.82ms/pp428; p4 49.9/0.461/46.93ms/pp64; p5 69.4/0.788/47.09ms/pp79 -> mean 62.72 / 0.721 / 49.48 ms.
  - `fn-new-nof2`: p1 54.6/0.745/57.68ms; p2 69.1/0.885/52.00ms; p3 65.0/0.799/50.98ms; p4 47.3/0.465/49.87ms; p5 61.0/0.706/50.35ms -> mean 59.39 / 0.720 / 52.18 ms.
- Single-prompt CLI trap (`draft-placement/round2-and-27b.txt`): 27B old vs new CLI gave 62.1/61.3 (`dd228997b103`) vs 53.7/52.9 (`4f76946736c2`) due to greedy text divergence and acceptance swing (10-18%), motivating the rule never to use single-prompt `llama-cli` MTP TG for A/B.
- KLD & unit checks (Flash Next, same-build base, 4 chunks): `-ub 256` vs `-ub 512` KLD `4.66e-3`; `-ts 0.52,0.48` vs `1/1` KLD `4.74e-3`; P4 per matmul family `4.7-5.1e-3`; F2 at `-b 8 -ub 8` (2 chunks) `4.4e-3`; same top token 98.0-98.9%, PPL unchanged. P4 controls: matches rocBLAS to ~2e-6 rel (max diff 1e-5); running P4 but keeping rocBLAS output gives KLD 0; scaling `ssm_alpha` by `1 +- 3e-7` gives `4.7e-3` (proving model sensitivity floor). `DSV4_HC_MIX` 12/12 vs CPU, `MUL_MAT` 2520/2520 pass.
- Rejected/unbuilt from round 1 profile review: F1 (load-time rect copies misread as runtime), Q1 (later shown already fused), N3 (not built), T1 (unfinished WIP: wrong output from token 1, 3282 -> 3115 launches).

### 2026-11-06 Flash Next UD-IQ3_XXS at 100K with MTP3 and grouped MoE default

Lands grouped MoE as default on gfx906 (`298882544`, `GGML_CUDA_FLASHNEXT_MOE=0` disables). Archived origin: `old-ignore/2026-11-06/iq3-100k-mtp/`.

- Source: `32a8e415b` + `298882544`; builds `build-conv-20261106` and `build-fnspeed-20261106`.
- Model: UD-IQ3_XXS rev `d96201194f10` + Q8_0 n-gram table; draft `mtp-Qwen3.8-Flash-Next-Q8_0` on `-devd ROCm1`; F16 KV, ctx 100352, `-ts 0.52,0.48`, `--spec-draft-n-ubatch 64`, `-b 1024 -ub 512`.
- Workload: standard `flake.nix` `llama-cli` (1 run each unless noted, Grade B); `llama-bench` at depth 8192 (2 alternating rounds of r2, Grade A).

| Build | Mode | PP t/s | TG t/s | Change | Output sha |
| --- | --- | ---: | ---: | ---: | --- |
| `conv` | plain | 407.8 | 41.7 | - | `8db6ec2802ff` |
| `conv` | MTP2 | 402.6 | 48.9 | +17% vs plain | `cdc496d26b1a` |
| `conv` | MTP2 + grouped MoE (env) | 387.1 | 51.8 | +6% vs MTP2 | same as MTP2 |
| `conv` | MTP3 | 376.3 | 49.1 | - | `3dd833063ecc` |
| `conv` | MTP3 + grouped MoE (env) | 439.3 | 52.8 | +8% vs MTP3 | same as MTP3 |
| `fnspeed` | **MTP3, grouped MoE default (2 runs)** | 438.2 / 439.5 | **52.2 / 52.5** | +26% vs plain | same as MTP3 |
| `fnspeed` | `llama-bench` plain @8K: copy kernels vs 2D memcpy | 515.5 / 515.4 | 40.53 / 40.37 | none (Grade A) | - |

- Verdict: grouped MoE landed (+6-8% TG on MTP verify, byte-identical output, 132 repack-vs-CPU cases pass). Avoiding HIP 2D memcpy in `ggml_cuda_cpy` gave 0 gain (initial profile's 15.6%/19% `__amd_rocclr_copyBufferRectAligned` was load-time weight upload).

### 2026-11-06 Flash Next big quants: Q5_0/IQ4_XS repack, Q8_0 n-gram table, 100K and MTP fit

Archived origin: `old-ignore/2026-11-06/flashnext-big-quants/`.
- Source: `s2-iq-signs` branch + Q5_0/IQ4_XS repack conversion (`ebd76ced1`), then native Q5_0 (`3e19b6adc`, exact 22 B/32); build `build-conv-20261106`.
- Models: new upload rev `d96201194f10` UD-IQ3_XXS, UD-Q3_K_XL (gate/up 76% IQ3_XXS / 17% IQ4_XS / 7% IQ2_S; down 51% IQ4_NL / 48% Q5_0), UD-IQ4_XS (gate/up 52% IQ3_XXS / 44% IQ4_XS; down 74% Q5_0 / 24% IQ4_NL), plus unsloth Q8_0 n-gram table shard 4. Drafts: unsloth Q8_0 sidecar, bartowski Q4_0 sidecar.
- Settings: 8K and conversion `llama-bench` rows used Q8_0 KV (pre-rule, r3 or 2 alternating rounds of r2); all fit and MTP `llama-cli` (`flake.nix`, 1 run each) rows used F16 KV.

| Build | Mode | PP t/s | TG t/s | Change | PPL / Note |
| --- | --- | ---: | ---: | ---: | --- |
| S2 | 27B Q8_0 plain @8K (ref) | 617.7 | 36.8 | reference | - |
| S2 | UD-IQ3_XXS plain @8K | 524.7 | 38.4 | - | - |
| S2 | UD-Q3_K_XL plain @8K | 509.0 | 39.7 | - | - |
| S2 | UD-IQ4_XS `-ncmoe 2` @8K | 426.1 | 30.5 | - | - |
| conv off | UD-Q3_K_XL @8K | 470.4 (1 outlier) | 39.23 | - | PPL 1.9170 |
| conv on | UD-Q3_K_XL @8K | 541.0 | 40.12 | +2.3% TG (~+6% PP) | PPL 1.9150 |
| conv on + Q8_0 table | UD-Q3_K_XL @8K | 533.1 | 40.54 | +1% TG | PPL 1.9108 |
| conv off | UD-IQ4_XS `-ncmoe 3` @8K | 398.4 | 29.56 | - | PPL 1.9236 |
| conv on | UD-IQ4_XS `-ncmoe 3` @8K | 434.4 | 30.58 | +9% PP, +3.5% TG | PPL 1.9222 |
| native Q5_0 | UD-Q3_K_XL + Q8 tbl, plain, 100K, `-ub 256` | 409.1 | 42.1 | - | 0.02-0.03 GiB free/GPU |
| native Q5_0 | same, 64K / 32K (`-ub 256`) | 416.4 / 417.0 | 43.8 / 43.9 | - | - |
| native Q5_0 | same, 32K, `-ub 512` | 472.6 | 41.2 | - | - |
| native Q5_0 | MTP2, Q4_0 draft ROCm1, experts on CPU, 64K, `-ts 0.505,0.495` | 382.9 | 34.9 | slower than plain | rejected |
| native Q5_0 | **MTP2, full Q4_0 draft ROCm1, 32K, `-ts 0.515,0.485`, `-ub 256`, draft ub 64** | 390.5 | **51.5** | +17% vs plain 32K | MTP3 & ub512 OOM |

- Memory per GPU on Q3_K_XL at 100K F16 KV `-ub 256`: weights 29.9 GiB (0.98 GiB mirrored), main KV 1176 MiB, QSA indexer key cache 588 MiB (mirrored), compute 364 MiB, recurrent state 56 MiB, AR staging 64 MiB; free 0.02-0.03 GiB. Full Q4_0 draft needs 2.3-2.5 GiB on one GPU. 132 repack-vs-CPU + 234 `test-backend-ops` cases pass.

### 2026-11-06 S2: Cheaper IQ sign handling on gfx906 (Grade A)

Replaces emulated `__vcmpne4` + `__vsub4` with `sign_mask4` / `iq_apply_signs` (`(g ^ m) + (m & 0x01010101)`) in IQ2_S mat-vec and IQ2_S/IQ3_XXS MMQ tile loads. Archived origin: `old-ignore/2026-11-06/s2-iq-signs/`.

- Source: `1785a4513` (branch `s2-iq-signs` on `02980eb3d`); baseline `bc0ce9691` (`build-catchup-20261105-213622`).
- Model: Flash Next UD-IQ3_XXS rev `c8b5954a88c2` (pre-2026-11-04 upload), standard Flash Next (`-b 1024 -ub 512`).
- Workload: `llama-bench` PP2048/TG128, r5 alternating (Grade A); 96-token greedy `llama-completion`.

| Build | Mode | PP2048 t/s | TG128 t/s | Change | Output sha |
| --- | --- | ---: | ---: | ---: | --- |
| baseline (`bc0ce9691`) | plain | 541.45 +- 1.84 | 41.17 +- 0.11 | - | `d9c2f5346efc` |
| candidate (`1785a4513`) | plain | 557.97 +- 1.09 | 43.27 +- 0.15 | **+3.05% PP / +5.11% TG** | same (bit-exact) |

- Correctness: exhaustive host check (all IQ2_S grid words x 256 sign bytes + IQ3_XXS) found 0 mismatches; `test-backend-ops` passed 115/115 IQ2_S/IQ3_XXS `MUL_MAT`/`MUL_MAT_ID` cases on ROCm0 (n=1..64). Landed.

### 2026-11-05 Catch-up to official 4b77dee05 and 27B config sweep / 32K depth

Archived origins: `old-ignore/2026-11-05/catchup-4b77dee05/`, `config-sweep/`, `depth32k/`.

- Source: `bc0ce9691` (merge of official `4b77dee05`), build `build-catchup-20261105-213622`; baseline `build-ilp` (`e04bc2e38`, old master / rebuild stage A).
- Model: 27B Q8_0 rev `4ca720788d1e`; DFlash draft `z-lab/Qwen3.8-27B-DFlash2` Q8_0 rev `2d9571f8ce46`.
- A3 hardware check (sysfs read 2026-11-05): ROCm1 already at max 178 W cap (ROCm0 225 W).

#### A. Catch-up regression screen (`bc0ce9691` vs `e04bc2e38`, 1 run each, Grade B)

| Build | Mode | PP t/s | TG t/s | Change | Output sha |
| --- | --- | ---: | ---: | ---: | --- |
| baseline (`e04bc2e38`) | plain `llama-bench` | 697.55 | 37.75 | - | - |
| candidate (`bc0ce9691`) | plain `llama-bench` | 696.88 | 37.90 | -0.1% / +0.4% | - |
| baseline | MTP2 (`flake.nix` CLI) | 598.0 | 64.0 | - | `dd228997b103` |
| candidate | MTP2 (`flake.nix` CLI) | 584.5 | 64.1 | -2.3% / +0.2% | same |
| baseline | DFlash5 (`flake.nix` CLI) | 584.0 | 66.8 | - | `9f2493efdd9b` |
| candidate | DFlash5 (`flake.nix` CLI) | 574.4 | 67.4 | -1.6% / +0.9% | same |

- TP smoke checks passed at n=1 and n=2048; 32-token `llama-completion` coherent.

#### B. 27B speculative depth sweep (2K & 32K) and ubatch 4096 check (`bc0ce9691`)

32K prompt: first 116,000 bytes of `src/llama-context.cpp` at `bc0ce9691` (~32K tokens), `-c 40960`. Ubatch test: `llama-bench` PP4096 `-b 4096 -ub 2048,4096` r3.

| Mode | PP t/s | TG t/s | Change | Output sha |
| --- | ---: | ---: | ---: | --- |
| plain, 2K CLI | 587.4 | 36.8 | - | - |
| MTP2, 2K CLI (2 runs) | 584.5 / 582.1 | 64.1 / 63.9 | - | `dd228997b103` both |
| MTP3, 2K CLI | 579.0 | 61.5 | -3.8% vs MTP2 | `39f8c45d0e87` |
| DFlash5, 2K CLI (2 runs) | 574.4 / 570.6 | 67.4 / 67.7 | +5.9% vs MTP2 | `9f2493efdd9b` both |
| DFlash7, 2K CLI | 570.7 | 62.5 | -7.7% vs DFlash5 | same as DFlash5 |
| plain, 32K CLI | 541.2 | 29.4 | -20% vs 2K | `b94840dcade5` |
| MTP2, 32K CLI | 512.3 | 42.7 | -33% vs 2K | `c8ae52851843` |
| DFlash5, 32K CLI | 512.0 | 39.9 | -41% vs 2K | same as plain |
| `llama-bench` PP4096, `-ub 2048` / `-ub 4096` (r3) | 684.1 / 606.3 | - | **-11.4%** (Grade A) | `-ub 4096` rejected |

### 2026-10-29 Fork rebuild on official `36a0e2ffc`, sweeps, profiles, and official-head `418f905a4` checks

Major consolidation day creating the clean squashed fork (`f5031e739` on official `36a0e2ffc`). Archived origins: `old-ignore/2026-10-29/{rebuild,a1-sweep,n1-dflash,tg-host-gap,head-merge-validation,quant-placement-compare,iq2s-single-token-experiment,grouped-moe-integration,strata-review,optlab-review}/`.

#### A. Streamlined rebuild stages U, A, B, C, D (`36a0e2ffc`)

Matched `llama-bench`, `-sm tensor`, 50/50, FA on. 27B: `-b/-ub 2048`, PP2048/TG128, r2. Flash Next (pre-11-04 upload `c8b5954a88c2`): Q8_0 KV, `-b/-ub 1024/512`, r2 + separate r5. All builds use `max-ilp` + CXX `-O3` unless noted.

| Stage | Description (diff vs upstream) | 27B PP2048 | 27B TG128 | FN PP2048 | FN TG128 | FN r5 PP / TG |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| U | Official `36a0e2ffc` (0 diff) | 448.60 | 32.77 | 302.99 | 39.66 | 304.33 +- 3.78 / 39.76 |
| A | Old master `e04bc2e38` (on `418f905a4`, ~18.9k+/1.2k-, 123 files) | 684.56 +- 0.31 | 36.82 +- 0.16 | 527.79 +- 2.01 | 38.99 +- 0.47 | - |
| B | Pass 1 cleanup + B1/B12 fixes + peer-AR opt-in (17655+/1213-, 121 files) | 682.92 / 684.19 | 36.96 / 36.61 | 525.73 / 528.13 | 38.34 / 39.56 | 509.92 +- 17.95 / 39.23 +- 0.13 |
| C (no `max-ilp`/`-O3`) | Pass 2 without `-tps` pipeline stages, missing flags | 669.23 +- 0.96 | 35.70 +- 0.02 | 514.19 +- 5.19 | 40.69 +- 0.38 | - (flags proved worth 2-3%) |
| C3 (fixed flags) | Pass 2 without `-tps` stages (15735+/1177-, 118 files) | 685.60 +- 1.32 | 36.58 +- 0.21 | 520.22 +- 1.14 | 36.93 +- 2.78 | 516.43 +- 7.96 / 38.79 +- 0.97 |
| D (`f5031e739`) | Pass 3: DSV4/MiniMax/MiMo back to upstream (14894+/924-, 109 files) | **685.34 +- 1.52** | **36.67 +- 0.22** | **527.63 +- 0.50** | **39.09 +- 0.15** | 520.11 +- 7.34 / 39.26 +- 0.26 |

- D vs U: +52.8% 27B PP, +11.9% 27B TG, +74.1% FN PP, FN plain TG neutral. Greedy 96-token output byte-identical across B, C, D (and U for 27B; FN diverges after ~30 tokens due to repack/fusion rounding).

#### B. Speculative decoding on Stage D (`llama-cli`, `flake.nix`, 512 out, c16384, 1 run each, Grade B)

Grouped MoE disabled; MTP uses `--spec-draft-backend-sampling` (Flash Next uses Q8_0 sidecar via `-md`):

| Model | Mode | Prompt t/s | Generation t/s | vs plain TG |
| --- | --- | ---: | ---: | ---: |
| 27B Q8_0 | plain | 462.0 | 36.2 | - |
| 27B Q8_0 | MTP2 | 514.0 | 60.7 | +68% |
| 27B Q8_0 | MTP3 | 557.8 | 58.0 | +60% |
| Flash Next IQ3_XXS | plain | 348.0 | 37.2 | - |
| Flash Next IQ3_XXS | MTP2 | 339.1 | 49.6 | +33% |
| Flash Next IQ3_XXS | MTP3 | 349.6 | 44.5 | +20% |

#### C. A1 config & env-knob sweep (TG128 on master, 1/1 baseline in same run, mostly 1 rep, Grade B)

| Setting | 27B Q8_0 TG t/s (1/1 baseline) | Flash Next TG t/s (1/1 baseline) |
| --- | --- | --- |
| `-ts 0.52/0.48` | 36.20 (37.42); retest r2: 36.06, 36.16 (37.13, 37.01) | 37.80 (37.46) |
| `-ts 0.48/0.52` | 35.52 (37.39) | 36.64 (40.08) |
| `-ts 0.56/0.44` | 34.92 (37.40) | not run |
| `-t 16` | 36.61 (36.93) | 39.44 (37.64) |
| `-t 8` | 36.86 (37.30) | default |
| `GGML_CUDA_AR_SLABS=0` | 37.39 (37.16-37.45) | 36.79 (37.52-37.58) |
| `GGML_SCHED_SHARE_VIEW_INPUTS=0` | 37.11 (37.16-37.37) | 36.12 (37.20-37.27) |
| `GGML_META_PARALLEL_DISPATCH=0` | 35.83 (37.37) | 35.82 (37.27-37.99) |
| `LLAMA_SHEXP_SPLIT=0` | n/a | 36.94 (37.52) |

#### D. N1 DFlash replay check, N2 host-gap diagnosis, and scale-fold attempt

- **N1 DFlash draft replay check** (27B DFlash5, 512 tok): draft ctx rebuilt graph 273 times at 0.36 ms each (98 ms of 8.5 s = 1.1%), BUT runs on single-device ROCm0 where `ggml-cuda` HIP graph prop-compare already replays it (691-node generate graph replayed 67 times with only 2 re-captures; 58-node inject graph also replayed). Upper bound <= 0.7 ms of 80 ms step (< noise: baseline 57.3 vs instrumented 60.1 t/s, text identical). Closed by bound.
- **N2 per-token host-gap diagnosis & experiments** (`mx-final/build-rb`, `rocprofv3` TG32 last 20 tokens):
  - Profiled TG: 27B 25.3 (unprofiled 36.6), FN 21.4 (unprofiled 39.1). Confirmed token graph replayed (2 `hipGraphLaunch`/token). Profiled both-GPU-idle: 27B 1.04 ms/tok, FN 4.48 ms/tok (~70 small `hipMemcpyAsync`, ~10 syncs).
  - N2a (parallel replay launch on worker threads, `llama-bench` TG128 r2): 27B `36.63 +- 0.31 -> 36.67 +- 0.17`, FN `39.14 +- 0.07 -> 38.78 +- 0.07` -> reverted.
  - N2b (`GGML_SCHED_INPUT_STAGING_F32_MAX=1048576` to stage 20 KB-262 KB F32 split inputs async, TG128 r3 alternating): 27B `37.09, 37.13 -> 37.11, 37.13`; FN `38.43, 38.66 -> 38.45, 38.12` -> delta ~0, not changed.
- **Scale-fold attempt (P1a)**: folding `1/hc` SCALE after `hc_attn_down`/`hc_ffn_down` (Q8_0 10240x320) into repacked mat-vec built and matched output, but TG32 trace had identical kernel counts (116,873 on Agent 1, 10,145 `scale_f32` on both builds; r5 TG -0.5%), so fusion never fired. Branch dropped.

#### E. Flash Next 64-token TG profile (`fe34c2bf` candidate library)

Unprofiled TG 36.71, profiled 28.93 (-21%); 63 steady repeats (2.178 s window, both GPUs active 1.905 s):

| Kernel group | ROCm0 sum | ROCm1 sum | Two-GPU union | Union/repeat |
| --- | ---: | ---: | ---: | ---: |
| Non-expert Q6_K/type-14 and q8_0-repacked mat-vecs | 377 ms | 401 ms | 628 ms | 9.97 ms |
| TP AllReduce | 327 ms | 65 ms | 336 ms | 5.33 ms |
| IQ2_S, IQ4_NL and IQ3_S expert mat-vecs | 199 ms | 215 ms | 295 ms | 4.69 ms |
| Float mat-vecs | 195 ms | 194 ms | 278 ms | 4.41 ms |
| Routing and MoE reduction | 100 ms | 109 ms | 152 ms | 2.41 ms |
| Position/attention, hyper-connections, GDN/SSM | 142 ms | 153 ms | 274 ms | 4.35 ms |

ROCm1 per-token detail: `quantize_q8_1<32>` 484 calls 2.33 ms; `scale_f32` 303 calls 1.77 ms; float mat-vecs 288 calls 3.08 ms; Q6_K & Q8_0 repacked mat-vecs 496 calls 6.36 ms.

#### F. Official-head `418f905a4` port validation, MMQ traversal fix, quant/placement & IQ2_S checks

- **Official-head `418f905a4` merge validation** (candidate `12229543ad` vs accepted baseline `84932540a` / `fbdb024841`, Flash Next IQ3_XXS rev `c8b5954a`, grouped MTP2, 2097 prompt / 128 out, `-lm dio`, c100096, Q8 KV, b/ub 1024/512, 1 pair):
  - Supplemental 27B Q8_0 baseline (`llama-bench` PP2048/TG128): **690.256 / 36.802 t/s**.
  - Unrepaired `418f905a4` port: baseline **345.09 / 37.68** -> candidate **247.31 / 40.22** (**-28.3% PP**, +6.7% TG; acc 73/107 = 0.68224, L 2.35, exact text match `ea8578dc`).
  - Preserved 2026-10-25 `flashnext-fast` binary re-check on same DIO command: baseline **349.53 / 39.18** vs preserved `flashnext-fast` **350.50 / 39.47** (acc 73/107, exact text `7c9cebe8`).
  - **W17 (`terra-port-fix`)**: restoring row-first MMQ tile traversal on gfx906 non-stream-K fixed PP: baseline **347.97 / 36.77** -> traversal candidate **351.30 / 39.51** (acc 73/107, exact text `7c9cebe8`, Grade B).
- **W18 (`iq2s-single-token-experiment`)**: two-row IQ2_S MMVQ for gfx906 single-token even rows (3 CPU-ref tests pass; doubled `flake.nix` 4,143 prompt / 256 out, `-t 20 -tb 20`, no MTP, 1 pair, Grade B): baseline (`a4666a05`) **396.35 PP / 37.20 TG** (39.21 s wall) -> candidate (`fe34c2bf`) **398.80 PP / 38.08 TG** (+0.62% PP, +2.37% TG, 41.44 s wall, identical text `ac812967`). (An earlier run was invalidated due to RPATH loading candidate libs.)
- **Flash Next quant, CPU MoE offload, and `-sm layer` check** (binary `9384bc4b`, `flake.nix` 2097 prompt / 128 out, c100096, Q8 KV, b/ub 1024/512, `-t 8 -tb 8`, `taskset -c 0-7`, no MTP, 1 run each, Grade B):
  - IQ3_XXS GPU experts (`-sm tensor`): **377.50 PP / 38.94 TG** (30.49 s wall, 5.47 GiB RSS).
  - IQ3_XXS GPU experts (`-sm layer`): **353.76 PP / 30.99 TG** (-6.3% PP, -20.4% TG, 23.37 s wall, 3.22 GiB RSS).
  - IQ4_XS first 8 MoE layers on CPU (`ROCm_Host` 10.7 GiB), op-offload ON: **197.59 PP / 24.42 TG** (43.68 s wall, 15.8 GiB RSS).
  - IQ4_XS first 8 MoE layers on CPU, `--no-op-offload`: **204.92 PP / 25.15 TG** (42.49 s wall, 15.8 GiB RSS).
- **Grouped-MoE integration check**: checkpoint `84932540a` + 7 runtime/test files; 121-file patch replay verified; 18/18 canonical IQ2_S/IQ4_NL CPU-ref tests passed on ROCm0 (repacked harness timed out at 110 s shared cap).
- **External code reviews**: Strata (`870aedd9e1eb`, MIT) yielded ideas S1-S5 (S2 landed 2026-11-06); `llama-cpp-optimization-lab` (`5408613e`, NVIDIA CPU/PCIe offload) yielded L1 and caution against non-e2e micro-wins.

### 2026-10-25 Flash Next grouped-MoE / GPU-feedback experiment, 100K/MTP fit, and 27B CLI

Archived origins: `old-ignore/2026-10-25/{flashnext-fast/README.md,2026-10-25-qwen3.8-flash-next-gfx906.md,2026-10-25-flash-next-mtp-short-check.md,2026-10-25-qwen3.8-27b-flake-cli.md}`.

#### A. `flashnext-fast` isolated experiment (grouped MoE verify & GPU feedback)

- Source: isolated worktree on `fdc8e5b65cd6120b2f89e509d5f354bda4a3a30b` (`build-fast`, TheRock ROCm 10.2.0, `max-ilp`/`-O3`).
- Workload: `flake.nix` (2,097 framed prompt tokens, SHA-256 `fe8d514fa094...`), 256 outputs, greedy seed 42, c100096, Q8_0 KV, 50/50 TP. Flash Next UD-IQ3_XXS rev `c8b5954a88c2` + self-contained Q8_0 MTP sidecar (`-devd ROCm1`), MTP2, `-b 1024 -ub 512`. Order: control, grouped, device, both, then reverse (2 runs each, Grade A/B).

| Flash Next MTP2 setting | Mean PP t/s | Mean TG t/s | TG range | Runs |
| --- | ---: | ---: | ---: | ---: |
| Single-GPU draft (`-devd ROCm1`), both off | 330.3 | 47.65 | 47.3-48.0 | 2 |
| Single-GPU draft, **grouped MoE only (`GGML_CUDA_FLASHNEXT_MOE=1`)** | 372.9 | **50.45 (+5.9%)** | 49.1-51.8 | 2 |
| Single-GPU draft, device feedback only (`LLAMA_QWEN4_MTP_DEVICE=1`) | 376.4 | 48.35 | 48.3-48.4 | 2 |
| Single-GPU draft, both enabled | 326.6 | 49.55 | 47.3-51.8 | 2 |
| Tensor-parallel draft control, both off | 379.2 | 47.30 | 47.3-47.3 | 1 |

- All 9 Flash Next runs produced identical generated text, accepting 157/194 draft proposals (80.928%), mean length 2.62 (depth acceptance 0.876 / 0.742).
- 27B Q8_0 (`4ca720788d1e`) regression check (same prompt/c100096/Q8 KV, `-b/-ub 2048`, built-in MTP2, order off/on/on/off):
  - Both flags off: mean PP 575.8, mean TG **52.60** (52.3-52.9), 2 runs.
  - Both flags on: mean PP 577.0, mean TG **53.30** (53.3-53.3), 2 runs (identical text across all 4 MTP2 runs).
  - Plain decode, both flags on: PP 585.0, TG **35.60** (1 run). Original binary checks: plain 34.7 TG, MTP2 48.7 TG.
- Unit checks: 18/18 canonical IQ2_S/IQ4_NL CPU-ref cases + 24/24 repacked IQ4_NL CPU-ref cases (`gfx906-test-flashnext-moe`, max NMSE < 1.1e-14) passed. Grouped MoE accepted for integration (2026-10-29) and made default (2026-11-06); GPU feedback left experimental.

#### B. Flash Next 100K capacity, Strata comparison, and matched MTP2 check

- Build: `build-ilp/bin/llama-cli` on upstream `344be0a60` overlay / `b11929-fdc8e5b65`, TheRock ROCm 10.2.0, single runs (Grade B).
- Long 4K prompt (`flake.nix` x2 = 4,092 raw tokens, SHA-256 `d121f276660e...`), `-n 4096`, `-c 100096`, Q8_0 KV, UD-IQ3_XXS (`c8b5954a88c2`), host PLE (`LLAMA_PLE_SHARD` unset):

| Engine / Mode | Context | Batch / ubatch | PP t/s | TG t/s | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| mx-llama.cpp UD-IQ3_XXS plain | 100096 | 2048 / 2048 | 552.3 | 36.7 | ~31.3/34.3 GB per GPU (~2.8 GiB free); 262144 1-tok alloc passed |
| Strata UD-IQ3_XXS plain (external ref) | 9216 | - | 312.3 | 39.6 | 4,394 prompt / 4,096 gen; different context & prompt framing |
| mx-llama.cpp UD-IQ3_XXS plain (short 2046-tok prompt) | 100096 | 2048 / 2048 | 434.7 | 38.0 | Short check |
| Short 256-out check (`b11929-fdc8e5b65`): 27B Q8_0 plain | 100096 | 2048 / 2048 | 558.9 | 35.2 | `flake.nix` 2,046 raw tok |
| Short 256-out check (`b11929-fdc8e5b65`): FN IQ3_XXS plain | 100096 | 2048 / 2048 | 454.9 | 38.4 | Same prompt |
| Matched long 4K->4K (`b11929-fdc8e5b65`): FN IQ3_XXS plain | 100096 | 1024 / 512 | 396.9 | 37.3 | Control for MTP2 (2048/2048 MTP OOM by 195.5 MiB on ROCm0) |
| Matched long 4K->4K (`b11929-fdc8e5b65`): FN IQ3_XXS MTP2 (Q8_0 sidecar) | 100096 | 1024 / 512 | 368.5 (-7%) | **42.9 (+15%)** | ~0.6-0.7 GiB free/GPU |

- UD-IQ4_XS 100K attempt: failed allocation on ROCm0 with Q8 KV; with Q4 KV, `-b/-ub 256`, `-ts 45/55`, prefill buffer failed short 440 MiB on ROCm1.

#### C. Qwen3.8-27B Q8_0 `flake.nix` n4000 CLI at `-c 100096`

- Build: `build-ilp` build 11776 / commit `766f7c5e0`, 27B Q8_0 (`4ca720788d1e`), `-b/-ub 2048`, `-c 100096` (~6K populated), `flake.nix`, `-n 4000`, greedy seed 42, single runs (Grade B):
  - **MTP depth 2** (`--spec-draft-backend-sampling`): **549.7 PP / 56.8 TG**.
  - **MTP depth 3** (`--spec-draft-backend-sampling`): **545.2 PP / 60.0 TG**.
  - **DFlash2 Q8_0, max draft 7**: **543.7 PP / 59.9 TG**.

### 2026-10-12 Two-GPU TP optimization marathon, speculative decoding push, and three-slot validation

Archived origins: `old-ignore/2026-10-12/notes/{2026-10-12-work-status.md,2026-10-12-tp-profile-findings.md,2026-10-12-roofline.md,2026-10-12-three-slots-before-fix.md}`, `old-ignore/2026-10-12/tp-profile/bench-*.md`, `old-ignore/2026-10-12/2026-10-12-three-slots.md`.

#### A. Initial baseline (`build-ilp`, HEAD `8d73dc74e`, build `5d6e2c412` 11547) & `-ts` sweep

Model: `unsloth/Qwen3.8-27B-GGUF:Q8_0` (27.04 GiB, 27.32 B), `-ngl 999 -fa on -b 2048 -ub 2048 -p 2048 -n 128 -r 2` unless noted (Grade B):

| Config | PP t/s | TG128 t/s | Note |
| --- | ---: | ---: | --- |
| Single GPU ROCm0 (`-sm none -b/-ub 512`, PP512) | 437.22 +- 2.14 | 22.76 +- 0.01 | ~660 GB/s weight stream |
| TP `-ts 50/50` (baseline `bench-tp-50-50` / `bench-ts-sweep`) | 545.24 +- 2.60 / 546.67 +- 0.34 | 30.17 +- 0.12 / 30.22 +- 0.13 | Baseline before Oct 12 changes |
| TP `-ts 53/47` | 579.89 +- 2.64 | 30.42 +- 0.24 | - |
| TP `-ts 56/44` | 591.33 +- 0.71 | 30.17 +- 0.19 | +8% PP pre-duplex/slabs |
| TP `-ts 58/42` / `60/40` | 570.9 / 561.0 | - | - |
| TP `-b/-ub 4096` PP4096: `50/50` / `56/44` | 526.8 / 563.4 | - | ub 4096 slower |
| TP `GGML_CUDA_TP_OVERLAP=1`: `50/50` / `56/44` | 543.7 / 587.4 | - | No-op (HIP never compiled `#elif` branch) |

- Initial `rocprofv3` PP profile (`-ts 50/50`, 518 t/s profiled, 3.88 s ubatch): `mmq_gemm_repacked` (493 calls) 1698 ms ROCm0 / 2018 ms ROCm1 (+19% slower MI50, ~33 int8 TOPS = ~56% dp4a peak on MI60); `gated_delta_net_chunked_cuda` (48) 207/243 ms; `concat_non_cont` (48) 71/75 ms; `quantize_mmq_q8_1` (493) 66/67 ms; all kernels 2259/2625 ms; AR D2H 378/378 ms; AR H2D 397/383 ms (0.78 s exposed copies, ~2.7 GB each way).
- Initial `rocprofv3` TG profile (`-p 0 -n 64`, 29.2 t/s = 34.3 ms/tok): `mul_mat_vec_q8_0_repacked` (432 calls) 19.3 ms (~715 GB/s; 64 fused gate+up = 8.5 ms); `ggml_cuda_ar_kernel` (128) 4.1 ms ROCm0 / 1.7 ms ROCm1; `rms_norm_f32` (305) 1.9 ms; `quantize_q8_1` (257) 1.1 ms; LM head 0.95 ms; `k_bin_bcast` (176) 0.75 ms; ~1870 kernels/tok; `GGML_META_PROFILE=1` revealed meta whole-token graph never engaged without P2P (129 per-subgraph dispatches, 37.7 ms host compute). Also confirmed `-bs` backend sampling is a no-op under `-sm tensor` (CLI 458.2/30.4 base vs 464.4/30.0 `-bs`).

#### B. Core TP optimization bundle (W04-W09, 2026-10-12 01:07-01:41, Grade B)

Step-by-step measured progression on 27B Q8_0 (`-ts 50/50`, vs baseline worktree `../mx-base` HEAD `8d73dc74e`):

| Step / Change | Metric | Before | After | Note |
| --- | --- | ---: | ---: | --- |
| W04: Capture-safe host-staged AR (device counters + `prepare`/`launch_rank`) -> whole-token HIP graph captures | `llama-bench` TG64; CLI 256 tok | 30.2 (CLI 460.8/29.9) | **34.77** (CLI 439.4/34.4) | Text byte-identical |
| W05 + W07 + W09: Fused `add+rms_norm+mul` (+Q8_1 producer), duplex copy-engine AR (PP 599->647), tiled-transpose concat | `llama-bench` r3 PP2048 / TG128; CLI 256 | 551.8 / 30.16 (CLI 460.8/29.9) | **649.2 / 35.81** (CLI 539.5/35.2) | W05 part of TG 34.8->36.2; text byte-identical |
| W06: `GGML_RP_MMV_ROWS=8` (was 16) | `llama-bench` TG128 | 36.2 | **36.8** | Made default (`-ts 56/44` gave 647/34.6) |
| W08: Slab-overlapped prefill AR (`GGML_CUDA_AR_SLABS` 0 / 2 / 4 / 8) | `llama-bench` r2 PP2048; CLI 256 | 651.1 (0) / 670.7 (2) | **682.0 (4)** / 680.1 (8); CLI **558.2 / 36.0** | +4.7% PP, TG neutral, text identical |
| Post-slab `-ts` sweep (`llama-bench` r2 PP2048 / TG128) | `50/50` vs `53/47` vs `56/44` | 683.6 / 36.8 (50/50) | 702.2 / 35.6 (53/47); 704.1 / 34.8 (56/44) | Keep 50/50 for TG |

- Patch kit exported (`llama-head-08ffba21e`, 124 files, 0 mismatches). `test-backend-ops` passed CONCAT (+3 transposed cases), RMS_NORM, ADD, MUL_MAT.

#### C. Speculative decoding push (Rounds 1-3 + single-GPU DFlash, 2026-10-12 01:44-15:02, Grade B)

- **DFlash2 quick look (`-n 256`, n-max 7)**: 423.2 PP / 39.3 TG (L=3.86, pos acc 0.80/0.64/0.47/0.33/0.26/0.21/0.15; step ~98 ms = verify(8) 57-62 + draft gen 27 + prepare 17). Target batch scaling (`llama-bench -p N -n 0`): 1 tok 31.6 ms, 2: 34.1, 3: 38.5, 4: 40.5, 5: 44.9, 8: 57.1 ms.
- **W10 narrow-batch Q8_0 mat-vec (2-8 cols)**: planar activations (`GGML_RP_NC_PLANAR`) + 16-lane rows (`GGML_RP_NC_L16`) + `__launch_bounds__`: `pp4` 40.6 -> 37.2 ms, `pp6` 48.7 -> 40.8-41.4 ms, `pp8` 57.0 -> 45.7-45.9 ms. Round 3 LDS activation staging (`GGML_RP_NC_LDS`): 6-col kernel 75.4 -> 73.1 us (-3%), df5 n256 CLI TG 37.8 -> 38.4. Verify `rocprofv3` GPU1 busy (`p1/p3/p6`): all kernels 29.7 / 33.8 / 39.6 ms (mat-vec 19.9 / 22.3 / 25.9 ms = 1.12x at 3 cols, 1.30x at 6 cols).
- **W11/W12 draft/MTP repack & prefill fixes**: `LLAMA_DRAFT_REPACK` df5 n512 TG 42.9 -> 45.9 (draft gen 20.6 -> 17 ms/step); `LLAMA_SPEC_LAZY_PREFILL` mtp3 n512 PP 512 -> 549.3 (TG 52.9); `LLAMA_MTP_REPACK` mtp3 n512 TG 52.9 -> 54.2 (PP 555); M-RoPE tap accum (`LLAMA_TAP_ACCUM_MROPE`) + tail deferral (`LLAMA_DFLASH_DEFER_TAIL`) + skip dup D2H: df5 n64 PP **433.8 -> 547.8** (L=3.71 unchanged; prompt gather 130->21 ms, tgtsync 206->0 ms).
- **Tried and rejected during Rounds 1-3**: `RPL=4` (spilled), `GGML_RP_NC_FUSION=1` (slower 6-8 cols), `#pragma unroll 2` (6-col mat-vec 25.1 -> 28.9 ms), multi-row planar Q8_1 producer in norm (norm slowed by same 0.6 ms), `LLAMA_DFLASH_MIRROR_OUTPUT=1` (TG 45.9 -> 43.1), `--spec-draft-p-min` 0.5/0.6 (TG 30.7/28.6), MTP depth 1 (47.7) / 4 (47.2), content-keyed meta token graph (L collapsed 2.95 -> 1.11-1.6), MTP fused catch-up (TG 51.0 vs 51.8), n-gram drafting screen at n512 (MTP2 555.2/54.4, `ngram-mod,draft-mtp` 559.6/54.7, `ngram-simple,draft-mtp` 553.1/48.7, `ngram-mod` alone 567.1/35.6 = plain).
- **Final-boss `llama-cli -f flake.nix -n 4000` (c32768, b/ub 2048, greedy seed 42) progression**:

| Session / Arm (n=4000) | PP t/s | TG t/s | Mean acc L | Text vs plain |
| --- | ---: | ---: | ---: | --- |
| Start (cold GPUs): plain | 564.2 | 35.6 | - | reference |
| Start: dflash7 / dflash5 / mtp3 | 420.9 / 420.6 / 510.6 | 40.1 / 45.4 / 48.9 | n/a | diverges ~char 1436 |
| Round 1 end: plain (warm GPUs 71-72C) | 558.3 / 556.3 | 31.7 / 32.3 | - | reference |
| Round 1 end: **mtp2 + `--spec-draft-backend-sampling`** | **545.5 / 553.2** | **56.7 / 60.3** | 2.60 | diverges char 2548 |
| Round 1 end: mtp2 (earlier) / mtp3 | 542.9-546.5 / 548.7-557.1 | 56.1-57.7 / 53.8-55.5 | 2.60 / 2.86 | - / char 1232 |
| Round 1 end: dflash5 / dflash7 | 428.7 / 425.3 | 54.5 / 51.6 | 3.33 / 3.59 | char 1232 / - |
| Round 2 (post M-RoPE tap fix): plain | 568.0 | 34.8 | - | reference |
| Round 2: mtp2 + dbs | 546.0 | 55.1 | 2.60 | char 2548 |
| Round 2: dflash4 / dflash5 / dflash6 | 535.2 / 533.5 / 534.4 | 50.9 / 51.4 / 49.9 | 3.13 / 3.33 / 3.46 | char 1232 |
| Round 3 (post `NC_LDS`): mtp2 + dbs confirm | 558.3 | **61.6** | 2.60 | char 2548 (variance vs 55.1) |

- **General prose essay prompt** (printing press 1450-1700, `-n 512`, greedy): plain 36.6, MTP2+dbs 56.5, DFlash n-max 5 **59.4 / 60.0** (L=3.81, pos acc 0.84/0.68/0.53/0.43/0.32, draft gen 15.4 ms/step), DFlash n-max 7 55.8.
- **W16 Duplex-AR race fix (`src_read_done` event, 13:23)** + **W13 Single-GPU DFlash draft, private repacked head (4.3 -> 1.6 ms), async lag-one replay (4096 window)** (14:14):
  Five diverse prompts (`-n 512`, greedy; CLI PP / TG / L):

| Prompt (prompt tokens) | Plain (PP/TG) | MTP2 (PP/TG/L) | DFlash5 TP draft (PP/TG/L) | DFlash5 single-GPU draft (PP/TG/L) |
| --- | --- | --- | --- | --- |
| Long-doc summary `docs/build.md` (11808) | 614.4 / 34.8 | 598.2 / 52.8 / 2.53 | 602.5 / 37.7 / 2.90 | 598 / 47.4 / 2.90 |
| CUDA code review (4333) | 594.2 / 34.0 | 575.5 / 52.6 / 2.39 | 577.8 / 47.3 / 3.42 | 577 / 57.6 / 3.45 |
| Python code gen (126) | 180.3 / 36.4 | 172.9 / 48.8 / 2.11 | 161.5 / 46.8 / 2.87 | 157 / 52.1 / 2.87 |
| Multi-part math proof (213) | 212.0 / 36.5 | 211.2 / 58.5 / 2.52 | 191.3 / 63.0 / 4.15 | 195 / 71.7 / 4.15 |
| Tutorial `docs/function-calling.md` (8066) | 620.7 / 35.0 | 602.2 / 49.0 / 2.28 | 609.6 / 40.5 / 2.98 | 606 / 48.8 / 2.98 |
| **Mean TG** | **35.3** | **52.3** | **47.1** | **55.5 (+3.2 vs MTP2)** |

- **Greedy logprob correctness characterization** (`llama-server`, `/v1/chat/completions`, greedy):
  - 3 prompts x 384 tokens (1,152 tokens/mode): **0/1152 non-argmax tokens** in plain, MTP2, and DFlash5. Mean chosen logprob: plain -0.3116, MTP2 -0.3114, DFlash5 -0.3151. Every first divergence occurred at a near-tie (plain top-1 vs top-2 margin 0.0028-0.0475 nats) where spec picked plain's #2 due to 0.001-0.007 nat batched-verify rounding.
  - 3,000-token `flake.nix` check: Plain 32.5 TG (0/3000 non-argmax, 48 near-ties < 0.05 nats), MTP2 52.9 TG (0/3000 non-argmax, 1st diff tok 315 margin 0.033), DFlash5 52.0 TG (0/3000 non-argmax, 1st diff tok 30 margin 0.074); zero 50-token repeats.

#### D. Three-slot validation and context limits (2026-10-12, W15)

- Source: `a39e959d9c362d66c3702aa582e5f51febdab757` + fixes (`libllama.so` SHA256 `50b0872a...`, `libllama-common.so` `8a494529...`). 27B Q8_0, 2x 32,752 MiB gfx906, F16 KV, `-np 3`, `-ts 50/50`, `-b/-ub 2048`.
- Fixes: persistent tap tensors allocate 2049 rows (matching rounded reservation for 3 slots); DFlash `in_prompt` tracked per sequence so one slot's prompt never defers another slot's rejected verification rows past rollback (pre-fix isolation at `-c 196608 -b/-ub 2046`: default had 2 draft-cache errors, wave 0/1 43.44/8.60 s; `LLAMA_DFLASH_DEFER_TAIL=0` had 0 errors, 43.73/9.29 s).
- Capacity & concurrent two-wave regression (`check.py`: Wave 0 = 11,728 / 4,617 / 19-tok prompts x 256 out; Wave 1 = 19 / 20 / 19-tok prompts x 128 out; greedy seed 42):

| Mode | Max `-c` (per slot) | Peak free GPU0 / GPU1 (MiB) | Wave 0 / Wave 1 (s) at max `-c` | Wave 0 / Wave 1 (s) at equal `-c 196608` | Acceptance / Argmax |
| --- | ---: | --- | --- | --- | --- |
| Plain | 479,232 (159,744) | 515.6 / 522.9 | 38.43 / 7.76 | **38.37 / 7.75** (fastest 3-slot) | 0 non-argmax (1150 entries) |
| MTP2 | 354,816 (118,272) | 567.6 / 554.5 | 44.18 / 9.95 | 44.06 / 9.89 | 447/629, 226/308; 0/1152 non-argmax |
| DFlash5 (fixed, 50/50) | 242,688 (80,896) | 566.2 / 7582.8 | 42.83 / 8.41 | 44.16 / 8.55 (pre-fix `TAP_STATIC=0`) | 0 errors, 0/1152 non-argmax |
| DFlash5 (fixed, 48.25/51.75) | 338,688 (112,896) | 437.9 / 598.9 (pre-fix) | 45.16 / 8.64 (fixed) | - | 0 errors |

- Lower-capacity fixed DFlash check at `-c 230400`: 43.21 / 8.44 s, 1080.5 / 8033.4 MiB free. Plain KV+arena slope: ~38.2 kB/token/GPU.

### 2026-10-11 Head merge GDN comparison (`5d6e2c412`, build 11547)

Archived origin: `old-ignore/2026-10-11/head-merge/*.jsonl` (timestamps `2026-10-12T06:07:16Z` to `06:15:59Z`).
Model: `Qwen3.8-27B-Q8_0.gguf` rev `4ca720788d1e`, 2x gfx906 `-sm tensor`, `-fa on`, F16 KV, `-b/-ub 2048`, `-t 4`, depth 0, PP2048 / TG256, r1 (Grade B):

| Arm (`5d6e2c412`, build 11547) | Timestamp (UTC) | PP2048 t/s (ns) | TG256 t/s (ns) |
| --- | --- | ---: | ---: |
| `before` (pre-merge) | 06:07:16-21Z | 499.66 (4,098,795,428) | 31.54 (8,117,889,064) |
| `merged-upstream-gdn` | 06:13:51-56Z | 482.94 (4,240,718,338) | 30.43 (8,412,546,405) |
| `merged-gfx906-gdn` | 06:15:55-59Z | **506.89** (4,040,313,590) | 30.01 (8,529,607,781) |

- Keeping gfx906 chunked GDN beat upstream GDN by +5.0% on PP2048 (506.89 vs 482.94).

### 2026-10-01 Port kits (`e358d591` & `0bc845d3`), MTP planning fix, and chunked GDN snapshot fix

Archived origins: `old-ignore/2026-10-01/patches/llama-head-e358d-to-mx-20261001/{README.md,benchmark.jsonl}`, `llama-head-0bc845-to-mx-38b4/README.md`.

- Model: `Qwen3.8-27B-Q8_0.gguf` rev `fe1e2a23d973`, 2x gfx906 `-sm tensor`, `-fa on`, F16 KV, `-b/-ub 4096`, `-t 4`, depth 16384, PP2048 / TG256, `--no-warmup` (Grade B):
  - Older Sep 10 reference (cited): **414.24 PP / 30.28 TG**.
  - Pre-merge 2026-10-01 checkpoint (`38b4da7ba` on `0bc845d3`, r1 default): **431.13 PP / 27.81 TG**.
  - Same checkpoint with `GGML_ENABLE_CUSTOM_AR=0` (r2 mean): **435.14 PP / 28.43 TG**.
  - Merged `e358d591` + MX `8535054eb` (build 11449, tree `6f1050df1`, timestamp `2026-10-01T18:01:55Z`, r1): **424.5733 PP2048** (4,823,666,528 ns) / **28.8573 TG256** (8,871,250,781 ns) (-1.52% PP, +3.77% TG vs pre-merge; -4.7% TG vs Sep 10 ref).
- **MTP planning fix (`8a6351ddf`)**: moving target/draft scheduler reservation into initialization before first prompt on a 2,097-token prompt (single runs): **243.23 PP / 37.3 TG -> 397.18 PP / 34.1 TG**.
- **Chunked GDN snapshot correctness fix (`8a6351ddf`)**: custom chunked GDN ignored `keep_rs_t` (only wrote final state, leaving 2 rollback slots unwritten for MTP2); fixed and verified via 4 new `test-backend-ops` `GATED_DELTA_NET` cases (`K=3`, chunked & serial fallback) on ROCm0.

### 2026-09-29 HIP host-staged AllReduce regression check (`AGENTS.md`)

- Source: `AGENTS.md` head-merge check note dated 2026-09-29 (Grade D, no separate raw log retained).
- Finding: excluding HIP from the host-staged AllReduce path in `ggml/src/ggml-cuda/allreduce.cu` reduced Qwen3.8-27B Q8_0 **PP2048 by 21%** and **TG256 by 38%** on two gfx906 GPUs (matching the exact magnitude of the 2026-09-10 HIP guard removal gain below).

### 2026-09-10 MX + upstream merge campaign, HIP AllReduce guard fix, and Muse-Glimmer / Gemma-4 benchmarks

Archived origins: `old-ignore/undated/legacy/RESULTS.md`, `old-ignore/2026-09-10/benchmark_results/*.jsonl`.

#### A. Qwen3.8-27B Q8_0 head-merge campaign (`mx-upstream-2026-09-09`)

- Inputs: MX `0c81bd502` / HEAD `c0df68229`, upstream `434ddbbc0`, PR 27825 `ee5b990d1`. RCCL OFF.
- Canonical protocol: Qwen3.8-27B Q8_0, 2x MI60, `-sm tensor`, `-fa on`, `-b/-ub 4096`, actual cached depth 16384, PP2048, TG256, r1, `--no-warmup` (Grade B).

| Arm | PP2048 t/s | TG256 t/s | Finding / Attribution |
| --- | ---: | ---: | --- |
| Clean upstream `434ddbbc0` (`build-norccl`, 4096/4096) | 196.9662 | 20.1327 | Clean upstream baseline at 16K depth |
| Clean upstream `434ddbbc0` (2048/512) | 179.5703 | 19.8510 | Sample ns: PP 11.405 s, TG 12.896 s |
| Merged MX + upstream baseline | 324.4369 | 18.3208 | Used meta fallback AR (see correction below) |
| PR 27825 labelled arm (rebuilt validation) | 325.7131 (322.2327) | 18.7649 (18.4168) | Re-attributed: stale HIP guard kept fallback active |
| PR 27825 labelled + `LLAMA_SHEXP_SPLIT=0` | 314.0392 | 17.6764 | Reject: shared-expert split helps PP & TG |
| PR 27825 labelled + `GGML_RP_GENERIC_MMV=1` | 315.6193 | 18.4499 | Reject: repacked MMV beats generic |
| Private exact BF16 overlap (2048/2048) | 325.0877 | 18.5718 | Reject on this head |
| Afternoon isolation: repacking off (`-nr 1`) | 130.4371 | 16.9108 | Disabling repack loses 59.9% PP, 9.49% TG |
| Afternoon isolation: serial dispatch (`GGML_META_PARALLEL_DISPATCH=0`) | 317.0269 | 18.7604 | -2.60% PP, +0.42% TG -> keep parallel |
| Afternoon isolation: fresh default control (fallback AR) | 325.4868 | 18.6829 | Control before guard fix |
| **W01: HIP overlap-only guard removed (11 lines); generic AR active** | **394.1701** | **25.8895** | **+21.10% PP, +38.57% TG** (5.196 s PP, 9.888 s TG) |

- Proof of execution: TG32 trace showed zero `ggml_cuda_ar` launches before the 11-line guard removal; after removal, `gfx906-test-tp --n 1 --check` recorded four `ggml_cuda_ar_kernel<float, __hip_bfloat16>` launches, and `--n 2048 --check` passed in 67,643 us (vs 81,601 us before; n=1 before was 38,351 us).
- 64K winner gate (pre-guard-fix PR 27825 arm): depth 16384 `325.7131 / 18.7649` -> depth 65536 **216.0927 PP (-33.65%) / 16.2924 TG (-13.17%)** (fit on 2x 32 GB MI60).

#### B. Muse-Glimmer-30B Q8_0 and Gemma-4-31B-it Q8_0 JSONL runs (2026-09-10 23:34-23:42Z)

Protocol: 2x gfx906 `-sm tensor`, `-fa on`, F16 KV, `-b/-ub 4096`, `-t 4`, depth 16384, PP2048 / TG256, r1 (Grade B).

| Stage / Build (commit, build#) | Model & HF rev | Timestamp (UTC) | PP2048 t/s (ns) | TG256 t/s (ns) |
| --- | --- | --- | ---: | ---: |
| `02-rocm-host-allreduce` (`9b06d8c7a`, 10900) | Muse-Glimmer-30B Q8_0 (`faa5b025c584`) | 23:41:26-42:16Z | 375.06 (5,460,477,023) | 29.43 (8,698,671,484) |
| `03-mxxm-repack-moe` (`fd04166db`, 10902) | Muse-Glimmer-30B Q8_0 (`faa5b025c584`) | 23:40:04-45Z | **450.57** (4,545,328,641) | **29.55** (8,664,428,354) |
| `04-vr-bench-harness` (`0abd145ec`, 10903) | Muse-Glimmer-30B Q8_0 (`faa5b025c584`) | 23:36:40-37:23Z | 453.13 (4,519,720,591) | 27.20 (9,410,215,733) |
| `03-mxxm-repack-moe` (`fd04166db`, 10902) | Gemma-4-31B-it Q8_0 (`c1ac76e99d55`) | 23:38:19-39:17Z | 318.76 (6,424,990,473) | **22.50** (11,376,209,280) |
| `04-vr-bench-harness` (`0abd145ec`, 10903) | Gemma-4-31B-it Q8_0 (`c1ac76e99d55`) | 23:34:53-35:51Z | **322.71** (6,346,268,575) | 21.18 (12,084,723,722) |

(Note: `02-rocm-host-allreduce_gemma-4-31b-q8.jsonl` was empty.)

### 2026-09-01 Upstream `0eadefebd` + MX `63ed8cd1a` merge validation

Archived origin: `old-ignore/undated/legacy/MERGE_2026-09-01.md`.
Build: Release, HIP gfx906, RCCL ON, `GGML_CCACHE=OFF`, `GGML_NATIVE=ON`, `LLAMA_BUILD_GFX906_TESTS=ON`.

- **Qwen3.8-27B Q8_0** (`-sm tensor`, DirectIO, internal AR, BF16 overlap, `-b 2048`, `-c 16384`, `flake.nix` CLI): Prompt **320.0 t/s**, Generation **11.3 t/s**.
- **Qwen3.8 Flash-Next (`qwen4exp A3B`) UD-IQ3_XXS** (first recorded run; 76.32 GiB file, 26.82 GiB lazy-mapped PLE, ~50.2 GiB resident GPU buffers; `-sm layer`, lazy DirectIO, `-b 2048`, depth 4096):
  - `PP512`: **175.811 t/s**; `TG128`: **23.297 t/s**; `PP2048`: **223.366 t/s**; `TG512`: **23.436 t/s**.
  - `flake.nix` CLI smoke test: **223.7 PP / 23.2 TG**.
- **Q8 overlap harness (`gfx906-test-tp`) fixed-slab & slab-size sweep**:
  - Fixed 1024-col slabs: `n=1` off F32 pass 49.744 ms; `n=2048` off F32 pass 73.855 ms; `n=2048` on F32 pass 53.248 ms; `n=2048` on BF16 pass **41.861 ms**.
  - Experimental compute-slab knob (rejected): 512 cols BF16 pass 42.469 ms; 256 cols incorrect output; 2048 cols illegal GPU memory access.
- CTest note: multi-GPU `test-llama-archs` hit RCCL `ncclGroupEnd()` failure; single-GPU `HIP_VISIBLE_DEVICES=0` passed in 66.66 s.

### 2026-08-14 MX head `46b95d97e` merge and first Qwen3.8-27B Q8_0 validation

Archived origins: `old-ignore/undated/legacy/RESULTS.md`, `old-ignore/2026-08-13/results/2026-08-13-gfx906-q8.csv`.

- Source: MX head `46b95d97e-dirty` (build 10279) + standalone `q8-tp-overlap.patch` (SHA-256 `080ecd80b352...`).
- Model: `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-Q8_0.gguf` (29,036,089,344 tensor bytes, SHA-256 `a680f44a0692...`; 506 Q8_0 + 360 F32 tensors; 65 `ffn_down.weight` Q8_0 tensors + integrated `nextn` MTP tensors).
- Protocol: 2x MI60, `-sm tensor`, `-fa on`, `-b/-ub 2048`, `-lm dio`, depth 16384, PP2048 / TG1024, r1, `--no-warmup`, BF16 overlap env vars set (Grade B):
  - Focused checks: `n=1` fallback pass 34,570 us; `n=2048` BF16 overlap pass 43,938 us.
  - Full-model `llama-bench`: **PP2048 319.4779 t/s**, **TG1024 18.4011 t/s**, score (`TG + PP/10`) **50.3489**, estimated 16K+1K scenario **106.9326 s**.
- Qwen3.8 MTP check (5,072-byte synthetic engineering prompt SHA-256 `e1140a07...`, greedy seed 42, 1024 generated tokens, 1 run/arm):
  - `--spec-draft-n-max 2`: **243.9 PP / 29.7 TG**.
  - `--spec-draft-n-max 4`: **243.7 PP / 28.8 TG**.

### 2026-08-12 to 2026-08-13 Initial gfx906 Q8_0 exact BF16 TP overlap campaign (Qwen3.6-27B)

Earliest recorded campaign on this 2x MI60 host (archived origins: `old-ignore/2026-08-13/results/2026-08-13-gfx906-q8.csv`, `old-ignore/undated/legacy/RESULTS.md`).

- Source: `b8efb9510` / `b8efb9510-dirty` (build 10275), patch SHA-256 `080ecd80b352...`.
- Models: `unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0` (27.04 GiB, rev `5cb35eb3dcbf`; 506 Q8_0, 0 F16, 360 F32) and `UD-Q8_K_XL` (33.31 GiB; 291 Q8_0, 119 F16, 456 F32).
- Protocol: 2x MI60, `-ngl 99 -fa on -sm tensor -dev ROCm0/ROCm1 -lm dio -b 2048 -ub 2048 -pg 2048,0 -pg 0,1024 -d 16384 -r 1 --no-warmup`, 3 s cooldown (Grade B). Score = `TG + PP/10`; `scenario_seconds = 16384/PP + 1024/TG`.

| Model | Arm | Commit (build) | PP2048 t/s | TG1024 t/s | Score | 16K+1K (s) | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.6-27B Q8_0 | C0 `production_reference` | `b8efb9510` (10275) | 282.9760 | 19.8199 | 48.1175 | 109.5641 | Reference |
| Qwen3.6-27B Q8_0 | C1 `same_build_overlap_off` | `b8efb9510-dirty` (10275) | 283.0380 | 18.4718 | 46.7756 | 113.3220 | Same-build control |
| Qwen3.6-27B Q8_0 | `broad_hip_transport` (all collectives) | `b8efb9510-dirty` (10275) | 262.5141 | 14.0310 | 40.2824 | 135.3930 | Reject |
| Qwen3.6-27B Q8_0 | C2 `f32_overlap` (matching nodes) | `b8efb9510-dirty` (10275) | 276.3395 | 18.0294 | 45.6633 | 116.0856 | Reject |
| Qwen3.6-27B Q8_0 | **C3 `bf16_overlap_final` (matching nodes)** | `b8efb9510-dirty` (10275) | **318.7780** | **19.6266** | **51.5044** | **103.5702** | Keep opt-in (+12.65% PP) |
| Qwen3.6-27B UD-Q8_K_XL | `overlap_off` | `b8efb9510-dirty` (10275) | 281.0242 | - | - | - | Control |
| Qwen3.6-27B UD-Q8_K_XL | `bf16_overlap_final` | `b8efb9510-dirty` (10275) | 314.8519 | 15.7750 | 47.2601 | 116.9502 | Quality-first (+12.04% PP, -19.63% TG vs Q8_0) |

- Focused AllReduce correctness checks (all-ones matrix/activation, expected output 17408, copy-staged HIP path):
  - Initial broad HIP transport, overlap off: `n=1` pass 26,751 us; `n=2048` pass 68,423 us.
  - F32 two-slab overlap (pre-gate): `n=2048` pass 52,684 us.
  - BF16-wire two-slab overlap (final gate): `n=2048` pass **42,105 us**.
