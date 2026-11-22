# Performance records

Append-only log of dated runs, newest first. The live summary is in [PERFORMANCE.md](PERFORMANCE.md). Full commands, identities and raw data are in the linked [old-ignore](old-ignore/README.md) folders.

### 2026-11-21 Inkling 2 Mini, late: NVFP4 repacked experts, banded prefill attention, faster mix kernels

In plain words: the exact NVFP4 file now runs at 836 t/s prompt and 45.8 t/s generation on one GPU (490.8 / 30.5 this afternoon), close to the IQ4_NL file, and long prompts need 5.5 GB less memory. 27B and Flash Next did not change. Code on top of the record below: `06aa25f25` + `d37d86dd7` + `2162c74dd` (banded relative-bias attention, port of unsloth #386 plus a tile-kernel path for gfx906, default on for prefill), `a99a7930b` (wide-block `res_mix` / `rms_norm_cat`), `0ba4be7f1` (NVFP4 experts on the repacked path). Old build `build-u1sort-20261121` (`400989a0d`), new build `build-inkling-final-20261121` (`0ba4be7f1`). `llama-bench -r 2`, order old, new, new, old, below 60C; raw data in `build-inkling-final-20261121/validation/final/`, script [final_validation_2.sh](investigation-notes/2026-11-21-inkling2-mini/final_validation_2.sh). How the wins were found and what is left for each model: [launch-floor-playbook.md](launch-floor-playbook.md).

| Model | Config | PP2048 old | PP2048 new | TG128 old | TG128 new |
| --- | --- | ---: | ---: | ---: | ---: |
| 27B Q8_0 | 2 GPUs, `-sm tensor`, ub 512 | 613.5 | 614.5 (+0.2%) | 37.14 | 37.21 (+0.2%) |
| Flash Next UD-IQ3_XXS | 2 GPUs, `-sm tensor`, ub 512 | 655.0 | 657.7 (+0.4%) | 49.90 | 50.18 (+0.6%) |
| Inkling 2 Mini IQ4_NL | ROCm0, ub 512 | 564.2 | 878.6 (+55.7%) | 36.28 | 47.75 (+31.6%) |
| Inkling 2 Mini Q8_0 | ROCm0, ub 512 (1 run each) | 566.1 | 880.1 (+55.5%) | 34.47 | 44.41 (+28.8%) |
| Inkling 2 Mini NVFP4 + Q8_0 | ROCm0, ub 512 | 490.8 (a) | 836.3 (+70.4%) | 30.5 (a) | 45.80 (+50.2%) |

(a) merge-only build `build-catchup-20261121-e6d6`; the old build cannot run NVFP4. PP2048 at `-ub 2048`, new build: IQ4_NL 1137.7 (old 578.1), Q8_0 1137.3 (old 575.9), NVFP4 1081.1 (merge-only 509.7).

Per change, same binary or two builds, alternated:

| Change | File | PP2048 | TG128 |
| --- | --- | ---: | ---: |
| Wide-block `res_mix_f32` / `rms_norm_cat_f32` (26.7 and 11 us per call before) | IQ4_NL | 771.1 / 776.9 | 43.57 / 47.73 (+9.5%) |
| Banded bias, unsloth scalar kernels only (first port) | IQ4_NL | 761 / 383 (-50%) | 47.3 / 47.5 |
| Banded bias in the tile kernel, ub 512 | IQ4_NL | 775.4 / 879.6 (+13.4%) | 47.7 / 47.7 |
| Banded bias in the tile kernel, ub 2048 | IQ4_NL | 772.9 / 1135.7 (+47%) | - |
| NVFP4 experts repacked (`GGML_CUDA_REPACK_NVFP4=0` / default), ub 512 | NVFP4 | 637.8 / 737.3 (+15.6%) | 37.9 / 45.5 (+20%) |

Banded attention at 16K context (`llama-batched-bench -npp 16384 -ub 2048`): PP 576.5 / 686.9, TG 34.3 / 34.6, compute buffer 8,665 / 3,170 MiB. Small ubatches (token generation) stay on the dense bias; this is why TG does not change and why TG falls with context (36.4 t/s at 4K context against 45.8 at an empty one).

Tried, this approach did not work as a default: the unsloth banded kernels alone. Their fast prefill kernel needs NVIDIA matrix units; on gfx906 only the scalar kernels run and PP halves. They stay as the fallback (decode shapes, quantized K/V, kv_rows); the gfx906 path adds the bias lookup inside `flash_attn_tile`.

Quality (NVFP4 file, `llama-perplexity -c 4096 --chunks 2`, KLD against the first run): the same config twice gives KLD 0 (deterministic). Dense ub 512 vs dense ub 2048: mean KLD 0.0049, same top token 97.8%. Banded vs dense: 0.0055 (ub 512), 0.0060 (ub 2048), 0.0054 (ub 96), same top 97.0-97.6%. NVFP4 repacked vs generic kernels: 0.0054, same top 97.2%, PPL ratio 0.9997 +/- 0.0019. So every change sits at the level this model already moves when only the batch size changes. Open: why a pure batching change moves this model that much (guess: near-tie flips in the top-8 of 128 expert routing). PPL over 8 chunks, dense / banded: IQ4_NL 2.8307 / 2.8305 (ub 512), 2.8288 / 2.8283 (ub 2048); NVFP4 2.8210 / 2.8261; Q8_0 dense 2.8268.

`test-backend-ops` on ROCm0: MUL_MAT 2727/2727, MUL_MAT_ID 1162/1162, FLASH_ATTN_EXT 4420/4420, FLASH_ATTN_EXT_BANDED 71/71, CLIP_ROWS 9/9, RES_MIX 9/9, LOOP_GATE 3/3, RMS_NORM_CAT 6/6, SCONV_RES 12/12. Gap: the harness does not allocate the repack buffer type, so the NVFP4 repacked kernels are covered only by the model-level checks above. Known old failure, not from this work: generic NVFP4 (also Q2_K, Q3_K) MUL_MAT_ID with one token and 2 used experts.

NVFP4 repack memory: ROCm0 weights 16,320 MiB generic, 17,625 MiB repacked (+1,305 MiB): two decoded F16 scales per 32 values against 1 byte per 16 in the file. Untried: keep the raw UE4M3 bytes and decode in the kernel (VRAM-neutral).

NVFP4 server and multi-user, new build, ROCm0: `llama-server` with mmproj Q8_0, `-c 16384`: text TG 37.1 t/s (was 30.4), image request PP 460.0 (was 389.5), TG 35.0, server stays up. `llama-batched-bench -npp 4096 -npl 1,2,4 -ub 2048`: NVFP4 PP 991 / 746 / 559, aggregate TG 36.4 / 51.0 / 87.7; IQ4_NL PP 1032 / 768 / 570, TG 37.5 / 53.5 / 97.1 (IQ4_NL on the old build: PP 507 / 429 / 378, TG 30.6 / 44.7 / 79.2). CPU check (`--device none -ngl 0`) ends without an assert.

Cost model check: steady-state token generation is 98% GPU-busy with kernels back to back; a launch costs about 2.2 us in the app (2.0 us bare on the MI60, 2.6 us on the MI50; `rocprofv3` shows 4.2 us). Captured graphs are active for this model. See the playbook.

### 2026-11-21 Inkling 2 Mini: crash fix, fused ops, NVFP4 baseline

In plain words: Inkling 2 Mini no longer crashes on images, runs about 37% faster on prompts and 19% faster on generation, and now has an exact NVFP4 file. 27B and Flash Next did not change. Code: upstream `e6d624b9a` merged (`4c5738580`), crash fix `ddca7dd9b`, fused ops `3253927e4`, `b6c927fd1`, `f17889023`, `85bb8a0f7`. Old build `build-u1sort-20261121` (`400989a0d`), new build `build-sconvfuse-20261121` (`85bb8a0f7`). `llama-bench -r 2`, order old, new, new, old, start temperature below 60C; raw data in `build-sconvfuse-20261121/validation/final/`, script [final_validation.sh](investigation-notes/2026-11-21-inkling2-mini/final_validation.sh).

| Model | Config | PP2048 old | PP2048 new | TG128 old | TG128 new |
| --- | --- | ---: | ---: | ---: | ---: |
| 27B Q8_0 | 2 GPUs, `-sm tensor`, ub 512 | 614.4 | 611.0 (-0.6%) | 37.09 | 37.24 (+0.4%) |
| Flash Next UD-IQ3_XXS | 2 GPUs, `-sm tensor`, ub 512 | 658.2 | 655.9 (-0.3%) | 50.02 | 50.48 (+0.9%) |
| Inkling 2 Mini IQ4_NL | ROCm0, ub 512 | 564.2 | 775.1 (+37.4%) | 36.42 | 43.35 (+19.0%) |
| Inkling 2 Mini Q8_0 | ROCm0, ub 512 (1 run each) | 564.3 | 772.7 (+36.9%) | 34.30 | 40.33 (+17.6%) |
| Inkling 2 Mini NVFP4 + Q8_0 | ROCm0, ub 512 | 490.8 (a) | 641.0 (+30.6%) | 30.5 (a) | 35.43 (+16.2%) |

(a) The old build cannot run NVFP4 (it lacks #32669 / #32704); this column is `build-catchup-20261121-e6d6` (merge only, no fused ops), measured earlier the same evening, not alternated with the new runs. PP2048 at `-ub 2048` on the new build: IQ4_NL 774.2 (was 578.1), NVFP4 662.1 (was 509.7 on the merge-only build).

Per fused op, Inkling 2 Mini IQ4_NL on ROCm0, same binary with the kill switch off / on (mean of 2, order off, on, on, off):

| Op | Replaces | PP2048 | TG128 |
| --- | --- | ---: | ---: |
| `CLIP_ROWS` | 9-kernel q/k/r norm clip, 3 per layer | 565.3 / 567.6 | 36.07 / 39.15 (+8.5%) |
| `RES_MIX` | concat chain, rms norm, two matmuls, softmax of the block residual mix | 568.8 / 751.6 (+32%) | 39.54 / 40.45 (+2.3%) |
| `LOOP_GATE` + `RMS_NORM_CAT` | elementwise parts of the loop controllers (12 kernels to 5) | 754.5 / 772.9 (+2.4%) | 40.47 / 41.43 (+2.4%) |
| `SCONV_RES` | conv, dynamic taps, residual and scale of the short convolutions | 774.0 / 777.7 | 41.56 / 43.26 (+4.1%) |

Greedy text with a switch on and off matches for 35-48 tokens, then can differ by a word (float summation order). `test-backend-ops` on ROCm0: MUL_MAT 2727/2727, MUL_MAT_ID 1152/1152, CLIP_ROWS 9/9, RES_MIX 9/9, LOOP_GATE 3/3, RMS_NORM_CAT 6/6, SCONV_RES 12/12.

Tried, this approach did not work: one kernel for the whole loop controller with its three matmuls, one thread block per token. TG128 40.5 to 11.7 t/s, PP unchanged. A single block cannot stream the controller weights. Untried: two kernels (a block per gate value, a block per output slice). [Patch and notes](old-ignore/2026-11-21/loop-ctrl-fused/README.md).

NVFP4 file (`$MODELS_DIR/inkling2-mini/Inkling-2-Mini-NVFP4-Q8_0.gguf`, 17.57 GB, 18.8 GB VRAM at `-c 4096`): made with `llama-quantize --allow-requantize --tensor-type 'blk\.([0-9]|1[0-9])\.ffn_(gate_up|down)_exps=nvfp4' <Q8_K_XL.gguf_file> <out> Q8_0`. Coherent greedy text. It is 17% (PP) and 18% (TG) behind IQ4_NL / 12% behind Q8_0 on TG, all from the generic NVFP4 expert kernels (rocprof, merge-only build, NVFP4 vs Q8_0: `mul_mat_vec_q<NVFP4>` 8.1 vs 4.6 ms per token; `mul_mat_q<NVFP4>` 2.87 vs 1.72 s per 2048-token prompt; data in `build-catchup-20261121-e6d6/validation/inkling2-mini/nvfp4-speed/`). The 144 BF16 loop controller tensors do not matter (F32 copy: PP 488.5 vs 490.9, TG 30.5 vs 30.65).

NVFP4 server and multi-user, new build, ROCm0: `llama-server` with mmproj Q8_0, `-c 16384`: text 30.4 t/s TG (bench 35.4, so about 4.7 ms per token of server overhead is still unexplained), image request (2750 prompt tokens) PP 389.5, TG 28.5, server stays up. `llama-batched-bench -npp 4096 -npl 1,2,4 -ub 2048`: PP 551 / 475 / 414, aggregate TG 29.7 / 43.7 / 68.6 (IQ4_NL on the old build: PP 507 / 429 / 378, TG 30.6 / 44.7 / 79.2).

Image crash: fork bug, `s_copy_main` / `s_write` not filled for the Inkling input class; see the [handoff](handoff.md). Check: `llama-completion --device none -ngl 0 -n 6` on the IQ4_NL file ends without the `get_rows` assert.

### 2026-11-21 catch-up to official 46eb1742d

Merge `296a9e2c0` plus build fixes, then cleanup (peer AllReduce archived, about 30 sweep-only env knobs removed) and unsloth #264 (device MoE id counting sort). Builds: `build-merge-20261121` (merge only), `build-cleanup-20261121` (`698f466dd`), `build-u1sort-20261121` (`400989a0d` code, now `build-current`), each against `build-catchup-20261118-134459`. Cooled below 60C, order base/new/new/base, llama-bench r3. Raw data: each build's `validation/`.

| Model | Test | base | merge | cleanup | change (cleanup) |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B Q8_0 | PP2048 | 700.2 / 700.1 | 698.8 | 698.9 | -0.2% |
| Qwen3.8-27B Q8_0 | TG128 | 37.33 / 37.35 | 37.31 | 37.23 | -0.3% |
| Qwen3.8-27B Q8_0 | MTP2 server 5-prompt mean | 63.46 t/s, 40.31 ms/step, acc 0.797 | - | 63.92 t/s, 40.02 ms/step, acc 0.797 | ms/step -0.7% |
| Flash Next UD-IQ3_XXS | PP2048 (-b 1024 -ub 512) | 656.7 / 652.3 | 656.6 | 657.8 | +0.9% |
| Flash Next UD-IQ3_XXS | TG128 | 50.17 / 50.33 | 50.55 | 50.55 | +0.4% |
| Flash Next UD-IQ3_XXS | MTP3 server 5-prompt mean | 59.44 t/s, 49.49 ms/step, acc 0.660 | - | 59.04 t/s, 49.81 ms/step, acc 0.660 | ms/step +0.6% (base spread 1.9%) |
| 27B, 4 users x 16K | PP / 4-way TG | 577.2 / 84.26 | - | 572.8 / 82.24 | -0.8% / -2.4% (one pair, see repeat below) |
| Flash Next, 4 users x 16K | PP / 4-way TG | 475.6 / 84.66 | - | 486.9 / 85.98 | +2.4% / +1.6% |

Unsloth #264 on the same build (`GGML_CUDA_MOE_COUNTING_SORT=0` vs default), Flash Next: PP2048 at `-ub 512` 657.3 vs 657.0 (0.0%), TG128 50.31 vs 50.32; PP2048 at `-b 2048 -ub 2048` 877.4 vs 895.5 (+2.1%, both on runs above both off runs). MTP3 against the cleanup build: 59.57 vs 59.91 t/s, 49.32 vs 49.08 ms/step, acc 0.660 both; greedy Flash Next text identical. The first port trapped on out of range ids and aborted on ROCm1 under `-sm tensor` (meta expert slices); fixed in `400989a0d`.

27B 4 x 16K repeat, order A,B,B,A, base vs `build-u1sort-20261121`: PP 576.3 / 568.0 vs 566.3 / 571.8 (-0.5%), 4-way TG 85.57 / 82.27 vs 82.27 / 82.46 (-1.8%). Only the first, coldest run (start 49C, others 56-59C) is faster; the last base run equals the new build (82.27), so the drop is the start temperature, not the code. Both pairs ran base first from cold.

TP check pass at n 1 and 2048 for each build; 27B greedy text identical; `test-backend-ops` MUL_MAT_ID, TOP_K, SET_ROWS, GATED_DELTA_NET, GET_ROWS pass on ROCm0 (1152 and 352 cases with #264).

Merge notes: two conflicts (`top-k.cu` ours, `mmid.cu` fork overloads with upstream `ne_store`). Silent traps fixed: `expert_bounds` last entry `nex_prev + n_dumped`, `GGML_OP_COUNT` 101 (upstream SLEEP plus DSV4_HC_MIX), `build_attn_iswa_impl` needs `gctx.build_cpy_k`. Core delta vs upstream: +15276/-1039 in 121 files (was +16426/-995 in 124 against edec462cc), with #264 adding +715.

### 2026-11-18 catch-up to official edec462cc

Build `build-catchup-20261118-134459` (merge 992b4580f, final d7ea110cb) against `build-kvrows-20261116`, same binaries layout, cooled below 60C, order base/new/new/base. Raw data: `build-catchup-20261118-134459/validation/`.

| Model | Test | base | new | change |
| --- | --- | ---: | ---: | ---: |
| Qwen3.8-27B Q8_0 | PP2048 (llama-bench r2) | 715.2 / 709.1 | 713.3 / 711.9 | 0.0% |
| Qwen3.8-27B Q8_0 | TG128 | 37.91 / 37.89 | 37.97 / 37.89 | +0.1% |
| Qwen3.8-27B Q8_0 | MTP2 server 5-prompt mean | 65.40 t/s, 39.11 ms/step, acc 0.797 | 66.11-66.26 t/s, 38.59-38.68 ms/step, acc 0.797 | +1.1% |
| Flash Next UD-IQ3_XXS | PP2048 (-b 1024 -ub 512) | 663.7 / 664.3 | 671.9 / 672.9 | +1.3% |
| Flash Next UD-IQ3_XXS | TG128 | 50.50 / 50.73 | 51.16 / 50.24 | 0.0% |
| Flash Next UD-IQ3_XXS | MTP3 server 5-prompt mean | 59.17 / 61.00 t/s, 50.87 / 49.31 ms/step, acc 0.682 | 59.14 / 60.05 t/s, 49.68 / 48.94 ms/step, acc 0.660 | ms/step -1.3% |
| 27B, 4 users x 16K (batched-bench) | PP / 4-way TG | 577.9 / 82.6 | 572.9 / 81.4 | -0.9% / noise |
| Flash Next, 4 users x 16K | PP / 4-way TG | 474.1 / 83.5 | 478.7 / 84.5 | +1.0% / +1.2% |

TP check pass at n 1 and 2048; 27B greedy text correct. Flash Next greedy text changed (acceptance 0.682 -> 0.660 on the same prompts) while 27B did not; the likely cause is upstream GCN stream-k MMQ (#30022) changing the sum order of the IQ prefill matmuls (27B prefill uses the repacked GEMM). Not verified; a PPL/KLD A/B would settle it.

Merge notes: upstream #31617 removed the HIP shim that took 32-bit masks in raw `__shfl_*_sync`; 7 fork calls now use `ggml_cuda_shfl*`. The generic top-k MoE kernel is now 64-lane on gfx906; the fork's with_norm sum order is kept (bit-identical for n_expert_used <= 32). gfx906 keeps row-first MMQ for dense and MoE.

pi adapter (harness-hub Qwen3.8-27B-pi-adapted, F16 LoRA on 27B Q8_0): runtime `--lora` aborts under `-sm tensor` (`ggml-backend-meta.cpp:900`, no split rule for `lora_a`/`lora_b`); `-sm layer` asks for a 47 GB compute buffer on ROCm0 and fails. Not fixed yet.

### 2026-11-06 F2, P4, toolchain check and acceptance-aware speculative runs

Land the first two Fable-review kernels, check a suspected 27B MTP2 regression, and measure speculative modes without acceptance noise. Kind: A/B experiment and profile.

- Source: `52bcba210` (F2 `0920eb0ac` + P4); baseline `a27e4931e`. Toolchain A/B: same source built with ROCm 10.2.0a20261012 (clang `6da73151`, side venv `~/infer/toolchains/rocm-10.2.0a20261012`) and 10.3.0a20261105 (clang `36c1564f`, the TheRock venv since 2026-11-06 00:33).
- Model: 27B Q8_0 rev `4ca720788d1e`; Flash Next UD-IQ3_XXS rev `d96201194f10` with the Q8_0 n-gram table, Q8_0 MTP sidecar.
- Settings: standard; Flash Next MTP3 at ctx 100352, `-ts 0.52,0.48`, draft `-devd ROCm1`, `--spec-draft-n-ubatch 64`.
- Workload: llama-bench r2; speculative rows from [`spec_accept.sh`](scripts/spec_accept.sh) (5 prompts, 256 tokens, one pass each).

Load-excluded profiles (window after the last gap above 20 ms; rocprofv3 slows TG 7-24%): Flash Next TG runs about 3280 launches per token per GPU at about 8 us each, 86-89% busy, so it is launch-bound; MTP3 at 100K leaves ROCm0 22% idle; Flash Next PP is 33% repacked GEMM, 26% stock `mul_mat_q` for the IQ experts and 19% rocBLAS SGEMM; 27B TG is 70% Q8_0 mat-vec at 94% busy.

| Build | Mode | PP t/s | TG t/s | ms/step | Accept | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F2 off | Flash Next llama-bench @8K | 514 | 40.8 | - | - | - |
| F2 on | Flash Next llama-bench @8K | 510 | 46.0 | - | - | +12% TG |
| P4 off | Flash Next llama-bench @8K (with F2) | 508.9 | 41.8 (tg32) | - | - | - |
| P4 on | Flash Next llama-bench @8K (with F2) | 563.8 | 41.9 (tg32) | - | - | +11% PP |
| F2 off, new clang | Flash Next MTP3 100K | - | 59.4 | 52.18 | 0.720 | - |
| F2 on, new clang | Flash Next MTP3 100K | - | 62.7 | 49.48 | 0.721 | -5.2% step |
| F2 on, old clang | Flash Next MTP3 100K | - | 60.8 | 51.36 | 0.728 | +3.8% step vs new |
| F2 on, draft ROCm0 `-ts 0.46,0.54` | Flash Next MTP3 100K | - | 63.2 | 49.94 | 0.737 | +0.9% step |
| old clang | 27B MTP2 | - | 64.5 | 40.28 | 0.816 | - |
| new clang | 27B MTP2 | - | 62.5 | 40.27 | 0.777 | same step |
| old / new clang | 27B llama-bench | 712.1 / 705.2 | 37.77 / 37.71 | - | - | -1.0% PP |

KLD (Flash Next, base = same build, 4 chunks unless noted): `-ub 256` vs `-ub 512` 4.66e-3; `-ts 0.52,0.48` vs `1/1` 4.74e-3; P4 per matmul family 4.7-5.1e-3; F2 at `-b 8 -ub 8` (2 chunks) 4.4e-3; same top token 98.0-98.9% in all, PPL unchanged.

P4 KLD controls: the kernel output matches rocBLAS to about 2e-6 relative (max diff 1e-5 at max |value| 5-60). Running the kernel but keeping the rocBLAS result gives KLD exactly 0 (no side effects). Keeping rocBLAS but scaling `ssm_alpha` by (1 +- 3e-7) gives 4.7e-3, so the model is sensitive to about 1e-6 noise at these sites and any change of F32 sum order lands near 5e-3. All of these are measured against the IQ3 build itself, not a higher-precision reference. Not yet measured: KLD against a Q8_0/BF16 reference with P4 on and off. A Sonnet review found no kernel defects and judged default-on safe. Open on `main`: the dispatch lacks `ne11 <= 65535` (grid y limit), `src0->ne[0] == src1->ne[0]` and a restriction to `hc_*`/`ssm_alpha`/`ssm_beta` by name, so other models' small F32 weights (e.g. MoE routers with under 128 experts) also take this path.

Verdict: F2 and P4 are landed. Flash Next MTP3 at 100K now matches 27B Q8_0 MTP2 (62.7 vs 62.5 mean TG over the same prompts). The new ROCm nightly does not slow 27B; single-prompt llama-cli MTP numbers swing 10-18% with the greedy text and must not be used for A/B. The draft device does not matter (validated wrong). F1, Q1, N3 are validated wrong. T1 is unfinished (wrong output). Not covered: long prompts that fill 100K, DFlash on the new tree, F2 tuning for T=4. Raw: [f2-p4-accept](old-ignore/2026-11-06/f2-p4-accept/).

### 2026-11-06 Flash Next UD-IQ3_XXS at 100K with MTP

Get MTP plus 100K context on a Q3-class Flash Next quant, as fast as possible. Kind: A/B experiment and profile.

- Source: `32a8e415b` plus grouped MoE on by default (this commit); builds `build-conv-20261106` and `build-fnspeed-20261106`.
- Model: UD-IQ3_XXS rev `d96201194f10` with the Q8_0 n-gram table; draft unsloth `mtp-Qwen3.8-Flash-Next-Q8_0` (sidecar) on ROCm1 (`-devd ROCm1`).
- Settings: standard Flash Next with ctx 100352, target `-ts 0.52,0.48`, `--spec-draft-n-ubatch 64`. Q3_K_XL does not fit this (see the big-quants record); UD-IQ3_XXS has about 5.9 GiB less in GPU weights.
- Workload: standard llama-cli, one run each unless noted; llama-bench rows at depth 8192, 2 alternating rounds of r2.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| conv | plain | 407.8 | 41.7 | - | - | `8db6ec2802ff` |
| conv | MTP2 | 402.6 | 48.9 | +17% | - | `cdc496d26b1a` |
| conv | MTP2 + grouped MoE (env) | 387.1 | 51.8 | +6% vs MTP2 | - | same as MTP2 |
| conv | MTP3 | 376.3 | 49.1 | - | - | `3dd833063ecc` |
| conv | MTP3 + grouped MoE (env) | 439.3 | 52.8 | +8% vs MTP3 | - | same as MTP3 |
| fnspeed | **MTP3, grouped MoE default (two runs)** | 438.2 / 439.5 | **52.2 / 52.5** | +26% vs plain | - | same as MTP3 |
| fnspeed | llama-bench plain @8K, copy kernels vs 2D memcpy | 515.5 / 515.4 | 40.53 / 40.37 | none | - | - |

Verdict: grouped MoE is landed as the default (`GGML_CUDA_FLASHNEXT_MOE=0` disables): +6-8% TG on MTP verify, byte-identical output, 132 repack-vs-CPU cases pass. Best 100K setup: MTP3 with the Q8_0 sidecar, about 52.4 TG (27B Q8_0 MTP2 is 64). Avoiding the HIP 2D memcpy in `ggml_cuda_cpy` is validated wrong (no change); the `__amd_rocclr_copyBufferRectAligned` time in that profile (15.6% of PP, 19% of MTP3 kernel time) was the load-time weight upload, which the profile window included (corrected on 2026-11-06, see the [F2/P4 record](#2026-11-06-f2-p4-toolchain-check-and-acceptance-aware-speculative-runs)). Other profile findings: stock `mul_mat_q` for the IQ gate/up experts about 21% of PP; the F32 MoE router (`ffn_gate_inp`, rocBLAS SGEMM at about 1.2 TFLOPS) about 15% of PP. Not covered: long prompts filling 100K, acceptance rates. Raw: [iq3-100k-mtp](old-ignore/2026-11-06/iq3-100k-mtp/).

### 2026-11-06 Flash Next big quants: repack conversion, Q8_0 n-gram table, 100K and MTP fit

Make UD-Q3_K_XL and UD-IQ4_XS (new upload, rev `d96201194f10`) fast, with 100K context and MTP if they fit. Kind: A/B experiment and fit study.

- Source: branch `s2-iq-signs`: S2, Q5_0/IQ4_XS repack (conversion `ebd76ced1`, then native Q5_0 `3e19b6adc`); build `build-conv-20261106`.
- Model: UD-Q3_K_XL and UD-IQ4_XS; "Q8_0 table" = the unsloth `engram/` Q8_0 n-gram table swapped in as shard 4 (`~/models/flashnext/`). Drafts: unsloth `mtp-...-Q8_0` sidecar, bartowski `mtp-...-Q4_0`.
- Settings: standard Flash Next, except where noted. The 8K and conversion rows used Q8_0 KV (before the F16-only rule); all fit and MTP rows use F16 KV.
- Workload: llama-bench PP2048/TG128 at depth 8192 (r3, or 2 alternating rounds of r2 for the A/B); llama-cli rows are the standard `flake.nix` workload, one run each.

Expert weights per quant (GGUF headers): UD-Q3_K_XL gate/up IQ3_XXS 76% / IQ4_XS 17% / IQ2_S 7%, down IQ4_NL 51% / Q5_0 48%; UD-IQ4_XS gate/up IQ3_XXS 52% / IQ4_XS 44%, down Q5_0 74% / IQ4_NL 24%. Before this change Q5_0 and IQ4_XS ran on the stock kernels.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| S2 | 27B Q8_0 plain @8K | 617.7 | 36.8 | reference | - | - |
| S2 | UD-IQ3_XXS plain @8K | 524.7 | 38.4 | - | - | - |
| S2 | UD-Q3_K_XL plain @8K | 509.0 | 39.7 | - | - | - |
| S2 | UD-IQ4_XS `-ncmoe 2` @8K | 426.1 | 30.5 | - | - | - |
| conv off | UD-Q3_K_XL @8K | 470.4 (one outlier) | 39.23 | - | - | PPL 1.9170 |
| conv on | UD-Q3_K_XL @8K | 541.0 | 40.12 | +2.3% TG | - | PPL 1.9150 |
| conv on + Q8_0 table | UD-Q3_K_XL @8K | 533.1 | 40.54 | +1% TG | - | PPL 1.9108 |
| conv off | UD-IQ4_XS `-ncmoe 3` @8K | 398.4 | 29.56 | - | - | PPL 1.9236 |
| conv on | UD-IQ4_XS `-ncmoe 3` @8K | 434.4 | 30.58 | +9% PP, +3.5% TG | - | PPL 1.9222 |
| native Q5_0 | UD-Q3_K_XL + Q8_0 table, plain, 100K, `-ub 256` | 409.1 | 42.1 | - | - | - |
| native Q5_0 | same, 64K | 416.4 | 43.8 | - | - | - |
| native Q5_0 | same, 32K | 417.0 | 43.9 | - | - | - |
| native Q5_0 | same, 32K, `-ub 512` | 472.6 | 41.2 | - | - | - |
| native Q5_0 | MTP2, Q4_0 draft on ROCm1, experts on CPU, 64K, `-ts 0.505,0.495` | 382.9 | 34.9 | slower than plain | - | - |
| native Q5_0 | **MTP2, full Q4_0 draft on ROCm1, 32K, `-ts 0.515,0.485`, `-ub 256`, `--spec-draft-n-ubatch 64`** | 390.5 | **51.5** | +17% vs plain 32K | - | - |

Verdict: the conversion is landed (exact for Q5_0, IQ4_XS scale rounded to f16 at about 2^-12; same accuracy as the stock kernels against the CPU reference). UD-IQ4_XS does not fit on the GPUs: 2-3 CPU expert layers cost about 20-25% TG, mostly sync and transfer per CPU layer. MTP with the draft experts on the CPU is validated wrong (slower than plain). The best working setups are plain Q3_K_XL at 100K (42.1 TG) and MTP2 at 32K (51.5 TG, no memory margin; MTP3 and `-ub 512` ran out of memory). The PPL text is too easy to separate the quants; use the unsloth KLD table for quality.

Memory per GPU at 100K, F16 KV, `-ub 256` (allocation log and sysfs peaks): weights 29.9 GiB (0.98 GiB of it mirrored: `ffn_gate_inp`, `hc_*`), main KV 1176 MiB (split by heads), QSA indexer key cache 588 MiB (one key head, so mirrored), compute 364 MiB, recurrent state 56 MiB, AllReduce staging 64 MiB; free 0.02-0.03 GiB. `llama_kv_cache` and compute buffer log lines report per-GPU sizes. A full Q4_0 draft needs about 2.3-2.5 GiB on one GPU; `-sm row` is gone upstream (`af546bb40`). The unsloth "shared" drafts lack `token_embd` and do not load in this tree. Not covered: full-length 32K prompts with MTP2, multi-slot server, acceptance rates. Raw: [flashnext-big-quants](old-ignore/2026-11-06/flashnext-big-quants/).

### 2026-11-06 S2: cheaper IQ sign handling on gfx906

Replace the emulated byte ops `__vcmpne4` + `__vsub4` with `sign_mask4` and `iq_apply_signs` (`(g ^ m) + (m & 0x01010101)`) in the IQ2_S mat-vec and the IQ2_S and IQ3_XXS MMQ tile loads. Kind: A/B experiment.

- Source: branch `s2-iq-signs` on `02980eb3d`; baseline `bc0ce9691` (build `build-catchup-20261105-213622`).
- Model: Flash Next UD-IQ3_XXS rev `c8b5954a88c2` (the pre-2026-11-04 upload).
- Settings: standard Flash Next.
- Workload: llama-bench PP2048/TG128, r5 alternating; one 96-token greedy `llama-completion` per build.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | plain | 541.45 +- 1.84 | 41.17 +- 0.11 | - | - | sha `d9c2f5346efc` |
| candidate | plain | 557.97 +- 1.09 | 43.27 +- 0.15 | +3.05% / +5.11% | - | same |

Verdict: landed. The ranges do not overlap, and output is byte-identical. Correctness: an exhaustive host check of all IQ2_S grid words x 256 sign bytes and IQ3_XXS cases found 0 mismatches (grid bytes are 0x01..0x3e, so the +1 never carries), and `test-backend-ops` passed all 115 IQ2_S/IQ3_XXS `MUL_MAT`/`MUL_MAT_ID` cases on ROCm0 (n = 1..64). Not covered: the new Flash Next upload, MTP, grouped MoE. The merge build's Flash Next numbers here (541 / 41.2) are above the 2026-10-29 rows (527.6 / 39.1), but that is not a matched comparison. Raw: [s2-iq-signs](old-ignore/2026-11-06/s2-iq-signs/).

### 2026-11-05 27B config sweep and 32K depth

Find the best 27B speculative mode, test a larger ubatch, and check speed at a 32K prompt. Kind: A/B experiment.

- Source: `bc0ce9691` (merge of official `4b77dee05`), build `build-catchup-20261105-213622`.
- Model: 27B Q8_0 rev `4ca720788d1e`; DFlash draft `z-lab/Qwen3.8-27B-DFlash2` Q8_0 rev `2d9571f8ce46`.
- Settings: standard 27B. 32K runs use ctx 40960. The ubatch test is llama-bench PP4096 with `-b 4096 -ub 2048,4096`.
- Workload: speculative runs are standard, one run each. The 32K prompt is the first 116,000 bytes of `src/llama-context.cpp` at `bc0ce9691` plus a request to explain it (about 32K tokens at 3.66 bytes/token). ubatch r3.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| candidate | plain, 2K | 587.4 | 36.8 | - | - | - |
| candidate | MTP2, 2K (two runs) | 584.5 / 582.1 | 64.1 / 63.9 | - | - | `dd228997b103` both |
| candidate | MTP3, 2K | 579.0 | 61.5 | -3.8% vs MTP2 | - | `39f8c45d0e87` |
| candidate | DFlash5, 2K (two runs) | 574.4 / 570.6 | 67.4 / 67.7 | +5.9% vs MTP2 | - | `9f2493efdd9b` both |
| candidate | DFlash7, 2K | 570.7 | 62.5 | -7.7% vs DFlash5 | - | same as DFlash5 |
| candidate | plain, 32K | 541.2 | 29.4 | -20% vs 2K | - | `b94840dcade5` |
| candidate | MTP2, 32K | 512.3 | 42.7 | -33% vs 2K | - | `c8ae52851843` |
| candidate | DFlash5, 32K | 512.0 | 39.9 | -41% vs 2K | - | same as plain |
| candidate | PP4096, ub 2048 / 4096 | 684.1 / 606.3 | - | -11.4% | - | - |

Verdict: DFlash5 is the fastest 27B mode at a 2K prompt and MTP2 at 32K; the crossover was not measured. Deeper drafts (MTP3, DFlash7) are slower. ubatch 4096 is validated wrong for 27B PP. Plain TG loses 20% at 32K, which motivates D1. Repeats at 2K varied by 0.2-0.3 t/s TG; noise at 32K was not measured. DFlash output matches plain greedy at 32K. Not covered: Flash Next, acceptance rates (`--simple-io` prints none), the crossover depth. Raw, including generated text: [config-sweep](old-ignore/2026-11-05/config-sweep/), [depth32k](old-ignore/2026-11-05/depth32k/).

### 2026-11-05 catch-up to official 4b77dee05

Check that the merge keeps 27B speed and output. Kind: regression screen.

- Source: `bc0ce9691` (merge of official `4b77dee05`; build made before the doc-only commits); baseline `build-ilp`, `build_commit e04bc2e38` (old master, stage A of the rebuild), run with `LD_LIBRARY_PATH=build-ilp/bin`.
- Model: 27B Q8_0 rev `4ca720788d1e`; DFlash draft `z-lab/Qwen3.8-27B-DFlash2` Q8_0 rev `2d9571f8ce46`.
- Settings: standard 27B. MTP2 adds `--spec-draft-backend-sampling`.
- Workload: standard, one run each, baseline then candidate per mode.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | plain, llama-bench | 697.55 | 37.75 | - | - | - |
| candidate | plain, llama-bench | 696.88 | 37.90 | -0.1% / +0.4% | - | - |
| baseline | MTP2 | 598.0 | 64.0 | - | - | sha `dd228997b103` |
| candidate | MTP2 | 584.5 | 64.1 | -2.3% / +0.2% | - | same |
| baseline | DFlash5 | 584.0 | 66.8 | - | - | sha `9f2493efdd9b` |
| candidate | DFlash5 | 574.4 | 67.4 | -1.6% / +0.9% | - | same |

Verdict: no regression; speculative output is byte-identical between builds. The CLI prompt rate varies more than llama-bench PP (see below), so the -2% is not a signal. TP smoke checks passed at n=1 and n=2048; a 32-token `llama-completion` generation was coherent. Not covered: Flash Next (MoE PP after the MMQ tile layout change #31047), grouped MoE, multi-slot server. Raw, including generated text: [catchup-4b77dee05](old-ignore/2026-11-05/catchup-4b77dee05/).

### 2026-10-29 A1 config sweep (TG)

TG128 on the master build, each candidate next to a 1/1 baseline. ROCm0 is taken to be the MI60 (PCI 03:00, the lower bus; HIP orders devices by bus). Most runs are one repeat; the 27B 0.52/0.48 retest is two r2 runs. The sweep script passed `-ts` and `-t` twice, so llama-bench ran both values; the numbers are the candidate rows from the JSON. [Scripts and JSON](old-ignore/2026-10-29/a1-sweep/).

| Setting | 27B Q8_0 t/s (1/1 in same run) | Flash Next t/s (1/1 in same run) |
| --- | --- | --- |
| `-ts 0.52/0.48` | 36.20 (37.42); retest 36.06, 36.16 (37.13, 37.01) | 37.80 (37.46) |
| `-ts 0.48/0.52` | 35.52 (37.39) | 36.64 (40.08) |
| `-ts 0.56/0.44` | 34.92 (37.40) | not run |
| `-t 16` | 36.61 (36.93) | 39.44 (37.64), one sample |
| `-t 8` | 36.86 (37.30) | default |
| `GGML_CUDA_AR_SLABS=0` | 37.39 (baselines 37.16-37.45) | 36.79 (37.52-37.58) |
| `GGML_SCHED_SHARE_VIEW_INPUTS=0` | 37.11 (37.16-37.37) | 36.12 (37.20-37.27) |
| `GGML_META_PARALLEL_DISPATCH=0` | 35.83 (37.37) | 35.82 (37.27-37.99) |
| `LLAMA_SHEXP_SPLIT=0` | n/a | 36.94 (37.52) |

### 2026-10-29 N1 DFlash draft replay check

27B Q8_0, DFlash5, 512 tokens, temporary build counter and HIP-graph logs. The draft rebuilds its graph every call (273 builds, 0 reused) at 0.36 ms each: 98 ms of an 8.5 s run. The draft now runs as a single-device context on ROCm0, not on the meta backend, so meta `tg_lookup` and uids do not apply. The ggml-cuda HIP-graph path replays it anyway: the generate graph (691 nodes) replayed 67 times with 2 re-captures; the inject graph (58 nodes) also replays. The 129 captures and 17400 `hipLaunchKernel` in the trace come from the target's TP lanes. Upper bound of the fix: about 0.7 ms of an 80 ms step, below the noise (DFlash5 57.3 and 60.1 t/s on equal builds). [Notes and instrumentation patch](old-ignore/2026-10-29/n1-dflash/).

### 2026-10-29 per-token host gap (N2)

`rocprofv3` kernel, HIP runtime and memcpy trace of TG32 on master, last 20 tokens. Profiled TG was 25.3 (27B) and 21.4 (Flash Next) against 36.6 and 39.1 unprofiled, so host times are upper bounds. [Record](old-ignore/2026-10-29/tg-host-gap/diagnosis.md), [attribution follow-up](old-ignore/2026-10-29/tg-host-gap/option1.md).

- 27B: both GPUs idle 1.04 ms/token, mostly one 0.45-0.6 ms gap after the end-of-token sync (logits readback, then 4-5 rounds of event wait, small H2D copies and `hipStreamSynchronize`).
- Flash Next: both GPUs idle 4.48 ms/token. About 70 small `hipMemcpyAsync` and about 10 stream syncs before each launch, about 1.7 ms host time.
- llama's per-token `set_input` writes only to pinned host buffers. All pre-launch GPU copies come from the scheduler's split-input path; most already use the fork's async staging, and the rest are synchronous only because of the 4096 B F32 staging cap. Raising the cap (N2b) and overlapping lane launches (N2a) changed nothing.

### 2026-10-29 scale-fold attempt

Folding the `1/hc` SCALE after `hc_attn_down`/`hc_ffn_down` (Q8_0 [10240 x 320], repacked) into the repacked Q8_0 mat-vec store built and matched master output, but a TG32 kernel trace showed identical kernel counts on both builds (116873 on Agent 1, 10145 `scale_f32`), so the fusion never fired. The branch was dropped without finding why. Each `scale_f32` takes about 4.2 us with no dispatch gap before it.

### 2026-10-29 ideas from external reviews

- [Strata](https://github.com/Niko1221/Strata) (MIT, `870aedd9e1eb`): code-only review; its gfx906 results are community MI50 reports at IQ1_M-IQ3_XXS and are not comparable. Gave S1-S5. Lower value: fused gate/up + SwiGLU + q8_1 epilogue (off on gfx906 in Strata), attention query LDS swizzle (+1.2% PP at 64K only), HIP graph node priorities (NVIDIA only). Accuracy-costing features excluded. [Review](old-ignore/2026-10-29/strata-review/README.md).
- [llama-cpp-optimization-lab](https://github.com/Abzolute1/llama-cpp-optimization-lab) (`5408613e`, no license, NVIDIA only, experts in host RAM, own numbers not reproduced): mostly CPU/PCIe expert offload, which does not apply. Gave L1 and the rejected ngram-mod fallback. Its rejection of CPU micro-kernel wins that did not show end to end is a caution for S2 and S4. [Review](old-ignore/2026-10-29/optlab-review/README.md).

### 2026-10-29 rebuild on official 36a0e2ffc

Matched llama-bench, `-sm tensor`, 50/50, FA on. 27B: batch/ubatch 2048, PP2048/TG128, r2. Flash Next: Q8_0 KV, batch/ubatch 1024/512, r2 plus a separate r5. A is the old master on 418f905a4; B-D are rebuild stages. All builds use `-mllvm -amdgpu-sched-strategy=max-ilp` and CXX `-O3` (without them 27B is about 2-3% slower). [Rebuild record](old-ignore/2026-10-29/rebuild/README.md).

| Stage | 27B PP2048 | 27B TG128 | Flash Next PP2048 | Flash Next TG128 | Flash Next r5 PP / TG |
| --- | ---: | ---: | ---: | ---: | --- |
| U: official 36a0e2ffc | 448.60 | 32.77 | 302.99 | 39.66 | 304.33 +- 3.78 / 39.76 |
| A: old master e04bc2e38 | 684.56 | 36.82 | 527.79 | 38.99 | - |
| B: cleanup and fixes | 682.92 / 684.19 | 36.96 / 36.61 | 525.73 / 528.13 | 38.34 / 39.56 | 509.92 +- 17.95 / 39.23 |
| C: no pipeline stages | 685.60 | 36.58 | 520.22 | 36.93 +- 2.78 | 516.43 +- 7.96 / 38.79 |
| D: DSV4/MiniMax/MiMo back to upstream | 685.34 | 36.67 | 527.63 | 39.09 | 520.11 +- 7.34 / 39.26 |

Greedy output (96 tokens) was byte-identical across B, C and D. 27B output matches official byte for byte; Flash Next matches for about 30 tokens and then diverges, consistent with different kernel rounding (repack, fusions), not checked further.

#### Speculative decoding on D (one run each)

`llama-cli`, greedy, `flake.nix` prompt (about 2K tokens), 512 output tokens, context 16384. Flash Next uses the `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` sidecar via `-md`. MTP runs use `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-backend-sampling`. Grouped MoE was not enabled. Script: [mtp_check.sh](scripts/mtp_check.sh).

| Model | Mode | Prompt t/s | Generation t/s | vs plain TG |
| --- | --- | ---: | ---: | ---: |
| 27B Q8_0 | plain | 462.0 | 36.2 | - |
| 27B Q8_0 | MTP2 | 514.0 | 60.7 | +68% |
| 27B Q8_0 | MTP3 | 557.8 | 58.0 | +60% |
| Flash Next IQ3_XXS | plain | 348.0 | 37.2 | - |
| Flash Next IQ3_XXS | MTP2 | 339.1 | 49.6 | +33% |
| Flash Next IQ3_XXS | MTP3 | 349.6 | 44.5 | +20% |

Draft acceptance was not recorded (`--simple-io` prints none). The CLI prompt figure varies more than llama-bench PP.

### 2026-10-29 Flash Next TG profile

64-token capture on the IQ2_S candidate library (`fe34c2...`); matched unprofiled 36.71 TG, profiled 28.93 (-21%). 63 steady graph repeats; in the final 2.178 s window both GPUs had dispatches active for 1.905 s. Per-GPU sums and unions overlap, so they cannot be added into a critical path. The raw trace and `summarize.py` are outside the repository.

| Kernel group | ROCm0 sum | ROCm1 sum | Two-GPU union | Union per repeat |
| --- | ---: | ---: | ---: | ---: |
| Non-expert Q6_K/type-14 and q8_0-repacked mat-vecs | 377 ms | 401 ms | 628 ms | 9.97 ms |
| TP AllReduce | 327 ms | 65 ms | 336 ms | 5.33 ms |
| IQ2_S, IQ4_NL and IQ3_S expert mat-vecs | 199 ms | 215 ms | 295 ms | 4.69 ms |
| Float mat-vecs | 195 ms | 194 ms | 278 ms | 4.41 ms |
| Routing and MoE reduction | 100 ms | 109 ms | 152 ms | 2.41 ms |
| Position/attention, hyper-connections, GDN/SSM | 142 ms | 153 ms | 274 ms | 4.35 ms |

ROCm1 per-token detail: `quantize_q8_1<32>` 484 calls 2.33 ms; `scale_f32` 303 calls 1.77 ms; float mat-vecs 288 calls 3.08 ms (main specialization 264 calls, 2.90 ms); Q6_K and Q8_0 repacked mat-vecs 496 calls 6.36 ms.

### 2026-10-29 earlier checks on the official head

- Official-head port: an early unrepaired 418f port measured Flash Next MTP2 PP -28%; the cause was a changed gfx906 MMQ launch traversal. After restoring row-first traversal: 347.97 / 36.77 baseline vs 351.30 / 39.51, identical output and acceptance. [Traversal check](old-ignore/2026-10-29/head-merge-validation/terra-port-fix/README.md), [validation](old-ignore/2026-10-29/head-merge-validation/README.md).
- Quant and placement: IQ3_XXS with GPU experts 377.50 / 38.94; IQ4_XS with 8 CPU MoE layers 197.59 / 24.42 (op-offload on) and 204.92 / 25.15 (off), one run each, quant and placement changed together. Layer split 353.76 / 30.99. [Record](old-ignore/2026-10-29/quant-placement-compare/README.md).
- IQ2_S single-token path: 396.35 / 37.20 baseline vs 398.80 / 38.08 candidate, one pair, output matched, 4,143-token prompt, 256 generated. [Record](old-ignore/2026-10-29/iq2s-single-token-experiment/README.md).
- Grouped-MoE integration: build, replay and 18 canonical reference cases passed; the repacked harness did not finish. [Record](old-ignore/2026-10-29/grouped-moe-integration/README.md).

### Before 2026-10-29

| Date and workload | Setting | PP / TG | Source identity |
| --- | --- | --- | --- |
| Oct 12, synthetic PP2048/TG128 | Qwen3.8-27B Q8_0, plain | about 682 / 36.8 | local changes, commit unknown |
| Oct 25, `flake.nix`, 4000 outputs | 27B MTP2 | 549.7 / 56.8 | build `11776`, `766f7c5e0` |
| Same workload | 27B MTP3 | 545.2 / 60.0 | same |
| Same workload | 27B DFlash7 | 543.7 / 59.9 | same |
| Oct 25, short CLI, 256 outputs | 27B plain | 558.9 / 35.2 | `b11929-fdc8e5b65` |
| Oct 25, Flash Next long CLI | IQ3_XXS plain | 396.9 / 37.3 | same |
| Same workload | MTP2, Q8_0 sidecar | 368.5 / 42.9 | same |
| Oct 25, Flash Next MTP2, short CLI | grouped MoE off / on | 330.3 / 47.65 -> 372.9 / 50.45 | local experiment on `fdc8e5b65` |

One run per setting. The Oct 25 27B runs used about 6K populated tokens despite a 100K context setting, so they do not test a full context. Speculation helped single-request workloads; plain decoding was fastest in the [three-request mixed workload](old-ignore/2026-10-12/2026-10-12-three-slots.md). Target-logprob checks support greedy selection within a run, not general quality equivalence. DFlash vision, other quants, quantized KV and full contexts are untested. Records: [27B CLI](old-ignore/2026-10-25/2026-10-25-qwen3.8-27b-flake-cli.md), [Flash Next MTP](old-ignore/2026-10-25/2026-10-25-flash-next-mtp-short-check.md), [grouped MoE](old-ignore/2026-10-25/flashnext-fast/README.md), [Oct 12 profiles](old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md), [Oct 12 work log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md).
