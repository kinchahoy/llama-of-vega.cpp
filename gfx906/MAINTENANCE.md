# Fork Maintenance & Architecture Guide

Everything needed to build, preserve, debug, extend, and merge `llama-of-vega.cpp` against upstream `llama.cpp`. Current upstream base: **`e6d624b9a`** (merged 2026-11-21; core delta vs `e6d624b9a` excluding `gfx906/` is **139 files, +19,046 / -1,205 LOC**).

Read [GOALS.md](GOALS.md) for priorities and [HARDWARE.md](HARDWARE.md) for measured gfx906 limits. Always follow `AGENTS.md` and `skills/code-review/SKILL.md`.

## Build flags & build-retention policy

Use the local TheRock ROCm environment with exact comparison flags (`max-ilp` and `-O3` are worth 2-3% on 27B):

```bash
# Optional: source your ROCm / TheRock environment if not on PATH
# source "${THEROCK_ENV:-/path/to/therock.sh}"
cmake -S . -B "$CAND" \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_GFX906_TESTS=ON \
  -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build "$CAND" --target llama-cli llama-completion llama-server llama-bench llama-batched-bench gfx906-test-tp gfx906-test-flashnext-moe -j 8
```

- **Build-retention policy**: Retain **1-2 well-tested performant timestamped builds** in addition to the active working build (`build-current` and `build-previous` symlinks in the repo root, e.g. `build-current -> build-<tag>-<date>`).
- **Never rename or move a build directory**: CMake embeds the absolute build path into binary `RUNPATH`, so renaming breaks shared-library lookup. Point symlinks with `ln -sfn` instead. Never rebuild a retained baseline directory in place.
- Keep `GGML_HIP_RCCL=OFF` on this host (MI50 sits on chipset PCIe lanes; no P2P). Keep `GGML_HIP_GRAPHS=ON`.

## Fork feature & file delta inventory

| Feature | Purpose / Notes | Main source | Kill switch / Default |
| --- | --- | --- | --- |
| Repacked weights & gfx906 MMQ/MMVQ | 2-plane layout + dp4a MMV/GEMM for 10 quants (incl. Q5_0, IQ4_XS, NVFP4 3D experts); row-first gfx906 MMQ traversal, J cap <=64 | `ggml/src/ggml-cuda/q8_repack/`, `mmq.cuh`, `mmvq.cu` | `--no-repack`; `GGML_CUDA_REPACK_NVFP4=0`; `GGML_CUDA_REPACK_CONVERT=0` |
| IQ2_S narrow 1-tok MMVQ & S2 IQ signs | 2-row gfx906 path for even rows & 1 col; word-op `iq_apply_signs` (bit-exact, +5.1% FN TG) | `ggml/src/ggml-cuda/mmvq.cu`, `mmq.cuh` | default on |
| Grouped MoE weight reuse | Decode each IQ2_S / IQ4_NL expert weight once across up to 8 verify tokens | `ggml/src/ggml-cuda/flashnext-moe.cuh`, `q8_repack/flashnext-moe.cuh` | `GGML_CUDA_FLASHNEXT_MOE=0` (default on) |
| Host-staged AllReduce & TP runtime | Graph-safe device counters, duplex H2D/D2H streams, prefill slabs (`AR_SLABS=4`), concurrent TP lanes, whole-token graph replay | `ggml/src/ggml-cuda/allreduce.{cu,cuh}`, `ggml-backend-meta.cpp`, `ggml-backend.cpp` | `GGML_CUDA_AR_SLABS=0`, `GGML_META_PARALLEL_DISPATCH=0` |
| Two-launch fused norm & Inkling ops | Two-launch `add_rms_norm_mul` (27B) & `res_mix` (Inkling) at small rows; `GGML_OP_{CLIP_ROWS,RES_MIX,LOOP_GATE,RMS_NORM_CAT,SCONV_RES}` + ring recurrent state (`upload_s_copy`, `fill_s_write`) | `ggml/src/ggml-cuda/norm.cu`, `clip-rows.cu`, `res-mix.cu`, `loop-ctrl.cu`, `sconv-res.cu`, `src/models/inkling2.cpp` | `LLAMA_INKLING_{CLIP,MIX,CTRL,SCONV}_FUSED=0` |
| Fused hyper-connection mix (F2) | `GGML_OP_DSV4_HC_MIX` (3 launches vs 7 for 1-8 tokens) | `ggml/src/ggml-cuda/` (hc-mix), `src/models/qwen4exp.cpp` | `GGML_CUDA_HC_MIX_FUSED=0` |
| Small-M F32 GEMM (P4) | Split-K F32 GEMM on GCN for M<128, N>16 before rocBLAS fallback (dispatch guards still open; see [PERFORMANCE.md](PERFORMANCE.md)) | `ggml/src/ggml-cuda/ggml-cuda.cu` | `GGML_CUDA_SGEMM_F32=0` |
| Inkling 2 banded attention | Port of unsloth #386 (`GGML_OP_FLASH_ATTN_EXT_BANDED`) + fast gfx906 tile path (`use_banded` in `fattn-tile.cuh`, 162 LOC, default on for prefill). **Cleanup note**: `fattn-banded.cu` is **1,306 LOC** of slow non-matrix-unit fallback that halves PP on gfx906; prune once tile path handles SWA ring-wrap | `ggml/src/ggml-cuda/fattn-tile.cuh`, `fattn-common.cuh`, `fattn-banded.cu`, `src/llama-kv-cache.cpp`, `src/models/inkling2.cpp` | `LLAMA_INKLING_BANDED_ATTN=0` |
| Inkling 2 DSpark drafting | Port of ggml-org#32218 (`3a8d768df`, ~240+ LOC), opt-in; currently slower than plain until 4-tok verify & BF16 Markov head kernel work | `src/models/inkling2.cpp`, `common/speculative.cpp` | opt-in (`--spec-type`) |
| GCN sparse tile FA (QSA gather) | Port of ggml-org#30370: `flash_attn_mask_to_sparse_indices` lists gathered by the tile kernel through the `kv_rows` row gather; FN QSA layers at >= ~8K ctx (+20% PP @ 32K) | `ggml/src/ggml-cuda/fattn-tile.cuh` (`use_sparse`), `fattn.cu` (`ggml_cuda_flash_attn_ext_tile_shall_use_sparse`) | `GGML_CUDA_SPARSE_FA=0` |
| AMD `kv_rows` FA & decode occupancy | `use_kv_rows` template in `fattn-tile.cuh` for F16 K/V (+25% 27B 4x16K PP); `__launch_bounds__(128, 4)` on HIP `flash_attn_ext_vec` (`efb85a1a5`) | `ggml/src/ggml-cuda/fattn.cu`, `fattn-tile.cuh`, `fattn-vec.cuh` | `LLAMA_KV_ROWS_DISABLE=1` |
| MoE id counting sort (unsloth #264) | Device counting sort for `n_tokens >= 512`, skips out-of-slice TP expert ids, emits `expert_bounds` | `ggml/src/ggml-cuda/mmid.cu` | `GGML_CUDA_MOE_COUNTING_SORT=0` |
| Speculative replay, rollback & TP splits | MTP prompt replay + KV-only prefill, DFlash persistent taps + single-GPU drafter, plane-indexed multi-slot recurrent rollback, tied output head split, shared-expert split, host n-gram/PLE placement, deterministic wide-row `top-k.cu` | `common/speculative.cpp`, `src/llama-context.cpp`, `llama-graph.cpp`, `llama-memory-recurrent.cpp`, `ggml/src/ggml-cuda/top-k.cu` | default on |

Lineage: built on [mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp), [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906), and Stastez's host-staged ROCm AllReduce (PR #27825). Removed permanently in the 2026-10-29 rebuild and 2026-11-21 cleanup (do **not** restore during merges): multi-stage pipeline TP (`-tps`), fork-only DeepSeek V4/V4.1 / MiniMax-M2 / MiMo-2 code, peer-write AllReduce (`tp-allreduce.cu` / `GGML_CUDA_PEER_AR`), and ~30 sweep-only env knobs (`GGML_RP_*`, `GGML_CUDA_TP_OVERLAP`, `LLAMA_DFLASH_*`, etc.).

## Repacked-weight layout & quant coverage

Supported formats (10 total, checked by `ggml_cuda_repack_tensor_supported` for whole non-view 2D/3D weights on gfx906):
- `ne0 % 32 == 0`: **`Q8_0`**, **`MXFP4`**, **`IQ4_NL`**, **`Q5_1`**, **`Q5_0`**, plus **`NVFP4`** (3D MoE experts only; dense NVFP4 stays generic).
- `ne0 % 256 == 0`: **`Q4_K`**, **`Q5_K`**, **`Q6_K`**, **`IQ4_XS`** (converted at upload to `IQ4_NL` store layout via `ggml_cuda_repack_store_type`).
- Never repacked: `IQ1_*`, `IQ2_*`, `IQ3_*`, `Q4_0`, `Q4_1`.

| Feature / Path | 27B Q8_0 | Flash Next UD-IQ3_XXS / Q3_K_XL / IQ4_XS | Inkling 2 Mini (NVFP4 / IQ4_NL / Q8_0) | Gemma 4 31B (Q8_0 / K-quants) |
| --- | --- | --- | --- | --- |
| Repacked weights & GEMM/MMV | All weights | Non-expert Q6_K/Q8_0 + IQ4_NL/Q5_0/IQ4_XS experts | All dense Q8_0/IQ4_NL + 3D NVFP4 experts (+1.3 GiB VRAM for F16 scales) | All Q8_0 / K-quants (avoid Q4_0/Q4_1: no repack/fusion) |
| Fused 1-tok GLU / bias MMV | Dense Q8_0 GLU | MoE GLU for repacked down/gate/up (1-tok only) | Dense Q8_0 + MoE GLU (1-tok only) | Dense Q8_0 only (not K-quants) |
| Generic / narrow IQ paths | n/a | IQ2_S 2-row MMVQ, S2 signs, grouped MoE (2-8 tok), generic MMVQ for IQ2_S/IQ3_XXS/Q2_0 | Dense BF16 loop controllers (`mul_mat_vec_f` / rocBLAS) | n/a |
| TP & special splits | `-sm tensor`, AR slabs, `kv_rows` | `-sm tensor`, shared-expert split, host n-gram table | Single-GPU ROCm0 (no `-sm tensor` for `inkling2_loop`) | `-sm tensor`, tied 262K vocab output split |

### Core `q8_repack/` invariants

Everything outside `ggml/src/ggml-cuda/q8_repack/` includes only `repack.cuh` (from `ggml-cuda.cu`). Preserve upload, allocation, layout, dispatch, and consumers together -- canonical kernels cannot read repacked weights:

1. **Two-plane layout** (`repack-common.cuh`):
   - `qs` plane: `ne1 * qs_stride` bytes where `qs_stride` adds `+16` bytes whenever `row_bytes % 128 == 0` (dodges HBM channel bank conflicts while preserving `uint4` alignment).
   - `d` scale plane: separate plane so multi-row MMV lanes read scales contiguously. NVFP4 reuses the IQ4_NL nibble plane with `split_scales` (two F16 scales per 32 values).
   - Multi-expert (`ne2 > 1`) concatenates per-expert layouts at stride `repack_gcn_nbytes(type, ne0, ne1)`. Weights are write-once (`get_tensor == nullptr`).
2. **Four compute paths** (`repack-kernels.cuh`, `mul-mat.cu`, `mul-mat-id.cu`):
   - Single-token MMV (`ne11 == 1` / `n_tokens == 1`): `LANES=64` dense, `16` MoE plain, `32` MoE fused GLU (`<8,4,true,32,true>` avoids the 64-VGPR cap of 1024-thread blocks).
   - Narrow dense MMV (`mul_mat_vec_repacked_nc`): `2 <= ne11 <= 2*MMQ_RP_Q8_MMV_MAX_TOKENS` (16, dense only); 9..16 tokens run as two balanced chunks of <= 8 (`GGML_CUDA_RP_NC_SPLIT=0` restores the cap of 8). MoE narrow MMV still stops at 8 (`MU1(b)` in [PERFORMANCE.md](PERFORMANCE.md)).
   - 32-wide GEMM (`w32`): `8 < ne11 < 128` dense or `n_assign < 2*BN*n_expert` MoE.
   - 64-wide GEMM (`w64`): `ne11 >= 128` dense or high-assignment MoE. Multi-token GEMM reads `quantize_mmq_q8_1_cuda` reinterpreted as `block_q8_1_mmq_h` (`float d4[4]`, pinned by 3 `static_assert`s).
3. **Fused MMV guards**: Fires ONLY at 1 token when BOTH gate and up are repacked (`should_fire(src0) && should_fire(gate->src[0])`). MoE fused GLU covers no-bias `{MM_ID, MM_ID, GLU}` only; scale-fusion helpers in `ggml-cuda.cu` skip repacked weights.
4. **Persistent view cache** (`repack_view_get_cached`): Views of repacked tensors are re-packed once into `cudaMalloc`'d buffers keyed by `(data, ne0, ne1, ne2)` and never freed (pool reuse breaks captured HIP graphs).
5. **CMake & meta registration**: `file(GLOB "q8_repack/*.cu")` lives in BOTH `ggml-cuda/CMakeLists.txt` and `ggml-hip/CMakeLists.txt`. `ggml-backend-meta.cpp` composes per-lane repack bufts, whitelists supported types, and bypasses chunk-splicing in `set_tensor_async`.

## Merge maintenance index

| Feature | Preserve across merges | Final checks when affected |
| --- | --- | --- |
| Repack & MMQ/MMVQ | Upload, layout, dispatch, consumers, meta type whitelist, row-first gfx906 MMQ & J cap | TP smoke, greedy gen, matched PP/TG; NVFP4 KLD on/off |
| Narrow IQ2_S & grouped MoE | IQ2_S even-row 1-col guard; grouped MoE default on for narrow IQ2_S/IQ4_NL with original routing & fallback | `gfx906-test-flashnext-moe`, Flash Next MTP output |
| AllReduce & TP lanes | First `#if !defined(GGML_USE_MUSA)` branch in `allreduce.cu` compiled for HIP with no env switch; graph-safe counters, duplex/slabs, no new per-ubatch host sync | `gfx906-test-tp --n 1` and `--n 2048`, matched PP/TG |
| F2 (`DSV4_HC_MIX`) & P4 (`SGEMM_F32`) | Op enum + `GGML_OP_COUNT` asserts, CPU ref, meta whitelist, hc repack exemption; P4 GCN-only small-M before rocBLAS | `test-backend-ops -o DSV4_HC_MIX -o MUL_MAT`, FN PP & `-b 8 -ub 8` KLD |
| Inkling fused ops & banded FA | All 5 ops + `GGML_OP_FLASH_ATTN_EXT_BANDED` + `fattn-tile.cuh` `use_banded` + ring state (`upload_s_copy`, `fill_s_write`) + `can_reuse` | `test-backend-ops`, Inkling ROCm0 + `--device none`, PP2048 ub 512/2048, banded KLD |
| AMD `kv_rows` FA & MoE counting sort | `use_kv_rows` template (plain kernels keep VGPRs), `fattn.cu` F16 TILE, meta `src[5]`; counting sort skips out-of-slice ids & writes `expert_bounds` | `test-backend-ops -o FLASH_ATTN_EXT -o MUL_MAT_ID`, [scripts/bench_multiuser.sh](scripts/bench_multiuser.sh) |
| Speculative replay & rollback | Draft repack, split KV, per-seq taps (ub 2048 can need 2049 tap rows), plane-indexed multi-slot delta-net rollback, sequential ubatch split on rollback | Targeted MTP/DFlash check & [scripts/spec_accept.sh](scripts/spec_accept.sh) |

## The 15 silent merge traps

Every one of these merged or compiled cleanly on past upstream catch-ups but silently broke correctness or speed. Check all 15 on every merge:

1. **Single MMQ MoE `J` selection**: MMQ MoE uses linear `tile_spec` from `mmid.cu` (upstream #31047). Kernel `J` MUST equal `J_best` from `ggml_cuda_mmq_get_J_best` (which also holds the gfx906 `J <= 64` cap). Never add a second `J` selection in `mul_mat_q_switch_J`.
2. **Repack/grouped-MoE `expert_bounds` overload**: Repack and grouped MoE read per-expert `expert_bounds`, not `tile_spec`. Keep the `ggml_cuda_launch_mm_ids_helper` overload in `mmid.cu` that writes `expert_bounds`.
3. **Recurrent conv rollback rings**: Upstream `build_conv_state_store` writes only `min(n_tokens, K)` rollback slots (bounded by `cell.rs_max`). Preserve the fork's `s_write` (delta-net) and `s_write_conv` (qwen4exp) ring branches and `valid_depth`.
4. **DFlash feature buffer names**: Upstream `layer_inp_dev` is the device feature buffer; the fork's per-layer persistent taps are `layer_inp_tap`. Keep the names separate.
5. **DFlash default tap pipeline & single-GPU draft**: Upstream `LLAMA_DFLASH_DEV=1` stays opt-in so the fork tap/accum pipeline remains default. A DFlash draft with its own head must NOT set `ctx_other` (keeps it single-GPU).
6. **Meta backend repack type whitelist**: Every new repacked type MUST be added to `ggml_backend_meta_device_supports_op` (`ggml-backend-meta.cpp`), otherwise under `-sm tensor` weights silently stay un-repacked with zero errors.
7. **HIP >= 7 `__ballot_sync` wave64 truncation**: HIP >= 7 defines `__ballot_sync(mask, p)` as `__ballot(p) & mask`, so `0xFFFFFFFF` silently drops lanes 32-63 on wave64 gfx906 (compiling out our `vendors/hip.h` fallback). Dual-32-lane kernels must use 64-bit `__ballot(p)` shifted by the half-wave (`TOPK_MOE` broke here).
8. **Per-layer K/V rotations**: K/V rotations are per-layer constants (`inp->self_k_rot[il]`), not graph inputs. Local MTP KV-only paths must index by layer and call `build_rope`.
9. **View-input root copy flag**: Upstream `hv_tensor_requires_copy` (#31419) skips unflagged split inputs. The fork `GGML_SCHED_SHARE_VIEW_INPUTS` path adds a view's root to `split->inputs` and MUST also flag the root, or the root copy is never uploaded.
10. **Deterministic wide-row `top-k.cu`**: Keep the fork `top-k.cu` (count/offsets/gather) over upstream #31084: upstream only stabilizes the tie set, whereas our backend sampler and DFlash rely on exact output order (HIP lacks CUB, so rows >1024 cols hit this).
11. **MTP deferred prefill vs `dev_rows`**: Upstream `LLAMA_MTP_DEV_ROWS=1` (#31927) stays opt-in because it disables the fork's deferred prefill (+18% TTFT).
12. **`mm_ids_helper` `it_compact` reset**: Since upstream `b94959a64` (#32505), `mm_ids_helper` resets `it_compact` after each store dump. The final `expert_bounds` entry in `mmid.cu` MUST be `nex_prev + n_dumped`, NOT `it_compact` (auto-merge silently drops the last expert's rows).
13. **`build_attn_iswa_impl` free function**: Upstream `8af68d486` (#32429) moved the iswa `build_attn` body into free function `build_attn_iswa_impl`. Fork replica-aware `build_cpy_k`/`build_get_k` calls auto-merge into it and MUST be qualified as `gctx.build_*`.
14. **Out-of-slice MoE expert ids under `-sm tensor`**: Each TP GPU sees expert ids outside its expert slice. Any device MoE id sort or helper MUST skip out-of-range ids, never trap (broke ROCm1 on the first #264 port).
15. **`supports_op` must accept empty tensors**: A ubatch without outputs gives the last layer 0 tokens. If `supports_op` rejects `ne == 0` (e.g. a `>= 1` shape guard), the scheduler moves the node to CPU and `-sm tensor` crashes in `ggml_backend_sched_copy_input` (null call). The backend skips empty nodes anyway (`ggml_cuda_is_view_or_noop`). Broke F2 `DSV4_HC_MIX` on 2026-11-22 (FN prefill >= 8K).

## Weekly upstream catch-up procedure

1. **Clean start**: Start from a clean, committed `main`. Keep `BASE` pointing at `build-current` (resolved real path; never touch it).
2. **Merge in git**:
   ```bash
   git config rerere.enabled true
   git config rerere.autoupdate false
   OLD_HEAD=$(git rev-parse HEAD)
   BASE="$(readlink -f "$PWD/build-current")"
   CAND="$PWD/build-catchup-$(date +%Y%m%d-%H%M%S)"
   OUT="$CAND/validation"
   mkdir -p "$OUT"
   printf '%s\n' "$OLD_HEAD" > "$OUT/baseline-source.txt"
   printf '%s\n' "$BASE" > "$OUT/baseline-build.txt"
   git fetch upstream
   UPSTREAM_HEAD=$(git rev-parse upstream/master)
   printf '%s\n' "$UPSTREAM_HEAD" > "$OUT/upstream-source.txt"
   git merge --no-commit --no-ff "$UPSTREAM_HEAD"
   ```
   Resolve conflicts, run `git rerere`, and review auto-merged callers against all 15 silent merge traps above.

### One final validation pass

1. **Static diff (no GPU)**:
   - Run `git diff --check`, `git diff --cached --check`, check `git ls-files -u` is empty and no conflict markers exist.
   - Diff graph builders and kernel geometry (`blocks`, `threads`, `launch_bounds`, `LDS`, fusion conditions) and predict any cost change from [HARDWARE.md](HARDWARE.md).
   - Confirm `ggml/src/ggml-cuda/allreduce.cu` compiles the host-staged path for HIP by default with no env switch (`AGENTS.md` check; excluding HIP cost -21% PP / -38% TG on 2026-09-29).
2. **Build once**: Configure and build `$CAND` using the exact build flags at top of this doc. Record `git diff --binary HEAD > "$OUT/candidate-changes.patch"`, `CMakeCache.txt`, `sha256sum`, and `ldd` outputs. Unset `GGML_CUDA_PEER_AR` and `GGML_CUDA_TP_OVERLAP`.
3. **One end-to-end comparison** (start below 60 C, `--offline`, identical cached files):
   ```bash
   for label in baseline candidate; do
     if [ "$label" = baseline ]; then BUILD="$BASE"; else BUILD="$CAND"; fi
     timeout --kill-after=5s 110s "$BUILD/bin/llama-bench" \
       -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
       -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
       -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -d 0 -r 1 -o json \
       > "$OUT/$label.json" 2> "$OUT/$label.log"
   done
   ```
4. **Affected features only**:
   - AllReduce / TP touched: `"$CAND/bin/gfx906-test-tp" --n 1 --check && "$CAND/bin/gfx906-test-tp" --n 2048 --check` (both must report `check=pass`).
   - Numerics / sampling touched: `llama-completion --temp 0 --top-k 1 -s 42 -n 32 -no-cnv --simple-io --no-display-prompt`.
   - Flash attention, KV cache, ubatch splitting, or server slots touched: [scripts/bench_multiuser.sh](scripts/bench_multiuser.sh) (`4x16K` unified KV + 4-way TG; `REPS=2` if within 3%). Reference 2026-11-15 baselines (`kv_rows` on vs off):
     - 27B Q8_0: **577.8 / 577.5 PP**, **80.86 / 80.93 4-way TG** (off: 462.3 PP, 81.1 TG). Drop toward 462 PP means `kv_rows` stopped engaging.
     - Flash Next UD-IQ3_XXS: **470.5 / 473.1 PP**, **84.17 / 83.90 4-way TG** (off: 472.2 PP, 84.2 TG).
   - MoE / speculative / Inkling touched: `test-backend-ops`, `gfx906-test-flashnext-moe`, [scripts/spec_accept.sh](scripts/spec_accept.sh), or targeted model KLD.
5. **Promotion**: Once validated and committed, update symlinks (`ln -sfn "$BASE" build-previous && ln -sfn "$CAND" build-current`) without renaming build directories.

## How to add a fused op

Minimal complete reference commit: `85bb8a0f7` (`GGML_OP_SCONV_RES`).

**Files to touch**:
1. `ggml/include/ggml.h`: add enum value before `GGML_OP_COUNT` and builder prototype.
2. `ggml/src/ggml.c`: add op name, symbol, builder implementation, and bump **both** `static_assert(GGML_OP_COUNT == ...)` checks.
3. `ggml/src/ggml-cpu/ggml-cpu.c`, `ops.h`, `ops.cpp`: add CPU reference implementation (required for unit tests and `--device none` validation).
4. `ggml/src/ggml-cuda/<op>.cuh` + `<op>.cu`: HIP/CUDA kernel and dispatch function.
5. `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatch switch case + `supports_op`.
6. `ggml/src/ggml-backend-meta.cpp`: add to meta backend whitelist (otherwise `-sm tensor` silently falls back to CPU or aborts).
7. `tests/test-backend-ops.cpp`: add reference test cases.
8. `src/models/<model>.cpp`: wire up behind a one-time-read env kill switch so the old graph remains the A/B and KLD reference.

**gfx906 kernel design rules**:
- **Cut along weight matmuls**: never put a weight matmul inside a one-block fused kernel (one-block loop controller dropped Inkling TG 40.5 -> 11.7 t/s).
- **Respect the one-block rule** ([HARDWARE.md](HARDWARE.md#the-one-block-rule)): use 1024 threads + warp reductions + single-pass sums for rows up to ~5,000 floats; use the two-launch form (~5.4 us flat) above ~5,000 floats.
- **Zero-copy contiguous outputs**: lay out multi-consumer outputs so each consumer takes a contiguous view without `ggml_cont` copies.
- **No dense bias/mask operands when small**: without matrix units, compute small relative biases or masks inline inside the attention tile kernel.
- **Verify graph replay**: a pointer or shape varying across tokens disables HIP graph replay; confirm zero "warmup reset" lines with `GGML_CUDA_GRAPH_DEBUG=1 -v`.
- **Validate thoroughly**: run `test-backend-ops`, test `--device none -ngl 0` (catches bad `get_rows` indices), and run `--kl-divergence` (not just greedy text) for any attention or non-bit-exact change.
