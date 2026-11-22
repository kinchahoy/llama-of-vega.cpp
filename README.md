# llama.cpp (gfx906 MI50 / MI60 Performance Fork)

**Status datestamp: 2026-11-22** | Upstream base: [`ggml-org/llama.cpp` `e6d624b9a` (`b12623`)](https://github.com/ggml-org/llama.cpp/commit/e6d624b9a)

A community fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for long-context (50-100K+ token), 1-4 parallel slot agentic inference on legacy AMD **gfx906** GPUs (**Radeon Instinct MI60 / MI50**, 32 GiB HBM2 each, no matrix units / MFMA).

It is not affiliated with or endorsed by the upstream llama.cpp project. The full upstream README is preserved in [README.llama.cpp.md](README.llama.cpp.md).

## Design goals & quality focus

- **Target workload**: Long-context agentic inference (**50-100K+ tokens**, where both prompt processing and token generation matter) across **1-4 parallel slots** on legacy 32 GiB AMD gfx906 hardware.
- **Approach**: Prioritize reusable, model-independent kernel and architecture wins (attention, GDN/SSM, MoE, repacked mat-vec/GEMM, multi-slot scaling) over fragile parameter tuning, with a secondary goal of keeping custom code maintainable for upstream catch-up merges and eventual upstreaming.
- **Quality & accuracy first**:
  - **No KV cache quantization**: All benchmarks and presets keep **F16 KV cache** by default.
  - **Full model fidelity**: No lossy expert dropping or approximation hacks; speculative decoding (MTP / DFlash / DSpark) preserves exact target-model output.
  - **Verification**: Kernel changes are checked with CPU-reference unit tests (`test-backend-ops`) and logit-level KL-divergence (`llama-perplexity --kl-divergence`), not just greedy text smoke tests.

## Current state & performance (2026-11-22)

Measured on **2x gfx906 (MI60 + MI50)**, TheRock ROCm, F16 KV cache (no KV quantization), `-fa on`, cool-start alternated ABBA runs (`llama-bench` / `llama-batched-bench`):

| Model | Hardware & Mode | PP2048 | TG128 / Serving | Notes |
| --- | --- | ---: | ---: | --- |
| **Qwen3.8-27B Q8_0** | 2 GPUs, `-sm tensor`, `ub 512` (`ub 2048`) | **614.0** (~697) | **37.96** plain | ~+53% PP / +14% TG vs upstream; MTP2 ~62-65 t/s; DFlash5 ~69-71 t/s (short prompt); 4x16K unified KV (`kv_rows`): **577.8 PP (+25%) / 80.9 4-way TG** |
| **Qwen3.8 Flash Next UD-IQ3_XXS** | 2 GPUs, `-sm tensor`, `ub 512` | **651.0** | **50.46** plain | >+100% PP / ~+29% TG vs upstream; MTP3 @ 100K ctx (Q8_0 sidecar on ROCm1): **62.7 t/s** |
| **Inkling 2 Mini NVFP4 + Q8_0** | 1 GPU (MI60), `ub 512` (`ub 2048`) | **835.7** (1081) | **46.97** plain | +70% PP / +54% TG over initial merge; 4K/16K ctx TG: **43.12 / 40.53 t/s** (`flash_attn_ext_vec` `(128,4)` fix) |
| **Inkling 2 Mini IQ4_NL** | 1 GPU (MI60), `ub 512` | **878.6** | **48.98** plain | +56% PP / +35% TG over initial merge |

### Key additions over upstream `e6d624b9a`

- **gfx906 repacked weight path (`ggml/src/ggml-cuda/q8_repack/`)**: 10 weight formats (`Q8_0`, `MXFP4`, `IQ4_NL`, `IQ4_XS`, `NVFP4`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`) tuned for GCN `v_dot4` without MFMA, plus narrow multi-column decode/verify mat-vec and fused gate+up+SwiGLU.
- **Host-staged duplex & slab-overlapped AllReduce (`ggml/src/ggml-cuda/allreduce.cu`)**: Default HIP tensor-parallel path for PCIe boards without P2P/RCCL (`-DGGML_HIP_RCCL=OFF`), plus whole-token HIP graph replay and parallel lane dispatch.
- **Unified-KV `kv_rows` tile FlashAttention on gfx906** (+25% 4x16K PP) and HIP `(128, 4)` occupancy launch bounds on `flash_attn_ext_vec` (+13-21% decode TG at 4K-16K).
- **Fused GCN kernels & MoE speedups**: `DSV4_HC_MIX` (F2), small-M tiled split-K F32 GEMM (P4), word-op IQ sign handling (S2), one-wave `topk_moe_wave_cuda<512, 10>` (B1), grouped MoE verify (`flashnext-moe.cuh`), two-launch wide reductions (`add_rms_norm_mul`, `res_mix`), Inkling 2 fused ops (`CLIP_ROWS`, `RES_MIX`, `LOOP_GATE`, `RMS_NORM_CAT`, `SCONV_RES`) and tile-kernel banded prefill attention.
- **Speculative decoding & rollback**: Multi-slot recurrent state rollback (`snapshot_ring`), single-GPU DFlash/MTP sidecar support, and opt-in Inkling 2 DSpark port (upstream PR `#32218`; currently slower than plain decode on gfx906, off by default).

### Known caveats (2026-11-22)

- Tested primarily on **gfx906 (MI50 / MI60)** with TheRock ROCm (`-DGGML_HIP_RCCL=OFF`). Wave32/NVIDIA/RDNA guards are in place for `DSV4_HC_MIX`, and `SGEMM_F32` is gated to GCN, but non-gfx906 targets are not regularly benchmarked here.
- **Open high-leverage gaps** tracked in [vr/PERFORMANCE.md](vr/PERFORMANCE.md): long-context QSA tile skipping & 6:1 GQA head grouping (`G1`) at 32K-100K, the `>8` token cliff for 4-slot + speculative verify (`MU1(b)`), Inkling PP4K+ SWA ring-wrap fallback & `flash_attn_tile<128,128,64>` register spill, and Q8-reference KLD check for `SGEMM_F32` (`P4`).

## Build & documentation

```bash
# Source your ROCm / TheRock environment first if needed
cmake -S . -B build-gfx906 \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_VR_EXPERIMENTS=ON \
  -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build-gfx906 --target llama-cli llama-server llama-bench llama-batched-bench vr-test-gfx906-tp vr-test-flashnext-moe -j 8
```

Detailed fork docs live under `vr/`:
- [vr/README.md](vr/README.md) - quickstart, run commands, and directory index
- [vr/GOALS.md](vr/GOALS.md) - target workload, priorities, and accuracy posture
- [vr/HARDWARE.md](vr/HARDWARE.md) - calibrated gfx906 constants (occupancy, streaming, cross-GPU exchange)
- [vr/PERFORMANCE.md](vr/PERFORMANCE.md) - live baselines, audit findings, testing rules, and worklist
- [vr/BENCHMARK-TIMELINE.md](vr/BENCHMARK-TIMELINE.md) - full dated benchmark timeline (2026-08-12 to present)
- [vr/MAINTENANCE.md](vr/MAINTENANCE.md) - fork feature delta, repack layout, silent merge traps, and upstream catch-up guide

## Credits & prior art

This fork builds directly on upstream `llama.cpp` and borrows ideas, reference implementations, or benchmarks from the wider open-source inference community:

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** - upstream inference engine, HIP/CUDA backends, GDN/FlashAttention/speculative infrastructure, and PR `#32218` (DSpark).
- **[unslothai/llama.cpp](https://github.com/unslothai/llama.cpp)** - PR `#264` (device counting sort for MoE IDs), PR `#386` (Inkling 2 banded relative-bias attention interface), and dynamic GGUF quantizations.
- **[mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp)** & **[mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)** - gfx906 tensor-parallelism, whole-token HIP graph replay, weight-repack lineage, and `-mllvm -amdgpu-sched-strategy=max-ilp` toolchain tuning.
- **[iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906)** - early `q8_repack` gfx906 kernel lineage.
- **[Niko1221/Strata](https://github.com/Niko1221/Strata)** - gfx906 IQ sign-handling word-op ideas (`S2`) and MoE roofline observations.
- **[thejacer/slop.cpp](https://github.com/thejacer/slop.cpp)** - Inkling 2 Mini gfx906 tensor parallelism, load-time weight folding, fused ops, and async input-upload ring ideas.
- **[alex4300/llama.cpp-gfx906-opt](https://github.com/alex4300/llama.cpp-gfx906-opt)** & **[milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp)** - gfx906 multi-column LDS-staged GEMV, GCN `DKQ=256` FlashAttention tuning, and Vega 20 MMQ/word-op ideas.

---

For full upstream documentation, build options, supported models, and license information, see **[README.llama.cpp.md](README.llama.cpp.md)** (and [LICENSE](LICENSE), MIT).
