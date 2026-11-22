# vr/ - gfx906 Inference Guide

Local `llama.cpp` fork (`main` branch, upstream base `e6d624b9a`) tuned for two 32 GiB AMD gfx906 GPUs (**ROCm0 = MI60** 64 CU / CPU PCIe lanes; **ROCm1 = MI50** 60 CU / chipset PCIe lanes). Primary targets: Qwen3.8-27B Q8_0, Qwen3.8 Flash Next, Inkling 2 Mini, and upcoming Qwen 4 at 50-100K+ context and 1-4 parallel slots.

## Build

Always use the TheRock ROCm environment and `max-ilp` + `-O3` flags (missing them loses 2-3% on 27B). Never rename a build directory (`RUNPATH` is absolute); point `build-current` / `build-previous` symlinks instead.

```bash
source $THEROCK_ENV
cmake -S . -B build-new \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_VR_EXPERIMENTS=ON \
  -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build-new --target llama-cli llama-completion llama-server llama-bench llama-batched-bench vr-test-gfx906-tp vr-test-flashnext-moe -j 8
```

## Quick run & live server

Active symlinks in repo root: `build-current -> build-dspark-20261122` (`3a8d768df`, opt-in DSpark), `build-previous -> build-final-20261122` (`ae32e5e58`).

```bash
# Single-request CLI (Qwen3.8-27B Q8_0, 50/50 TP; add --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling for MTP2)
./build-current/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on \
  -b 2048 -ub 2048 -c 32768 -t 20 -tb 20 --temp 0 --top-k 1 -s 42

# Router server (serves qwen27b, qwen27b-dflash, flashnext, flashnext-long, gemma4)
vr/server/serve.sh
# Or systemd unit: systemctl status|start|stop llama-model-server
```

Keep HIP graphs on and RCCL off on this host (MI50 is on chipset PCIe lanes; default host-staged AllReduce runs with no env switch).

## Directory map

| Path | Purpose |
| --- | --- |
| [GOALS.md](GOALS.md) | **Read first**: owner mission, 50-100K+ 1-4 slot workload, priority hierarchy, accuracy & deviation rules |
| [HARDWARE.md](HARDWARE.md) | Measured gfx906 constants (Q1 occupancy, Q2 streaming, Q3 DMA exchange), one-block rule, MI60/MI50 asymmetry, cross-GPU portability |
| [PERFORMANCE.md](PERFORMANCE.md) | Live baselines, 2026-11-22 BA-4 audit & skeptical findings, method/testing rules, token breakdowns, prioritized worklist, 28 lost routes, retired claims |
| [BENCHMARK-TIMELINE.md](BENCHMARK-TIMELINE.md) | Append-only chronological benchmark history and dated experiment records |
| [MAINTENANCE.md](MAINTENANCE.md) | Fork feature/file inventory, 10-quant repack layout & invariants, **14 silent merge traps**, weekly upstream catch-up & validation, how to add a fused op |
| [server/serve.sh](server/serve.sh) | Router `llama-server` launcher (uses `build-current` and `models.ini`) |
| [server/models.ini](server/models.ini) | Validated unified-KV multi-user presets for `qwen27b`, `qwen27b-dflash`, `flashnext`, `flashnext-long`, `gemma4` |
| [scripts/kernel_budget.py](scripts/kernel_budget.py) | Anchor-segmented `rocprofv3` SQLite analyzer: exact steady counts, Q1 occupancy, spill/under-fill/MI50-skew flags |
| [scripts/bench_multiuser.sh](scripts/bench_multiuser.sh) | Cool-start `<60 C` 4x16K unified-KV + 4-way TG comparison for 27B and Flash Next |
| [scripts/spec_accept.sh](scripts/spec_accept.sh) / [scripts/spec_accept.py](scripts/spec_accept.py) | 5-prompt `llama-server` speculative A/B reporting acceptance and ms/verify step |
| [bench/q1_occupancy.cpp](bench/q1_occupancy.cpp) | Standalone microbench: VGPR/LDS/`__launch_bounds__` residency & dispatch rate |
| [bench/q2_stream.cpp](bench/q2_stream.cpp) | Standalone microbench: HBM2 streaming GB/s by `(S, G, T)`, chunk vs run, FMA/dot4/F16 |
| [bench/q3_exchange.cpp](bench/q3_exchange.cpp) | Standalone microbench: host-staged D2H/H2D/duplex DMA latency & bandwidth |
| [bench/one_block.cpp](bench/one_block.cpp) | Standalone microbench: one-block cost curve, threads/block, barriers vs phases, 2-kernel form |
| [bench/launch_floor.cpp](bench/launch_floor.cpp) | Standalone microbench: bare HIP launch cost on MI60/MI50 & multi-stream throughput |
| [test-gfx906-tp.cpp](test-gfx906-tp.cpp) | Focused AllReduce / TP smoke test (`vr-test-gfx906-tp --n 1 --check` and `--n 2048 --check`) |
| [test-flashnext-moe.cpp](test-flashnext-moe.cpp) | Focused grouped-MoE CPU-reference test (`vr-test-flashnext-moe`) |
