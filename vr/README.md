# gfx906 guide

Local enhancements for two 32 GiB gfx906 GPUs (MI50/MI60), primarily tested with Qwen3.8-27B Q8_0. Built on [MX](https://github.com/mixa3607/mx-llama.cpp) and [iacopPBK's gfx906 work](https://github.com/iacopPBK/llama.cpp-gfx906); see [credits](IMPROVEMENTS.md#built-on). The active base is official llama.cpp `e6d624b9a` (merged 2026-11-21; the patch kit is for `46eb1742d`, see [catch up](how-to-catchup.md)). The code started as the [2026-10-29 rebuild](old-ignore/2026-10-29/rebuild/README.md): upstream plus cleanup and fixes, without pipeline stages, and with DeepSeek V4/V4.1, MiniMax-M2 and MiMo-2 on upstream code. Source snapshots and benchmark identities are recorded in the performance reports.

| Guide | Purpose |
| --- | --- |
| [Improvements](IMPROVEMENTS.md) | What changed, why it helps, and what to revisit |
| [Performance](PERFORMANCE.md) | One live document, in process order: [current numbers](PERFORMANCE.md#current-numbers), [the model](PERFORMANCE.md#the-model), [how to test](PERFORMANCE.md#how-to-test), [what worked](PERFORMANCE.md#what-worked), [routes that lost](PERFORMANCE.md#routes-that-lost), [worklist](PERFORMANCE.md#worklist), [retired claims](PERFORMANCE.md#retired-claims) |
| [Performance records](PERFORMANCE-RECORDS.md) | Append-only dated runs, newest first |
| [Handoff](handoff.md) | State and next steps |
| [Process](PROCESS.md) | How performance is found: the four stages (learn the hardware once, predict from data on disk, design on paper in batches, one confirmation per batch), the order of work, the after-merge rule, how to add a fused op |
| [Hardware card](HARDWARE-CARD.md) | The single source of measured hardware constants (launch cost, one-block cost, bandwidth, noise), the one-block rule, and checks against real kernels |
| [Token budget](TOKEN-BUDGET.md) | Where one decode token goes, step by step, per model (profiled, growing record) |
| [Hardware guide](HARDWARE-GUIDE.md) | What gfx906 can and cannot do, and which optimization tricks carry over from newer GPUs |
| [Catch up](how-to-catchup.md) | Merge upstream; static diff first, then the one end-to-end comparison and only the checks of affected features; patch export is optional |
| [Current patch kit](patches/llama-head-46eb1742d/README.md) | Replayable overlay for the `46eb1742d` base |
| [Rebuild record, 2026-10-29](old-ignore/2026-10-29/rebuild/README.md) | Stages A-D, cumulative patches, results, status log and what was dropped |
| [Benchmark scripts](scripts/README.md) | Build flags, llama-bench, output and MTP checks used for the rebuild |
| [Server presets](server/models.ini) | `vr/server/serve.sh`: one llama-server (router mode) that switches between 27B (MTP2 or DFlash5), Flash Next (MTP3, 100K) and Gemma 4 per request (`"model": "qwen27b"`, `"qwen27b-dflash"`, `"flashnext"`, `"gemma4"`). Smoke test, 2026-11-10 (not repeated after later merges; B, one 96-token reply each): 51 / 61 / 48 / 32 t/s |
| [Previous patch kit](old-ignore/2026-10-29/patches/llama-head-418f905a4/README.md) | Historical overlay for the old `418f905a4` base |
| [Historical records](old-ignore/README.md) | Dated benchmarks, experiments, and superseded workflows |

Process in one sentence: learn the hardware once with synthetic microbenchmarks, predict each kernel's cost from its launch geometry in saved traces, design changes on paper with a predicted saving, and confirm each batch with one trace and one alternated run; kernel and code work comes first and parameter tuning is the last stage, with the owner's permission ([PROCESS.md](PROCESS.md)).

## Priorities and separate work

1. Qwen3.8-27B at Q8_0 is the primary performance target.
2. Gemma 4 at Q8 and Qwen Next at a quantization that fits the two GPUs are secondary targets.

`flashnext-fast` produced the accepted grouped-MoE weight-reuse path, now included in the source and active patch set. It is on by default for eligible gfx906 MoE verification since 2026-11-06; `GGML_CUDA_FLASHNEXT_MOE=0` disables it. Its separate GPU-feedback experiment stays in the original worktree. See [the evidence](old-ignore/2026-10-25/flashnext-fast/README.md).



## Build

Use the local TheRock environment with gfx906 support. From the repository root:

```sh
source $THEROCK_ENV
cmake -S . -B build-ilp \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build-ilp --target llama-cli llama-server llama-bench -j 8
```

Keep the `CMAKE_HIP_FLAGS` and `CMAKE_CXX_FLAGS` values above for every comparison build. A build without them was 2-3% slower on 27B in the 2026-10-29 rebuild.

One GPU is on CPU PCIe lanes and the other on motherboard lanes. Keep RCCL off and use host-staged AllReduce on this topology. Revisit RCCL after fixing the lane placement; this is a host constraint, not a permanent gfx906 rule.

## Run

This example requires the cached model. Keep a 50/50 tensor split while comparing changes.

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on \
  -b 2048 -ub 2048 -c 32768 -t 20 -tb 20 --temp 0 --top-k 1 -s 42
```

Twenty CPU threads are the requested comparison setting for this host, not a claim that 20 is optimal for every workload.

For single-request MTP, append `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling`. This is a useful starting point, not a universal winner; concurrency and prompt length change the result. Use the same model/GPU flags with `llama-server` for serving, setting host and port explicitly.

Keep HIP graphs enabled and RCCL off for the current mixed PCIe-lane topology. HIP uses the upstream host-staged AllReduce path plus the local graph-safe counters and slab/duplex overlap hooks; the old `GGML_CUDA_TP_OVERLAP` switch is removed. The opt-in peer-write AllReduce is archived in [old-ignore/2026-11-21/peer-ar](old-ignore/2026-11-21/peer-ar/README.md). Most local optimizations are enabled by default; historical environment sweeps are in the archive. Multi-stage TP (`-tps`) is removed; see [the rebuild changes](IMPROVEMENTS.md#2026-10-29-rebuild-removed-and-changed).

## Validation

For weekly updates, first have the human commit and push the reviewed current state, then fetch and merge upstream in the normal checkout with rerere enabled. Resolve and review the merge, diff the graph and kernel geometry without the GPU, build once in a new directory, and run the one end-to-end PP and TG comparison with the pre-merge commit that AGENTS.md requires; run other checks only for features the diff affects. The human commits and pushes the accepted merge after the checks pass. Keep the last two successful build directories, including their libraries, for comparisons. Git merges are the primary update mechanism; export a patch kit only when a portable snapshot or release checkpoint is useful. Kits describe their recorded base and target and may lag behind the branch. See [the catch-up routine](how-to-catchup.md).

The focused TP harness is built with `-DLLAMA_BUILD_VR_EXPERIMENTS=ON`, target `vr-test-gfx906-tp`; it runs only when AllReduce or tensor-split code is affected. Longer checks and performance comparisons require explicit user OK. The final report includes baseline/candidate PP2048 and TG128 side by side. The worktree-based `catchup-script.sh` is optional.

`main` is the final branch. Use `experiment/topic` for unfinished work and `archive/YYYY-MM-DD/topic` for retained historical branches; see the [cleanup record](old-ignore/2026-10-29/cleanup.md).
