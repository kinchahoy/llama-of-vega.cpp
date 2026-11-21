# Plan: merge upstream 46eb1742d, take useful unsloth work, shrink the gfx906 delta

Plain-language summary:

1. Merging the latest official llama.cpp (46eb1742d, 79 new commits) into our fork is small: git predicts only 2 conflicting files, both in the GPU MoE/top-k code.
2. One auto-merged file has a silent bug that the conflict markers do not show: our MoE "expert bounds" helper now under-reports the last expert's row count. The fix is one word (`it_compact` -> `n_dumped`).
3. Upstream does not yet replace any big piece of our fork. Most of our 16.4k added lines are still needed for the speed gains (repacked weights, two-GPU tensor parallel, speculative decoding).
4. We can still cut about 2,000-3,000 lines without losing speed: archive the unused peer-to-peer AllReduce (1.4k lines, opt-in only), and delete sweep-only environment knobs and off-by-default experiments.
5. From unsloth, only a few items matter for our hardware: the device-side MoE id sort (#264) for Flash Next prompt speed, a crash guard for empty CUDA graphs (#241), and shared MTP heads for Flash Next (#240) to save memory. Most other pins are NVIDIA-only or for models we do not run.

Scope: planning only. Nothing was built, run, committed or checked out. Conflict predictions come from `git merge-tree --write-tree HEAD 46eb1742d` (result tree `6026edc63af52147c958201a9af58f38ad371b55`). Items not verified are marked "unverified".

Repos and identities:

| Name | Path | Ref |
| --- | --- | --- |
| OURS | $REPO_ROOT | main 4411bbc2a, merge-base edec462cc, 77 commits ahead (72 non-merge + 5 merges; 28 touch code, 44 doc-only) |
| LLAMAHEAD | $HOME/infer/llamahead | ggml-org master 46eb1742d (2026-11-21), 79 commits after edec462cc |
| UNSLOTHHEAD | $HOME/infer/unslothhead | e85442867; pins in `scripts/unsloth/pr-set.json` (30 entries) |

---

## Part 1. Merge OURS with upstream 46eb1742d

### 1.1 Predicted conflicts

`git merge-tree --write-tree --name-only HEAD 46eb1742d` reports exactly two content conflicts. Everything else auto-merges (about 40 files with overlapping edits).

| File | Upstream cause | Resolution |
| --- | --- | --- |
| `ggml/src/ggml-cuda/top-k.cu` (2 hunks, lines ~248 and ~295 of the merged file) | b94959a64 (#32505, duplicate expert IDs) replaces a local `popc` lambda with the new `ggml_cuda_popc` helper and keeps `warp_ties` in upstream's tie gather | Take OURS for both hunks. The fork's count/offsets/gather kernel replaced upstream's tie gather (merge trap: "fork top-k.cu is kept over upstream #31084"). Confirm no leftover upstream `warp_ties`/`nwarps` locals and that the fork kernel does not need `ggml_cuda_popc`. |
| `ggml/src/ggml-cuda/mmid.cu` (1 hunk, line ~274) | b94959a64 rewrote `mm_ids_helper`: a bounded shared-memory store (`ne_store`), multi-pass `dump_store`, `n_dumped` counter, no `nbytes_shared <= smpbo` assert | Keep the fork's `launch_mm_ids_helper_switch` and the two `ggml_cuda_launch_mm_ids_helper` overloads (tile_spec and expert_bounds). Use upstream's new launch arguments: `(ids, ids_src1, ids_dst, tile_spec, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse, ne_store)`. |

### 1.2 Silent merge trap found by inspection (auto-merged, wrong)

In the merged `mmid.cu` kernel (merged line ~226):

```cpp
    if (expert_bounds) {
        expert_bounds[expert] = nex_prev;
        if (expert == int(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;   // wrong after b94959a64
        }
    }
```

After b94959a64, `it_compact` is reset to 0 after each `dump_store`. The total for this expert is now in `n_dumped`. Change to `nex_prev + n_dumped`. If this is missed, the repacked MoE GEMM and the grouped-MoE path (both read `expert_bounds`) skip the last expert's rows. Add this to the "Silent merge traps" list in `vr/IMPROVEMENTS.md`.

Also confirm the `if (!tile_spec) return;` guard stays after the bounds write and before the tile loop. It auto-merged in the right place.

### 1.3 Upstream changes in edec462cc..46eb1742d that touch our hot files

| Upstream commit | Files that overlap our delta | Effect on the fork | Action |
| --- | --- | --- | --- |
| b94959a64 CUDA: duplicate expert IDs (#32505) | mmid.cu, top-k.cu, fattn.cu, common.cuh, ggml.h, mmf.cu/.cuh, ggml-cuda.cu | Conflicts above. Adds `ggml_cuda_popc` (32/64-bit mask aware) in common.cuh. mm_ids_helper now handles any n_tokens (no smem assert) | Resolve as above. Later: use `ggml_cuda_popc` in fork kernels that hand-code `__popcll` (W64/B1 code) when touched |
| d4f5c2604 CUDA: 64-bit dst offsets in mul_mat_q (#32374) | mmq.cuh (4 lines) | Auto-merged. Our gfx906 row-first traversal lives in mmq.cu, not touched | Check that the gfx906 launch path in `mmq.cu` passes the 64-bit offsets the same way (semantic check) |
| e3c3f666a hip: RDNA1 MMVQ table (#32389), ea84d7090 Orin MMVQ (#31268) | mmvq.cu | Auto-merged next to our Q8_0 `return 2` rows-per-block and the two-row IQ2_S path | Check `calc_rows_per_block` / table selection still returns the fork value for `GGML_CUDA_CC_VEGA20` |
| 9fbe30c2a MUL_MAT_ID precision in fallback (#31345), 4b69c79cc GDN supports_op (#32660), c7468107e set_n scatter (#32489) | ggml-cuda.cu | Small. 4b69c79cc changes the GDN supports_op check; our chunked GDN kernel (`gated_delta_net_chunk.cu`) is dispatched from that op | Check that the new supports_op still accepts the shapes our chunk path and meta whitelist use (27B and Flash Next are both GDN hybrids) |
| 7fce5439f BF16 ROPE and SET_ROWS (#32387) | set-rows.cu, ops.cpp, ggml.c | Auto-merged next to K1's float4 F32 set_rows | Run `test-backend-ops -o SET_ROWS` in the final pass |
| 0a7675746 sticky NaN in recurrent state (#31913) | llama-graph.cpp (`build_rs`), llama-memory.cpp, llama-ext.h | `ggml_scale_inplace(.., 0)` becomes `ggml_fill_inplace(.., 0.0f)`. Merged tree has the fill. The meta backend already handles `GGML_OP_FILL` (ggml-backend-meta.cpp:1471) | Generation on both models under `-sm tensor` covers it. Check the rollback snapshot planes for any other zero-by-scale (none found by grep) |
| 01592dc58 drop hparams/prec_policy/hdmd from llm_graph_params (#32596) | llama-graph.h/.cpp, llama-context.cpp | Grep of the merged tree finds no fork use of the removed fields | None beyond build |
| 652d559ed graph node budget into model classes (#32648), 4c4a0f0a4 MoE cache node reserve (#32636) | llama-context.cpp, models.h, dflash.cpp | `graph_max_nodes()` merged cleanly. Fork token graphs, DFlash and MTP contexts use it | Watch for "graph node budget" asserts with DFlash/MTP and Flash Next (512 experts) |
| 4783b1145 indexer key rotation into model classes (#32653) | llama-kv-cache.cpp, models.h | Next to kv_rows (7ddd49693) and KV replica code | Build; multi-user check covers kv_rows |
| 8af68d486 Inkling 2 mask (#32429), 29b52d398 Inkling 2 Mini (#32694), 55d323507 Domyn | arch/model/vocab/gguf-py | Model support only. Fork has no Inkling code | Take upstream |
| ac830a9cb backend scheduler tests (#27258) | ggml.h, ggml.c, ggml-cpu, tests | New tests run the scheduler; fork changes `ggml-backend.cpp` and `ggml-alloc.c` heavily | Build and run the new scheduler tests once (CPU only, no GPU) to catch fork scheduler regressions |
| 7716ec8cb deprecate ggml_ftype (#32647) | ggml.h, ggml.c | Repack code does not use ggml_ftype (unverified, grep before build) | Build |

Upstream did not touch, in this range: `common/speculative.cpp`, `src/llama-memory-recurrent.cpp`, `fattn-tile.cuh`, `fattn-vec.cuh`, `allreduce.cu`, `ggml-backend-meta.cpp`, `q8_repack/`, `mmq.cu`. MTP/DFlash state and the AllReduce invariant are therefore low risk for this merge. Still confirm the invariant: HIP enters the first `#if !defined(GGML_USE_MUSA)` branch of `allreduce.cu`, no env switch.

### 1.4 Where upstream now provides an equivalent (drop candidates)

None of the large fork features are replaced by upstream in this range. Checked:

| Fork feature | Upstream status at 46eb1742d | Verdict |
| --- | --- | --- |
| Chunked GDN (`gated_delta_net_chunk.cu`, 371 lines) | Upstream has `gated-delta-net-mma.cu`, enabled on CDNA only (47fe13331 #31327); it needs MFMA, gfx906 has none | Keep |
| top-k count/offsets/gather | Upstream radix top-k has deterministic ties only (#31084); CUB DeviceTopK (#28713) is CUDA-only | Keep |
| kv_rows on AMD tile kernel | Upstream kv_rows (#29510) still not on non-MFMA AMD | Keep |
| CUDA graph key with shape | Upstream still keys by `nodes[0]` only | Keep (see unsloth #241 below) |
| HIP host-staged AllReduce | Upstream has the base (#27825); fork adds graph counters, duplex, slabs | Keep local extensions |
| `ggml_cuda_popc` | New upstream helper | Use it instead of hand-written popc when fork kernels are edited (tiny) |

### 1.5 Ordered merge procedure (follows vr/how-to-catchup.md)

1. Human commits and pushes the current state (main is clean at 4411bbc2a).
2. Create the worktree `.claude/worktrees/catchup-46eb1742d` on branch `catchup-46eb1742d` from HEAD. Set `OLD_HEAD`, `BASE=$(readlink -f build-current)` (now `build-catchup-20261118-134459`), `CAND=$PWD/build-catchup-<date>`, `OUT`. rerere on.
3. `git fetch upstream`; confirm `upstream/master` is 46eb1742d (or pin that SHA); `git merge --no-commit --no-ff 46eb1742d`.
4. Resolve `top-k.cu` (take ours, both hunks) and `mmid.cu` (fork switch + upstream launch args), then apply the `n_dumped` fix in 1.2. `git rerere`, `git add` the two files.
5. Semantic review of auto-merged hot files, in this order: `mmid.cu` kernel body, `mmvq.cu` (VEGA20 rows), `mmq.cuh`/`mmq.cu` 64-bit offsets, `ggml-cuda.cu` GDN supports_op and MUL_MAT_ID fallback, `set-rows.cu`, `llama-graph.cpp` `build_rs`, `llama-context.cpp` `graph_max_nodes`, `llama-kv-cache.cpp`.
6. Walk the "Silent merge traps" list in IMPROVEMENTS.md (J selection once, expert_bounds overload, `s_write`/`s_write_conv`, `layer_inp_tap`, DFlash dev opt-in, meta repack whitelist, HIP ballot, K/V rotation per layer, `hv_tensor_requires_copy`, fork top-k, MTP dev rows opt-in).
7. `git diff --check`, `git diff --cached --check`, `git ls-files -u` empty, no conflict markers.
8. One build (flags from the guide, `-DLLAMA_BUILD_VR_EXPERIMENTS=ON`), then `vr-test-gfx906-tp --n 1 --check` and `--n 2048 --check`.
9. With user OK: one 27B generation, the matched PP2048/TG128 pair, and (because mmid/top-k/MoE changed) a Flash Next PP/TG pair plus `test-backend-ops -o MUL_MAT_ID`, `-o TOP_K`, `-o SET_ROWS`, `-o GATED_DELTA_NET`. Multi-user check (`vr/scripts/bench_multiuser.sh`) because kv-cache files changed.
10. Human commits the merge; update `build-current`/`build-previous` symlinks; add the mmid trap to IMPROVEMENTS.md.

Do the merge first, before any of the shrink work in Part 3, so the merge A/B measures only upstream changes.

---

## Part 2. Unsloth candidates

How pins are recorded: `scripts/unsloth/pr-set.json` lists 30 pinned commit URLs (rules in its `_doc`). Tools: `scripts/unsloth/pin_merge.py`, `pin_contract.py`, `repin.py`, `additive_merge.py`; CI in `.github/workflows/unsloth-pr-set-lint.yml`, `unsloth-pin-preflight.yml`, `unsloth-repin-bot.yml`. The nightly build is upstream plus these pins. The pinned commits are not ancestors of unsloth master, so map a pin with `git diff $(git merge-base <pin> upstream/master) <pin>` in UNSLOTHHEAD. Most numbers are unslothai PRs; ggml-org#24423 and ggml-org#32218 are upstream PRs. The 456 commits on unsloth master are pin maintenance and CI/packaging; ignore them.

### 2.1 Candidates worth taking or evaluating

| # | Source (pin commit, key commits) | Files, size | Upstream at 46eb1742d | HIP / gfx906 | Expected effect | Risk | How to bring in |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U1 | unsloth #264, pin fd6fae58b (00f... series; key: a66db92dd device permutation, 1df1cecf0 counting sort, 8cdf169a3 gating, 58ff937e3 repeated ids, 8c14e99d8/97b8fc58a get_rows) | mmid.cu +446, getrows.cu +154, ggml-cuda.cu +91, mmid.cuh +16, tests +65; 7 files +712/-64. Based on upstream 8bf88ba93 (after b94959a64) | No | Generic CUDA; uses `ggml_cuda_shfl` helpers and ballots. Its own expert_bounds-writing `ggml_cuda_launch_mm_ids_helper` overload is the same idea as the fork's overload | Flash Next PP: the single-warp-per-expert helper walks all n_tokens x 10 ids for each of 512 experts; the counting sort from 512 tokens on should cut that. Also related to ledger M2 (`repack_tile_off<<<1,1>>>`). Gate thresholds were tuned on NVIDIA only (T4/L4/B200) | Medium: rewrites mmid.cu again, interacts with fork J selection and expert_bounds consumers; wave64 masks must be checked | After the merge. Port as one patch on top of upstream; replace the fork expert_bounds overload with #264's. A/B Flash Next PP2048 (and 27B as control, it has no MoE). Watch for an upstream PR of the same work (unverified, `gh` not available on this host) |
| U2 | unsloth #241, pin ee25de406: 7fa42ff5e (graph key with no nodes), 271f94738 (key by shape), 0e8ffdc72 (drop graph cache cap) | common.cuh, ggml-cuda.cu; ~+52/-17 | No | Applies to HIP graphs | Fork already keys by `nodes[0]`, `n_nodes`, `nodes[0]->ne[1]` (ggml-cuda.cu:3924, B15). Fork key dereferences `nodes[0]` without an `n_nodes == 0` check: take only 7fa42ff5e's guard | Low | Hand-port the 2-line guard. Propose the shape key upstream jointly; both forks found the same spec-decode thrash |
| U3 | unsloth #240, pin 924234599: 36175a832 (qwen4exp MTP heads sharing target token_embd/output), 235c558d2, d2dd44f2c, 08490ff69 | llama-context.cpp +29, llama-cparams.h +1, qwen4exp.cpp +13; +38/-5 | No (upstream has Flash Next and MTP, not shared-head GGUFs) | Model-level | Ledger D2: unsloth "shared" Flash Next drafts save about 1.8 GiB, which is the limit for Flash Next MTP at long context | Medium: fork's `load_private_head` (`LLAMA_DFLASH_PRIVATE_HEAD`) copies the head for a single-GPU draft because the target head lives in a meta buffer under `-sm tensor`. Borrowing works only when the draft can read the target tensors, so on the ROCm1 sidecar the saving may disappear (unverified) | Port after the merge; test Flash Next MTP3 fit and step time at 100K |
| U4 | unsloth #251, pin 827478f5a (74db68e04, cd3c2847e, 07208276e threshold 0.80, 3f09f1e49) | common/fit.cpp +141, arg.cpp, common.cpp, llama-moe-cache.cpp +52; +212/-23 | Partial: upstream has the MoE cache (#30966, #32636); `--moe-cache-mib auto` and the 0.80 resident threshold are unsloth-only | Backend-neutral | No PP/TG change at a fixed config. Helps ledger E1 (Flash Next IQ4_XS fit) | Low, but touches common/fit | Optional, only if E1 is pursued |
| U5 | unsloth #385, pin 8825dc242 (7980f5ca4 chunk Inkling 2 relative-bias attention over query rows) | inkling2.cpp +73, llama-graph, kv-cache; +87/-12 | No; upstream has Inkling 2 (5c008e000 #32217), mask fixes (8af68d486 #32429, arrives with this merge), Inkling 2 Mini (29b52d398 #32694, arrives with this merge) | Generic graph code | Lower memory for Inkling 2 dense rel-bias mask. Not a focus model | Low | Take only if Inkling 2 is run; otherwise wait for upstream |

### 2.2 Inkling status

| Item | Upstream 46eb1742d | Unsloth only |
| --- | --- | --- |
| Inkling 2 text + vision | yes, 5c008e000 (#32217) | DSpark Inkling 2 draft and FA folded bias: ggml-org#32218 (open upstream PR, pin 78d779424, +818/-33) and unslothai #390 (pin 8aa2f23f6, +1342/-57) |
| Inkling 2 mask fixes | yes, 8af68d486 (#32429), in this merge | - |
| Inkling 2 Mini | yes, 29b52d398 (#32694), in this merge | - |
| Relative-bias chunking | no | #385 (U5) |
| Banded flash attention for rel-bias | no | #386 (pin 523e46559; 7c3655ce7 adds `GGML_OP` FLASH_ATTN_EXT_BANDED and `fattn-banded.cu`, 889 lines). Only the tensor-core prefill kernel (`flash_attn_ext_banded_mma`, PTX `mma.sync` + `cp.async`, TURING_MMA_AVAILABLE) is NVIDIA only. The f32 kernel and split-KV decode pair use only WARP_SIZE/ggml_cuda_shfl and are the dispatch fallback, so the op should run on HIP for prefill and decode (not compiled). Gain: no dense n_kv x n_ubatch x n_head mask. Cost: new op, ~1.1k lines, sits on #378. Take #378 chunked dense path first; try banded op only if Inkling 2 shows a memory/speed problem, A/B vs dense |
| Looped Inkling 2 variant | no | #389 (pin 3b34f4f2f, +586/-44) |
| Inkling 1 (TML) | no | #378 (pin e65c374cb), #386 |

Recommendation: rely on upstream for Inkling 2. Do not carry Inkling pins in this fork.

### 2.3 Evaluated and not worth taking for gfx906

| Item | Why not |
| --- | --- |
| CUB DeviceTopK | Already upstream (71ad0590f #28713) and CUDA-only. HIP builds have no CUB; hipCUB/rocPRIM top-k is unverified and TB1 shows the 12 radix launches cost only 70-90 us per call |
| CUDA graph chunking | Already upstream (280141e6d #30804, fix 9ff0d7e90); fork has `kernel_funcs_chunks` |
| HIP 64-bit warp mask repin (#378, #264, bf063b959) | bf063b959 changes only `pr-set.json`; the mask code could not be identified (unverified). Upstream b94959a64 now adds `ggml_cuda_popc`, and the fork already fixed the ballot trap in B1 (457d4470b) |
| IQ1_M MMQ (#304, pin e4e1eba07, +265/-3) | Adds RDNA2/3/4 tables only, no GCN entry; no focus model uses IQ1_M |
| IQ1_XS/XXS/XXXS types (#61, pin b74bd0876, +878 in ggml-cuda) | New quant types; adds GCN MMQ table entries only for those types |
| Banded FA (#386) | See 2.2 |
| #157 cudaMemcpyDefault in cpy 2D (+10/-1), #149, #158 (HIP iGPU) | Small HIP fixes for other setups; #157 could matter for HIP 2D copies but F1 showed rect copies are load-time only. Optional |
| #95 penalties sampler (+118/-13) | CPU sampler speed with penalties only |
| GDN, fattn, mmvq, argsort, allreduce, split-mode tensor | No pin touches them |

---

## Part 3. Shrink the fork delta

### 3.1 Current delta by feature (non-doc: `git diff edec462cc HEAD -- . ':!vr' ':!docs'`, 124 files, +16426/-995)

| Feature group | Files | +/- | Category |
| --- | --- | --- | --- |
| Repack (q8_repack/*, 9 new files incl. README 210) | 9 | +4969/-0 | (d) keep; README moves to vr (c) |
| Meta backend, scheduler, alloc (ggml-backend-meta.cpp +1669/-123, ggml-backend.cpp +404/-15, ggml-alloc.c +308/-21, sched-impl.h, backend-impl.h) | 6 | +2402/-160 | (d) keep core; (c) knobs |
| llama runtime: context, graph, kv-cache, memory-recurrent/hybrid, ext, cparams | 16 | +1739/-157 | (d) keep (MTP, DFlash, split KV, rollback, kv_rows) |
| Peer-write AllReduce (tp-allreduce.cu 1027, .cuh 192), opt-in `GGML_CUDA_PEER_AR=1` | 2 | +1219/-0 | (c) archive |
| ggml-cuda glue (ggml-cuda.cu +998/-86, common.cuh, graph.cuh) | 3 | +1112/-91 | (d) keep; (c) about 200 lines of peer-AR glue (ggml-cuda.cu ~1439-1640, 1874-1879) and knobs |
| Speculative (common/speculative.cpp +856/-83, .h, examples/speculative) | 3 | +884/-92 | (d) keep; (c) knobs |
| Model/loader/vocab/gguf (llama-model.cpp +342/-40, vocab +54/-18, qwen4exp/qwen35/delta-net-base, gguf-split +32/-18) | 15 | +672/-103 | (d) mostly; (b) gguf-split fix; (c) PLE prefault |
| F2 hc-mix (dsv4-hc.cu +478) plus CPU reference ops (+127) | 5 | +606/-0 | (d) keep, measured +12% FN TG |
| Chunked GDN | 3 | +443/-16 | (d) keep |
| MMQ/MMVQ/MMID/vecdot/quantize | 11 | +417/-124 | (d) keep (row-first traversal, J cap, IQ2_S two-row, S2) |
| topk/argsort (topk-moe B1 +217, top-k +118/-71, argsort +72) | 4 | +410/-91 | (d) keep; (b) B1 and top-k order fix are upstream candidates |
| Other CUDA kernels (norm +101, getrows +116, concat +69, set-rows +81, ...) | 9 | +383/-14 | (d) mostly; (b) B12 getrows fix |
| AllReduce host-staged extensions (allreduce.cu +314/-77, .cuh) | 2 | +322/-77 | (d) keep (invariant) |
| Grouped MoE (flashnext-moe.cuh 190 + q8_repack 59) | 2 | +249/-0 | (d) keep |
| P4 small-M F32 GEMM | 2 | +151/-0 | (d) keep; (b) GCN candidate |
| kv_rows tile on AMD | 4 | +143/-57 | (d) keep; (b) upstream candidate |
| tests/test-backend-ops.cpp | 1 | +127/-2 | (d) keep (or upstream with the ops) |
| Server (+74/-5), llama-bench (+9), CMake, AGENTS/CLAUDE/README | ~12 | +148/-6 | (d) small |
| One-line `event_query` slots in 15 other backends | 15 | +15/-0 | (c) move to proc-address (open item) |

### 3.2 Category (a): redundant with upstream, drop

Little is redundant at 46eb1742d:

- `ggml/include/ggml-backend.h` (+2/-1): a pure line-wrap change of `ggml_backend_meta_device` arguments. Revert. -3 lines.
- Fork expert_bounds overload in `mmid.cu`/`mmid.cuh` (+46/-11): keep for now; drop when U1 (#264) or an upstream equivalent lands.
- `LLAMA_TP_BACKEND_SAMPLING` (SP1a): already removed in d7ea110cb. Only the AllReduce 16-byte fallback from 6f21fe163 remains, and that stays.
- Local popc lambdas: use upstream `ggml_cuda_popc` when touched (a few lines).

Expected: about -10 to -60 lines. No A/B needed.

### 3.3 Category (b): upstream or unsloth candidates (reduce delta over time)

| Item | Fork location, size | Route |
| --- | --- | --- |
| B1-style gguf-split fix (split string literal at newline) | tools/gguf-split/gguf-split.cpp +32/-18 | Small upstream bug-fix PR (human-written) |
| B12 getrows fix: unsharded IQ4_NL/MXFP4 gather with `ne00 % 256 != 0` wrote zeros | ggml/src/ggml-cuda/getrows.cu part of +116/-11 | Upstream bug-fix PR if it reproduces upstream (unverified whether upstream has the bug) |
| AllReduce fallback for sizes not a multiple of 16 bytes (6f21fe163) | ggml-cuda.cu +6 | Upstream PR; also affects upstream's HIP path |
| CUDA graph key by shape (B15) | ggml-cuda.cu:3924 | Joint upstream proposal with unsloth #241 |
| kv_rows tile kernel for non-MFMA AMD | fattn-tile.cuh, fattn-vec.cuh, fattn.cu, +143/-57 | Upstream PR candidate; clear benefit (27B 4x16K PP 462 -> 578) |
| GCN small-M F32 GEMM (P4) | sgemm-f32.cu 146 | Upstream candidate restricted to GCN; needs other-model evidence |
| `topk_moe_wave` 512x10 (B1) and the wave64 ballot fix (457d4470b) | topk-moe.cu +217/-16 | The ballot fix is a clear upstream bug-fix; the kernel is gfx906-specific |
| top-k output order fix | top-k.cu +118/-71 | Upstream discussion; sampler/DFlash depend on order |
| GCN MMQ tuning (row-first, J cap) | mmq.cu +201/-87 | Discussion 23881; per-quant small PRs |
| MoE ids (expert_bounds) | mmid.cu | Replace with unsloth #264 if it goes upstream |

Open upstream PRs could not be searched (`gh` is not installed on this host): unverified. If all these land: about -600 to -900 lines.

### 3.4 Category (c): shrink or simplify (default path unchanged)

| Change | Lines | Needs A/B? |
| --- | --- | --- |
| C1. Archive peer-write AllReduce: export `tp-allreduce.cu/.cuh` plus the ggml-cuda.cu glue (CustomARContext/Plan, `tp_custom_ar_*` calls, `GGML_ENABLE_CUSTOM_AR`, `GGML_TP_AR_*`, `HSA_FORCE_FINE_GRAIN_PCIE`, `GGML_TP_AR_KTIMING` fprintfs) as a patch under `vr/old-ignore/2026-11-xx/peer-ar/`, with a README saying how to restore it for ledger A4 (after moving ROCm1 to CPU lanes) | about -1219 new-file lines, about -200 glue | No perf A/B (opt-in path, off by default). Required: TP smoke n=1/2048 and the default 27B PP/TG pair, to prove the host-staged path did not depend on the glue. Check that the graph-replay counters and slab hooks in `allreduce.cu` do not use `CustomARContext` |
| C2. Remove sweep-only knobs and their non-default branches: `GGML_RP_NC_PLANAR`, `GGML_RP_NC_LDS`, `GGML_RP_NC_L16`, `GGML_RP_MMV_ROWS` (q8_repack/mul-mat.cu), `GGML_SCHED_INPUT_STAGING_{F16,F32,I64,}_MAX`, `GGML_SCHED_NON_GRAPH_INPUT_STAGING_MAX`, `GGML_SCHED_N_COPIES` (ggml-backend.cpp), `GGML_CUDA_CUBLAS_CHUNK_MIB`, `GGML_CUDA_EAGER_BLAS_HANDLE`, `GGML_CUDA_FUSE_ADD_RMS_NORM`, `GGML_CUDA_Q8_1_PRODUCER`, `GGML_CUDA_Q8_1_CACHE`, `GGML_CUDA_MOE_WEIGHTED_REDUCTION`, `GGML_CUDA_CONCAT_TRANSPOSED`, `GGML_CUDA_REPACK_KQUANT_MOE_FUSION`, `GGML_GALLOC_LAYOUT_CACHE`, `GGML_META_TOPK_BCAST`, `GGML_META_INNER_GRAPHS_EAGER_OFF`, `LLAMA_TAP_STATIC`, `LLAMA_TAP_EMBD_SKIP`, `LLAMA_TAP_ACCUM_MROPE`, `LLAMA_TAP_ACCUM_MIN`, `LLAMA_SPEC_LAZY_PREFILL`, `LLAMA_DFLASH_DEFER`, `LLAMA_DFLASH_DEFER_TAIL`, `LLAMA_DFLASH_ASYNC_REPLAY`, `LLAMA_DSPARK_*` windows, `LLAMA_KV_REPLICA` | about -150 to -400 (unverified; depends on how much code sits behind each "off" branch) | Build and the standard pair only. Behavior at defaults is unchanged by construction; diff review must show each removed branch was the non-default one |
| C3. Remove off-by-default experiments: `GGML_META_NO_COMM` (debug butterfly fallback), `GGML_SCHED_STAGE_SIMPLE_GPU_INPUTS`, `LLAMA_PLE_PREFAULT`, `LLAMA_DRAFT_PIPELINE` and `LLAMA_MTP_DRAFT_PIPELINE` ring for drafts (only if no preset in `vr/server/models.ini` sets them; unverified), `GGML_TP_AR_ONESHOT`/`GGML_TP_AR_NO_GATE` (NCCL-only, RCCL off) | about -100 to -300 (unverified) | No perf A/B. Archive each removed block as a patch with a "how to retry" note (memory rule: failed attempts are archived, not lost) |
| C4. Keep these kill-switches (they are in the merge index and used for A/B): `GGML_CUDA_FLASHNEXT_MOE`, `GGML_CUDA_SGEMM_F32`, `GGML_CUDA_HC_MIX_FUSED`, `GGML_CUDA_REPACK_CONVERT`, `GGML_K1_STATE_STORE`, `GGML_K1_SET_ROWS_VEC`, `LLAMA_KV_ROWS_DISABLE`, `GGML_CUDA_AR_SLABS`, `LLAMA_SHEXP_SPLIT`, `GGML_SCHED_SHARE_VIEW_INPUTS`, `LLAMA_MTP_DEV_ROWS`, `LLAMA_DFLASH_PRIVATE_HEAD`, `LLAMA_MTP_REPACK`, `LLAMA_DRAFT_REPACK`, `LLAMA_PLE_SHARD`, `LLAMA_GDN_NO_CHUNK_SNAPSHOTS`, `LLAMA_MTP_ADAPTIVE_DISABLE`, `LLAMA_MTP_NO_KV_ONLY`, `LLAMA_MTP_DEFERRED_PREFILL`, `GGML_CUDA_PEER_AR` (until C1) | 0 | - |
| C5. A2 knobs: `LLAMA_SHEXP_SPLIT`, `GGML_SCHED_SHARE_VIEW_INPUTS`. Ledger says the PP effect was never measured. If PP2048 r3 (27B) / r5 (Flash Next) shows no difference, delete the knob and, if the off state is equal, the whole feature | 0 to -150 (unverified) | Yes (see 3.7) |
| C6. Move `ggml/src/ggml-cuda/q8_repack/README.md` (210) to `vr/` and link it from the code with one line | -210 from core | No |
| C7. `event_query` via the proc-address table instead of 15 one-line backend edits (open item in IMPROVEMENTS.md) | -15 +about 10 | No; build only |
| C8. Move ggml-cuda.cu glue into separate files (open item): repack dispatch, q8_1 cache, slab state, fusion glue | 0 net, but smaller conflict surface in the most-edited upstream file | No perf A/B; TP smoke and the standard pair |

Total (c): about -1,900 to -2,500 lines of core delta (C1 is the only firm number, about -1,420).

### 3.5 Category (d): must keep (rough line counts)

| Feature | Lines | Why |
| --- | --- | --- |
| Repack upload/layout/dispatch/kernels | about 4,760 (without README) | 27B +53% PP, +12% TG; Flash Next +74% PP vs official |
| Meta backend lanes, token graph replay, scheduler/alloc integration | about 2,200 | Lanes worth about 1.5 t/s TG on both models; whole-token HIP graphs |
| MTP/DFlash/DSpark runtime, split KV, rollback, taps, kv_rows | about 2,500 (context, graph, kv-cache, memory, speculative) | MTP2 +68% 27B, Flash Next +33%; multi-slot rollback correctness |
| ggml-cuda glue minus peer AR | about 800 | Fusions, q8_1 cache, slab overlap, repack dispatch |
| Host-staged AllReduce extensions | about 320 | Invariant: -21% PP / -38% TG without HIP host staging |
| F2, P4, chunked GDN, grouped MoE, B1, S2, K1, IQ2_S two-row, kv_rows tile | about 1,900 | All landed with measured gains in PERFORMANCE.md |
| Model integration (PLE placement, shared-expert split, tied head split, MTP mirrors) | about 550 | Needed for 27B/Flash Next/Gemma 4 under `-sm tensor` |

### 3.6 Target structure and commit-count reduction

The history on main cannot shrink without rewriting it. Two options; the human decides:

- Option A (recommended, no history rewrite): keep main's merge history. After the 46eb1742d merge and the Part 3 cleanups, export a new patch kit `vr/patches/llama-head-46eb1742d/` as an ordered per-feature stack. This kit is the "small stack" view, and it is what reviewers and future rebuilds read. Move `vr/patches/llama-head-36a0e2ffc/` to `vr/old-ignore/`.
- Option B (like the 2026-10-29 rebuild): build a fresh branch on 46eb1742d from the stack below, prove `git diff <rebuilt> <main-after-cleanup> -- . ':!vr'` is empty, then replace main (needs the human's push; it changes the published history of the personal fork).

Proposed stack (12 patches, file-grouped where possible so they replay cleanly):

| # | Patch | Main files | Approx. lines |
| --- | --- | --- | --- |
| 1 | ggml: op and backend interface additions (DSV4_HC_MIX op, event_query, CPU reference ops) | ggml.h, ggml.c, ggml-cpu/*, backend-impl.h | 200 |
| 2 | ggml-backend: scheduler input staging, share-view inputs, alloc layout cache | ggml-backend.cpp, ggml-alloc.c, sched-impl.h | 650 |
| 3 | ggml-backend-meta: TP lanes, token graph replay, repack whitelist, kv_rows src[5] | ggml-backend-meta.cpp | 1,650 |
| 4 | ggml-cuda: host-staged AllReduce extensions (graph counters, duplex, slabs) | allreduce.cu/.cuh, ggml-cuda.cu AR part | 450 |
| 5 | ggml-cuda: gfx906 repack (upload, layout, GEMM/GEMV, MoE) | q8_repack/*, getrows, ggml-cuda.cu dispatch | 5,000 |
| 6 | ggml-cuda: gfx906 MMQ/MMVQ (row-first traversal, J cap, IQ2_S two-row, S2 signs), mmid expert_bounds | mmq*, mmvq.cu, vecdotq, mmid* | 500 |
| 7 | ggml-cuda: fused kernels (norm/q8_1 producer, F2 hc-mix, P4 sgemm, concat, set_rows K1) | norm.cu, dsv4-hc.cu, sgemm-f32.cu, concat.cu, set-rows.cu | 900 |
| 8 | ggml-cuda: MoE routing and sampling kernels (B1 topk_moe_wave, top-k order, argsort wide rows, grouped MoE) | topk-moe.cu, top-k.cu, argsort.cu, flashnext-moe.cuh | 700 |
| 9 | ggml-cuda: chunked GDN with snapshots | gated_delta_net*.cu | 440 |
| 10 | ggml-cuda: kv_rows tile on AMD | fattn-*.cuh, fattn.cu | 150 |
| 11 | llama: split KV, recurrent rollback, taps, MTP/DFlash state, kv_rows selection, model integration | src/* | 2,400 |
| 12 | common/server/tools: speculative replay, DSpark/DFlash/MTP options, gguf-split fix, llama-bench | common/*, tools/*, examples/* | 1,050 |

Tests ride with their feature patch.

Estimated delta vs LLAMAHEAD:

| State | Non-doc lines added (approx.) | Commits/patches carried |
| --- | --- | --- |
| Now (vs edec462cc) | +16,426 / -995, 124 files | 77 commits since base (28 code) |
| After merge 46eb1742d (resolution adds about 0-20) | about +16,430 / -1,000 | 78 |
| After (a) + (c) | about +13,900 to +14,500 / -950, about 120 files | Option A: kit of 12 patches; Option B: 12 commits on 46eb1742d |
| After (b) items land upstream (long term) | about +13,000 to +13,600 | 10-12 |

Overlap with UNSLOTHHEAD: none of the fork's large features exist in unsloth pins. The only overlap is the MoE ids helper (fork expert_bounds overload vs unsloth #264) and the shape graph key (fork B15 vs unsloth #241).

### 3.7 A/B checks for the human

Use `build-current` as A and the candidate as B. Follow the memory rules: cool both GPUs below 60C before each run, alternate A,B,B,A, run each GPU program with `setsid --wait`, and report PP and TG for both models with old vs new.

Standard pair (both models). Run it once after the merge, once after the Part 3 cleanup, and for U1/U3:

```bash
source $THEROCK_ENV
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE LLAMA_KV_ROWS_DISABLE
HUB=$HF_HUB_CACHE
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
MFN=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
for B in "$A" "$CAND" "$CAND" "$A"; do
  setsid --wait timeout --kill-after=5s 300s "$B/bin/llama-bench" -m "$M27" \
    -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
    -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -d 0 -r 3 -o json
done
for B in "$A" "$CAND" "$CAND" "$A"; do
  setsid --wait timeout --kill-after=5s 300s "$B/bin/llama-bench" -m "$MFN" \
    -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
    -b 1024 -ub 512 -t 8 -p 2048 -n 128 -d 8192 -r 5 -o json
done
```

The Flash Next settings (`-b 1024 -ub 512 -t 8`, depth 8K) follow `vr/scripts/bench_multiuser.sh` and the @8K ledger rows. Unsloth replaced the IQ3_XXS file on 2026-11-04; use the cached snapshot above for both builds.

Which items need which checks:

| Item | Standard pair | Extra |
| --- | --- | --- |
| Merge 46eb1742d | yes, both models | `test-backend-ops -o MUL_MAT_ID`, `-o TOP_K`, `-o SET_ROWS`, `-o GATED_DELTA_NET`, `-o FLASH_ATTN_EXT`; `vr/scripts/bench_multiuser.sh "$OUT/multiuser" "$BASE" "$CAND"`; Flash Next MTP3 grouped-MoE output identity (expert_bounds feed it) |
| C1 peer AR archive | yes (27B required, Flash Next recommended) | TP smoke n=1/2048 |
| C2/C3 knob and experiment removal | yes, both | One MTP2 27B `vr/scripts` spec_accept run, because tap/DFlash knobs are in speculative code |
| C5 A2 knobs | PP2048 r3 27B, r5 Flash Next, with `LLAMA_SHEXP_SPLIT=0` and `GGML_SCHED_SHARE_VIEW_INPUTS=0` on B only (same build, ENV_B style) | - |
| U1 unsloth #264 port | yes; Flash Next PP2048 is the target, 27B is the control | `test-backend-ops -o MUL_MAT_ID`, `-o GET_ROWS`; Flash Next PP at `-ub 2048` too |
| U2 graph-key guard | build and TP smoke only | - |
| U3 shared MTP heads | no plain pair | Flash Next MTP3 at 100K: fit (VRAM per GPU), ms/step and acceptance vs the current sidecar preset |

---

## Recommended order

1. Merge 46eb1742d (Part 1), with the mmid `n_dumped` fix and the trap note. Validate and accept.
2. Cleanup commit set C1 (peer AR archive), C6, C7, (a). One validation pass.
3. C2/C3 knob and experiment removal, in small commits per file group, each archiving what it removes. One validation pass at the end.
4. C5 A2 measurement; drop knobs that show no effect.
5. U2 guard (tiny). Then U1 (#264 port) as a measured experiment for Flash Next PP. U3 if Flash Next MTP at long context is a priority.
6. Export the 46eb1742d patch kit as the 12-patch stack (Option A), or rebuild (Option B) if the human wants a short history.
7. Human-written upstream PRs for the (b) bug fixes: gguf-split, B12 getrows, AllReduce 16-byte fallback, wave64 ballot fix, then kv_rows on AMD.

## Unverified

- Open upstream/unsloth PRs matching fork features (`gh` not installed).
- Line savings for C2, C3 and C5 (depends on the code behind each non-default branch).
- Whether unsloth #264's NVIDIA-tuned gates help on gfx906, and whether its ballot code is wave64-safe.
- Whether U3 shared heads save memory under `-sm tensor` with the ROCm1 sidecar draft.
- The code behind unsloth's "HIP 64-bit warp masks" repin (bf063b959 only edits `pr-set.json`); ggml-org#24423 pin 604a809e is not fetched locally.
- Whether any `vr/server/models.ini` preset sets `LLAMA_DRAFT_PIPELINE`/`LLAMA_MTP_DRAFT_PIPELINE` (check before C3).
