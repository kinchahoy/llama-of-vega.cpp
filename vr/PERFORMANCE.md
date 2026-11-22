# Performance evidence

The one live performance document, ordered like the [process](PROCESS.md): what is fast now, the cost model with where each token goes, how to test, what worked, the routes that lost, and the worklist of priced ideas. Hardware constants are only in [HARDWARE-CARD.md](HARDWARE-CARD.md). Old dated runs are in [PERFORMANCE-RECORDS.md](PERFORMANCE-RECORDS.md) and are not edited. Ids (B1, S2, A4, SP1, Q3, TB2 ...) are kept from the old idea ledger.

Contents: [Current numbers](#current-numbers), [The model](#the-model), [How to test](#how-to-test), [What worked](#what-worked), [Routes that lost](#routes-that-lost), [Worklist](#worklist), [Retired claims](#retired-claims).

## Current numbers

2026-11-22 final run, two builds alternated: old `build-inkling-final-20261121` (`0ba4be7f1`) and new `build-final-20261122` (`ae32e5e58`). `llama-bench -r 2`, order old, new, new, old, each run started below 50C. Record: [2026-11-22](PERFORMANCE-RECORDS.md#2026-11-22-decode-attention-occupancy-two-launch-reductions-hardware-card). Start of the Inkling work for comparison (`build-u1sort-20261121`, 2026-11-21): IQ4_NL 564.2 / 36.28.

| Model | Config | PP2048 (old / new) | TG128 (old / new) | Build | Grade |
| --- | --- | --- | --- | --- | --- |
| 27B Q8_0 | 2 GPUs, `-sm tensor`, ub 512 | 614.7 / 614.0 | 37.35 / 37.96 | `build-final-20261122` | A (four alternated pairs) |
| Flash Next UD-IQ3_XXS | 2 GPUs, `-sm tensor`, ub 512 | 655.9 / 651.0 | 50.28 / 50.46 | `build-final-20261122` | A |
| Inkling 2 Mini NVFP4 + Q8_0 | ROCm0, ub 512 | 836.0 / 835.7 | 45.94 / 46.97 | `build-final-20261122` | A |
| Inkling 2 Mini IQ4_NL | ROCm0, ub 512 | 880.2 / 878.6 | 47.79 / 48.98 | `build-final-20261122` | B (one run each) |
| Inkling 2 Mini NVFP4, 4096-token context | ROCm0, `llama-batched-bench -npl 1` | 746 to 819 (not explained) | 38.19 / 43.12 | `build-final-20261122` | A |
| Inkling 2 Mini NVFP4, 16384-token context | ROCm0, `llama-batched-bench -npl 1` | 657.1 / 658.6 | 33.50 / 40.53 | `build-final-20261122` | B (one run each) |

To refresh: run the [after-merge validation](how-to-catchup.md#one-final-validation-pass) on the new build against the last good build, with the order and temperature rules below, replace this table, and add the full run as a new record in PERFORMANCE-RECORDS.md. Older rows (2026-10-29 to 2026-11-06) stay in the records file.

## The model

Token time = launches x launch cost + real kernel work. Every constant (launch cost, profiler cost, the one-block cost formulas, bandwidth, thermal drift, noise) is on the [hardware card](HARDWARE-CARD.md#constants) and is not repeated here. Kernels fall into three classes (see `vr/scripts/kernel_budget.py`): at the launch floor (only fewer launches help), one block and above the floor (apply the [one-block rule](HARDWARE-CARD.md#the-one-block-rule)), and many blocks above the floor (real work; compare bytes with the [mat-vec bandwidth](HARDWARE-CARD.md#constants)). The value of one removed launch is on the card; the "launch floor" rows below are launches x that value (C, derived). Grades: A alternated, B one trace or microbench. Raw data: `build-inkling-final-20261121/validation` (builds not in every worktree). Checks of the model against real kernels: [HARDWARE-CARD.md](HARDWARE-CARD.md#checks-of-the-card-against-real-kernels).

### Where the token goes

Per token, both GPUs summed for the kernel tables (traced, so times are trace times; see [Profiler caveats](#profiler-caveats)). Sources: 2026-11-21 traces (`build-banded-20261121/validation/ppl/prof-27b` and `.../prof-fn`) and the 2026-11-21 launch counts.

#### 27B Q8_0 (2 GPUs, tensor split)

| Item | Value | Grade |
| --- | --- | --- |
| Launches per token per GPU | 1,645 (3,290 on both GPUs, traced) | B, one trace |
| Launch floor | 3.6 ms of the 26.9 ms token, 13% | C, derived |
| GPU busy | 96-99% (kernels back to back) | B, one trace |
| `mul_mat_vec_q8_0_repacked<8, 8, false, 64, false>` | 761 calls, 30.5 us per call, 23.2 ms per token, 40% | B |
| `mul_mat_vec_q8_0_repacked<8, 8, false, 64, true>` (fused FFN) | 132 calls, 127.4 us, 16.8 ms, 29% | B |
| `ggml_cuda_ar_kernel` | 264 calls, 20.0 us, 5.3 ms, 9% | B |
| `add_rms_norm_mul_f32<1024>` | 264 calls, 11.3 us, 3.0 ms, 5% (before the two-launch form, see [What worked](#what-worked)) | B |
| `quantize_q8_1<32>` | 266 calls, 4.5 us, 1.2 ms, 2% | B |
| About ten kinds of tiny kernels of the recurrent layers, 99 calls each | about 1,000 calls, 4.5 us, 4.5 ms, 8% | B |
| What limits it | weight mat-vec: 69% of traced kernel time, at about 90% of the roofline; AllReduce kernel 9%; the reachable kernel gain is small | B |

#### Flash Next UD-IQ3_XXS (2 GPUs, tensor split)

| Item | Value | Grade |
| --- | --- | --- |
| Launches per token per GPU | 2,811 (5,621 on both GPUs, traced) | B, one trace |
| Launch floor | 6.2 ms of the 19.8 ms token, 31% | C, derived |
| GPU busy | 83-88%; an idle share exists and is unexplained | B, one trace |
| `mul_mat_vec_rp` (experts) | 510 + 92 calls, 15.4 us, 9.3 ms, 21% | B |
| `ggml_cuda_ar_kernel` | 197 calls, 25.8 us, 5.1 ms, 11% | B |
| `quantize_q8_1<32>` | 598 calls, 4.6 us, 2.7 ms, 6% | B |
| `mul_mat_vec_f<float, float, 1, 128>` | 344 calls, 6.9 us, 2.4 ms, 5% | B |
| `dsv4_hc_mix_down_f32<1, 5>` | 197 calls, 11.6 us, 2.3 ms, 5% | B |
| `dsv4_hc_mix_up_f32<2, 1>` | 197 calls, 10.7 us, 2.1 ms, 5% | B |
| `mul_mat_vec_q` (generic, two kinds) | 53 + 43 calls, 34 us, 3.3 ms, 7% | B |
| `unary_op_kernel`, `k_bin_bcast`, `scale_f32`, `dsv4_hc_mix_prep`, `dsv4_hc_post` | 370 + 274 + 227 + 197 + 197 calls, 4.2 us, 5.3 ms, 12% | B |
| What limits it | launch cost and some idle host time; experts 21%, AllReduce kernel 11%, hyper-connection mix about 11 us per call | B |

#### Inkling 2 Mini (ROCm0, one GPU)

| Item | Value | Grade |
| --- | --- | --- |
| Launches per token, start of 2026-11-21 | 7,219; floor 15.9 ms of a 27.5 ms token, 58% | C (floor derived) |
| Launches per token, after the fused ops (IQ4_NL, 47.7 t/s) | 4,128; floor 9.1 ms of a 21.0 ms token, 43% | C (floor derived) |
| GPU busy | 98% (kernels back to back; trace gaps zero at p99) | B, one trace |
| Captured HIP graphs | active for generation (debug log shows warmup complete, no unstable disable) | B |
| Value of a launch per layer pass removed | about 0.1 ms per token (0.5%; 48 passes x the launch cost) | C, derived |
| `flash_attn_ext_vec` with context | 15.6 us to 85.8 us per call at 4K context: 93% of the generation growth (32.8 t/s at 4K against 37.4 at 16 tokens, both traced; 36.4 t/s untraced); the bias chain kernels do not grow. Fixed on HIP by launch bounds (128, 4), see [What worked](#what-worked) | B, one trace pair, 2026-11-22 |
| Per-kernel table | none in the docs yet (gap): run `vr/scripts/kernel_budget.py` on a trace | - |
| What limits it | the launch cost and kernel time on the GPU queue, not the host | B |

### Other measured facts

- A fused kernel with one 256-thread block per token and a tree reduction per sum cost 26.7 us per call. Rewritten with a wide block and warp reductions it gave TG 43.57 to 47.73 on Inkling IQ4_NL (A). A one-block kernel that also did three weight matmuls took TG from 40.5 to 11.7 (A).
- Host `perf` is blocked on this machine (perf_event_paranoid=4). Use host timestamps or kernel gaps.
- Inkling 2 Mini output noise (KLD) is in [Quality checks](#quality-checks).

### Numbers that disagree

Several quantities appear with different values in older text. They measure different things (different denominators) or were not reconciled. Do not pick one without checking what it divides by.

| Quantity | Value | What it measured | Date | Source |
| --- | --- | --- | --- | --- |
| 27B share of the bandwidth roof | 75% | weight-stream time (28.7 GB at 650-715 GB/s, 20-21 ms) over the 27.2 ms token | undated text, cites the 2026-10-12 profile | old "What limits each model" text (see [records][rec-1029-x]) |
| | 77% | same weight-stream time over the unprofiled 26.70 ms budget token | 2026-11-10 | [TOKEN-BUDGET.md](TOKEN-BUDGET.md) |
| | 89-90% | repacked mat-vec kernel over the best pure stream (760-800 GB/s) at the real shapes | 2026-11-10 | [Q3 microbench](old-ignore/2026-11-10/exp-q3/README.md) |
| | about 90% | mat-vec over "the roofline", about 800 GB/s | 2026-11-21 | [PROCESS.md](PROCESS.md) |
| 27B mat-vec bandwidth | 650-715 GB/s | large Q8 mat-vecs per card | undated | [HARDWARE-GUIDE.md](HARDWARE-GUIDE.md) section 5, old limits text |
| | 715 / 685 GB/s | ROCm0 / ROCm1 mat-vec in the profiled critical path | 2026-11-10 | [TOKEN-BUDGET.md](TOKEN-BUDGET.md) |
| | 760-800 GB/s | best pure stream at the real shapes (a microbench ceiling, not the kernel) | 2026-11-10 | [Q3 microbench](old-ignore/2026-11-10/exp-q3/README.md) |
| | about 800 GB/s | reached by the repacked mat-vec (28.6 GB at 800 GB/s is 17.9 ms of 26.9 ms) | 2026-11-21 | [PROCESS.md](PROCESS.md) |
| 27B AllReduce share | 6% | critical-path exchange 1.6 ms of 26.7 ms | 2026-11-10 | [TOKEN-BUDGET.md](TOKEN-BUDGET.md) |
| | 9% | AllReduce kernel time (264 calls x 20 us, 5.3 ms), share of kernel time summed over both GPUs | 2026-11-21 | [PROCESS.md](PROCESS.md) |
| | 4-7 of 27 ms | AllReduce plus post-AR gaps plus peer wait | 2026-10 (A4 row) | A4 in [Worklist](#both-models-and-infrastructure) |
| | 2-2.5 ms | the part of that A4 could reach (exchange 1.62 ms plus part of ROCm1's small-kernel lag) | 2026-11-10 | A4 in [Worklist](#both-models-and-infrastructure) |
| 27B launches per token per GPU | 1595 | decode column, 27B at depth 0 | 2026-11-10 | [TOKEN-BUDGET.md](TOKEN-BUDGET.md) |
| | 1,645 (3,290 on both GPUs) | traced decode | 2026-11-21 | [PROCESS.md](PROCESS.md) |
| Flash Next launches per token per GPU | 2908 | decode at 8K depth, ROCm1 | 2026-11-10 | [TOKEN-BUDGET.md](TOKEN-BUDGET.md) |
| | 2,811 (5,621 on both GPUs) | traced decode | 2026-11-21 | [PROCESS.md](PROCESS.md) |
| | about 3280, at about 8 us each | load-excluded profile, before F2 variants | 2026-11-06 | [F2/P4 record][rec-1106-f2] |
| | 3282 -> 3115 | T1 WIP branch, before F2 | 2026-11-06 | T1 in [Worklist](#flash-next) |
| 27B cost of depth, 2K or 0 to 32K | +6.8 ms per token | llama-cli, plain TG 36.8 -> 29.4 | 2026-11-05 | [records][rec-1105-s] |
| | 3.17 ms per token | llama-bench depth 0 -> 32K, unprofiled (26.70 -> 29.87) | 2026-11-10 | [TOKEN-BUDGET.md](TOKEN-BUDGET.md) |
| Profiled TG slowdown | 21-45% | Flash Next 2026-10-29 traces | 2026-10-29 | see [Profiler caveats](#profiler-caveats) |
| | 7-24% | load-excluded profiles | 2026-11-06 | see [Profiler caveats](#profiler-caveats) |

## How to test

Short form. The process, its stages and budgets are in [PROCESS.md](PROCESS.md); the record format (with the prediction fields) is at the top of [PERFORMANCE-RECORDS.md](PERFORMANCE-RECORDS.md#record-format). The after-merge regression screen is in [how-to-catchup.md](how-to-catchup.md#one-final-validation-pass); it does not decide whether an optimization works. Scripts: [scripts](scripts/README.md).

### Stage 4 confirmation

Per batch of changes: unit tests for the touched ops, one short trace that compares measured with predicted per kernel (a mismatch above about 20% updates the [card](HARDWARE-CARD.md)), and one alternated end-to-end run per affected model. At most one build and about ten GPU-minutes per batch. Parameter tuning is not part of a batch; it is the last stage and needs owner permission ([Order of work](PROCESS.md#order-of-work)).

### Grades

Every performance statement in the docs carries one grade.

- **A**: alternated or repeated A/B on the real model, with numbers.
- **B**: single or one-order measurement, microbench, or a baseline from another time.
- **C**: estimate or derivation.
- **D**: assertion, no measurement found.

If the measurement behind a number cannot be found, grade it lower and say so. Do not use the words "validated", "proven", "always", "never", "cannot", "only", "big win", or the phrases "largest ... lever" and "honest ... total" for anything below grade A. For a lost experiment write "tried: this approach did not work", say what a different approach could try, and do not write "the idea is wrong".

### Status words

- **Worked**: in main, with the evidence and its grade.
- **Did not work**: one specific implementation lost a test that could show the effect. The row says what was tried, the numbers, why it lost (if measured), and what was not tried. A different approach may still win.
- **Not actually tested**: the change was not built, or just a premise was checked.
- **Hypothesis**: not tested, or a test could not show the effect (the change did not fire, wrong metric, one run inside the noise).
- **Retired**: a claim that later evidence contradicts or replaces. Listed so nobody believes it again.

### Test rules

A test marks an approach as "did not work" if it does all of these:

1. Confirms the change runs: kernel counts in a `rocprofv3 --kernel-trace`, a debug counter, or a changed output where one is expected. The scale-fold attempt measured a build where the fusion did not fire.
2. Measures the metric the change targets. AllReduce slab overlap is a PP change, so a TG run does not judge it.
3. Beats the noise: matched builds, same settings, baseline and candidate alternated A B B A. Start below 60C. On 2026-11-21 benchmarks drifted 15-30% when the GPUs were hot (B). The older notes said 27B TG varies about +-0.3 t/s between runs; that was measured on cool, rested GPUs. Flash Next TG ranged 33.3-40.1 t/s across one-repeat baselines on 2026-10-29 and Flash Next PP varied up to about 18 t/s (B), so for Flash Next use r5 alternating, or a profile that shows the change in kernel time.
4. For speculative modes, use [`spec_accept.sh`](scripts/spec_accept.sh) (llama-server, 5 prompts) and compare ms per verify step, not single-prompt llama-cli TG. A rounding change flips the greedy text and with it the acceptance: on 2026-11-06 one llama-cli prompt moved 27B MTP2 from 62 to 54 and Flash Next MTP3 from 53 to 63 with step time unchanged (B).
5. Check quality for changes that are not bit-exact (next section).

Keep the 50/50 split for A/B work. Ask for explicit OK before performance runs.

### Quality checks

- Greedy text equality is a smoke test, not evidence of equal quality. Speculative runs diverge from plain after about 1500-2000 bytes (batched verify flips near-tie tokens), so compare speculative runs to speculative runs.
- Use `llama-perplexity` and `llama-perplexity --kl-divergence` against the same build with the change off. Flash Next has a KLD noise floor of about 4.5e-3 with the same top token about 98.7% (B): harmless changes such as `-ub 256` or `-ts 0.52,0.48` reach it, and so does a 3e-7 relative change in one F32 matmul. Judge a change against that floor and PPL, not against 1e-3. Changes that act on small batches (decode, verify) need the KLD run at `-b 8 -ub 8`.
- Inkling 2 Mini: output moves by mean KLD 0.0049 when the ubatch size is the one change (512 vs 2048), and the same config twice gives KLD 0 (B, 2 chunks). Greedy-text equality and KLD or PPL differences of that size (for example PPL 2.8307 vs 2.8305) do not separate a bug from noise on that model.
- For an attention change also run a logit-level check (KLD), not only greedy text.

### Profiler caveats

- `rocprofv3` adds about 1.2 us to every kernel and shows a 4.2 us floor for a launch, while the cost in the app is about 2.2 us (B). Use traces for kernel counts and ranking, not absolute time.
- Profiled TG slowdown has two ranges that are not reconciled: 21-45% (older How to test text, from the 2026-10-29 Flash Next profile at 21% and the N2 trace at 31-45%; [2026-10-29 profile][rec-1029-prof], [N2][rec-1029-n2]) and 7-24% ([2026-11-06 F2/P4 record][rec-1106-f2], [TOKEN-BUDGET.md](TOKEN-BUDGET.md) method). Profiled PP changes by 2-4% (TOKEN-BUDGET.md). Treat profiled ms as an upper bound.
- Host `perf` is blocked on this machine (`perf_event_paranoid=4`). Use host timestamps around graph compute, or the gaps between kernels in a trace.

## What worked

Sorted by model, then by effect. Effects are quoted as the source has them. Grades follow the test rules above. "Switch" is the kill switch that turns the change off. "Origin" is where the idea came from and "Attempts" is how many failed tries came before it landed, both from [the history dataset](investigation-notes/2026-11-22-process-retro/history-retro.md) (attempts count only tries a record names; `unknown` where the history is squashed). Rows W05, W07, W10 and W13 were missing and were added on 2026-11-22 from the Oct 12 work log; the other 2026-10-12 items of the dataset (W06, W09, W11, W12, W15, W16) and the 2026-11-11 to 2026-11-15 items (W26, W27) are not added yet.

| Id | Change in plain words | Model | Measured effect | Grade | Origin of the idea | Attempts before landing | Switch | Record |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F2 | Fused hyper-connection mix op `GGML_OP_DSV4_HC_MIX` for 1-8 tokens (3 launches instead of 7) | Flash Next | TG128 @8K 40.8 -> 46.0 (+12%, 2 rounds); MTP3 100K step time 52.2 -> 49.5 ms; KLD at `-ub 8` 4.4e-3 (noise floor). Not tuned for T=4 (47 us per call). Order of the 2 rounds not given | B | profile (Fable review of our profiles) | 1 | `GGML_CUDA_HC_MIX_FUSED=0` | [F2/P4][rec-1106-f2] |
| P4 | Tiled split-K F32 GEMM for small-M matmuls on GCN (rocBLAS is slow for these) | Flash Next | PP2048 @8K 508.9 -> 563.8 (+11%), PP512 +12%, TG unchanged; KLD 4.7-5.1e-3 (noise floor). Dispatch guards still open | B | profile (Fable review) | 0 | `GGML_CUDA_SGEMM_F32=0` | [F2/P4][rec-1106-f2] |
| L-RPK | Repack Q5_0 and IQ4_XS on gfx906 | Flash Next | UD-Q3_K_XL @8K +2.3% TG, about +6% PP; UD-IQ4_XS +3.5% TG, +9% PP; PPL unchanged. Baseline PP had one outlier; native Q5_0 not timed separately | B | unknown | 0 | `GGML_CUDA_REPACK_CONVERT=0` | [big quants][rec-1106-q] |
| S2 | Cheaper IQ sign apply (`iq_apply_signs`) in IQ2_S mat-vec and IQ2_S/IQ3_XXS MMQ tile loads; bit-exact | Flash Next | r5 alternating +5.1% TG, +3.1% PP, same output. measured on the old UD-IQ3_XXS upload | A | fork-port (Strata review, idea only) | 0 | none named | [S2][rec-1106-s] |
| L-GM | Grouped MoE for 2-8 token verify, default on since 2026-11-06 | Flash Next | about 5.9% mean TG (Oct 25, MTP2); UD-IQ3_XXS at 100K: MTP2 48.9 -> 51.8, MTP3 49.1 -> 52.8; byte-identical output. Single runs. Applies to IQ2_S and IQ4_NL expert weights | B | unknown | 0 | `GGML_CUDA_FLASHNEXT_MOE=0` | [100K][rec-1106-m] |
| B1 | One-wave MoE routing kernel `topk_moe_wave_cuda<512,10>` (512 experts, top-10, rows <= 8); bit-exact | Flash Next | 14.9 vs 28.8 us per call; TG128 @8K 45.71 -> 47.53 (+4.0%), 4 alternating rounds, r5 | A | profile (Fable review) | 0 | none named (other shapes keep the old kernel) | [record](old-ignore/2026-11-10/play-b1/README.md) |
| L-IQ2 | gfx906 two-row IQ2_S one-token expert path | Flash Next | +2.37% TG in one matched pair; the note says "not repeatability evidence" | B | unknown | 0 | none named | [IMPROVEMENTS.md](IMPROVEMENTS.md) |
| U1 (R-CU1) | Unsloth #264 MoE id counting sort for 512 or more tokens | Flash Next | PP2048 at `-ub 2048` 877.4 vs 895.5 (+2.1%); 0.0% at `-ub 512`. Inside noise, not shown (the 18 t/s gap equals the stated Flash Next PP noise) | B | fork-port (unsloth #264) | 1 | `GGML_CUDA_MOE_COUNTING_SORT=0` | [catch-up 46eb1742d][rec-1121-u] |
| R-100K | MTP3 with a Q8_0 sidecar at 100K context on UD-IQ3_XXS | Flash Next | 52.2 / 52.5 TG, +26% vs plain; two llama-cli runs (llama-cli TG swings 10-18% with the greedy text). 5-prompt server mean MTP3 59.4 -> 62.7 with F2 | B | n/a (config) | n/a | none | [100K][rec-1106-m] |
| R-F2P4 | Result of F2 and P4 at long context | Flash Next, 27B | FN MTP3 at 100K 62.7 vs 27B MTP2 62.5 t/s; different models and acceptance | B | n/a (check) | n/a | none | [F2/P4][rec-1106-f2] |
| R-HEAD | Row-first MMQ traversal restored in the official-head port | Flash Next | MTP2 PP / TG 347.97 / 36.77 -> 351.30 / 39.51, identical output; the unrepaired port had PP -28%. Single pair | B | bug (matched run showed PP -28%) | 0 | none | [earlier checks][rec-1029-e] |
| K1 | One `set_rows` for the recurrent conv snapshots | 27B | MTP2 spec_accept 40.38 -> 38.95 ms/step (-3.5%), mean TG 62.4 -> 64.7, same acceptance. One pass each way; Flash Next not measured | B | profile (Fable review) | 0 | `GGML_K1_STATE_STORE=0`, `GGML_K1_SET_ROWS_VEC=0` | [F2/P4][rec-1106-f2] |
| L-AR | AllReduce slab overlap | 27B | PP2048 651 (off) -> 682 (4 slabs), text identical; TG neutral in the 2026-10-29 sweep. PP not re-measured on the rebuild tree | B | profile | 0 | `GGML_CUDA_AR_SLABS=0` (default 4) | [A1 sweep][rec-1029-a1] |
| L-MTP | MTP2 with `--spec-draft-backend-sampling` as the speculative default | 27B, Flash Next | 27B +68% vs plain, Flash Next +33%; one run each. Best depth flips between single runs (MTP2, MTP3, DFlash5) | B | differential (depth and mode sweep) | 0 | flags | [rebuild][rec-1029-rb] |
| R-32K | Speculative mode by prompt length | 27B | DFlash5 best at a 2K prompt (67.4 / 67.7), MTP2 best at 32K (42.7 vs 39.9); one or two llama-cli runs, crossover not measured | B | n/a (config) | n/a | flags | [27B sweep][rec-1105-s] |
| R-IN1 | All Inkling work of 2026-11-21 together | Inkling 2 Mini | IQ4_NL PP2048 564.2 -> 878.6 (+55.7%), TG128 36.28 -> 47.75 (+31.6%); NVFP4 490.8 / 30.5 (merge-alone build, not alternated) -> 836.3 / 45.80 (+70.4% / +50.2%) | A (NVFP4 old column B) | n/a (aggregate) | n/a | per change below | [late record][rec-1121-late] |
| R-IN4 | Banded relative-bias attention inside the tile kernel (port of unsloth #386 plus a gfx906 path), default on for prefill | Inkling 2 Mini | IQ4_NL PP2048 ub 512: 775.4 / 879.6 (+13.4%); ub 2048: 772.9 / 1135.7 (+47%); TG unchanged by design; 16K compute buffer 8,665 -> 3,170 MiB (B) | A | fork-port (unsloth #386) | 1 | `LLAMA_INKLING_BANDED_ATTN=0` | [late record][rec-1121-late] |
| R-IN6 | NVFP4 experts on the repacked path (layout variant of IQ4_NL, two F16 scales per 32 values) | Inkling 2 Mini | NVFP4 PP2048 637.8 / 737.3 (+15.6%), TG128 37.9 / 45.5 (+20%); costs +1,305 MiB VRAM | A | differential (NVFP4 against Q8_0 trace) | 0 | `GGML_CUDA_REPACK_NVFP4=0` | [late record][rec-1121-late] |
| R-IN3 | Wide-block `res_mix_f32` and `rms_norm_cat_f32` (were 26.7 us and 11 us per call) | Inkling 2 Mini | IQ4_NL TG128 43.57 -> 47.73 (+9.5%), PP2048 771.1 / 776.9 | A | profile | 0 | none for the rewrite (op switches `LLAMA_INKLING_MIX_FUSED=0`, `LLAMA_INKLING_CTRL_FUSED=0`) | [late record][rec-1121-late] |
| R-IN7a | `CLIP_ROWS` fused op (replaces a 9-kernel clip, 3 per layer) | Inkling 2 Mini | IQ4_NL TG128 36.07 / 39.15 (+8.5%), PP2048 565.3 / 567.6; mean of 2, off, on, on, off | A | profile | 0 | `LLAMA_INKLING_CLIP_FUSED=0` | [crash fix][rec-1121-c] |
| R-IN7b | `RES_MIX` fused op | Inkling 2 Mini | PP2048 568.8 / 751.6 (+32%), TG128 39.54 / 40.45 (+2.3%) | A | profile | 0 | `LLAMA_INKLING_MIX_FUSED=0` | [crash fix][rec-1121-c] |
| R-IN7c | `LOOP_GATE` + `RMS_NORM_CAT` fused ops (12 kernels to 5) | Inkling 2 Mini | PP2048 754.5 / 772.9 (+2.4%), TG128 40.47 / 41.43 (+2.4%) | A | profile | 1 | `LLAMA_INKLING_CTRL_FUSED=0` | [crash fix][rec-1121-c] |
| R-IN7d | `SCONV_RES` fused op (short convolutions, 4 taps) | Inkling 2 Mini | PP2048 774.0 / 777.7, TG128 41.56 / 43.26 (+4.1%) | A | profile | 0 | `LLAMA_INKLING_SCONV_FUSED=0` | [crash fix][rec-1121-c] |
| R-IN2 | Check that 27B and Flash Next did not change | 27B, Flash Next | 27B +0.2% PP and +0.2% TG, Flash Next +0.4% PP and +0.6% TG: no change | A | n/a (check) | n/a | - | [late record][rec-1121-late] |
| I-5 | Host-staged AllReduce is the default HIP path (no environment switch) | 27B | AGENTS.md quotes -21% PP2048 and -38% TG256 on two gfx906 for Qwen3.8 27B Q8_0 when HIP is excluded, 2026-09-29. No record of that run was found in vr/ | D (no record found) | upstream-port (PR 27825) | unknown | none | AGENTS.md |
| L-GR | Whole-token HIP graph replay | 27B, Flash Next, Inkling | 2026-10-29 trace shows two `hipGraphLaunch` per token (D, no effect number); graphs are active for Inkling generation (B) | D / B | profile | 0 | build flag `GGML_HIP_GRAPHS` | [rebuild][rec-1029-rb] |
| L-PD | Parallel lane dispatch | 27B, Flash Next | off costs "about 1.5 t/s TG on both models"; one repeat per run; inside noise, not shown (Flash Next TG noise is 33.3-40.1) | B | fork-port (mx-llama) | unknown | `GGML_META_PARALLEL_DISPATCH=0` | [A1 sweep][rec-1029-a1] |
| R-FLAG | Build flags `max-ilp` and `-O3` | 27B | a build without them was 2-3% slower in the 2026-10-29 rebuild | B | unknown | unknown | build flags | [rebuild][rec-1029-rb] |
| R-CU2-4 | Catch-up merges (4b77dee05, edec462cc, 46eb1742d) | 27B, Flash Next | no regression found: -0.1% / +0.4% (4b77dee05), Flash Next PP +1.3% (edec462cc, cause not verified), 27B 4x16K -2.4% / -1.8% was the cold first run (ABBA repeat); one run each | B | n/a (regression screens) | n/a | - | [4b77dee05][rec-1105-c], [edec462cc][rec-1118], [46eb1742d][rec-1121-u] |
| A3 | ROCm1 power cap raised | both | already at its maximum, 178 W (ROCm0 225 W), by sysfs read 2026-11-05; nothing to change | B | n/a (check) | n/a | - | [records](PERFORMANCE-RECORDS.md#2026-11-05-27b-config-sweep-and-32k-depth) |
| W05 | Fused add + rms_norm + mul with a Q8_1 producer at the post-AllReduce site (found in the Oct 12 work log, not in the old table) | 27B | part of TG 34.8 -> 36.2 (work log); CLI text identical | B | profile | 0 | none named | [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md) |
| W07 | Duplex copy-engine AllReduce (H2D on its own stream); a race fix for it (`src_read_done` event) landed the same day | 27B | PP 599 -> 647 (work log) | B | profile (P2a: AR PCIe traffic not overlapped) | 0 | none named | [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md) |
| W10 | Narrow Q8_0 mat-vec for MTP verify: planar activations, 16-lane rows, LDS-staged activations | 27B | pp8 57.0 -> 45.9 ms, pp6 48.7 -> 41.4, pp4 40.6 -> 37.2; df5 CLI 37.8 -> 38.4 (work log) | B | profile (rocprof) | 0 (unroll 2 dropped the same session) | none named | [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md) |
| W13 | Single-GPU DFlash draft with a private repacked head and async lag-one replay | 27B | 5-prompt mean TG 55.5 vs MTP2 52.3, PP unchanged (work log) | B | profile (GPU busy 67%, about 12 host gaps per step) | 1 (mirrored output head, slower at 43.1) | none named | [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md) |
| IK1 | `flash_attn_ext_vec` launch bounds (128, 4) on HIP: the kernel used more than 128 VGPRs and ran 2 blocks per compute unit | Inkling 2 Mini | NVFP4 TG at 4K context 38.3 -> 41.4 t/s, at 16K 33.6 -> 39.0; short context unchanged. Commit `efb85a1a5` | A at 4K, B at 16K | profile (trace pair: 93% of the context growth in this kernel) | 0 (a wrong mechanism, R-IN9, was retired the same week) | none | commit `efb85a1a5` |
| R-IN10 | Two-launch form of `res_mix` and `add_rms_norm_mul` at small row counts (partial sums, then many blocks), chosen by the one-block rule | Inkling 2 Mini, 27B | 27B Q8_0 TG128 37.35 to 37.96 (+1.6%, A, four alternated pairs below 50C), PP2048 614.7 / 614.0; Inkling 2 Mini NVFP4 TG128 45.94 to 46.97 (+2.2%, A), PP 836.0 / 835.7; Flash Next 50.28 / 50.46 (inside noise). Kernel time in a trace: `res_mix` 17.0 to 9.5 us, `add_rms_norm_mul` 11.37 to 9.05 us (B) | pending | profile + card (one-block rule) | 0 (27B-ARN tried three micro-variants earlier, see [Routes that lost](#routes-that-lost)) | none named | commit `ae32e5e58`; trace in [HARDWARE-CARD.md](HARDWARE-CARD.md#checks-of-the-card-against-real-kernels) |

## Routes that lost

Each entry is a ROUTE (an idea), not a verdict. A lost implementation does not close a route. Every entry gives the theory that predicted a gain, what was really built and how much effort it took, the result, why it lost (marked `tested`, `argued` or `conjecture`), the best-case bound for any implementation, and a status. Many attempts were one quick first version. Grades: A alternated A/B on the real model; B single or one-order measurement or microbench; C estimate; D assertion.

Status words: `OPEN - one implementation lost` (default); `NARROWED` (experiments ruled out named causes); `CLOSED BY BOUND` (the best-case bound is below the noise, or two independent competent implementations failed for the same understood reason); `NOT TESTED` (never built, or a premise check only); `SUPERSEDED BY A WIN` (a later implementation of the same route won; the history stays).

Reference numbers used in the bounds: a launch costs about 2.2 us in the app (A; 1.5 to 1.9 us saved per launch removed by a kill switch; the profiler shows 4.2 us); mat-vec reaches about 800 GB/s per GPU (B); token time and launches per token per GPU: 27B Q8_0 26.9 ms and 1,645, Flash Next 19.8 ms and 2,811, Inkling 2 Mini IQ4_NL 21.0 ms and 4,128 (B, one trace each). TG noise between runs: 27B about +-0.3 t/s when alternated and cool (0.8%); Flash Next about 1 t/s (2%); benchmarks drift 15 to 30% when the GPUs are hot. One thread block runs on one compute unit (1 of 64); a kernel that streams megabytes through one block is far below 800 GB/s (argued; a microbench is running, so this stays a conjecture until its result is added).

### Summary

Counts: 10 OPEN, 9 NARROWED, 6 NOT TESTED, 2 CLOSED BY BOUND, 1 SUPERSEDED BY A WIN (28 routes). Order: OPEN (largest bound first, by share of token time or gain), NARROWED, NOT TESTED, CLOSED BY BOUND, SUPERSEDED BY A WIN. Bounds are best cases for any implementation, not predictions.

| Id | Name | Status | Best-case bound | One-line reason |
| --- | --- | --- | --- | --- |
| L-CMOED | MTP draft experts on the CPU | OPEN | up to +17% TG (the MTP2 gain at 32K, B) | one config; CPU sync per step reasoned, not isolated |
| R-IN8 | One-block Inkling loop controller | OPEN | 2.3 ms per token (11% of 21.0 ms) | one block streams 2.4 MB of weights; a two-kernel form was not built |
| TB1 | Bitonic top-k for the QSA indexer | OPEN | 0.75 ms (3.8% of 19.8 ms), launches only (C) | one-row bitonic is slower; the fused radix route was not built |
| M1 | DFlash device feature rows | OPEN | about 2 ms of a 55 ms step (3.6%, C) | upstream path run as is; 200 MiB copy per step |
| L-Q81 | Multi-row planar Q8_1 producer | OPEN | 0.29 ms (1.1% of 26.9 ms) | cost moved into the norm kernel; no numbers kept |
| L-UB4K | Ubatch 4096 for 27B PP | OPEN | 0.3% of PP | one sweep point, cause not known |
| R-IN5 | Unsloth banded kernels alone (Inkling PP) | OPEN | realised by another kernel: PP +13.4% (ub 512), +47% (ub 2048), A | weak first port looked like a dead idea; a tile-kernel version won |
| L-SML | `-sm layer` on Flash Next | OPEN | no bound | one run, reason not recorded |
| L-O12 | Oct 12 small items (five) | OPEN | no bound | five items, three numbers, no reasons |
| L-NGM | n-gram speculation fallback | OPEN | no bound | one screen run without numbers; user decision on file |
| Q3 | Faster 27B Q8_0 mat-vec | NARROWED | 1.9 ms (7% of 26.9 ms) | more loads, more rows, k-split lose; block-contiguous read not built |
| A1-27B | Uneven `-ts` for 27B TG | NARROWED | about 1.1 ms (4%), profiled lag | both directions lose; lag is in small kernels the split does not scale |
| L-CPY | Avoid the HIP 2D memcpy | NARROWED | about 4% of MTP3 kernel time | the rect copies are tensor-split staging, not a hot path |
| N2a | Overlap the two lanes' graph launches | NARROWED | 27B idle share 1-4% (0.27-1.1 ms) | stagger is not on the critical path; Flash Next idle share open |
| N2b | Async staging of small F32 inputs | NARROWED | 0.4 ms (27B, C) | delta about 0 on both models; premise was profiler cost |
| A7 | Prefill GEMM occupancy | NARROWED | no bound | two variants with spills lose; spills not separated from reuse loss |
| L-D0 | MTP sidecar on ROCm0 | NARROWED | no bound | the +4% was acceptance; step 49.9 vs 49.5 ms |
| L-MDR | MTP device rows | NARROWED | no bound | step identical; disables deferred prefill (+18% TTFT) |
| L-NIGHT | ROCm nightly slows 27B MTP2 | NARROWED | no bound | step 40.28 vs 40.27 ms; drop was acceptance |
| SP1a | Backend sampling under `-sm tensor` | NOT TESTED | 3.7-7.4% per MTP2 step (C estimate) | aborted at load; no sampling ran |
| N3 | Two-half prefill AllReduce pipelining | NOT TESTED | +5.7% PP (exposed AR 5.4%, B) | nothing built |
| T1 | Fewer small launches in Flash Next decode | NOT TESTED | 0.88 ms (4.4% of 19.8 ms) | unfinished WIP, wrong output |
| F1 | Tensor-split host traffic off blit kernels | NOT TESTED | about 4% of MTP3 kernel time (B) | premise failed; nothing built |
| Q1 | Fuse AllReduce, add, norm, q8_1 | NOT TESTED | 0.8 ms (3.0% of 26.9 ms) | two partial variants; fold into the AR kernel not built |
| R-IN9 | Dense bias chain explains the slowdown with context | NOT TESTED | 0.26 ms (0.9%) of the growth | no implementation; explanation retired |
| N1 | Keep both DFlash draft graphs | CLOSED BY BOUND | 0.7 ms of 80 ms (0.9%) vs spread 4.8% | premise check: draft already replays |
| L-MGC | Content-keyed meta token graph for the draft | CLOSED BY BOUND | 0.7 ms of 80 ms (0.9%), from N1 | wrong output, then the premise was replaced |
| 27B-ARN | Rewrite `add_rms_norm_mul_f32<1024>` | SUPERSEDED BY A WIN | 0.94 ms (3.5% of 26.9 ms) | not arithmetic-bound, not input-bound; Q8_1 stage costs about 3 us. 2026-11-22: the two-launch form predicted by the one-block rule lowered the kernel from 11.37 to 9.05 us in a trace |

### OPEN routes

#### L-CMOED - MTP draft experts on the CPU
- Idea: put the draft head's expert weights in host memory so that MTP fits next to a large quant at long context.
- Why it should help (theory): fit. The plan was to free GPU memory; no speed theory is stated. The reachable gain is the MTP2 gain when the full draft sits on the GPU: 51.5 vs 43.9 plain at 32K, +17% (B, single runs).
- What was actually built: the existing `-cmoed` switch, one configuration (UD-Q3_K_XL, 64K, Q4_0 draft on ROCm1, experts on CPU, `-ts 0.505,0.495`), single runs. No variant, no tuning, no profile of the CPU step. Archive: the big-quants record.
- Result: MTP2 34.9 vs plain 43.8 TG (B).
- Why it lost: conjecture. The record says CPU sync per draft step costs more than speculation gains (reasoned); a related claim that CPU expert layers cost 20-25% was retired because quant and placement changed together (C).
- Best-case bound: up to the +17% measured at 32K with a full GPU draft (51.5 vs 43.9, B); the CPU step adds cost, so the real gain is lower.
- Route status: `OPEN - one implementation lost`.
- A better attempt would: put part of the draft experts on the CPU, or overlap the CPU expert work with the target's verify step, and time the CPU step on its own first. Source: [big quants][rec-1106-q].

#### R-IN8 - One-kernel loop controller for Inkling 2 Mini
- Idea: fuse the 12-kernel controller chain of the Inkling loop so that fewer launches run per looped pass.
- Why it should help (theory): 11 launches removed x 2.2 us = 24 us per controller (launch cost A, count B). Over 4 controllers x 24 looped passes that is the bound below.
- What was actually built: `GGML_OP_LOOP_CTRL`, one 256-thread block per token (used for 1-8 tokens) doing two norms, 64 gate dot products one after another with a tree reduction each, and two 64x2304 projections. First attempt, one variant, no tuning, alternated A/B on the real model. Tests passed (`test-backend-ops -o LOOP_CTRL` 2/2, greedy text equal for about 45 tokens). Archive: [notes](old-ignore/2026-11-21/loop-ctrl-fused/README.md), [patch](old-ignore/2026-11-21/loop-ctrl-fused/loop-ctrl-fused.patch).
- Result: TG128 40.5 -> 11.7 t/s (A); PP2048 737.5 / 751.5, unchanged (the path is off for large batches).
- Why it lost: argued. The kernel streams about 2.4 MB of F32 weights through one block. At a one-compute-unit share of bandwidth (800 GB/s / 64 = 12.5 GB/s) that is about 190 us per controller, against a saving of 24 us. Measured cost: (85.5 - 24.7) ms / 96 controllers = about 630 us per controller. The 64 sequential reductions would explain the rest (conjecture; the one-compute-unit share is a conjecture until the microbench result is added).
- Best-case bound: 11 launches x 4 controllers x 24 looped passes x 2.2 us = 2.3 ms per token, about 11% of 21.0 ms.
- Route status: `OPEN - one implementation lost`. The one-block form with the matmuls inside it is not worth repeating (argued: its cost is about 3 times the 190 us streaming estimate and about 26 times its saving). A later form that kept the three matmuls and fused only the elementwise parts gave PP +2.4% and TG +2.4% (A); it is in [What worked](#what-worked). The rule for CLOSED BY BOUND is not met (a bound of 11% is far above noise), so the route stays open.
- A better attempt would: use two kernels, a block per gate value (64 blocks) and then a block per output slice, so the matmuls stay parallel; microbench the two-kernel form first (IK4 in [Worklist](#inkling-2-mini)).

#### TB1 - Bitonic top-k for the QSA indexer
- Idea: replace the 12 radix launches of the decode top-k with a cheaper top-k.
- Why it should help (theory): the 12 radix launches cost 70-90 us per call (B) and the QSA indexer is 2.5 ms per token (3.0 at 32K) in the 2026-11-10 budget (B, profiled). The archive states no prediction for the bitonic kernel; the implied reasoning is one launch in place of 12.
- What was actually built: an existing bitonic path (`276834fdb`) enabled for one row up to 16,384 columns; a 31-line patch; the row sits in shared memory padded to a power of two. One variant, default parameters, microbench only (`test-backend-ops perf -o TOP_K`, six widths); the model A/B was not run. Archive: [record](old-ignore/2026-11-10/play-tb1/README.md), [patch](old-ignore/2026-11-10/play-tb1/bitonic-one-row.patch).
- Result: radix 71-87 us against bitonic 131 (2112 columns), 625 (8256), 883 us (16384) (B).
- Why it lost: not known. Conjecture: one row is one thread block doing O(n log^2 n) compare steps on one compute unit, the same one-block limit as R-IN8.
- Best-case bound: launches only, about 11 launches x 2.2 us = 24 us per call; if the 2.5 ms were all top-k at about 80 us per call that is about 31 calls per token, so 31 x 24 us = 0.75 ms, 3.8% of 19.8 ms (C; assumes the 2.5 ms is all top-k and mixes a profiled and an unprofiled base).
- Route status: `OPEN - one implementation lost`; route 2 (a fused radix kernel) was not built (TB1 route 2 in [Worklist](#flash-next)).
- A better attempt would: keep the radix work and fuse the 12 passes into one single-block kernel (the row is 8-64 KB, so it may fit in LDS), then time it against the 12-launch version.

#### M1 - DFlash device feature rows
- Idea: keep the DFlash feature rows on the device so that the host copy of the tap rows disappears.
- Why it should help (theory): the fork host path saves a small D2H/H2D tap copy; the archive expects a gain below 1-2 ms of a 55 ms step (C).
- What was actually built: nothing new. The upstream switch `LLAMA_DFLASH_DEV=1` was run on 27B as is, one run, shared host (plain TG was 16.7-17.4 t/s instead of about 37, so compare ms/step only). Archive: [record](old-ignore/2026-11-10/exp-spec/README.md).
- Result: 128.07 vs 55.03 ms/step, 2.3x slower (B).
- Why it lost: argued. Under `-sm tensor` the buffer is a mirrored meta tensor; the draft runs on ROCm0 alone, so the scheduler copies the whole 200 MiB (2048 x 5 taps x 5120 x F32, about 210 MB) to ROCm0 for every draft decode. Found by reading the code; no experiment isolates it.
- Best-case bound: about 2 ms of a 55 ms step (3.6%, C, from the archive); DFlash5 reps agreed within 0.3 ms (0.5%).
- Route status: `OPEN - one implementation lost`.
- A better attempt would: expose the ROCm0 simple tensor of the mirrored meta tensor as the feature source (today `ggml_backend_meta_buffer_simple_tensor` is static) and remove the draft's host wait on the target's written event.

#### L-Q81 - Multi-row planar Q8_1 producer (Oct 12)
- Idea: quantize several activation rows to Q8_1 in one launch for the narrow-batch (MTP verify) mat-vec.
- Why it should help (theory): the work log says it halves the quantize launches; no number is given.
- What was actually built: one variant, a multi-row planar producer for 2-8 rows; it was reverted. A planar quantize kernel that replaces quantize + convert was kept. No numbers, no patch kept. Archive: [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md), lines 33 and 50.
- Result: "halved quantize launches, norm kernel slower by the same amount"; no numbers (D).
- Why it lost: argued, from a later measurement. A separate quantize costs 4.2 us and the fused epilogue costs 3.5 us inside the norm kernel (7.1 + 4.2 = 11.3 us against 10.6 us fused, [Q1 record](old-ignore/2026-11-10/exp-q1/README.md), B), so moving the quantize into the norm kernel moves the cost rather than removing it.
- Best-case bound: for plain 27B TG, 133 `quantize_q8_1` launches per GPU (266 both GPUs, B) x 2.2 us = 0.29 ms, 1.1% of 26.9 ms, if the added kernel time were zero. This is above the 0.8% noise but small.
- Route status: `OPEN - one implementation lost`.
- A better attempt would: write Q8_1 from the producer's own epilogue where the data is already in registers, in place of a separate norm-block step (see the playbook q8_1 items in [Worklist](#flash-next)).

#### L-UB4K - `-b/-ub 4096` for 27B PP
- Idea: a larger ubatch reads the weights fewer times per prompt.
- Why it should help (theory): no stated theory. A weight-reuse argument: per GPU about 14.3 GB of weights are read once per ubatch, 17.9 ms at 800 GB/s, which is 0.6% of the 2.99 s that 2048 tokens take at 684 t/s (C).
- What was actually built: one `llama-bench` sweep point, `-b 4096 -ub 2048,4096`, r3. No profile of why. Archive: [27B sweep][rec-1105-s].
- Result: PP4096 r3: 606.3 +- 3.2 vs 684.1 +- 1.6 at ub 2048 (-11.4%) (A).
- Why it lost: not known.
- Best-case bound: doubling the ubatch halves a 0.6% weight-read share, so at most about 0.3% of PP (arithmetic above, C); r3 spread is 0.2-0.5%.
- Route status: `OPEN - one implementation lost`; the bound is small, so the route has little left to give.
- A better attempt would: nothing obvious for PP speed; if the larger ubatch is wanted for another reason, profile which buffer or kernel grows.

#### R-IN5 - Unsloth banded attention kernels alone for Inkling 2 prefill
- Idea: compute the relative bias inside the attention kernel instead of building a dense bias.
- Why it should help (theory): the dense bias chain is about a quarter of prompt kernel time (B, one trace).
- What was actually built: a port of unsloth #386 with its scalar and per-query kernels on HIP, because its fast prefill kernel needs NVIDIA matrix units. First port, one variant, alternated A/B. Record: [late record][rec-1121-late].
- Result: PP2048 761 -> 383 t/s (-50%), TG 47.3 / 47.5 (A).
- Why it lost: argued. Prompt attention is real work and the fallback handles one query at a time. PP2048 at 761 t/s is 2.69 s and at 383 t/s is 5.35 s, so 2.66 s extra per 2048 tokens against about 0.16 s of attention before (the 0.16 s figure came with the task; I did not find it in a record). That is roughly 18 to 20 times slower than the fork's tile kernel on the attention itself.
- Best-case bound: realised by another kernel. The same day a bias lookup inside the fork's own tile kernel gave PP +13.4% at ub 512 and +47% at ub 2048 (A), now default (see [What worked](#what-worked)).
- Route status: `OPEN - one implementation lost`. This is the example of a weak implementation that looked like a dead idea: the idea won with a different kernel.
- A better attempt would: nothing more for prefill. For decode see R-IN9 and IK1 in [Worklist](#inkling-2-mini).

#### L-SML - `-sm layer` instead of tensor split on Flash Next
- Idea: run layers split across the GPUs and skip the AllReduce.
- Why it should help (theory): no stated theory. The AllReduce kernel is 11% of Flash Next kernel time (B, one trace), which a layer split would remove.
- What was actually built: one `llama-bench` run with `-sm layer`, default parameters. Archive: [A1 sweep][rec-1029-a1].
- Result: 30.99 vs 38.94 TG (-20%), one run, outside the noise (B).
- Why it lost: not known. Conjecture: with a layer split one GPU works at a time, so the weight stream per token is not halved per GPU.
- Best-case bound: no bound (the AllReduce share is 11% of summed kernel time, but the cost of serial layers was not measured).
- Route status: `OPEN - one implementation lost`.
- A better attempt would: nothing obvious; a layer split with two requests in flight (a pipeline) is a different use case.

#### L-O12 - Oct 12 small items (five)
- Idea: five separate small ideas, run in one session: MTP fused catch-up, `nc` mat-vec unroll 2, `GGML_RP_NC_FUSION` (fused up/gate in the narrow mat-vec), MTP `p-min`, DFlash mirrored output head.
- Why it should help (theory): no stated theory in the log for any of them.
- What was actually built: each was one quick variant in one afternoon, mostly reverted, no patches kept. Numbers exist for three: MTP fused catch-up TG 51.0 vs 51.8 with the same acceptance (catch-up was already overlapped with host sampling); `p-min` TG 55 -> 29-31 (mtp3_p5 30.7, mtp4_p6 28.6); `GGML_RP_NC_FUSION` "no gain". Unroll 2 "worse, dropped"; the DFlash mirrored output head has no entry in the log. Archive: [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md).
- Result: as above, grade D (single runs in a log).
- Why it lost: not known for `p-min` (a drop of that size suggests a defect in the switch; conjecture). For the catch-up: argued from the log, the work was already hidden behind host sampling.
- Best-case bound: no bound.
- Route status: `OPEN - one implementation lost` (five routes in one entry; none was tuned).
- A better attempt would: take them one at a time with a kill switch and an alternated A/B; check `p-min` for a defect before reading it as a result.

#### L-NGM - ngram-mod speculation as a fallback when no MTP head is loaded
- Idea: use n-gram drafting when a model has no draft head.
- Why it should help (theory): no stated theory.
- What was actually built: one screen (n512, Oct 12) of `ngram-mod` alone. No numbers, no archive beyond one line in the [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md).
- Result: "ngram-mod alone = plain speed" (D).
- Why it lost: not known. The decision was the user's (rejected 2026-10-29; the record says do not propose it again), and that decision stands.
- Best-case bound: no bound.
- Route status: `OPEN - one implementation lost`; kept open here because no experiment closed it. The user decision is on file.
- A better attempt would: nothing obvious unless the user reopens it.

### NARROWED routes

#### Q3 - Faster 27B Q8_0 mat-vec
- Idea: close the gap between the repacked Q8_0 mat-vec and the streaming ceiling.
- Why it should help (theory): the best pure stream at the real shapes is 760-800 GB/s and the kernel is at 715 / 685 GB/s, so up to 10-11% headroom (B, microbench).
- What was actually built: a standalone microbench (`mvbench.hip`, real kernel, 16 layers, 108 mat-vecs, 5 passes) with variants: 2 to 5 loads in flight, 4 and 16 rows per block, k-split 2/4/8 (all but k-split bit-identical), a row-shaped stream with no math; plus one end-to-end A/B of `GGML_RP_MMV_ROWS` 8 vs 4 (alternating r3). The block-contiguous read with an LDS per-row re-reduce was tested as a stream, not as a kernel. Archive: [record](old-ignore/2026-11-10/exp-q3/README.md), [microbench](old-ignore/2026-11-10/exp-q3/mvbench.hip).
- Result: best pure stream 760-800 GB/s, kernel at 89-90% of it; 2-5 loads in flight -1 to -18%, 4 / 16 rows -1 / -70%, row split -3 to -40%; `GGML_RP_MMV_ROWS=4` 37.52 vs 37.45 TG (+0.3%, A, noise) (B for the microbench rows).
- Why it lost: tested for load depth and the dp4a math: a row-shaped stream with no math reaches 770-797 GB/s against the real kernel's 775 on the head matrix, so the gap is the one-row-per-wave access pattern and launch ramp. More loads and rows cost VGPRs and occupancy (argued from the numbers; U=5 590 vs 715 GB/s).
- Best-case bound: about 1.9 ms per token (C, the archive's figure), 7% of 26.9 ms; against 0.3 t/s noise (0.8%) it is above noise.
- Route status: `NARROWED`: more loads in flight, more rows per block, k-split and the math itself are ruled out as the cause of the gap. The block-contiguous read is untested.
- A better attempt would: build the contiguous per-block read with an LDS re-reduce as a real bit-exact kernel in `mvbench.hip` (Q3 route 2 in [Worklist](#27b-q8_0)). The knob `GGML_RP_MMV_ROWS` was removed 2026-11-21; the [knobs archive](old-ignore/2026-11-21/knobs/) has the patch.

#### A1-27B - Uneven `-ts` for 27B TG
- Idea: give the faster GPU (MI60, ROCm0) a larger share.
- Why it should help (theory): no stated theory. The older claim that the MI60 is 10-20% faster has no source (D, retired); a profile gives ROCm1 busy time 1.09 ms above ROCm0 per token (B, profiled).
- What was actually built: a sweep of five splits with `-ts` (0.52/0.48 with two retests, 0.48/0.52, 0.56/0.44; Oct 12: 53/47 and 56/44), single or r2 runs, no profile of the split and no change of the small kernels. Archive: [A1 sweep][rec-1029-a1].
- Result: 0.52/0.48 about -1 t/s (initial and two r2 retests: 36.20, 36.06, 36.16 vs 37.42, 37.13, 37.01), 0.48/0.52 -1.9 (35.52 vs 37.39), 0.56/0.44 -2.5 (34.92 vs 37.40); Oct 12: 53/47 35.6 and 56/44 34.8 vs 36.8 at 50/50, with PP 702.2 and 704.1 vs 683.6 (+2.7%, +3.0%) (A for TG).
- Why it lost: not known. Conjecture: the lag is in small kernels that both ranks run whole, so a split of the weights does not scale it.
- Best-case bound: the lag, about 1.1 ms of 26.9 ms (4%, B profiled; half of the profiled wait was profiler cost). A split does not divide the small kernels (conjecture).
- Route status: `NARROWED`: shifting work in either direction lowers TG, so imbalance in the weight work is not the cause.
- A better attempt would: target the cause, the small kernels on ROCm1 (K2, P3 in [Worklist](#27b-q8_0)); an uneven split for PP is a separate test (+3% PP for -5% TG on Oct 12).

#### L-CPY - Avoid the HIP 2D memcpy in `ggml_cuda_cpy`
- Idea: use copy kernels in place of `hipMemcpy2D` for rect copies.
- Why it should help (theory): a profile showed `__amd_rocclr_copyBufferRectAligned` at 15.6% of PP and 19% of MTP3 kernel time (B).
- What was actually built: copy kernels in place of the 2D memcpy, alternated r2 x2 on Flash Next UD-IQ3_XXS at 8K. Archive: [100K record][rec-1106-m].
- Result: PP 515.5 vs 515.4, TG 40.53 vs 40.37 (A).
- Why it lost: tested. A load-excluded profile showed the time was the load-time weight upload (`set_tensor_2d`); runtime rect copies are about 4% of MTP3 kernel time (B, [F2/P4 record][rec-1106-f2]). The rect copies come from tensor-split staging (argued).
- Best-case bound: about 4% of MTP3 kernel time (B); not below Flash Next noise (2%), so it is not a closure.
- Route status: `NARROWED`: the hot-path premise is ruled out; see F1 for the staging traffic.
- A better attempt would: avoid the tensor-split staging copy at its source (F1).

#### 27B-ARN - Rewrite `add_rms_norm_mul_f32<1024>` (27B, 264 calls per token, 11.3 us in the trace)
- Idea: make the fused add + norm + mul kernel as cheap as a plain row norm.
- Why it should help (theory): the data it touches is about 85 KB (0.1 us at full bandwidth), so the excess over the 4.2 us trace floor looked structural, as in `res_mix_f32` (26.7 us to about 5 us after a rewrite; TG +9.5%, A). Grade C by analogy.
- What was actually built: three micro-variants in about 30 minutes by one agent, profiled under `rocprofv3`, not benchmarked end to end; the patches were not kept (a gap; see the [note](old-ignore/2026-11-22/add-rms-norm-variants/README.md)). This is a quick, incomplete attempt. Earlier, Q1 also tried an unrolled epilogue (12.6 / 13.2 us).
- Result (B): registers 512x10: 14.8 us; float4 per thread 256x8 with branch-free loads: 11 to 13.5 us; same with a reciprocal in place of the divide: 10.7 to 11.5 us; baseline 11.3 us; without the Q8_1 output: 8.2 us.
- Why it lost: the premise was wrong on inspection: the kernel already used one wide block, warp reductions and one barrier, and the 4.4 us comparator (`rms_norm_f32<float, 32, ...>`) is a different instantiation doing a different job. Tested: not arithmetic-bound (divide vs reciprocal, no change), not input-volume-bound (reading a for b, no change); the Q8_1 stage costs about 3 us. Conjecture: a one-block kernel is limited by one compute unit.
- Best-case bound: (11.3 - 4.2) us x 132 calls per GPU = 0.94 ms per token, about 3.5% of 26.9 ms, if it reached the trace floor. Above the 0.8% noise.
- Route status: `SUPERSEDED BY A WIN` (2026-11-22). The two-launch form predicted by the one-block rule lowered the kernel from 11.37 to 9.05 us in a trace, see [What worked](#what-worked) (R-IN10, commit `ae32e5e58`). Earlier status, kept for the history: `NARROWED`: arithmetic and input volume are ruled out.
- A better attempt would: first measure the cost of a one-block kernel against the elements it touches (microbench running), then try the Q8_1 stage as a wave per row group or as part of the consumer. Related: Q1, and the 27B row in [Worklist](#27b-q8_0).

#### N2a - Overlap the two lanes' graph launches
- Idea: launch lane 1's graph from a worker thread so that lane 1 does not start about 2 ms after lane 0.
- Why it should help (theory): in a trace `hipGraphLaunch` blocked about 2.0 ms per lane, serial, and ROCm1 started about 2 ms after ROCm0 (B, profiled; the profiled TG was 25.3 against 36.6 unprofiled).
- What was actually built: launching lanes 1.. on the existing dispatcher worker threads (a 53-line diff); output byte-identical; one pair, r2. Archive: [diff](old-ignore/2026-10-29/tg-host-gap/parallel-launch-attempt.diff), [diagnosis](old-ignore/2026-10-29/tg-host-gap/diagnosis.md), [N2][rec-1029-n2].
- Result: 27B 36.63 +- 0.31 -> 36.67 +- 0.17; Flash Next 39.14 -> 38.78 (-0.4) (B, one pair).
- Why it lost: conjecture. In a profile the launches overlap but each slows to 1.2 / 2.5 ms (runtime lock contention); the record suggests the stagger is not on the critical path unprofiled. Consistent with the later trace: 27B is 96-99% GPU-busy.
- Best-case bound: the idle share. 27B 1-4% of 26.9 ms = 0.27-1.1 ms (B, one trace). Flash Next is 83-88% busy, so 12-17% idle (2.4-3.4 ms of 19.8 ms) whose cause is unexplained; no bound for it.
- Route status: `NARROWED`: the serial launch stagger is ruled out as the source of the 27B idle time. The Flash Next idle share is open (TB5 in [Worklist](#flash-next)).
- A better attempt would: measure the Flash Next idle share with host timestamps around graph compute (TB5).

#### N2b - Stage the remaining small F32 inputs asynchronously
- Idea: stage all pre-launch input copies through the async path, so that the host does not sync before each launch.
- Why it should help (theory): about 14 copies and 5 syncs per token on 27B and about 70 copies and 10 syncs on Flash Next before the launch (B, profiled); the archive estimates 0.4 ms (1.5%) on 27B and 1.5-2.5 ms (4-6%) on Flash Next as upper bounds (C).
- What was actually built: no code. The existing knob `GGML_SCHED_INPUT_STAGING_F32_MAX=1048576` stages every F32 input; TG128 r3, alternating, both models. Archive: [option 1](old-ignore/2026-10-29/tg-host-gap/option1.md); the knob was removed 2026-11-21 ([archived](old-ignore/2026-11-21/knobs/)).
- Result (B, alternating r3): 27B 37.09, 37.13 vs 37.11, 37.13; Flash Next 38.43, 38.66 vs 38.45, 38.12.
- Why it lost: conjecture. The record concludes the meta `set_tensor` path is not the cost; the original gap was largely profiler cost (see [Retired claims](#retired-claims)).
- Best-case bound: 0.4 ms on 27B (C, from the old profile whose gap was mostly profiler cost, so high); the 0.3 t/s noise is about 0.2 ms.
- Route status: `NARROWED`: the pre-launch host copies are ruled out as the critical path on both models (delta about 0).
- A better attempt would: batch the input tensors into one staging buffer and one upload per lane, or pipeline the next token's inputs (a design change); test after TB5 shows a host idle share on Flash Next.

#### A7 - Higher occupancy in the repacked Q8_0 prefill GEMM
- Idea: more waves per SIMD to hide latency in the prefill GEMM.
- Why it should help (theory): no stated theory in the archive; the implied premise is a latency-bound kernel.
- What was actually built: two variants behind `GGML_RP_GEMM_VAR`: V1 64x64 tile, 256 threads, `launch_bounds(256,3)`, 84 VGPR with 15 spilled; V2 64x64, 512 threads, `launch_bounds(512,4)`, 64 VGPR with 60 spilled; against the baseline 64x128 tile at 126 VGPR. Standalone harness `rpgemm.cpp` on 20 shapes, bit-identical, 3 rounds within 1%; `llama-bench` not run. Archive: [record](old-ignore/2026-11-10/play-a7/README.md), [patch](old-ignore/2026-11-10/play-a7/gemm-variants.patch).
- Result: GEMM n 2048, k 5120: V1 24-31% and V2 37-57% slower on both GPUs (B); baseline about 26 TOPS (ROCm0), 22.7 (ROCm1).
- Why it lost: conjecture. The archive reads it as VALU and LDS-issue bound with half the weight reuse and spills; no experiment separates the spills from the lost reuse.
- Best-case bound: no bound (the GEMM is 76% of the 27B PP critical path, B, but no peak was derived).
- Route status: `NARROWED`: more waves per SIMD at 64x64 tiles with spills is ruled out; a spill-free occupancy gain is untested.
- A better attempt would: raise occupancy without spilling (same tile, fewer registers by a different register blocking), or go the other way: larger tiles (TN 3-4) with a shorter BK, as the archive says.

#### L-D0 - Q8_0 MTP sidecar on ROCm0 instead of ROCm1
- Idea: put the draft on the other GPU.
- Why it should help (theory): an earlier single-prompt run showed +4%; ROCm0 is the card with the faster GEMM (21%, B). No further theory stated.
- What was actually built: `-devd ROCm0` with `-ts 0.46,0.54` (the balanced `-ts 0.50,0.50` runs out of memory), 5-prompt server run with F2, one pass. Archive: [F2/P4][rec-1106-f2], [placement data](old-ignore/2026-11-06/f2-p4-accept/draft-placement/).
- Result: 49.9 vs 49.5 ms/step (+0.9%), acceptance 0.737 vs 0.721 (B).
- Why it lost: argued. The earlier +4% was an acceptance effect; step time is equal within noise.
- Best-case bound: no bound.
- Route status: `NARROWED`: placement speed is ruled out at this memory split; a split that fits is untested.
- A better attempt would: nothing obvious unless more VRAM is free on ROCm0.

#### L-MDR - MTP device rows (`LLAMA_MTP_DEV_ROWS=1`) on 27B tensor split
- Idea: keep the MTP hidden rows on the device so that no host copy runs per step.
- Why it should help (theory): no stated theory (an upstream path).
- What was actually built: nothing new; the upstream switch, two runs each, alternated. Archive: [record](old-ignore/2026-11-10/exp-spec/README.md).
- Result: 40.76 / 40.39 vs 40.81 / 40.34 ms/step (+0.0%), same output; TTFT +18% (B).
- Why it lost: tested. The paths engage (log: `target h rows on the device: yes`) and the step is unchanged, so the host row copy is not on the critical path; deferred prefill is switched off, which raises TTFT.
- Best-case bound: no bound.
- Route status: `NARROWED`: the host row copy is ruled out as a cost of the MTP2 step on 27B.
- A better attempt would: nothing obvious; combining deferred prefill with device rows can at best match the default.

#### L-NIGHT - ROCm 10.3.0a20261105 nightly slows 27B MTP2
- Idea: a suspected regression, not a speedup: the new compiler might slow 27B.
- Why it should help (theory): not applicable; the suspicion came from llama-cli 64 -> 55 TG.
- What was actually built: the same source built with the old and the new clang, one pass each. Archive: [F2/P4][rec-1106-f2].
- Result: 27B MTP2 40.28 vs 40.27 ms/step, PP2048 712 vs 705, TG 37.8 vs 37.7; Flash Next MTP3 51.4 vs 49.5 (new faster) (B).
- Why it lost: tested. The drop was acceptance on a different greedy text (0.816 vs 0.777), step time equal.
- Best-case bound: no bound (a regression check).
- Route status: `NARROWED`: a compiler slowdown is ruled out for 27B at this noise level.
- A better attempt would: nothing; recheck after the next ROCm update.

### NOT TESTED routes

#### SP1a - Backend sampling under `-sm tensor`
- Idea: sample on the GPU with the vocabulary split across lanes, so that the host does not read back about 1 MB of logits per row.
- Why it should help (theory): without it every draft and target row samples 248,320 entries on the host; the host part of a 27B MTP2 step is about 5 ms (derived, C), est. 1.5-3 ms per MTP2 step (C).
- What was actually built: the removal of the refusal (a 27-line switch `LLAMA_TP_BACKEND_SAMPLING=1`) alone. It aborted twice at load, so no sampling ran: a tiny PARTIAL tensor broke the AllReduce unit assert (fixed, kept) and then `get_tensor_async` does not read PARTIAL back. Archive: [record](old-ignore/2026-11-18/sp1-tp-backend-sampling/README.md), [patch](old-ignore/2026-11-18/sp1-tp-backend-sampling/tp-backend-sampling-switch.patch).
- Result: abort at load; nothing measured (B).
- Why it lost: tested (the abort log shows the cause); the idea itself was not reached.
- Best-case bound: 1.5-3 ms per 40.3 ms MTP2 step = 3.7-7.4% (C; the CPU sampling share was not timed because `perf` is blocked).
- Route status: `NOT TESTED`.
- A better attempt would: build split-aware top-k / argmax (each lane takes its half, merge 2k candidates) and time the host sampler first with host timestamps (SP1 in [Worklist](#speculative-decoding)).

#### N3 - Two-half ubatch pipelining of the prefill AllReduce
- Idea: overlap one half's AllReduce with the other half's GEMM.
- Why it should help (theory): AllReduce exposed 5.4% of the 27B PP critical path (A6 profile, B); estimate +3-5% PP (C; "3-5% at best").
- What was actually built: nothing.
- Result: none.
- Why it lost: not applicable.
- Best-case bound: removing a 5.4% exposed share gives 1 / (1 - 0.054) = +5.7% PP (B profile, arithmetic).
- Route status: `NOT TESTED`.
- A better attempt would: prototype with pooled memory for two in-flight halves, bit-exact (A6, Q2, N3 in [Worklist](#27b-q8_0)).

#### T1 - Fewer small launches in Flash Next decode
- Idea: remove intermediate `scale_f32`, `quantize_q8_1` and elementwise launches.
- Why it should help (theory): the WIP cut launches 3282 -> 3115 (-167, -5.2%, B); at 2.2 us that is 0.37 ms (C).
- What was actually built: an unfinished WIP branch; wrong output from the first token; `llama-bench` crashed after a rebase. No timing.
- Result: none usable.
- Why it lost: not applicable (unfinished).
- Best-case bound: about 400 elementwise-chain launches per GPU per token x 2.2 us = 0.88 ms, 4.4% of 19.8 ms (C; the count is from the Hypotheses text).
- Route status: `NOT TESTED`.
- A better attempt would: build one fusion at a time behind a kill switch, check output after each (P1 in [Worklist](#flash-next)).

#### F1 - Move tensor-split host traffic off blit kernels
- Idea: avoid the rect-copy staging in tensor split.
- Why it should help (theory): `copyBufferRectAligned` looked like 15.6% of PP and 19% of MTP3 kernel time (B).
- What was actually built: nothing; a load-excluded profile. Record: [F2/P4][rec-1106-f2].
- Result: the premise failed. The time was the load-time weight upload (`set_tensor_2d`); runtime rect copies are about 4% of MTP3 kernel time (B).
- Why it lost: tested (the profile window was corrected).
- Best-case bound: about 4% of MTP3 kernel time (B).
- Route status: `NOT TESTED` (a premise check).
- A better attempt would: see L-CPY; the copies are tensor-split staging.

#### Q1 - Fuse AllReduce, residual add, RMS norm and q8_1 quantize (27B TG)
- Idea: do the post-AllReduce add, norm and quantize in one kernel (ideally inside the AllReduce kernel).
- Why it should help (theory): an external review estimated 2.5-3.5 ms and 400-500 launches per token, assuming the chain was unfused (C/D); wrong for this tree (it is already one kernel).
- What was actually built: a profile and two kernel variants, no end-to-end benchmark. B1 unrolled epilogue, row in registers: 12.6 / 13.2 us vs 10.6 / 10.9 us (slower). B2 column split over 5 blocks: not taken at run time, because the in-place ADD/MUL aliasing guard keeps it off (its matching greedy output came from the old kernel). Option (b), folding the work into the AR kernel's phase 3, was not built. Archive: [record](old-ignore/2026-11-10/exp-q1/README.md), patches [unrolled](old-ignore/2026-11-10/exp-q1/unrolled-v1.patch) and [split](old-ignore/2026-11-10/exp-q1/split-v2.patch).
- Result: the post-AR kernel is 130 x 10.7 us = 1.4 ms per token (B); B1 slower; the fused q8_1 epilogue costs 3.5 us inside the kernel (B).
- Why it lost: B1: the shuffle chains are not the bottleneck (argued from the timing); B2: aliasing blocks it (tested by a debug print).
- Best-case bound: at most about 0.8 ms above the launch floor (B), 3.0% of 26.9 ms; above the 0.8% noise.
- Route status: `NOT TESTED`.
- A better attempt would: do option (b) together with AR-LL, which rewrites the same kernel (needs a cross-block step in `ggml_cuda_ar_kernel`). Related: 27B-ARN.

#### R-IN9 - The dense bias chain makes Inkling 2 Mini generation slow down with context
- Idea: make the decode bias cheaper to recover the TG lost at long context.
- Why it should help (theory): the claim was written in the handoff and the 2026-11-21 late record without a test (D). The arithmetic that was skipped: at 4K context the bias is about 0.4 MB per stage against 16 MB of keys and values, about 2.5% (C, figures from the task text).
- What was actually built: nothing. It was an explanation, not an implementation.
- Result: a trace pair showed 93% of the growth in `flash_attn_ext_vec` (15.6 to 85.8 us per call) and no growth in the bias chain kernels (B). TG 36.4 t/s at 4K context untraced; under trace 32.8 t/s at 4K against 37.4 t/s at 16 tokens (B).
- Why it lost: tested; the explanation is retired (see [Retired claims](#retired-claims)).
- Best-case bound: the 7% of growth left for everything else is 0.07 x (1/32.8 - 1/37.4 s = 3.75 ms) = 0.26 ms, 0.9% of the traced 30.5 ms token (C, argued from B numbers).
- Route status: `NOT TESTED`. The live route is IK1 in [Worklist](#inkling-2-mini): the decode attention kernel.
- A better attempt would: see IK1.

### CLOSED BY BOUND routes

#### N1 - Keep both DFlash draft graphs allocated
- Idea: keep both draft graphs so that the rebuild does not block replay.
- Why it should help (theory): the old premise was that the rebuild blocks the replay (D).
- What was actually built: nothing; a premise check with a temporary build counter and HIP-graph logs (27B, DFlash5, 512 tokens). Archive: [N1][rec-1029-n1], [notes](old-ignore/2026-10-29/n1-dflash/notes.md).
- Result: the draft rebuilds its graph every call (273 builds) at 0.36 ms each, 98 ms of an 8.5 s run (1.1%); the draft already replays through HIP graphs (67 replays, 2 re-captures) (B).
- Why it lost: tested. The premise predates the single-GPU draft.
- Best-case bound: about 0.7 ms of an 80 ms step (0.9%) against a DFlash5 spread of 57.3 and 60.1 t/s on equal builds (4.8%).
- Route status: `CLOSED BY BOUND` (bound 0.9%, below the spread of 4.8%).
- A better attempt would: nothing obvious.

#### L-MGC - Content-keyed meta token graph cache for the draft (Oct 12)
- Idea: replay the draft's meta token graph.
- Why it should help (theory): the draft graph was rebuilt every call with a new uid, so the meta token graph did not replay (Oct 12 log, D).
- What was actually built: `GGML_META_TG_CONTENT_KEY`, one session step; it replays, but acceptance length fell 2.95 -> 1.1-1.6 (wrong output); the log says "debugging" and the work was not pursued. No patch is archived. Archive: [Oct 12 log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md).
- Result: acceptance length 2.95 -> 1.1-1.6 (D, archived log).
- Why it lost: not known (a defect in the key; conjecture). The draft was later found to run as a single-device context that already replays (N1).
- Best-case bound: the N1 figures: 0.7 ms of an 80 ms step, 0.9%, below the 4.8% spread.
- Route status: `CLOSED BY BOUND` (through N1; the one attempt failed for an unknown reason).
- A better attempt would: nothing obvious.

## Worklist

Every estimate here is grade C (derived from a profile or a model, not a measured gain). Estimates that were computed at a 4-5 us floor per removed launch are recomputed at the measured 2.2 us and show both numbers. Order in each table: estimated gain per effort, highest first. "Blockers" lists what stops the cheapest test. Host `perf` is blocked on this machine, so tests that need `perf` use host timestamps instead.

Columns added 2026-11-22. `Priced by` is one of `card` (the estimate follows from card constants; the arithmetic is shown briefly), `trace` (from one trace) or `guess` (no basis; the cell says which card constant or measurement would price it). `Kind` is `kernel/code` or `parameter (last stage, needs owner permission)` ([Order of work](PROCESS.md#order-of-work)). Under 1% predicted: do not build it. Parameter rows are in the last table.

### Batch 1

The first batch, from the first `kernel_budget.py` run and the card checks (2026-11-22). All five are kernel/code work.

| Id | Idea | Basis | Est. gain (C) | Cheapest test | Blockers | Priced by | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BA-1 | Flash Next `topk_moe_wave_cuda<512, 10>` runs as one block of 32 threads at 14.1 us; apply the one-block rule (split over blocks or widen) | 49.1 calls per token per GPU (B, one trace, `kernel_budget.py`) | 0.49 ms per token per GPU, about 2.5% of 19.8 ms: 49.1 x (14.1 - 4.2) us | kernel sandbox microbench of the split form, then one trace | none | `trace` (+ card): 49.1 x (14.1 - 4.2) us = 0.49 ms | kernel/code |
| BA-2 | Flash Next four other one-block kernels: `lightning_indexer_kernel_vec`, `k_argsort_f32_i32`, `rope_multi`, `top_k_mask_kernel` | listed by `kernel_budget.py` as one-block kernels above the floor (B, one trace); with B1-1 the total excess is 0.92 ms per token per GPU | 0.41 ms, about 2% | kernel sandbox, then one trace | none | `trace` (+ card): excess over the floor per call from the trace | kernel/code |
| BA-3 | Inkling 2 Mini: take views of the mix output and loop messages without copies (same item as IK2) | about 5 launches per layer pass (B, 2026-11-21 trace) | about 5 launches per layer pass x 48 passes x 2.2 us = about 0.5 ms, about 2.5% of 21.0 ms | build behind a kill switch; one trace | none | `card`: 5 x 48 x 2.2 us = 0.53 ms | kernel/code |
| BA-4 | An occupancy scan of every kernel from the trace columns (registers, shared memory, threads per block), to find kernels like `flash_attn_ext_vec` before the 4K trace did | the occupancy bug in `flash_attn_ext_vec` (2 blocks per compute unit) was a registers problem that the trace already showed | not priced until the occupancy constants are on the card | the occupancy calibration (Q1 on the [card](HARDWARE-CARD.md#calibrations)), then a script over the trace | needs Q1 | `guess`: priced once the occupancy constants are on the card | kernel/code |
| BA-5 | Block-size tuning of the two new two-launch kernels (`res_mix`, `add_rms_norm_mul`) | traced 9.5 and 9.05 us against the predicted 7.8 us ([card checks](HARDWARE-CARD.md#checks-of-the-card-against-real-kernels)); block size was not tuned | 1.3 to 1.7 us per call | kernel sandbox (compiles in seconds), not a full rebuild | none | `card`: traced minus predicted = 9.05 - 7.8 to 9.5 - 7.8 us | kernel/code |

### Inkling 2 Mini

What limits it: GPU 98% busy with kernels back to back (B, one trace), so the limit is the launch cost and kernel time on the GPU queue. At 4,128 launches per token the launch floor is about 43% of the 21.0 ms token (C, derived: launches x 2.2 us). Generation also slows with context because one kernel, `flash_attn_ext_vec`, grows from 15.6 to 85.8 us per call (B, one trace pair). Each launch per layer pass removed is worth about 0.1 ms per token (0.5%; 48 passes x 2.2 us, C).

| Id | Idea | Basis | Est. gain (C) | Cheapest test | Blockers | Priced by | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IK1 (new) | Make the decode attention kernel `flash_attn_ext_vec` cheaper at long context (for example compute the bias inside it, or skip masked work) | 93% of the generation growth with context is this kernel (B, 2026-11-22 trace pair); 32.8 t/s at 4K context vs 37.4 at 16 tokens, both under trace; 36.4 t/s at 4K untraced | up to the traced gap, about 4.6 t/s at 4K context (37.4 vs 32.8) | per-call time of the kernel against context length in a trace; check graph reuse after any change | a fix is in progress; see [handoff.md](handoff.md) | `trace`: done in part: see What worked (`efb85a1a5`); the traced gap is the price | kernel/code |
| IK2 (new) | Take views of the mix output and loop messages without `ggml_cont` (lay RES_MIX output out as [n_embd, n_tokens, n_q]) | 5 launches per layer pass (playbook, count from a 2026-11-21 trace, B) | 2.5% TG (5 launches x 0.1 ms per token at 2.2 us) | build it behind a kill switch, A B B A | none | `card`: 5 launches x 48 passes x 2.2 us = 0.53 ms | kernel/code |
| IK3 (new) | Fold input norms into the producer; fuse MoE glue (router weights, activation, combine) | 10 launches per layer pass | 5% TG | same | none | `card`: 10 launches x 48 passes x 2.2 us = 1.06 ms | kernel/code |
| IK4 (new) | Loop controllers Q/K/V as two batched kernels (a block per gate value, then a block per output slice) | 8 launches per looped pass; R-IN8 shows the one-block form lost | 4% TG | microbench the two-kernel form first | none | `card`: launches removed x 2.2 us, minus the two-kernel form at about 5.4 us per pair | kernel/code |
| IK5 (new) | Short-conv reads without the concat; state write as one in-place kernel | 4-6 launches per pass | 2.5% TG | same | none | `card`: 4-6 launches x 48 passes x 2.2 us | kernel/code |
| IK6 (new) | One kernel for clip Q, clip K, clip R, norm V and the tau scale (packed output) | 4 launches per pass | 2% TG | same | none | `card`: 4 launches x 48 passes x 2.2 us | kernel/code |
| IK7 (new) | One mat-vec launch over the weights that share an input (`wq`, `attn_rel`, `attn_sink_proj`, `wqkv_gate`) | 4 launches per pass; hardest | 2.5% TG | microbench of one multi-matrix launch against 4 calls, bit-exact check | none | `card`: 4 launches x 48 passes x 2.2 us, minus the added kernel time | kernel/code |
| IK8 (new) | Server TG is below the bench on the same file (NVFP4: 37.1 vs 45.8 t/s) | measured gap (B); the cause is unknown; the context slowdown (IK1) may explain part (`-c 16384`) | up to 8.7 t/s (the gap) | same file with server vs bench at equal context; then `--reasoning off` and the sampler | none | `guess`: price it with a server trace and a bench trace at equal context | kernel/code |
| IK9 (new) | Keep raw UE4M3 NVFP4 scale bytes and decode in the kernel | NVFP4 repack costs +1,305 MiB VRAM (B) | -1.3 GB VRAM; no speed claim | build and compare PP/TG and VRAM | none | `guess`: memory only; no speed claim to price | kernel/code |
| IK11 (new) | Find why a pure batch-size change moves this model by KLD 0.005 | measured (B); the guess is near-tie flips in expert routing | none (understanding) | log the top-8 expert sets at two ubatch sizes, or fix the routing | none | `guess`: understanding, no gain claimed | kernel/code |

### Flash Next

What limits it: 83-88% GPU-busy (B, one trace), 2,811 launches per token per GPU, so launch cost and some idle host time both matter (the launch floor is about 31% of a 19.8 ms token, C, derived). The biggest kernel groups are experts 21%, AllReduce kernel 11% and hyper-connection mix about 11 us per call (B, one trace). Items TB3 and L2 (small-kernel chains) apply here too; they are in the 27B table.

| Id | Idea | Basis | Est. gain (C) | Cheapest test | Blockers | Priced by | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TB2, F2b, playbook FN item 1 | Wide-block rewrite of `dsv4_hc_mix_down` / `_up` (11.6 and 10.7 us per call, one pass, warp reductions); fuse `dsv4_hc_post` into the next `dsv4_hc_mix_prep`; fold down + up into one launch (last-block reduction) | 197 calls each at 11 us (B, 2026-11-21 trace); HC row 3.88 ms (14%) in the 2026-11-10 budget (B, profiled); the res_mix rewrite gave +9.5% on Inkling (A) by analogy; F2b: T=4 call 47 us vs 27 us at T=1 | rewrite part: about 1.3 ms per GPU, about 6% TG (by analogy). Launch part: 2 launches x 96 sites x 2.2 us = 0.42 ms (was 1.0-1.5 ms profiled at the old 4-5 us assumption, which included tails); post + prep removes 96 launches per token: est. 0.21 ms (was about 0.4 ms, 2%). F2b weight-stationary T>1 kernel: about 1.6 ms per MTP3 step (~3%) | one `-o DSV4_HC_MIX` backend test plus A B B A on the rewrite; sum order must stay, or KLD at `-ub 8` | none | `trace`: 197 calls at 11 us each in one trace; the launch part is card arithmetic (2 launches x 96 sites x 2.2 us = 0.42 ms) | kernel/code |
| Generic `mul_mat_vec_q` calls (playbook FN item 2) | Add the missing quant type as a repack layout variant (53 + 43 calls at about 34 us) | 2026-11-21 trace (B) | 4-8% TG | find which tensors and types fall to the generic kernel (names in the trace) | none | `trace`: 53 + 43 calls at about 34 us | kernel/code |
| P1, T1, playbook q8_1 and elementwise items | Remove intermediate `scale_f32` and `quantize_q8_1` passes: q8_1 cache keyed on the view root, swiglu writing q8_1 directly, one l2 norm over q and k; write q8_1 in the producer; fuse elementwise chains (about 400 launches per GPU per token) | about 480 `quantize_q8_1` (2.3 ms) and 303 `scale_f32` (1.8 ms) per token on ROCm1, profiled, counts predate F2 (B); T1 WIP: 3282 -> 3115 launches (-5.2%) but wrong output from token 1 (unfinished); 299 q8_1 launches per GPU per token in the 2026-11-21 trace | T1's 167 fewer launches: est. 0.37 ms (was 0.7-0.8 ms at the old 4-5 us assumption). `scale_f32` 303 x 2.2 us = 0.67 ms (was 1.8 ms profiled). The old range for P1 was 1.0-1.5 ms profiled; q8_1 in producer about 3% TG and elementwise chains about 4% (playbook, at 2.2 us). Sum of these overlaps TB2 and is not added | recount `quantize_q8_1` and `scale_f32` per token in a new trace (counts are from before F2); then bisect T1 with `GGML_CUDA_FEWER_LAUNCHES` masks on its branch | T1 branch gives wrong output (suspect: the l2 merge on strided q/k) | `card`: 167 launches x 2.2 us = 0.37 ms; 303 x 2.2 us = 0.67 ms | kernel/code |
| TB1 (route 2) | Fused single-block radix kernel for the QSA top-k | QSA indexer 2.5 ms (3.0 at 32K) in the 2026-11-10 budget (B, profiled); 12 radix launches cost 70-90 us per call | not estimated | microbench against the 12-launch version | the bitonic route lost (see Routes that lost) | `guess`: apply the one-block cost formula once the row length is known from the trace | kernel/code |
| FN-A | Weight-stationary grouped expert GEMM for PP (per-expert token lists, 8-16 q8_1 tokens per chunk in LDS, decode each weight once) | stock `mul_mat_q` gate/up is 22-23% of PP (94 calls per ubatch, about 2 ms each) in the 2026-11-10 PP budget (B); about 78 GB/s | +12-14% PP corrected (was +20% at ub 512) | go/no-go microbench of one layer against `mul_mat_q`; KLD check | L effort | `guess`: needs the streaming-efficiency calibration (Q2 on the card) | kernel/code |
| TB4 | Fold the MoE weighted reduce and shared-expert add into the routed down mat-vec | MoE reduce + combine 1.03 ms on ROCm1 vs 0.78 on ROCm0 (B, profiled) | 0.4-0.7 ms profiled (launch count not stated, not recomputed) | per-kernel times of `moe_weighted_reduction_f32` and `k_bin_bcast` in the FFN segment | none | `trace`: profiled times of the two kernels | kernel/code |
| TB5 | Build the QSA k-pool inputs and n-gram ids on the device, or reuse them across tokens | ROCm0's own token-start gap is 0.90 ms @8K and 1.22 @32K vs 0.25 / 0.32 on 27B (B, profiled); Flash Next is 83-88% busy, so some idle share exists | up to about 0.65 ms @8K (estimate) | host timestamps around graph compute (this is the proper N2 test) | `perf` blocked | `guess`: needs host timestamps around graph compute | kernel/code |
| W64 | True wave64 small kernels (one row per 64 lanes; `WARP_SIZE` is a logical 32, the gfx906 wave is 64) | candidate sites `norm.cu`, `binbcast.cu:406`, `softmax.cu:377`, `topk-moe.cu:389`; the 4-5 us glue kernels dominate | 0.3-1.5 ms per token (estimate; likely 0.1-0.5 us per call, unknown) | list kernels with workgroup size 32 in one trace (`Workgroup_Size_X`); microbench the top one | a 64-lane tree changes sum order (KLD) | `guess`: needs the threads-per-block effect for 32-thread kernels from the trace columns | kernel/code |
| P2 | Speed up tiny float projections (find the shapes, combine projections that share an input) | 344 `mul_mat_vec_f` calls at 6.9 us (2.4 ms) in the 2026-11-21 trace (B); 9.84 us on ROCm1 vs 6.97 on ROCm0 (2026-11-10) | up to about 2 ms (old count predates F2); likely below 1 ms | recount in a new trace | none | `trace`: 344 calls at 6.9 us in one trace | kernel/code |
| S1 | Stage the IQ codebook grid and q8_1 activations in LDS in the grouped expert gate/up kernel | Strata found a similar kernel compute-bound at 110-130 GB/s; ours reach about 150 GB/s useful | about 0.5 ms (a fraction of the 2.58 ms routed experts) | microbench | Strata numbers are not comparable | `guess`: needs a bytes-per-second curve for LDS-staged expert reads (Q2) | kernel/code |
| S4 | One-wavefront-per-row small mat-vec with a 64-lane butterfly, 1-8 columns | candidate for P2 | unknown (D) | check our block shape first | none | `guess`: needs the block shape of the small mat-vecs from the trace | kernel/code |
| M2 | Build the repacked MoE tile list from upstream `tile_spec` instead of `repack_tile_off<<<1, 1>>>` | `repack_tile_off` 44 calls per ubatch at 65-67 us = 5.7 ms per 1K, 0.34% of PP (B) | upper bound +0.3% PP | bit-for-bit output check | none | `trace`: 44 calls at 65-67 us per ubatch | kernel/code |
| P1a | Fold the constant 1/hc scale into the HC down mat-vec | r5 TG -0.5%, but the fusion did not fire (10145 `scale_f32` on both builds) (B) | 0.4 ms for this subset, 1.2 ms for all 303 scales (profiled, at about 4.2 us each) | get a lower `scale_f32` count in a trace first | none | `card`: launch count x 2.2 us; the profiled 0.4 / 1.2 ms used 4.2 us | kernel/code |
| L-GMR | Grouped-MoE gain on the current tree | not re-run since Oct 25 (the Landed row has the older numbers) | not estimated | MTP2 with `GGML_CUDA_FLASHNEXT_MOE=0/1`, r5 alternating, same acceptance | none | `guess`: needs the MTP2 on/off run | kernel/code |
| A5 | Expert parallelism: whole experts per GPU | routed experts 2.58 ms (B, budget); busier GPU runs about 6.2 whole experts against 5 equivalent today; a 640-row expert must run more than 1.25x faster than two halves | 1-2 ms, corrected to 0-0.5 ms for TG (PP is the better case) | one-GPU microbench: 10 experts x 320 rows against 6-7 experts x 640 rows | L effort; no upstream prior art found | `guess`: needs the one-GPU expert microbench and the Q2 streaming curve | kernel/code |
| D2, E1 | Fit: keep one copy of the QSA indexer key cache and router (D2, saves about 0.8 GiB at 100K); per-expert hot/cold placement across GPU and CPU (E1, about 1 ms per token per CPU layer sync) | memory arithmetic (C); upstream's MoE expert cache refuses `-sm tensor` | D2 not enough for 64K or 100K MTP on Q3_K_XL; E1 lets IQ4_XS fit | E1: UD-IQ4_XS `-sm layer -ncmoe 3 --moe-cache-mib N` against `-sm tensor -ncmoe 3` (30.6 TG) | needs tensor-split support in `llama-moe-cache` | `guess`: memory arithmetic; E1 needs the cross-GPU exchange constant (Q3) | kernel/code |

### 27B Q8_0

What limits it: 98% GPU-busy at 96-99% (B, one trace). 69% of traced kernel time is weight mat-vec at about 90% of the roofline (B), AllReduce kernel 9%, 1,645 launches per token per GPU (launch floor about 13% of the 26.9 ms token, C). The reachable kernel gain is small.

| Id | Idea | Basis | Est. gain (C) | Cheapest test | Blockers | Priced by | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Playbook 27B item 1 (`add_rms_norm_mul_f32<1024>`) | Rewrite the norm epilogue with a wide block and warp reductions (11.3 us per call against 4.5 us for a plain row norm) | 264 calls per token (B, 2026-11-21 trace); Q1 tried an unrolled epilogue and measured 12.9 us; three more micro-variants gave 10.7 to 14.8 us (route 27B-ARN in [Routes that lost](#routes-that-lost)) | about 0.9 ms per token per GPU, about 3% TG | `test-backend-ops` plus A B B A | none | `trace`: 264 calls at 11.3 us; superseded by the two-launch win (R-IN10) | kernel/code |
| TB3, L2, playbook 27B item 2 | Fused GDN decode step: conv, l2 norms, gates, gated delta net, gated norm in 2-3 launches instead of about 13 per layer; one launch for mat-vecs that share an input (GDN qkv, z, alpha, beta; attention q/k/v) | about 500 recurrent-chain launches per GPU per token; recurrent row 3.23 ms (27B@0) and 2.32 ms (FN@8K) in the profiled budget (B); 176 launches for L2 | TB3: about 10 launches per layer saved: 27B 48 layers x 10 x 2.2 us = 1.06 ms (was about 1.5 ms at the old assumption), Flash Next 36 layers x 10 x 2.2 us = 0.79 ms (was about 1.2 ms). Playbook: about 1.1 ms, about 4% for 500 launches. L2: 176 launches x 2.2 us = 0.39 ms (was 0.8-1.2 ms, corrected 0.4-0.7 ms at the old assumption) | dump the recurrent-segment kernel names and mean us of one layer from a 27B trace; go if at least 8 are under 6 us; microbench 4 calls vs one multi-matrix launch (bit-exact) | check upstream for a fused GDN decode step first | `card`: 10 launches x 48 layers x 2.2 us = 1.06 ms (27B) | kernel/code |
| Q3 (route 2) | Contiguous per-block reads with a per-row LDS re-reduce in the Q8_0 mat-vec (bit-exact) | the kernel is at 89-90% of the best stream (B) | about 1.9 ms per token is the most left (C) | `mvbench.hip` variant | none | `trace`: kernel at 89-90% of the best stream | kernel/code |
| A6, Q2, N3 | Two-half ubatch pipelining of the prefill AllReduce so one half's reduce overlaps the other's GEMM, after a re-profile of 27B PP (A6 re-profile is done) | A6 re-profile 2026-11-10: repacked GEMM 76% of the critical path (gate/up 33.5%, down 18.9%), chunked GDN 7.9%, AllReduce exposed 5.4%, ROCm1 GEMM 21% slower than ROCm0 (B) | N3: +3-5% PP corrected (was +8-12%, and "3-5% at best") | prototype with pooled memory for two in-flight halves, bit-exact | M effort | `trace`: A6 re-profile: exposed AllReduce 5.4% of PP | kernel/code |
| G1, D1, L1 | Long-context attention: group all 6 Q heads per KV head in `TILE` (G1); a small-batch vec path with several query columns per KV pass (D1); skip K/V work in `fattn-vec` for masked wavefronts (L1) | 27B: llama-bench depth 0 -> 32K costs 3.17 ms per token (26.70 -> 29.87, B); flash attention 3.34 ms at 32K, 209 us per layer per GPU against about 86 us for one F16 KV read (B, profiled); vec path at short context: `TILE` costs 28 us per call on 27B and 20-27 us on Gemma 4; L1 is a third-party +13% decode at 85K on NVIDIA, not reproduced here | G1: about 1-1.5 ms at 32K (corrected from up to about 3 ms); D1: about 0.3 ms (27B) and 1.2 ms (Gemma 4) at short context; L1: long context | rocprofv3 at 32K; go if attention per layer is at least 2x a single KV read; count the masked fraction for L1; DFlash5 verify still to profile | probably not bit-exact (KLD); source line anchors in the old rows were stale | `trace`: attention per layer at 32K from one trace | kernel/code |

### Both models and infrastructure

What limits it: the AllReduce exchange (1.01-1.62 ms critical-path per token, B, profiled, 2026-11-10) and ROCm1 small kernels running about 0.6 us longer than ROCm0 (B). The values for these quantities differ by source, see [Numbers that disagree](#numbers-that-disagree).

| Id | Idea | Basis | Est. gain (C) | Cheapest test | Blockers | Priced by | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P3 | Find the late rank before each AllReduce and speed up its preceding work | ROCm0 waits in AR 5.19 vs 1.03 ms per token on Flash Next and 4.1 vs 1.7 on 27B (Oct 12); 2026-11-10: about half of the profiled wait is profiler cost; real lag about 1.1 ms per token on 27B (each small kernel about 0.6 us longer on ROCm1); ROCm1 busy minus ROCm0 busy: 27B@0 1.09 ms, FN@8K 2.02, FN@32K 1.95, G4@0 1.36 (B, profiled). Lanes on the dispatcher thread and uneven `-ts` were tried: no TG gain | about 1.1 ms on 27B, reached through fewer small kernels, K2 and A4 | see K2 | none | `trace`: ROCm1 busy minus ROCm0 busy in the profiled budget | kernel/code |
| A4 | Move ROCm1 to CPU-attached PCIe lanes, then test peer AllReduce and RCCL | AR, post-AR gaps and peer wait are 4-7 of 27 ms per token on 27B (old row); 2026-11-10: the reachable part is the AR exchange (27B 1.62 ms, FN 1.01 ms critical) plus ROCm1's small-kernel lag if it comes from the chipset link (K2 tells) | about 2-2.5 ms on 27B, 2-3 ms on Flash Next, not 4-7 ms (C; hardware change not tested) | run K2 first | hardware change; the claim that RCCL does not work here rests on the topology assertion (D), RCCL was never measured in these docs | `guess`: hardware change; needs the cross-GPU exchange constant (Q3) and K2 first | kernel/code |
| AR-LL | NCCL-LL style AllReduce packets (8 B data plus an 8 B sequence word) so the peer polls the data: one PCIe trip instead of 2-3; bit-exact | `ggml_cuda_ar_kernel` 26.9 us for 5 KB on Flash Next (92 per GPU per token, 2.5 ms), 16.9 us on 27B (129 per token, 2.2 ms); critical exchange 1.62 ms (27B@0), 1.01 ms (FN@8K), 1.90 ms (G4@0) (B, profiled) | corrected to at most 0.6 ms (27B) and 0.5 ms (Flash Next); the earlier -1 ms and -0.7 ms are the whole exchange | time a variant of `vr-test-gfx906-tp` with the two system fences per block relaxed (`allreduce.cu` `__threadfence_system` lines; timing test); build LL if it reaches below about 8 us per call | none | `guess`: needs the host-staged exchange curve (Q3) | kernel/code |
| Program kernel | One generic kernel that runs a short list of elementwise steps (add, mul, scale, tanh, sigmoid, clamp) over a small tensor in one launch; the backend fuses any run of small elementwise nodes for every model | none measured (D) | not estimated | prototype on one chain | needs use counts so values read outside the run are kept | `guess`: needs a prototype and use counts | kernel/code |
| Thread per stream | Host thread per stream for independent branches of a layer | 1.9x launch throughput with a host thread per stream, 1.45x from one host thread (B, microbench) | not estimated | microbench on a real two-branch layer | each join costs an event wait | `guess`: the card has 1.45x and 1.9x for synthetic launches only; needs a real two-branch layer | kernel/code |
| TB6 | Gemma 4: find why the AllReduce after the FFN takes 19.4 us against 11.5 us after attention for the same vector | 2026-11-10 budget G4@0: AR exchange 1.16 vs 0.69 ms; attention glue 1.67 ms (B) | about 0.5 ms (AR) plus part of the attention glue | explain from per-AR trace times | secondary model | `trace`: per-AR times of one trace (19.4 against 11.5 us) | kernel/code |
| L-GCN | Upstream GCN Q8_0 MMQ rows 256 -> 512 threads (discussion 23881) | third-party: +19% Q8_0 pp512, -6.9% Q4_K (C); helps plain upstream, not the repacked path | none for this fork's repacked path | - | not reproduced | `guess`: third-party number; not reproduced | kernel/code |
| Ruled out by analysis (not tested) | Tree verify (48 recurrent GDN layers need forked state per branch); mirror Flash Next attention to drop one AllReduce per layer (+1.1 GB per GPU, no memory left at 100K); run replicated glue on ROCm0 alone (each glue block would need its own exchange of 10-20 us, more than ROCm1's few-us penalty) | analysis, no test (C / D) | - | - | not tested | `guess`: analysis only | kernel/code |

### Speculative decoding

What limits it: the verify step and the host side of each step. The split numbers in [TOKEN-BUDGET.md](TOKEN-BUDGET.md) are estimates (C); no speculative trace exists.

| Id | Idea | Basis | Est. gain (C) | Cheapest test | Blockers | Priced by | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SP1, SP1a | GPU greedy / top-k sampling under `-sm tensor`: each lane takes top-k over its vocab half and the 2k candidates merge (exact for greedy) | `llama_context::set_sampler` refuses backend sampling under tensor split, so 27B falls back to the CPU sampler: each draft and target row reads back about 1 MB of logits and samples 248,320 entries on the host; the SP1a attempt aborted at load (see Routes that lost); the host part of a 27B MTP2 step is about 5 ms (derived, C) | 1.5-3 ms per MTP2 step (4-7%), about 2 ms per DFlash step, about 0.5 ms per token plain server (estimate) | host timestamps around the sampler during `spec_accept.sh` | `perf record -g` (the old test plan) is blocked; the `GGML_META_TOPK_BCAST_LOGITS` switch named in the old row is not in the code; line anchors moved | `guess`: host part of a step is derived, not traced; needs host timestamps | kernel/code |
| SP2 | Reduced-vocabulary draft head (FR-Spec style), reusing upstream EAGLE3 `d2t` mapping | the full 248,320-row head is about 75% of each 27B MTP draft decode's bytes; LM head 0.88 ms (27B TP) and 0.52 ms (FN TP) per pass (B, budget) | 27B MTP2 -1.5 ms per step (~4%), Flash Next MTP3 -2.5 ms (~5%), DFlash -1.5 ms (~3%) (estimate) | CPU side: share of tokens in the top 32k/64k ids over the `spec_accept.sh` replies plus a code/prose corpus; go if at least 99% | none | `guess`: bytes of the head are known; needs the streaming efficiency (Q2) for the draft decode | kernel/code |
| SP4 | Stop charging the deferred DFlash draft prompt replay to generation | DFlash5 step is 48.1-49.0 ms on 22-29-token prompts but 60.3-66.9 ms on about 1,750-token prompts (B) | +15-25% TG on long-prompt requests (estimate) | `LLAMA_DFLASH_DEFER=0` against default, short prompts | the `LLAMA_DFLASH_DEFER` knob was removed 2026-11-21; the test needs it re-added | `guess`: needs the knob re-added and a run | kernel/code |
| MU1 | Speculation that scales with busy slots: (a) lower the draft length when several slots verify at once, (b) let the narrow repacked mat-vec take up to 16 columns (`MMQ_RP_Q8_MMV_MAX_TOKENS 8`) | 4 users at 16K: plain 4-way decode 81 t/s, server MTP2 60.9 t/s; 2026-11-11: no draft 43.9 vs MTP2 41.2 aggregate; different harnesses, not a matched A/B (B) | up to the 81 vs 61 gap (about +30% aggregate at 4 busy users, estimate) | SP3 extended to 16 columns, plus a matched server A/B of 4 users with MTP2 against no draft | none | `guess`: needs the verify cost curve to 16 columns; part (a) is a setting and waits for the owner, part (b) is code | kernel/code |
| L-DF10 | Find the about 10 ms host gap per DFlash step (inject/generate gap, sampling and accept, readbacks) | Oct 12 profile; not the draft rebuild (N1) | DFlash5 about 70 t/s if closed (Oct 12 estimate) | trace first | none | `guess`: needs a trace of one DFlash step | kernel/code |
| U3 | Load the unsloth shared MTP heads (borrow the target embedding and head) to save memory for Flash Next MTP | saves about 1.8 GiB; "unverified" under `-sm tensor` with the ROCm1 sidecar (C) | memory only | load test | none | `guess`: memory only | kernel/code |

### Parameter tuning - last stage, needs owner permission

Not on the active list. Moved here from the tables above on 2026-11-22 (no row deleted); open only after the owner agrees that kernel-level work is tapped out and gives permission. Rows that are kernel or code work on the speculative path stay in their tables above. MU1 is mixed: its part (b) is code and stays in the speculative table, part (a) (draft length) is a setting and waits here.

| Id | Idea | Basis | Est. gain (C) | Cheapest test | Blockers | Priced by | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IK10 (new) | Two GPUs: one Inkling instance per GPU for 2-4 slots (NVFP4 file needs about 19 GB of 32) | user workload: one slot about 70% of the time, 2-4 slots about 30%; `-sm tensor` is not implemented for `inkling2_loop` | not estimated | `llama-batched-bench` on two instances | tensor-split rules for the new tensor types do not exist | `guess`: needs a measured two-instance run; no card constant prices it | parameter (last stage, needs owner permission) |
| A2 | Defaults `LLAMA_SHEXP_SPLIT=1` and `GGML_SCHED_SHARE_VIEW_INPUTS=1` | off was 0.6-1.1 t/s slower, inside Flash Next noise (B, one repeat); PP not run | not estimated | PP2048 r5 alternating; drop a knob that shows no difference | none | `guess`: needs a PP A/B; no card constant prices a knob default | parameter (last stage, needs owner permission) |
| A1-FN | `-ts` for Flash Next TG, and for PP on both models | Flash Next TG one repeat: 0.52/0.48 37.80 vs 37.46, 0.48/0.52 36.64 vs 40.08 (B, inside noise); Oct 12: 56/44 gave +3% 27B PP for -5% TG | expect at most about 0.4 ms from a 4% shift (C) | Flash Next TG r5 alternating; PP2048 r3 at 56/44 | none | `guess`: needs an alternated TG run; no card constant | parameter (last stage, needs owner permission) |
| L-T16 | `-t 16` on Flash Next | 39.44 vs 37.64, one sample in the same run (B) | not estimated | r5 alternating against `-t 8` | none | `guess`: needs an alternated TG run; no card constant | parameter (last stage, needs owner permission) |
| K2 (both models) | `HIP_FORCE_DEV_KERNARG=1` (kernel arguments in VRAM) | ROCm1 runs every small kernel about 0.6 us longer while its mat-vec is 1.1% slower; bare launch 2.0 us on the MI60 vs 2.6 us on the MI50 (B, microbench); the cause may be PCIe latency on chipset lanes, not tested | 27B 0-1.1 ms, Flash Next 0-2.0 ms (profiled, low confidence) | TG r3 (27B) and r5 (Flash Next) alternating plus one trace each; go if ROCm1's excess busy time (1.09 / 2.02 ms) shrinks | none; env only | `card`: 0.6 us (2.6 - 2.0 bare, MI50 against MI60) x 1,645 launches = 0.99 ms upper bound on ROCm1, low confidence | parameter (last stage, needs owner permission) |
| SP3 | Measure the verify cost curve for 1-8 columns, then re-tune draft depth | server per-position acceptance (DFlash5 0.775/0.624/0.509/0.425/0.353, MTP2 0.843/0.690) with about 1.7 ms per extra verify column (Oct 12) predicts DFlash7 and MTP3 should win; the 2026-11-05 llama-cli runs measured DFlash7 -7.7% and MTP3 -3.8% (B) | 0-8% | `llama-bench -p 1,2,...,8 -n 0 -r 5` on both models; no code | none | `guess`: needs the verify cost curve for 1-8 columns | parameter (last stage, needs owner permission) |

### Added 2026-11-22: DSpark step cost (Inkling 2 Mini)

Kind: kernel/code. Priced by: guess (needs one trace of a speculative step). A DSpark step costs about 2.1 to 2.2 plain tokens (56 to 59 ms) and returns 1.9 to 2.2 tokens at the publisher's settings (A for the timing, [record](PERFORMANCE-RECORDS.md#2026-11-22-inkling-2-dspark-drafting-opt-in-not-a-speed-gain-yet)). If the verify batch of 4 tokens cost what one token costs plus its extra work, the step would be near 1.3 plain tokens and DSpark would give about 1.5 times TG (C, estimate). Cheapest test: one `rocprofv3` trace of a short DSpark run and `kernel_budget.py`, compared with a plain trace by kernel. Do not change `--spec-draft-n-max` or other settings first.

## Retired claims

Each row: what the old text claimed, what is known now, and where. Grades are in "now".

| Claimed | Now | Source |
| --- | --- | --- |
| The limit is a per-token host gap before each token graph (N2, TB5 in part, "No dispatch on either GPU: about 4.3 ms", launch ramps, "reduce host launch gaps") | steady-state generation is 98% GPU-busy with kernels back to back on Inkling, 96-99% on 27B (B, one trace each); the profiled gap was largely profiler cost. Flash Next is 83-88% busy, so an idle share exists and is unexplained (see TB5) | [The model](#the-model) |
| Every removed launch is worth the 4-5 us floor (ceilings of P1, T1, TB2, TB3, L2, W64) | the app cost is about 2.2 us; 4.2 us is the profiler floor; ceilings built on 4-5 us are about 2x high and are recomputed in Hypotheses (B) | [The model](#the-model) |
| A4 (move ROCm1 to CPU lanes) is the top lever, 4-7 of 27 ms | an untested hardware change; the 2026-11-10 correction was 2-2.5 ms on 27B (C) | A4 in [Worklist](#both-models-and-infrastructure) |
| "Official llama.cpp matches the fork on Flash Next plain TG because the fork speeds up byte-heavy paths" | an explanation without a test (D); Flash Next plain TG rose from 39.1 to 50.18 (B/A, rows F2, S2, B1, P4), all with fork kernels | [What worked](#what-worked) |
| Against official llama.cpp: +53% 27B PP, +12% 27B TG, +74% Flash Next PP, no Flash Next TG gain (2026-10-29, `36a0e2ffc`) | single stage runs on an old base (B); not re-measured; the Flash Next TG part is superseded | [rebuild record][rec-1029-rb] |
| The MI60 is 10-20% faster and has the better host link | no source (D); ROCm1 GEMM is 21% slower in PP but mat-vec 1.1% slower in TG; the 2.0 us vs 2.6 us bare launch cost differs by 0.6 us (B) | [HARDWARE-GUIDE.md](HARDWARE-GUIDE.md), [TOKEN-BUDGET.md](TOKEN-BUDGET.md) |
| Uneven `-ts` can give 2-3% | on 27B every uneven split measured slower (A1-27B, A); Oct 12: 56/44 gave +3% PP for -5% TG (B) | [Routes that lost](#routes-that-lost) |
| FP8/FP4 lose precision, so we skip them; they lose their speed point | NVFP4 is the Inkling checkpoint's own expert format; repacked it runs PP 836.3 against IQ4_NL 878.6 on the same GPU (A/B) | [What worked](#what-worked) |
| Removed knobs: `GGML_SCHED_INPUT_STAGING_F32_MAX`, `GGML_RP_MMV_ROWS`, `LLAMA_DFLASH_DEFER`, `GGML_META_TOPK_BCAST_LOGITS`, `GGML_CUDA_PEER_AR`, `GGML_CUDA_TP_OVERLAP`; about 30 sweep knobs archived 2026-11-21 | no longer in the code (0 hits); `GGML_CUDA_FEWER_LAUNCHES` exists on the unfinished T1 branch | [knobs archive](old-ignore/2026-11-21/knobs/), [IMPROVEMENTS.md](IMPROVEMENTS.md) |
| Peer-write AllReduce is an opt-in option | archived 2026-11-21; no live code path | [archive](old-ignore/2026-11-21/peer-ar/README.md) |
| Inkling generation slows with context because small ubatches use the dense bias chain (route R-IN9) | the bias chain kernels do not grow; 93% of the growth is `flash_attn_ext_vec` (B, one trace pair, 2026-11-22) | [The model](#the-model) |
| "About 41" t/s for Inkling 2 Mini short-context generation in `llama-batched-bench` | not a measurement; measured: 36.4 t/s at 4K context (untraced) and, under trace, 32.8 t/s at 4K against 37.4 t/s at 16 tokens (B) | [The model](#the-model) |
| "GPU 99% busy is not shown by rocprof; 25% busy" (a retraction in the 2026-11-21 handoff) | the retraction was wrong: the 25% figure included model load; steady-state is 98% busy (B) | [The model](#the-model) |
| "About 7,100 kernels, each tiny kernel 3.8 us" | the 3.8 us is profiler wall time per launch, not kernel duration; the app cost is about 2.2 us | [The model](#the-model) |
| 27B uses "about 75% of the bandwidth roof"; "a software 40-44 TG is a plausible target"; later 77%, 40-42, 90% | different denominators, not reconciled; targets are estimates (C) | [Numbers that disagree](#numbers-that-disagree) |
| "A measured 3-5 ms cut would give about 41-45 TG" and "a 3-5 ms cut gives about 50-56 TG" (Flash Next) | estimates (C), not measured; Flash Next already measures 50.18 | [What worked](#what-worked) |
| 2026-10-29 Flash Next attribution list (float mat-vec 3.1 ms, AllReduce 5.3 ms, ...) | replaced by the 2026-11-10 budget; the profile was 21% slower than unprofiled | [TOKEN-BUDGET.md](TOKEN-BUDGET.md), [profile][rec-1029-prof] |
| "Routed experts are 19% of active bytes; free expert weights would save under 1 ms" | the byte share is a measured tally (B, old upload); the time claim is an estimate (C) | [records][rec-1029-x] |
| "Speculation gains less on Flash Next because each verify token picks its own experts" | the gain numbers are B (MTP2 +33% vs +68%); the mechanism is an assertion (D) | [What worked](#what-worked) |
| "2-3 CPU expert layers cost about 20-25% TG, mostly sync and transfer" | quant and placement changed together in that run (C); mechanism not isolated | [big quants][rec-1106-q] |
| "Plain TG loses 20% at 32K, +6.8 ms per token, mostly attention" | not reproduced: llama-bench gives 3.17 ms (B); see [Numbers that disagree](#numbers-that-disagree) | [27B sweep][rec-1105-s] |
| "Judged default-on safe" (F2/P4 review) | a review opinion (D); the Q8-reference KLD was not measured and P4 dispatch lacks guards | [F2/P4][rec-1106-f2] |
| "DFlash5 is the fastest 27B mode at a 2K prompt" | one or two llama-cli runs (B); the same record says llama-cli TG swings 10-18% with the greedy text | [27B sweep][rec-1105-s] |
| "27B TG varies about +-0.3 t/s, so r3 alternating is enough" | benchmarks drift 15-30% when the GPUs are hot; use A B B A below 60C (B) | [How to test](#test-rules) |
| "Profiled TG was 21-45% slower" | two ranges exist, 21-45% and 7-24%; not reconciled | [Profiler caveats](#profiler-caveats) |
| "Byte-identical output" or "text identical" as evidence of no quality change; PPL 2.8307 vs 2.8305 as parity | a smoke test; on Inkling a batch-size change alone moves KLD about 0.005 (B) | [Quality checks](#quality-checks) |
| K2: "looks like per-dispatch PCIe latency on chipset lanes rather than the power cap" | untested (C/D); the 0.6 us bare launch difference between MI50 and MI60 is consistent with it, not proof | K2 in [Worklist](#27b-q8_0) |
| The old verdict wording "validated" plus "wrong" (F1, Q1, N3, draft device, 2D memcpy, ub 4096) in 2026-11-06 verdicts | the phrase meant "one implementation lost"; F1 was a premise check, N3 was not built, Q1 was a profile; see the status words | [Routes that lost](#routes-that-lost) |
| S3 (split HC down projection along K) about 3% | superseded by F2 (`dsv4_hc_mix_down` is a fused op now); probably obsolete | [records][rec-1106-f2] |
| S5 (router top-k in one wavefront) | merged into B1 | B1 in [What worked](#what-worked) |
| A6 (re-profile 27B PP) listed as open | done 2026-11-10; folded into the A6, Q2, N3 row | [Worklist](#27b-q8_0) |
| L-MMQC: `mmq-config-gfx906.cuh` is never selected | the file no longer exists (removed in the 2026-10-29 rebuild) | [IMPROVEMENTS.md](IMPROVEMENTS.md) |
| Open-section priority list (K2, TB1, TB5, B1, TB2, TB3, SP1, P1) of 2026-11-10 | repeated ledger rows with other numbers; TB1 is a tried item and B1 landed; replaced by the tables above | - |
| Oct 25 rows (grouped MoE 47.65 -> 50.45, 27B MTP3 60.0) | one run each, old trees (B); superseded by 2026-11-06 and later rows | [records][rec-1029-x] |
| Handoff target "TG from 28 ms toward 10 ms per token on one GPU" | reached about 21 ms (47.7 t/s) on Inkling IQ4_NL; the plan was replaced | [late record][rec-1121-late] |
| Rely on upstream for Inkling 2; do not carry Inkling pins; unsloth banded op "should run on HIP" | the fork carries the #386 port, fused ops and the NVFP4 repack; scalar banded kernels halve PP | [late record][rec-1121-late] |
| "Cut 2,000-3,000 lines without losing speed" (merge plan) | actual core delta +16426/-995 became +15276/-1039 | [catch-up record][rec-1121-u] |
| Larger quants and `-ctk q8_0` plans (temp-128k plan): "Q3_K_XL at 128K with Q8_0 KV leaves about 4 GiB" | measured 2026-11-06: Q3_K_XL with F16 KV at 100K leaves 0.02-0.03 GiB; quantized KV is not used (user rule) | [big quants][rec-1106-q] |
| "Larger quant is a quality experiment, not a speed improvement" | Q3_K_XL 39.7 vs IQ3_XXS 38.4 @8K (B) | [big quants][rec-1106-q] |
| Quantized KV and verify kernels with more than 8 rows "can leave the optimized path" (IMPROVEMENTS) | user rule: do not quantize the KV cache; the 8-column limit mechanism is known but not timed (MU1 hypothesis) | MU1 in [Worklist](#speculative-decoding) |
| My one-block-per-head explanation of the decode attention slowdown (`flash_attn_ext_vec` at 4K context) | the cost prediction was right (about 70 us against 85.8 us traced) and the mechanism was wrong: the kernel already split the keys over blocks; the limit was occupancy (more than 128 VGPRs, 2 blocks per compute unit); fixed by launch bounds (128, 4) on HIP (A at 4K, B at 16K) | [card checks](HARDWARE-CARD.md#checks-of-the-card-against-real-kernels), [What worked](#what-worked) |

More: [PERFORMANCE-RECORDS.md](PERFORMANCE-RECORDS.md) holds every dated record, unedited.

[rec-1121-late]: PERFORMANCE-RECORDS.md#2026-11-21-inkling-2-mini-late-nvfp4-repacked-experts-banded-prefill-attention-faster-mix-kernels
[rec-1121-c]: PERFORMANCE-RECORDS.md#2026-11-21-inkling-2-mini-crash-fix-fused-ops-nvfp4-baseline
[rec-1121-u]: PERFORMANCE-RECORDS.md#2026-11-21-catch-up-to-official-46eb1742d
[rec-1118]: PERFORMANCE-RECORDS.md#2026-11-18-catch-up-to-official-edec462cc
[rec-1106-f2]: PERFORMANCE-RECORDS.md#2026-11-06-f2-p4-toolchain-check-and-acceptance-aware-speculative-runs
[rec-1106-m]: PERFORMANCE-RECORDS.md#2026-11-06-flash-next-ud-iq3_xxs-at-100k-with-mtp
[rec-1106-q]: PERFORMANCE-RECORDS.md#2026-11-06-flash-next-big-quants-repack-conversion-q8_0-n-gram-table-100k-and-mtp-fit
[rec-1106-s]: PERFORMANCE-RECORDS.md#2026-11-06-s2-cheaper-iq-sign-handling-on-gfx906
[rec-1105-s]: PERFORMANCE-RECORDS.md#2026-11-05-27b-config-sweep-and-32k-depth
[rec-1105-c]: PERFORMANCE-RECORDS.md#2026-11-05-catch-up-to-official-4b77dee05
[rec-1029-a1]: PERFORMANCE-RECORDS.md#2026-10-29-a1-config-sweep-tg
[rec-1029-n1]: PERFORMANCE-RECORDS.md#2026-10-29-n1-dflash-draft-replay-check
[rec-1029-n2]: PERFORMANCE-RECORDS.md#2026-10-29-per-token-host-gap-n2
[rec-1029-x]: PERFORMANCE-RECORDS.md#2026-10-29-ideas-from-external-reviews
[rec-1029-rb]: PERFORMANCE-RECORDS.md#2026-10-29-rebuild-on-official-36a0e2ffc
[rec-1029-prof]: PERFORMANCE-RECORDS.md#2026-10-29-flash-next-tg-profile
[rec-1029-e]: PERFORMANCE-RECORDS.md#2026-10-29-earlier-checks-on-the-official-head
