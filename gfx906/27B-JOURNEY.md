# How a 27B model got fast on two legacy GPUs: the Qwen3.8-27B Q8_0 journey in llama-of-vega.cpp

_AI-drafted_

Status date: 2026-11-22. Every number below comes from the fork's own docs or code, with its date and, where the docs give one, an evidence grade: **A** = alternated A/B on the real model, **B** = single run, microbenchmark or trace, **C** = estimate, **D** = assertion without a saved record. "Estimate (mine)" marks back-of-envelope arithmetic done for this document, with the working shown. Paths are relative to the repo root.

The short version: in the one same-day comparison against official upstream (2026-10-29, B), prompt processing went from 448.60 to 685.34 tokens/s and plain decode from 32.77 to 36.67; decode stands at 37.96 today (2026-11-22, A). Speculative decoding lifts one user to about 56-61 tokens/s through the server, and four concurrent users now get about 98 tokens/s in total (2026-11-22, A). The interesting part is why each step worked, and why many plausible ideas did not.

## 1. The hardware: memory-rich, compute-poor

The machine has two AMD "Vega 20" datacenter cards, known to the compiler as `gfx906`: a Radeon Instinct MI60 (device `ROCm0`) and an MI50 (device `ROCm1`), 32 GiB each.

| Property | MI60 (ROCm0) | MI50 (ROCm1) | Source |
| --- | --- | --- | --- |
| Compute units (CUs) | 64 | 60 | HARDWARE.md |
| Power cap | 210 W (max 225 W) | 170 W (max 178 W) | BENCHMARK-TIMELINE.md host table |
| PCIe attachment | CPU root port, Gen3 x8 | chipset (PCH) root port, Gen4 x4 behind DMI | BENCHMARK-TIMELINE.md host table |
| Peak HBM2 streaming | 891 GB/s | 904 GB/s | HARDWARE.md, 2026-11-22, B |
| Last-level cache | 4 MB L2, 64 KB LDS per CU | same | HARDWARE.md portability table (D) |
| Matrix units | none | none | HARDWARE.md |

Some vocabulary, since the rest of the document leans on it:

- **CU (compute unit)**: the GPU's unit of execution. Per the occupancy formula in HARDWARE.md, each CU has 4 **SIMD** engines, and each SIMD keeps up to 8 **waves** resident (32 per CU).
- **Wave** (**wave64** here): 64 threads ("lanes") executing the same instruction in lockstep; NVIDIA's warp, but twice as wide. A **thread block** of T threads is `ceil(T / 64)` waves that can share scratch memory and synchronize.
- **VGPR (vector general purpose register)**: a register holding one 32-bit value per lane. Each SIMD has a pool of 256 that its resident waves must share, so a kernel that needs many registers leaves room for fewer waves. This fact drives several stories below.
- **LDS (local data share)**: 64 KB of fast on-chip scratch memory per CU.
- **HBM2**: stacked memory next to the GPU die. The spec figure is 1024 GB/s; the fork measured 87-88% of that.
- **`v_dot4`**: the one "AI" instruction these cards have. It treats two 32-bit words as four signed 8-bit numbers each and returns the sum of the four products plus an accumulator (`ggml_cuda_dp4a` in the code). There are no matrix units.
- **P2P (peer to peer)**: one GPU writing directly into the other's memory over PCIe. This host cannot: `hipDeviceCanAccessPeer` is 0 in both directions (HARDWARE.md Q3). Everything exchanged between the cards passes through host RAM at about 7 GB/s per direction (2026-11-22, B).

For comparison, the HARDWARE.md portability table (grade D, spec background) lists an RTX 3090 at about 936 GB/s, with tensor cores. So these cards have modern memory speed and old arithmetic. A workload limited by "how fast can I read the weights" runs well; one limited by "how many multiply-adds per second" runs poorly. The rest of this document is about arranging the work to be of the first kind.

The two cards are not symmetric, and the asymmetry is specific (HARDWARE.md "Skeptical corrections", 2026-11-22):

- Memory streaming is equal (891 vs 904 GB/s; the 27B mat-vec is only 1.1% slower on the MI50).
- Compute-bound prefill GEMM is about 21% slower on the MI50 (60 CUs and a lower power cap).
- Tiny kernels are slower on the MI50: bare launch cost 2.6 us vs 2.0 us (2026-11-21, B), suspected to come from the chipset PCIe lanes.

An earlier belief that "the MI60 is 10-20% faster everywhere" is now in the retired-claims list in PERFORMANCE.md.

One more fact: the cards throttle. At 98-104 C the clock drops to 938 MHz and everything runs 15-30% slower (HARDWARE.md, 2026-11-21, B). Section 10 shows how much trouble that caused.

## 2. The theory you need

### 2.1 What a forward pass costs

A transformer layer is mostly multiplications of activations by fixed weight matrices. Qwen3.8-27B in Q8_0 is 27.04 GiB for 27.32 B parameters (BENCHMARK-TIMELINE.md, 2026-10-12): about one byte per weight. There are two regimes:

- **Decode** (token generation, "TG"): one new token at a time. Each weight matrix multiplies a single activation vector: a matrix-vector product ("mat-vec" or MMV). Every weight is read once and used once.
- **Prefill** (prompt processing, "PP"): hundreds or thousands of prompt tokens at once (a "ubatch"). Each weight matrix multiplies a matrix of many columns: a general matrix-matrix product (GEMM). Every weight is read once and used once per token.

### 2.2 Roofline thinking

A roofline model says a kernel is limited by whichever runs out first: memory bandwidth or arithmetic throughput. For any kernel, ask which roof it is under and how close it is.

**Decode is under the bandwidth roof.** HARDWARE.md records the key number: the two cards together stream 28.6 GB of weights per token in 17.9 ms out of a 26.9 ms token (2026-11-21, B), which the card calls "~800 GB/s effective" per card.

Estimate (mine): 28.6 GB / 2 cards = 14.3 GB per card per token; 14.3 GB / 17.9 ms = 799 GB/s. At the measured peak of 891 GB/s the same bytes would take 14.3 / 891 = 16.0 ms. So a perfect mat-vec kernel could shave about 1.9 ms from a 26.3 ms token (1000 / 37.96). PERFORMANCE.md gives the same bound for its "Q3" route: at most 1.9 ms (7%).

That estimate explains the shape of the project. Decode weight streaming is already at about 90% of what the memory can deliver, and the old ALU keeps up: the Q2 microbenchmark shows `v_dot4` at 891 GB/s, the same as pure loads (2026-11-22, B). The remaining decode wins had to come from the other 8-9 ms per token, and from doing more than one token of useful work per pass over the weights.

**Prefill is under the compute roof.** Estimate (mine): in a 2048-token ubatch each card still reads its 14.3 GB only once, about 7 MB per prompt token, so bandwidth is irrelevant. What matters is multiply-adds per second: the 2026-10-12 profile put the repacked GEMM at about 33 int8 TOPS, about 56% of the `v_dot4` peak on the MI60 (B). That GEMM is 76% of 27B prefill time (PERFORMANCE.md), which is why the MI50's 21% deficit shows in PP and not in TG.

### 2.3 Occupancy

Occupancy is how many thread blocks are resident on a CU at once. More resident blocks means more memory requests in flight, which is what reaches the bandwidth roof. HARDWARE.md gives an exact formula, checked against the HIP API and a hardware counter (Q1 microbenchmark, 2026-11-22, B):

```
vgpr_alloc      = ceil(V / 4) * 4                      # V = VGPRs the kernel uses
waves_per_simd  = min(8, floor(256 / vgpr_alloc))
waves_per_block = ceil(T / 64)                         # T = threads per block
blocks_per_cu   = min(floor(4 * waves_per_simd / waves_per_block),
                      floor(65536 / LDS_bytes))
resident blocks = blocks_per_cu * CU_count             # 64 on MI60, 60 on MI50
```

Thresholds: up to 32 VGPRs gives 8 waves per SIMD, up to 64 gives 4, up to 128 gives 2, above 128 gives 1. A kernel that creeps past 128 registers halves its occupancy silently. And if the compiler is allowed fewer registers than the kernel needs, it "spills" values to scratch memory; in the microbenchmark, spills of 464/792/808 bytes cost +12/+25/+43% wall time even in a trivial loop.

### 2.4 Launch overhead and HIP graphs

A decoded token is 1,595 kernel launches per GPU (PERFORMANCE.md, steady state). Each launch has a fixed cost regardless of the work inside:

| How the kernel is launched | Cost per launch | Source |
| --- | --- | --- |
| Under the `rocprofv3` profiler | 4.2 us floor (profiler adds about 1.2 us) | HARDWARE.md, 2026-11-21, B |
| Bare eager launch from a host loop | 2.0 us (MI60), 2.6 us (MI50) | HARDWARE.md, 2026-11-21, B |
| Node in a captured, replayed HIP graph | 1.22 us pure floor; 1.5-1.9 us saved per removed launch including its trivial work | HARDWARE.md, 2026-11-21..22, A/B; the 1.22 us microbench source was not saved, the card says to rerun before relying on it |

A **HIP graph** (AMD's counterpart of a CUDA graph) records a sequence of kernel launches once and replays it with a single host call. The per-kernel host overhead goes away; a per-node dispatch cost remains.

Estimate (mine): 1,595 launches x 1.22 us = 1.9 ms, and x 1.7 us = 2.7 ms, out of a 26.3 ms token. Launch overhead is real but bounded, and a fusion that removes one launch per layer buys on the order of 0.1 ms.

### 2.5 The one-block rule

The one-block microbenchmark (HARDWARE.md, 2026-11-22, B) gives a blunt rule: a single thread block can pull only about 26-30 GB/s, roughly 1/30 of the card. A kernel that reduces a row inside one block costs about `2.9 us + 0.46 us per 1,000 floats`; spread over many blocks, equivalent work costs 2.3-2.8 us flat. So:

- Under about 5,000 floats: one block of 1024 threads is fine.
- Above about 5,000 floats: split into two launches (partial sums over many blocks, then a second pass), about 5.4 us flat up to 36,864 floats.
- Never stream weights through one block.

Section 6 shows this rule paying off on 27B.

## 3. Tensor parallelism without peer-to-peer

### 3.1 What gets split

One card alone runs this model at 23.42 tokens/s (2026-11-10, B): the mat-vec takes 37.40 ms per token at about 767 GB/s. Bytes are the bottleneck, so let two memory systems read half each. That is **tensor parallelism** (TP, `-sm tensor`): every weight matrix is cut in two and each card holds one half.

For a mat-vec `y = W x`, the split used here cuts along the input dimension, so each card holds part of every row (the meta backend's comment at `ggml/src/ggml-backend-meta.cpp:816`: "A source split along axis 0 holds only part of each row"). Each card computes a partial dot product for every output element:

```
            GPU 0                         GPU 1
   y0 = W[:, :K/2] . x[:K/2]      y1 = W[:, K/2:] . x[K/2:]
              \                          /
               +---- AllReduce (sum) ---+
                          |
                    y = y0 + y1   (now present on both cards)
```

Before the next nonlinear step both cards need the full sum, and that exchange-and-add is an **AllReduce**. On 27B it happens after the mixer (attention or recurrent block) and after the feed-forward block of every layer: 128 per token (PERFORMANCE.md).

Measured: 42.70 ms -> 26.70 ms per token, 1.60x from two cards (2026-11-10, B). Not 2x, because the non-weight work is replicated rather than halved, and the AllReduces cost time.

### 3.2 Host-staged AllReduce with kernel polling

Without P2P, the obvious implementation is: copy device-to-host, copy host-to-device, launch an add kernel, with the host CPU synchronizing each step. The Q3 microbenchmark prices that at 25-41 us for small payloads (2026-11-22, B). Times 128 per token, that would be 3-5 ms (estimate, mine).

The decode path avoids the host entirely. Both GPUs map the same pinned host memory and a single kernel on each card does three phases (`ggml/src/ggml-cuda/allreduce.cu:114-223`):

1. All threads convert their partial result to BF16 (16-bit floats, halving the bytes on the wire) and store it into the shared host buffer.
2. Thread 0 writes a token number into an "arrival" slot, then spins, reading the peer's arrival slot until it shows the same token:

```cpp
// ggml/src/ggml-cuda/allreduce.cu:181-193 (abridged)
ggml_cuda_ar_signal_set(my_slot, token);
__threadfence_system(); // make our signal visible system-wide
while (ggml_cuda_ar_signal_get(other_slot) != token) {
    __builtin_amdgcn_s_sleep(4);
}
```

3. All threads read the peer's values from host memory and add them. Both sides round their own value through BF16 first so the two cards end up with bit-identical sums.

The GPU itself waits for its peer; no host thread is in the loop. Traced cost is about 20 us per call on 27B, versus 40 us for the copy-based form at that payload size (HARDWARE.md).

For prefill the payload is megabytes, so the GPU's copy engine moves it while the compute engine keeps working. Two refinements landed on 2026-10-12 (B): separate streams per direction so transfers overlap ("duplex", PP 599 -> 647), and "slabs", where the GEMM result is produced in 4 pieces and each starts its transfer while the next is computed (PP2048 651.1 -> 682.0, +4.7%).

### 3.3 A cautionary tale: the guard that disabled it

This path came from upstream PR 27825. When first merged it did nothing: a stale preprocessor guard excluded HIP builds, and a slower fallback ran silently. Removing 11 lines gave 325.49 / 18.68 -> 394.17 / 25.89 tokens/s PP/TG at 16K depth: +21.10% PP, +38.57% TG (2026-09-10, B). The project instructions carry a note dated 2026-09-29 that excluding HIP from this path cost 21% PP2048 and 38% TG256 (grade D: no separate raw log; the timeline notes it matches the 2026-09-10 magnitude). Checking it is now mandatory after every upstream merge.

### 3.4 Making the whole token one graph

In the 2026-10-12 profile, whole-token graph replay never engaged: 129 separate dispatches per token. The cause was the AllReduce, which needed a fresh token number from the host on every call, so its arguments changed every time and nothing could be recorded once and replayed.

The fix was to move the counter onto the device:

```cpp
// ggml/src/ggml-cuda/allreduce.cu:139-147
// Token and slot come from a per-block device counter, not from the host.
// Both ranks run the same AR sequence, so their counters stay equal. This keeps the kernel valid in a replayed graph.
__shared__ int s_token;
if (tid == 0) {
    s_token = counter[bid] + 1;
}
```

Now the kernel's arguments are identical on every token, and all ~1,600 launches per card become one replayed graph (capture and replay at `ggml/src/ggml-cuda/ggml-cuda.cu:7741-7777`). Result: TG64 30.2 -> 34.77 tokens/s (2026-10-12, B). Dispatching both cards concurrently is worth another ~1.5 tokens/s (37.37 -> 35.83 when switched off, 2026-10-29, B).

## 4. Weight kernels

### 4.1 Q8_0 and why the stock layout hurts

Q8_0 stores weights in blocks of 32 signed bytes plus one 16-bit float scale: 34 bytes. A dot product of a weight row with an activation vector becomes, per block, `scale_w * scale_x * (integer dot product of 32 byte pairs)`, and the integer part is eight `v_dot4` instructions. Activations are quantized to bytes on the fly (the `quantize_q8_1` kernels).

The problem is the 34-byte block. The widest load moves 16 aligned bytes, and the interleaved 2-byte scales break that alignment.

### 4.2 The repacked layout

At model load, the fork rewrites every supported weight tensor into a two-plane layout (`ggml/src/ggml-cuda/q8_repack/repack-common.cuh`, described in MAINTENANCE.md):

```
canonical row:  [d][32 x int8][d][32 x int8][d][32 x int8] ...     (34 B blocks)

repacked:       qs plane:  row 0: [int8 int8 int8 ... ]  + pad
                           row 1: [int8 int8 int8 ... ]  + pad
                           ...
                d plane:   row 0: [d d d ...]  row 1: [d d d ...]  ...
```

Now every weight load is an aligned 16-byte load (four `v_dot4` operands), and a lane that handles several rows reads their scales from adjacent memory. One curious detail:

```cpp
// ggml/src/ggml-cuda/q8_repack/repack-common.cuh:31-38
template <typename T>
static __host__ __device__ inline T repack_qs_stride(const T ne0) {
    T rs = ne0;
    if (rs % 128 == 0) {
        rs += 16;
    }
    return rs;
}
```

If a row is a multiple of 128 bytes long, 16 padding bytes are added. The code comment says this is "so row starts do not alias HBM channels": a kernel reads many rows at the same offset at once, and with such a stride those addresses collide on the same memory channel. Shifting each row by 16 bytes spreads them out and keeps 16-byte alignment. This model's 5120-wide rows hit the condition. The docs give no isolated before/after number for the +16 alone, so treat the mechanism as the authors' stated rationale.

What the repack as a whole is worth: on 2026-09-10 turning it off cost 59.9% of PP and 9.49% of TG (B).

### 4.3 Four compute paths

The right kernel shape depends on how many tokens are in the batch. `ggml_cuda_mul_mat_repacked` (`q8_repack/mul-mat.cu:35`) dispatches to four paths (MAINTENANCE.md):

| Tokens in batch | Path | Idea |
| --- | --- | --- |
| 1 | single-token MMV | 64 lanes cooperate on one row; 8 rows per block |
| 2..8 (now 2..16, see section 7) | narrow multi-column MMV (`nc`) | read each weight once, dot it against every token's activations |
| above the narrow range, up to 127 | `w32` tiled GEMM | 32-token-wide tiles |
| 128 and up | `w64` tiled GEMM | 64-token-wide tiles |

**Single token.** The launch at `mul-mat.cu:18-24` uses blocks of 8 rows x 64 lanes. Its comment explains why: only one 16-wave group fits a CU, so smaller groups shorten the partly empty last dispatch round. Going from 16 to 8 rows per block was worth 36.2 -> 36.8 tokens/s (2026-10-12, B).

**Narrow multi-column.** When verifying speculative drafts (section 7) the batch is 3-6 tokens. The `nc` kernel reads each 16-byte weight chunk once and multiplies it against every column before moving on, so memory traffic stays near that of one token and only the cheap arithmetic grows. Geometry is chosen per width in `ggml_cuda_mul_mat_repacked_nc_t` (`mul-mat.cu:236-347`), which narrows from 64 lanes per row to 32 or 16 when rows are long enough; its comment notes that a 2048-wide row is too short to pay for narrower lane groups while a 5120-wide row is not. When this landed, an 8-token batch went from 57.0 ms to 45.9 ms (2026-10-12, B).

**Tiled GEMM.** For real prefill the kernel is `mmq_gemm_repacked` (`repack-kernels.cuh:946`), declared with `__launch_bounds__(64 * NRL, 2)`: 256 threads per block and at least 2 waves per SIMD, which caps it at 128 VGPRs.

### 4.4 Fusing gate, up and SwiGLU

The feed-forward block computes `down( silu(gate(x)) * up(x) )`, where `gate` and `up` are two large matrices applied to the same input. At one token, the fork runs both in a single kernel: each lane loads a chunk of `up` and the matching chunk of `gate`, reuses the same activation words for both, and applies the activation function at the end (`rp_mmv_fusion_epilogue`, `repack-kernels.cuh:59-84`). This saves launches and an intermediate buffer. It fires only at exactly one token and only when both matrices are repacked. In the 2026-10-12 trace, the 64 fused gate+up calls were 8.5 ms of the 19.3 ms mat-vec time.

### 4.5 Compiler flags

Two build flags are mandatory: `-mllvm -amdgpu-sched-strategy=max-ilp`, which asks the AMD backend to schedule for instruction-level parallelism, and `-O3`. Without them: PP 684 -> 669 (-2.2%), TG 36.8 -> 35.7 (-3.0%) (2026-10-29, B).

## 5. Attention

### 5.1 The KV cache stays F16

Attention needs the keys and values of every previous token: the **KV cache**. The fork keeps it in 16-bit floats and excludes quantized KV by policy (GOALS.md): accuracy first. There is little to regret. The one comparison in the timeline (on Flash Next, a different model, 2026-11-13, B) found Q8_0 KV 7% slower at 16K depth (34.2 vs 36.6 tokens/s) even though it halves KV memory: unpacking is arithmetic, the scarce resource here.

At depth 0, attention is 0.45 ms of a 27B token. At 32K context it is 3.34 ms, about 10% (2026-11-10, B, profiled), and plain TG drops from 37.45 to 33.48 tokens/s. Long context turns attention into a bandwidth problem of its own.

### 5.2 Two FlashAttention kernels

"FlashAttention" here means one kernel that streams over the KV cache in tiles and keeps a running softmax, instead of materializing a full score matrix. Two shapes are used:

- **vec** (`fattn-vec.cuh`): for decode. One or two query tokens against a long KV.
- **tile** (`fattn-tile.cuh`): for prefill. Many query columns per block against tiles of KV, without matrix units.

### 5.3 `kv_rows`: only read your own rows

When several server slots share one KV pool, their cells are interleaved in the same buffer. The simple way to keep them apart is a mask: scan the whole pool and add minus infinity to the scores of other users' cells. With four 16K-token users, each prefill then reads up to four times more KV than it needs.

`kv_rows` (a port of upstream #29510 to the gfx906 tile kernel) passes the kernel a sorted list of the rows that belong to this sequence. The kernel exploits the sorting:

```cpp
// ggml/src/ggml-cuda/fattn-tile.cuh:629-641 (abridged)
// kv_rows are sorted: if the tile is one run of cells, read it directly and gather only mixed tiles
const int r0 = KV_rows[k_VKQ_0];
if (r0 >= 0 && KV_rows[k_VKQ_0 + nbatch_fa - 1] - r0 == nbatch_fa - 1) {
    rows    = nullptr;
    k_row_0 = r0;
}
```

If a tile's rows are consecutive it is read as a plain contiguous block; only tiles that straddle two users' cells pay for an indexed gather.

Result (2026-11-15, A): four users x 16K prefill 462.3 -> 577.8 tokens/s, +25.0%; decode neutral. In a server test, per-user prefill speed without it fell 430 / 277 / 203 / 161 tokens/s as the pool filled; with it, 358 / 354 / 353 / 353 (those runs on hot, throttled GPUs).

Two details are instructive. A gate in `llama_kv_cache::get_n_kv_rows` (`src/llama-kv-cache.cpp:1556-1594`) turns the feature off for decode and draft verification, where the dense path reads each cell once for all slots and rows would read more. And the first two implementations lost: attempt 2 put the row logic into the shared tile kernel, which pushed every variant to the 128-VGPR cap with 57-97 spilled registers and cost 1.5% of single-user PP. The fix was a separate template instantiation (`use_kv_rows`) so the plain kernel keeps its registers.

### 5.4 The VGPR lesson: `__launch_bounds__(128, 4)`

On 2026-11-22 a trace showed `flash_attn_ext_vec` taking 85.8 us per call at 4K context (found on another model, Inkling 2 Mini, but the kernel is shared). The first diagnosis, from the one-block rule, was "one block per head is streaming 2.1 MB", predicting about 70 us. Close enough to look right, and wrong: the kernel already spread keys over many blocks. The real cause was that it used more than 128 VGPRs.

Apply the formula from section 2.3 (estimate, mine, using the HARDWARE.md formula): more than 128 VGPRs gives 1 wave per SIMD; blocks are 128 threads = 2 waves; so `floor(4 * 1 / 2) = 2` blocks per CU. HARDWARE.md records the same "2 b/CU".

The fix is one line:

```cpp
// ggml/src/ggml-cuda/fattn-vec.cuh:20-24
#ifdef GGML_USE_HIP
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 4) // more blocks per CU than 1-2: raises occupancy for decode
#else
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
#endif // GGML_USE_HIP
```

On HIP the second argument means "minimum waves per SIMD", so 4 caps the kernel at 64 VGPRs and the formula gives `floor(4 * 4 / 2) = 8` blocks per CU (the Q1 table's `T128_N200_L0K_M4` row confirms 8). On the model where it was found: +12.9% TG at 4K (A) and +21.0% at 16K (B). The docs record no 27B-specific number for this change. The lesson is general: read blocks, VGPRs, LDS and scratch from the trace before naming a mechanism.

### 5.5 GQA 6:1 and the open G1 idea

27B uses **grouped-query attention (GQA)** at 6:1: six query heads share each key/value head. Ideally a K/V tile is loaded once and all six heads are scored against it. The kernel's GQA fast path is compiled for group sizes that divide by 8, 4 or 2 (`fattn-tile.cuh`, `launch_fattn_tile_switch_ncols2`). Six divides only by 2, so two heads go per pass and the KV cache is read three times per token.

The proposed fix, "G1", pads the group to 8 so all six real heads share one pass, and four slots fit in one KV pass. Estimated gain (C): 1.7-2.0 ms per token at 32K and 5.5-6.5 ms at 100K. It is not built yet.

## 6. Small-kernel work

With the big kernels near their roofs, what remains is a long tail: only 433 of the 1,595 launches per token are weight mat-vecs (2026-11-10 trace). The rest are norms, adds, quantizes and state updates.

### 6.1 Two-launch reductions

After every AllReduce the model computes `rms_norm(a + b) * w` over a 5120-wide row and also produces the quantized bytes for the next mat-vec. This was already fused into one kernel, `add_rms_norm_mul_f32` (2026-10-12). But a normalization needs the sum of squares of the whole row before it can write any output, so the kernel ran as one 1024-thread block per row: 5,120 floats, just past the ~5,000 crossover of the one-block rule.

The 2026-11-22 version splits it (`ggml/src/ggml-cuda/norm.cu:934-981`, dispatch at `:1017-1022`): launch 1 runs 256-thread blocks that each add their slice and write one partial sum of squares; launch 2 runs blocks that each sum the few partials and write their slice of the output. More launches, less time: 11.37 -> 9.05 us traced per call, and 27B TG 37.35 -> 37.96 (+1.6%, A, 4 alternated pairs below 50 C), the first 27B decode kernel win in three weeks.

Earlier the same day, three micro-rewrites of the one-block kernel were neutral or slower. The structure was the problem, not the instructions.

### 6.2 What a removed launch is worth, and how the number kept changing

The working figure for "one fewer kernel launch" used to be 4-5 us. It came from profiler traces, where the smallest kernel shows 4.2 us, but 1.2 us of that is the profiler. A bare loop measures 2.0-2.6 us, and decode replays a graph, where the floor is 1.22 us per node. Fusion plans priced at 4.2 us overstated their gains by more than 2x; even 2.2 us over-prices graph-replayed decode by 1.2x-1.8x (HARDWARE.md).

PERFORMANCE.md keeps a "retired claims" table so these do not come back. A selection:

| Retired claim | What was actually true |
| --- | --- |
| Every removed launch saves 4-5 us | 4.2 us is the profiler floor; in a replayed graph it is 1.22 us, 1.5-1.9 us with trivial work |
| A 4.3 ms per-token host gap limits 27B | 27B is 96-99% GPU-busy in steady state; the gap was profiler cost |
| 27B runs 1,645 kernels per token | 1,595; the old script dropped a third of the launches and divided by the wrong token count |
| The 25-41 us copy microbenchmark proves decode AllReduce is at its floor | Category error: it measures copy-engine transfers plus host sync; decode AllReduce polls mapped memory from the GPU |

The lesson is not "profilers lie". A measurement instrument has its own cost model, and you calibrate it once (here with a bare-loop microbenchmark that runs in seconds) before trusting differences between its readings.

## 7. Speculative decoding

### 7.1 Why it works here

If decode is bandwidth-bound, the expensive thing is a pass over the weights, not the arithmetic done during it. The narrow multi-column kernel (section 4.3) lets one pass process several tokens. So if something cheap can guess the next few tokens, the big model can check all the guesses in one pass for little more than the price of one token.

The verify-width curve from 2026-11-22 (`llama-bench -p N -n 0`, ub 512) makes this concrete: a 1-token batch takes 30.2 ms and an 8-token batch 45.4 ms. Estimate (mine): (45.4 - 30.2) / 7 = about 2.2 ms per additional token, against about 30 ms for the first.

Verification keeps the output honest: the target model's own logits decide which draft tokens to keep, and the first wrong guess and everything after it are discarded. A 2026-10-12 check found 0 of 1,152 non-argmax tokens in plain, MTP2 and DFlash5 modes. Greedy text can still diverge from a non-speculative run, because batched arithmetic rounds differently by 0.001-0.007 nats and that flips near-ties.

### 7.2 MTP2

**MTP (multi-token prediction)** uses a small trained head shipped with the model. It takes the big model's last hidden state and the token just sampled, and predicts the following token; applied twice it drafts two tokens ("MTP2"). The driver is `common_speculative_impl_draft_mtp` (`common/speculative.cpp:2410`).

The measured step (2026-11-10, B): 40.6 ms producing 2.55 tokens on average at 0.777 draft acceptance, so 15.9 ms per token (62.8 tokens/s) versus 26.7 ms plain. Check (mine): 1 + 2 x 0.777 = 2.55. The docs' estimated split is about 31 ms for the 3-token verify, 4.5 ms for the two drafts and 5 ms of host sampling and rollback.

Deeper is not better. MTP3 was slower than MTP2 (61.5 vs 64.1 tokens/s, 2026-11-05, B): the third draft is accepted less often but always costs a draft step and a wider verify.

One supporting piece: the MTP head needs its own KV for the prompt. Building it per prefill chunk forced a synchronization each time, so the fork defers it and replays the whole prompt into the draft context once at the end (`finish_deferred_prefill`, `common/speculative.cpp:2831`).

### 7.3 DFlash

**DFlash** is a separate small draft model that reads hidden states tapped from the target and drafts up to 5 tokens per step: 54.9 ms for 3.88 tokens at 0.581 acceptance, about 70 tokens/s on short prompts (2026-11-10, B). Running the drafter on one GPU instead of across both made it competitive (5-prompt mean 47.1 -> 55.5 tokens/s, 2026-10-12, B). Its weakness is long context: at 32K, MTP2 gave 42.7 and DFlash5 39.9 tokens/s (2026-11-05, B, single runs). The default preset uses MTP2.

### 7.4 Today's finding: the 8 -> 9 token cliff

Now combine speculation with multiple users. Four users with MTP2 each need a 3-token verify, so the batch is 12 tokens. The narrow kernel was capped at 8 (`MMQ_RP_Q8_MMV_MAX_TOKENS`); above that, dispatch fell to the `w32` tiled GEMM, which, as the code comment says, "computes a full 32-wide tile whatever the batch is".

The curve (2026-11-22, `llama-bench -p N -n 0 -r 5`, ub 512):

| Tokens in batch | 1 | 8 | 9 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ms per batch, before | 30.2 | 45.4 | 93.6 | 95.5 | 98.1 |
| ms per batch, after | (unchanged) | (unchanged) | 62.8 | 66.9 | 78.8 |

A ninth token doubled the cost of the batch. Nine tokens in a 32-wide tile use 9/32 of the lanes: machinery built for compute-bound prefill was doing a bandwidth-bound job.

The fix (commit `ce504ae78`) adds no new kernel. For 9..16 tokens it runs the existing narrow kernel twice on two balanced halves:

```cpp
// ggml/src/ggml-cuda/q8_repack/mul-mat.cu:116-121
const int64_t n_chunks = (ne11 + MMQ_RP_Q8_MMV_MAX_TOKENS - 1) / MMQ_RP_Q8_MMV_MAX_TOKENS;
for (int64_t c = 0; c < n_chunks; c++) {
const int64_t c0 = ne11 * c / n_chunks;
const int64_t nc = ne11 * (c + 1) / n_chunks - c0;
const block_q8_1 * xq_c  = xq + c0 * x_stride;
float *            dst_c = dst_d + c0 * dst_s1;
```

Twelve tokens become 6 + 6. That reads the weights twice, which is wasteful on a bandwidth-bound machine: a true 12-column kernel would read them once. But two efficient passes beat one pass that idles most of its lanes: 95.5 -> 66.9 ms for the 12-token batch (-30%).

Serving result (4 users x 1.5K-token prompts, MTP2, ABBA with 2 seeds, grade A): total generation 76.4 / 76.3 / 75.8 / 74.1 tokens/s before versus 99.9 / 95.4 / 99.2 / 99.0 after. Means 75.7 -> 98.4, +30%; per user about 18.0 -> 22.9. Quality check against the old path: KL divergence 0.00042, top-1 agreement 98.97%. A kill switch (`GGML_CUDA_RP_NC_SPLIT=0`) restores the cap of 8.

About a dozen lines, reusing kernels that were already tested. The true wide kernel is left open.

## 8. Serving

`llama-server` runs **slots**: independent conversations decoded together in one batch. The `qwen27b` preset in `gfx906/server/models.ini` gives 4 slots one unified KV pool of 262,144 tokens: one user can use all of it, or four can share it.

What that buys (README serving table, `bench_serve.py`, greedy, single runs, 2026-11-22):

| Scenario | Per user | Total |
| --- | ---: | ---: |
| 1 user, short prompt | 56-61 t/s | - |
| 1 user, 100K-token context | 37.6 t/s | - |
| 2 users | about 39 t/s each | about 80 t/s |
| 4 users | about 23 t/s each | about 98 t/s |

Read the last rows carefully: with more users the total grows, not the per-user rate. Batching users into one weight pass is the same trick as speculative verification, and the two compete for the same narrow-batch headroom.

**The out-of-memory fix.** The same day's server check found a single 100K-token request crashing with `out of memory` on ROCm0. Loading the model by its Hugging Face name also loads its vision projector (about 0.9 GiB) onto ROCm0, leaving 0.93 GiB free. Setting `no-mmproj-offload = true` puts the projector on the CPU; free memory went 0.93 -> 2.07 GiB and the request ran. A quieter symptom of the same shortage: the morning's 4-user total (64.4 tokens/s) was below the afternoon baseline (75.7) because the attention workspace did not fit. A speed measured on a machine that is quietly out of memory is not a speed.

**The open problem: prefill starvation.** With four users each sending a 50K-token prompt, nothing crashed, but prefills ran one at a time (first tokens at 123 / 282 / 462 / 640 s) and users already generating dropped to 0.7-1.9 tokens/s for minutes (2026-11-22, B). A first probe capped the prefill chunk while others decode: at a cap of 128 tokens the decoding user got 4.0 instead of 1.9 tokens/s, while prefill slowed from 285 to 243 tokens/s (B, branch not merged). The docs conclude that chunk size is not the whole cause. For multi-user long-context work, this scheduling problem now matters more than any kernel.

## 9. The scoreboard

All rows are Qwen3.8-27B Q8_0, two GPUs, `-sm tensor`, F16 KV, from BENCHMARK-TIMELINE.md Section 1 and the dated records. "d" is the context depth already in the cache before the measurement.

| Date | What | Config | PP2048 t/s | TG t/s | MTP2, 1 user | 4-user total | Grade |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-09-10 | Clean upstream `434ddbbc0` | d16384, ub 4096, TG256 | 196.97 | 20.13 | - | - | B |
| 2026-09-10 | Fork merge + AllReduce guard fix | same | 394.17 | 25.89 | - | - | B |
| 2026-10-12 | Fork before the kernel push | d0, ub 2048, TG128 | 551.8 | 30.16 | - | - | B |
| 2026-10-12 | After graphs, fusions, duplex/slab AR, speculative push | same | 682.0 | 36.8 | 60.3-61.6 (CLI) | - | B |
| 2026-10-29 | Official upstream `36a0e2ffc` | d0, ub 2048, TG128, r2 | 448.60 | 32.77 | - | - | B |
| 2026-10-29 | Fork rebuilt on that upstream | same | 685.34 | 36.67 | 60.7 (CLI) | - | B |
| 2026-11-15 | `kv_rows` | d0; 4x16K multi-user script | 699.4 | 37.22 | - | 60.9 (server MTP2); 80.9 (plain batched bench) | A for the kv_rows A/B |
| 2026-11-21 | Catch-up, both ubatch sizes | d0 | 698.9 (ub 2048) / 614.5 (ub 512) | 37.23 / 37.21 | 63.9 | - | B |
| 2026-11-22 | Two-launch norm | d0, ub 512 | 614.0 | 37.96 | 62.1 (5-prompt, B); 56-61 (bench_serve, B) | 64.4 (memory-starved, B) | A |
| 2026-11-22 | 9..16 token split | server, 4 x 1.5K, MTP2 | - | - | - | 75.7 -> 98.4 | A |

Which comparisons are apples to apples:

- **Fork versus upstream**: only the two 2026-10-29 rows. Same day, flags and benchmark: +52.8% PP and +11.9% TG. The two 2026-09-10 rows are matched to each other, but at 16K depth with an older protocol, so they do not compare with later rows.
- **Before and after the October push**: the two 2026-10-12 rows.
- **The 30% multi-user gain**: the last row, alternated on one machine on one afternoon.

Which are not:

- PP "dropped" from about 699 to 614 only because reporting moved from ubatch 2048 to ubatch 512 on 2026-11-21; that row shows both on one build.
- The MTP2 column mixes `llama-cli` single prompts, a 5-prompt server suite and `bench_serve.py`. The docs warn that single-prompt speculative numbers swing 10-18% from acceptance noise alone.
- The 4-user column mixes workloads (a server check, a plain batched benchmark, and `bench_serve.py` with MTP2). Only the 75.7 -> 98.4 pair is a controlled comparison. An older script on 2026-11-11 recorded 41.2 (B).
- Upstream has moved since 2026-10-29 (the current base is `e6d624b9a`), and the docs contain no newer matched upstream run for 27B. Estimate (mine), mixing dates and so not a measured claim: 698.9 / 448.60 = +56% PP and 37.96 / 32.77 = +16% TG. The README states "~+53% PP / +14% TG vs upstream".
- The docs contain no upstream measurement of MTP2 or 4-user serving on this hardware, so those columns have no baseline.

## 10. The method

The list above could suggest a steady march. PERFORMANCE.md counts 28 routes that lost. What kept the project moving was the process, which it describes as a model-first workflow.

**Learn the hardware once.** Standalone microbenchmarks in `gfx906/bench/` measure occupancy, streaming, cross-GPU exchange, single-block cost and launch cost. The whole calibration took about 8 GPU-minutes, and its results live in HARDWARE.md as a table of constants. The rule: never spend an end-to-end run on a question a microbenchmark answers in seconds.

**Predict before you measure.** Every experiment starts with five lines: the mechanism, the predicted gain in ms per token from the hardware card, a cheap falsifier, a best-case bound, and what result would force a change to the card. Under 1% predicted, skip it. If measurement and prediction differ by more than about 20%, the card gets fixed. Section 5.4 is this loop working: the prediction was numerically close, mechanistically wrong, and caught by reading the launch geometry.

**ABBA, cool.** Because hot cards run 15-30% slower, every A/B starts below 60 C and alternates A B B A so drift cancels. This rule was paid for. An uncooled sweep on 2026-11-10 drifted to 31.7 tokens/s by its fifth configuration and was thrown away. A `kv_rows` attempt showed a false "+16% 4-way decode". The two-launch norm measured +2.9% hot and +1.6% cool and alternated. Cool run-to-run noise on 27B is about 0.3 tokens/s (0.8%), the smallest effect worth claiming.

**KLD, not greedy text.** "Does greedy decoding still print the same text?" is only a smoke test: a rounding change in the last bit flips a near-tie and the text diverges, while a real bug in a rarely used path might leave it intact. The fork runs `llama-perplexity --kl-divergence`, which compares full output distributions over many positions, paired with CPU-reference unit tests (`test-backend-ops`). For speculative modes it compares ms per verify step, never one prompt's tokens/s.

**Update the model, don't just log failures.** GOALS.md is explicit: a failed kernel must update the quantitative model of the GPU, and a lost implementation does not close a route. Three examples:

- *A faster mat-vec ("Q3").* In a standalone benchmark the decode mat-vec ran at 715-718 GB/s where the best possible stream for the same matrices was 795-800 (2026-11-10, B). The obvious fix is more loads in flight per lane. Measured: 2 / 3 / 4 / 5 loads gave 710 / 687 / 668 / 590 GB/s. Each extra in-flight load costs registers, and registers cost occupancy. The route was narrowed, not closed: a contiguous-block layout that needs no extra registers is untested, with a ceiling of 1.9 ms.
- *Uneven splits.* If the MI60 is stronger, give it more of the model (`-ts`). A cooled sweep (2026-11-10, A/B) showed every uneven split slowing decode: 1.1/1 gave +2.0% PP but -2.4% TG, and 1/1.03 lost on both. This updated the asymmetry model: streaming is equal on both cards, so decode has nothing to rebalance; the MI50's lag is in small replicated kernels that a weight split does not move.

The evidence grades are part of the method too. A "(B)" next to a number makes it obvious, months later, which conclusions rest on one run.

## 11. What is still open

- **G1, padded GQA** (section 5.5): read the KV once instead of three times for the 6:1 heads. Estimated 5.5-6.5 ms per token at 100K (C).
- **MoE narrow batches, and a true 9..16 column kernel**: the split in section 7.4 reads weights twice and covers dense models only. Mixture-of-experts (MoE) models still stop at 8 tokens: on Flash Next the 8 -> 9 jump is 67.2 -> 170.2 ms, and today's change only brought the 9-token batch to 145.9 ms.
- **Prefill starvation** (section 8): scheduling long prefills against other users' decode steps.
- **Smaller items**: fusing the recurrent layers' ~13 tiny kernels into 2-3 (estimated 0.6-0.8 ms per token, C); capping the decode AllReduce grid, where 5-6 idle blocks still execute three system-wide fences each (no gain estimate in the docs); GPU-side sampling under tensor split (estimated 1.5-3 ms per MTP2 step, C).

## Glossary

- **ABBA**: run order baseline, candidate, candidate, baseline, so slow drift (heat) cancels.
- **AllReduce (AR)**: exchange partial results between GPUs and sum them so each holds the full value.
- **CU, SIMD, wave, lane**: compute unit; one of its 4 execution engines; 64 threads executing together; one thread of a wave.
- **Decode / TG, prefill / PP**: generating tokens one at a time; processing the prompt in batches. TG128 and PP2048 are tokens/s over 128 generated or 2048 prompt tokens.
- **MoE**: mixture of experts; each token uses only a few of many expert weight blocks.
- **GEMM, MMV**: matrix times matrix; matrix times vector ("mat-vec").
- **GQA**: grouped-query attention; several query heads share one key/value head.
- **HIP graph**: a recorded kernel launch sequence replayed with one call (HIP is AMD's CUDA-like interface).
- **KLD**: Kullback-Leibler divergence between two builds' output distributions.
- **MTP, DFlash**: a draft head built into the model; a separate draft model fed by the target's hidden states.
- **Occupancy**: thread blocks resident per CU; set by VGPR count, block size and LDS use.
- **Q8_0 / Q8_1**: 8-bit formats with one scale per 32 values; Q8_1 is used for activations.
- **Repack**: rewriting weights at load into the two-plane layout the gfx906 kernels read.
- **Slot, ubatch**: one concurrent conversation in the server; the prompt tokens processed in one forward pass.
- **Spill**: register contents pushed to scratch memory because the kernel needs more registers than it is allowed.
- **Tensor parallelism (TP)**: splitting each weight matrix across GPUs so both work on every layer.
- **VGPR**: vector register; 256 per SIMD shared among resident waves.
