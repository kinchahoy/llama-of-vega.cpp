#include "res-mix.cuh"

#define RES_MIX_BLOCK_SMALL 256
#define RES_MIX_MAX_STATES (GGML_MAX_SRC - 1)

struct res_mix_ptrs {
    const float * p[RES_MIX_MAX_STATES];
};

template <int RES_MIX_BLOCK>
static __global__ void res_mix_f32(const float * query, const res_mix_ptrs ptrs, float * dst, const int n_states, const int64_t n_embd, const int64_t n_q) {
    const int64_t q = blockIdx.x % n_q;
    const int64_t t = blockIdx.x / n_q;
    const float * qv = query + q*n_embd;

    constexpr int n_warps = RES_MIX_BLOCK/WARP_SIZE;
    __shared__ float part[n_warps][2*RES_MIX_MAX_STATES];
    __shared__ float w[RES_MIX_MAX_STATES];
    __shared__ float inv_rms[RES_MIX_MAX_STATES];

    float ss[RES_MIX_MAX_STATES]  = {};
    float dot[RES_MIX_MAX_STATES] = {};
    for (int64_t i = threadIdx.x; i < n_embd; i += RES_MIX_BLOCK) {
        const float qi = qv[i];
#pragma unroll
        for (int s = 0; s < RES_MIX_MAX_STATES; ++s) {
            if (s < n_states) {
                const float xi = ptrs.p[s][t*n_embd + i];
                ss[s]  += xi*xi;
                dot[s] += xi*qi;
            }
        }
    }
#pragma unroll
    for (int s = 0; s < RES_MIX_MAX_STATES; ++s) {
        ss[s]  = warp_reduce_sum(ss[s]);
        dot[s] = warp_reduce_sum(dot[s]);
    }
    if (threadIdx.x % WARP_SIZE == 0) {
#pragma unroll
        for (int s = 0; s < RES_MIX_MAX_STATES; ++s) {
            part[threadIdx.x/WARP_SIZE][s] = ss[s];
            part[threadIdx.x/WARP_SIZE][RES_MIX_MAX_STATES + s] = dot[s];
        }
    }
    __syncthreads();
    if (threadIdx.x < n_states) {
        const int s = threadIdx.x;
        float sum_ss  = 0.0f;
        float sum_dot = 0.0f;
        for (int k = 0; k < n_warps; ++k) {
            sum_ss  += part[k][s];
            sum_dot += part[k][RES_MIX_MAX_STATES + s];
        }
        inv_rms[s] = 1.0f/sqrtf(sum_ss/n_embd + 1e-6f);
        w[s] = sum_dot*inv_rms[s]/sqrtf((float) n_embd);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_s = w[0];
        for (int s = 1; s < n_states; ++s) {
            max_s = fmaxf(max_s, w[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            w[s] = expf(w[s] - max_s);
            sum += w[s];
        }
        for (int s = 0; s < n_states; ++s) {
            w[s] = w[s]/sum*inv_rms[s];
        }
    }
    __syncthreads();

    for (int64_t i = threadIdx.x; i < n_embd; i += RES_MIX_BLOCK) {
        float acc = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            acc += w[s]*ptrs.p[s][t*n_embd + i];
        }
        dst[(t*n_q + q)*n_embd + i] = acc;
    }
}

#define RES_MIX_SPLIT_BLOCK 256

// split form, kernel 1: per (chunk, state, row) partial sum of squares and dot with the query
static __global__ void res_mix_partial_f32(const float * query, const res_mix_ptrs ptrs, float * part, const int64_t n_embd, const int64_t n_q, const int n_chunks) {
    const int     chunk = blockIdx.x;
    const int     s     = blockIdx.y;
    const int64_t q     = blockIdx.z % n_q;
    const int64_t t     = blockIdx.z / n_q;
    const int64_t i     = (int64_t) chunk*RES_MIX_SPLIT_BLOCK + threadIdx.x;

    float ss = 0.0f, dot = 0.0f;
    if (i < n_embd) {
        const float xi = ptrs.p[s][t*n_embd + i];
        ss  = xi*xi;
        dot = xi*query[q*n_embd + i];
    }
    ss  = warp_reduce_sum(ss);
    dot = warp_reduce_sum(dot);

    __shared__ float sh[2][RES_MIX_SPLIT_BLOCK/WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) {
        sh[0][threadIdx.x/WARP_SIZE] = ss;
        sh[1][threadIdx.x/WARP_SIZE] = dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.0f, b = 0.0f;
        for (int k = 0; k < RES_MIX_SPLIT_BLOCK/WARP_SIZE; ++k) {
            a += sh[0][k];
            b += sh[1][k];
        }
        float * o = part + (((int64_t) blockIdx.z*gridDim.y + s)*n_chunks + chunk)*2;
        o[0] = a;
        o[1] = b;
    }
}

// split form, kernel 2: each block sums the partials, does the softmax and writes its slice
static __global__ void res_mix_out_f32(const res_mix_ptrs ptrs, const float * part, float * dst, const int n_states, const int64_t n_embd, const int64_t n_q, const int n_chunks) {
    const int64_t q = blockIdx.y % n_q;
    const int64_t t = blockIdx.y / n_q;

    __shared__ float w[RES_MIX_MAX_STATES];
    __shared__ float inv_rms[RES_MIX_MAX_STATES];

    if (threadIdx.x < n_states) {
        const int s = threadIdx.x;
        const float * pp = part + ((int64_t) blockIdx.y*n_states + s)*n_chunks*2;
        float sum_ss  = 0.0f;
        float sum_dot = 0.0f;
        for (int k = 0; k < n_chunks; ++k) {
            sum_ss  += pp[2*k];
            sum_dot += pp[2*k + 1];
        }
        inv_rms[s] = 1.0f/sqrtf(sum_ss/n_embd + 1e-6f);
        w[s] = sum_dot*inv_rms[s]/sqrtf((float) n_embd);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_s = w[0];
        for (int s = 1; s < n_states; ++s) {
            max_s = fmaxf(max_s, w[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            w[s] = expf(w[s] - max_s);
            sum += w[s];
        }
        for (int s = 0; s < n_states; ++s) {
            w[s] = w[s]/sum*inv_rms[s];
        }
    }
    __syncthreads();

    const int64_t i = (int64_t) blockIdx.x*RES_MIX_SPLIT_BLOCK + threadIdx.x;
    if (i < n_embd) {
        float acc = 0.0f;
        for (int s = 0; s < n_states; ++s) {
            acc += w[s]*ptrs.p[s][t*n_embd + i];
        }
        dst[(t*n_q + q)*n_embd + i] = acc;
    }
}

void ggml_cuda_op_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];

    res_mix_ptrs ptrs = {};
    int n_states = 0;
    while (n_states < RES_MIX_MAX_STATES && dst->src[n_states + 1]) {
        ptrs.p[n_states] = (const float *) dst->src[n_states + 1]->data;
        n_states++;
    }

    const int64_t n_blocks = dst->ne[1]*dst->ne[2];
    if (n_blocks <= 64) {
        const int n_chunks = (query->ne[0] + RES_MIX_SPLIT_BLOCK - 1)/RES_MIX_SPLIT_BLOCK;
        ggml_cuda_pool_alloc<float> part(ctx.pool(), n_blocks*n_states*n_chunks*2);
        res_mix_partial_f32<<<dim3(n_chunks, n_states, n_blocks), RES_MIX_SPLIT_BLOCK, 0, ctx.stream()>>>(
            (const float *) query->data, ptrs, part.get(), query->ne[0], query->ne[1], n_chunks);
        res_mix_out_f32<<<dim3(n_chunks, n_blocks), RES_MIX_SPLIT_BLOCK, 0, ctx.stream()>>>(
            ptrs, part.get(), (float *) dst->data, n_states, query->ne[0], query->ne[1], n_chunks);
    } else {
        res_mix_f32<RES_MIX_BLOCK_SMALL><<<n_blocks, RES_MIX_BLOCK_SMALL, 0, ctx.stream()>>>(
            (const float *) query->data, ptrs, (float *) dst->data, n_states, query->ne[0], query->ne[1]);
    }
}
