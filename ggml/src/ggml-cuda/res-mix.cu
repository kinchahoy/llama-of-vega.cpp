#include "res-mix.cuh"

#define RES_MIX_BLOCK_SMALL 256
#define RES_MIX_BLOCK_LARGE 1024
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
        res_mix_f32<RES_MIX_BLOCK_LARGE><<<n_blocks, RES_MIX_BLOCK_LARGE, 0, ctx.stream()>>>(
            (const float *) query->data, ptrs, (float *) dst->data, n_states, query->ne[0], query->ne[1]);
    } else {
        res_mix_f32<RES_MIX_BLOCK_SMALL><<<n_blocks, RES_MIX_BLOCK_SMALL, 0, ctx.stream()>>>(
            (const float *) query->data, ptrs, (float *) dst->data, n_states, query->ne[0], query->ne[1]);
    }
}
