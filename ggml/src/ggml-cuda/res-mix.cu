#include "res-mix.cuh"

#define RES_MIX_BLOCK 256
#define RES_MIX_MAX_STATES (GGML_MAX_SRC - 1)

struct res_mix_ptrs {
    const float * p[RES_MIX_MAX_STATES];
};

static __global__ void res_mix_f32(const float * query, const res_mix_ptrs ptrs, float * dst, const int n_states, const int64_t n_embd, const int64_t n_q) {
    const int64_t q = blockIdx.x % n_q;
    const int64_t t = blockIdx.x / n_q;
    const float * qv = query + q*n_embd;

    __shared__ float buf[RES_MIX_BLOCK];
    __shared__ float w[RES_MIX_MAX_STATES];
    __shared__ float inv_rms[RES_MIX_MAX_STATES];

    for (int s = 0; s < n_states; ++s) {
        const float * x = ptrs.p[s] + t*n_embd;
        float ss  = 0.0f;
        float dot = 0.0f;
        for (int64_t i = threadIdx.x; i < n_embd; i += RES_MIX_BLOCK) {
            ss  += x[i]*x[i];
            dot += x[i]*qv[i];
        }
        for (int part = 0; part < 2; ++part) {
            buf[threadIdx.x] = part == 0 ? ss : dot;
            __syncthreads();
            for (int r = RES_MIX_BLOCK/2; r > 0; r >>= 1) {
                if (threadIdx.x < r) {
                    buf[threadIdx.x] += buf[threadIdx.x + r];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                if (part == 0) {
                    inv_rms[s] = 1.0f/sqrtf(buf[0]/n_embd + 1e-6f);
                } else {
                    w[s] = buf[0]*inv_rms[s]/sqrtf((float) n_embd);
                }
            }
            __syncthreads();
        }
    }

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

    res_mix_f32<<<dst->ne[1]*dst->ne[2], RES_MIX_BLOCK, 0, ctx.stream()>>>(
        (const float *) query->data, ptrs, (float *) dst->data, n_states, query->ne[0], query->ne[1]);
}
