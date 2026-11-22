#include "loop-ctrl.cuh"

#define LOOP_CTRL_BLOCK 256

static __global__ void loop_gate_f32(const float * cur, const float * message, const float * gain, const float * forget, float * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*LOOP_CTRL_BLOCK + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = cur[i] - tanhf(forget[i])*cur[i] + tanhf(gain[i])*message[i];
}

template <int RMS_CAT_BLOCK>
static __global__ void rms_norm_cat_f32(const float * a, const float * b, float * dst, const int64_t n, const float eps) {
    const int64_t t = blockIdx.x;
    const float * x[2] = { a + t*n, b + t*n };

    constexpr int n_warps = RMS_CAT_BLOCK/WARP_SIZE;
    __shared__ float part[n_warps][2];
    __shared__ float scale[2];

    float ss[2] = {};
    for (int64_t i = threadIdx.x; i < n; i += RMS_CAT_BLOCK) {
        ss[0] += x[0][i]*x[0][i];
        ss[1] += x[1][i]*x[1][i];
    }
    ss[0] = warp_reduce_sum(ss[0]);
    ss[1] = warp_reduce_sum(ss[1]);
    if (threadIdx.x % WARP_SIZE == 0) {
        part[threadIdx.x/WARP_SIZE][0] = ss[0];
        part[threadIdx.x/WARP_SIZE][1] = ss[1];
    }
    __syncthreads();
    if (threadIdx.x < 2) {
        float sum = 0.0f;
        for (int k = 0; k < n_warps; ++k) {
            sum += part[k][threadIdx.x];
        }
        scale[threadIdx.x] = 1.0f/sqrtf(sum/n + eps);
    }
    __syncthreads();

    for (int k = 0; k < 2; ++k) {
        for (int64_t i = threadIdx.x; i < n; i += RMS_CAT_BLOCK) {
            dst[t*2*n + k*n + i] = x[k][i]*scale[k];
        }
    }
}

void ggml_cuda_op_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst);
    loop_gate_f32<<<(n + LOOP_CTRL_BLOCK - 1)/LOOP_CTRL_BLOCK, LOOP_CTRL_BLOCK, 0, ctx.stream()>>>(
        (const float *) dst->src[0]->data, (const float *) dst->src[1]->data,
        (const float *) dst->src[2]->data, (const float *) dst->src[3]->data, (float *) dst->data, n);
}

void ggml_cuda_op_rms_norm_cat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float eps;
    memcpy(&eps, dst->op_params, sizeof(eps));
    const float * a = (const float *) dst->src[0]->data;
    const float * b = (const float *) dst->src[1]->data;
    if (dst->ne[1] <= 64) {
        rms_norm_cat_f32<1024><<<dst->ne[1], 1024, 0, ctx.stream()>>>(a, b, (float *) dst->data, dst->src[0]->ne[0], eps);
    } else {
        rms_norm_cat_f32<LOOP_CTRL_BLOCK><<<dst->ne[1], LOOP_CTRL_BLOCK, 0, ctx.stream()>>>(a, b, (float *) dst->data, dst->src[0]->ne[0], eps);
    }
}
