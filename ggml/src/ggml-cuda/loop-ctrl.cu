#include "loop-ctrl.cuh"

#define LOOP_CTRL_BLOCK 256

static __global__ void loop_gate_f32(const float * cur, const float * message, const float * gain, const float * forget, float * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*LOOP_CTRL_BLOCK + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = cur[i] - tanhf(forget[i])*cur[i] + tanhf(gain[i])*message[i];
}

static __global__ void rms_norm_cat_f32(const float * a, const float * b, float * dst, const int64_t n, const float eps) {
    const int64_t t = blockIdx.x;
    const float * x[2] = { a + t*n, b + t*n };

    __shared__ float buf[LOOP_CTRL_BLOCK];
    __shared__ float scale[2];

    for (int k = 0; k < 2; ++k) {
        float ss = 0.0f;
        for (int64_t i = threadIdx.x; i < n; i += LOOP_CTRL_BLOCK) {
            ss += x[k][i]*x[k][i];
        }
        buf[threadIdx.x] = ss;
        __syncthreads();
        for (int r = LOOP_CTRL_BLOCK/2; r > 0; r >>= 1) {
            if (threadIdx.x < r) {
                buf[threadIdx.x] += buf[threadIdx.x + r];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scale[k] = 1.0f/sqrtf(buf[0]/n + eps);
        }
        __syncthreads();
    }

    for (int k = 0; k < 2; ++k) {
        for (int64_t i = threadIdx.x; i < n; i += LOOP_CTRL_BLOCK) {
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
    rms_norm_cat_f32<<<dst->ne[1], LOOP_CTRL_BLOCK, 0, ctx.stream()>>>(
        (const float *) dst->src[0]->data, (const float *) dst->src[1]->data, (float *) dst->data, dst->src[0]->ne[0], eps);
}
