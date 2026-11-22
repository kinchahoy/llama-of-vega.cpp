#include "clip-rows.cuh"

#define CLIP_ROWS_BLOCK 128

static __global__ void clip_rows_f32(const float * x, const float * scale, float * dst, const int64_t ne0, const int64_t ne1, const float threshold) {
    const int64_t row = blockIdx.x;
    const float * xr  = x + row*ne0;
    const float * sr  = scale ? scale + (row % ne1)*ne0 : nullptr;

    float ss = 0.0f;
    for (int64_t i = threadIdx.x; i < ne0; i += CLIP_ROWS_BLOCK) {
        const float v = sr ? xr[i]*sr[i] : xr[i];
        ss += v*v;
    }

    __shared__ float buf[CLIP_ROWS_BLOCK];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = CLIP_ROWS_BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            buf[threadIdx.x] += buf[threadIdx.x + s];
        }
        __syncthreads();
    }

    const float factor = fminf(fmaxf(threshold/(sqrtf(buf[0]) + 1e-6f), 0.0f), 1.0f);
    for (int64_t i = threadIdx.x; i < ne0; i += CLIP_ROWS_BLOCK) {
        dst[row*ne0 + i] = (sr ? xr[i]*sr[i] : xr[i])*factor;
    }
}

void ggml_cuda_op_clip_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * scale = dst->src[1];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float threshold = ggml_get_op_params_f32(dst, 0);

    clip_rows_f32<<<ggml_nrows(a), CLIP_ROWS_BLOCK, 0, ctx.stream()>>>(
        (const float *) a->data, scale ? (const float *) scale->data : nullptr, (float *) dst->data, a->ne[0], a->ne[1], threshold);
}
