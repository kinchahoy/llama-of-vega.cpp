#include "sconv-res.cuh"

#define SCONV_RES_BLOCK 256
#define SCONV_RES_N_CONV 4

static __global__ void sconv_res_f32(const float * x, const float * weight, const float * cur, const float * dynamic, float * dst,
        const int64_t n_channels, const int64_t n_tokens, const int64_t n_seq_tokens, const int64_t x_ne0, const int64_t n_heads, const float scale) {
    const int64_t i = (int64_t) blockIdx.x*SCONV_RES_BLOCK + threadIdx.x;
    if (i >= n_channels*n_tokens) {
        return;
    }
    const int64_t c   = i % n_channels;
    const int64_t tok = i / n_channels;
    const int64_t s   = tok / n_seq_tokens;
    const int64_t t   = tok % n_seq_tokens;

    const float * xc = x + (s*n_channels + c)*x_ne0 + t;
    const float * w  = weight + c*SCONV_RES_N_CONV;

    float conv = 0.0f;
    for (int j = 0; j < SCONV_RES_N_CONV; ++j) {
        conv += xc[j]*w[j];
    }
    float acc = cur[i] + conv;
    if (dynamic) {
        const float * d = dynamic + (tok*n_heads + c/(n_channels/n_heads))*SCONV_RES_N_CONV;
        for (int lag = 0; lag < SCONV_RES_N_CONV; ++lag) {
            acc += xc[SCONV_RES_N_CONV - 1 - lag]*d[lag];
        }
    }
    dst[i] = acc*scale;
}

void ggml_cuda_op_sconv_res(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * dynamic = dst->src[3];
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));

    const int64_t n = ggml_nelements(dst);
    sconv_res_f32<<<(n + SCONV_RES_BLOCK - 1)/SCONV_RES_BLOCK, SCONV_RES_BLOCK, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) dst->src[1]->data, (const float *) dst->src[2]->data,
        dynamic ? (const float *) dynamic->data : nullptr, (float *) dst->data,
        dst->ne[0], dst->ne[1], dst->ne[1]/x->ne[2], x->ne[0], dynamic ? dynamic->ne[0]/SCONV_RES_N_CONV : 1, scale);
}
