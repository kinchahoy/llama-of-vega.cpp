#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_rms_norm_cat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
