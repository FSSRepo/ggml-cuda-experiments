#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "ggml/ggml.h"

int main(void)
{
    ggml_time_init();
    int KW = 3, KH = 3, IC = 640, OC = 640;
    int IW = 32, IH = 48, /* IC = 640 */ N = 1;

    int s0 = 1;
    int s1 = 1;
    int p0 = 1;
    int p1 = 1;
    int d0 = 1;
    int d1 = 1;

    // Initialize adata
    float* adata = (float*)malloc(KW * KH * IC * OC * sizeof(float));
    for (size_t i = 0; i < KW * KH * IC * OC; i++) {
        adata[i] = 1.0f;
    }

    // Convert adata to fp16 format
    uint16_t* hadata = (uint16_t*)malloc(KW * KH * IC * OC * sizeof(uint16_t));
    ggml_fp32_to_fp16_row(adata, hadata, KW * KH * IC * OC);

    // Initialize bdata
    float* bdata = (float*)malloc(IW * IH * IC * N * sizeof(float));
    for (size_t i = 0; i < IW * IH * IC * N; i++) {
        bdata[i] = 1.0f;
    }

    struct ggml_init_params params_ctx;
    params_ctx.mem_size = 200 * 1024 * 1024;
    params_ctx.mem_buffer = NULL;
    params_ctx.no_alloc = false;

    struct ggml_context* ctx = ggml_init(params_ctx);
    
    struct ggml_tensor* ha = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KW, KH, IC, OC);
    memcpy(ha->data, hadata, KW * KH * IC * OC * sizeof(uint16_t));

    struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, IC, N);
    memcpy(b->data, bdata, IW * IH * IC * N * sizeof(float));

    struct ggml_tensor* result = ggml_conv_2d(ctx, ha, b, s0, s1, p0, p1, d0, d1);

    struct ggml_cgraph gf = ggml_build_forward(result);

    ggml_graph_compute_with_ctx(ctx, &gf, 6);

    ggml_graph_print(&gf);

    const float* ref = (float*)(result->data);

    printf("conv2d:\n%.2f %.2f %.2f %.2f\n%.2f %.2f %.2f %.2f\n",
                ref[0], ref[1], ref[2],
                ref[3], ref[4], ref[5],
                ref[6], ref[7]);

    free(adata);
    free(hadata);
    free(bdata);

    ggml_free(ctx);

    return 0;
}