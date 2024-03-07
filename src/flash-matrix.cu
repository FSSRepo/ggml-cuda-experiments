#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include "cuda_info.h"
#include "utils.h"
#include "tensor-mma.h"
#include "flash-llama.h"
#include "flash_row_float.h"
#include "kernel_test.h"

#define PADD(x, n) (((x) + (n) - 1) & ~((n) - 1))

void test_llama() {
    // cpu data
    tensor* tensor_q =              load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\fa-cuda-q-256.tensor");
    tensor* tensor_k =              load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\fa-cuda-k-256.tensor");
    tensor* tensor_v =              load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\fa-cuda-v-256.tensor");
    tensor* tensor_mask =           load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\fa-cuda-mask-256.tensor");
    tensor* tensor_qkv_ref =        load_tensor_from_file("C:\\proyectos\\kernel-data\\tg\\fa-cuda-qkv-256.tensor");

    // test params
    int head_dim = 128, kv_size = 256, batch_size = 1, num_heads = 32;
    float* qkv_cpu =    (float*)malloc(head_dim * batch_size * num_heads * sizeof(float));
    float* qkv_cuda =   (float*)malloc(head_dim * batch_size * num_heads * sizeof(float));
    float* scores =     (float*)malloc(batch_size * kv_size  * num_heads * sizeof(float));
    half* value_T =     (half*)malloc(head_dim * kv_size  * num_heads * sizeof(half));

    float scale = 1.0f / sqrtf((float)head_dim);

    {
        // cpu
        for(int h = 0; h < num_heads; h++) {
            mulmat_cpu(
                ((float*)tensor_q->data) + (h * head_dim * batch_size),
                ((half*)tensor_k->data)  + (h * head_dim * kv_size),
                ((half*)tensor_mask->data),
                scores + (h * kv_size * batch_size),
                batch_size, kv_size, head_dim, scale, true);
            softmax(scores + h * kv_size * batch_size, kv_size, batch_size);
        }

        for(int h = 0; h < num_heads; h++) {
            mulmat_cpu(scores + (h * kv_size * batch_size),
                ((half*)tensor_v->data) + (h * head_dim*kv_size), nullptr,
                qkv_cuda + h * head_dim * batch_size, batch_size, head_dim, kv_size, 1.0f, false);
        }

        // permute
        for(int h = 0; h < num_heads; h++) {
            for(int b = 0; b < batch_size; b++) {
                for(int i = 0; i < head_dim; i++) {
                    qkv_cpu[b*num_heads*head_dim + h*head_dim + i] = qkv_cuda[h*batch_size*head_dim + b*head_dim + i];
                }
            }
        }
    }

    // load to cuda
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        half *d_key, *d_value, *d_padded_mask;
        float *d_qkv, *d_query;

        cudaMalloc((void **)&d_query,   head_dim * batch_size * num_heads * sizeof(float));
        cudaMalloc((void **)&d_key,     head_dim * kv_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_value,   head_dim * kv_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_padded_mask, PADD(batch_size, 32) * kv_size * sizeof(half));
        cudaMalloc((void **)&d_qkv,     head_dim * batch_size * num_heads * sizeof(float));

        // transpose value
        for(int h = 0; h < num_heads; h++) {
            for(int c = 0; c < head_dim; c++) {
                for(int r = 0; r < kv_size; r++) {
                    value_T[h * kv_size * head_dim + c * kv_size + r] = ((half*)tensor_v->data)[h * kv_size * head_dim + r*head_dim + c];
                }
            }
        }

        cudaMemcpyAsync(d_query, tensor_q->data,   head_dim * batch_size * num_heads * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_key,   tensor_k->data,   head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        // cudaMemcpyAsync(d_value, tensor_v->data,   head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_value, value_T,          head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_padded_mask,  tensor_mask->data,  PADD(batch_size, 32) * kv_size * sizeof(half), cudaMemcpyHostToDevice, stream);

        fill_buffer(qkv_cuda, 0.0f, head_dim * batch_size * num_heads);

        // launch llama.cpp implementation
        if(false) {
            constexpr int nqpb = 16;
            constexpr int ncpw = 128;

            const int nwarps = batch_size <= nqpb ? max(2, std::min((int) kv_size/ncpw, 8)) : 1;

            dim3 blocks_num((batch_size + nqpb - 1) / nqpb, num_heads, 1);
            dim3 block_dim(32, nwarps, 1);

            const size_t shmem_f_ = nqpb*(head_dim + nwarps*(ncpw + nqpb))*(sizeof(float)/2);

            // printf("ne00= %d, ne01= %d, ne02= %d, ne03= %d\nne10= %d, ne11= %d, ne12= %d, ne13= %d\nne31= %d, nb31= %d\nnb01= %d, nb02= %d, nb03= %d\nnb11= %d, nb12= %d, nb13= %d\nne0= %d, ne1= %d, ne2= %d, ne3= %d\n",
            //     head_dim, batch_size, num_heads, 1, // query
            //     head_dim, kv_size, num_heads, 1, // key value
            //     PADD(batch_size, 32), kv_size * 2, // mask
            //     head_dim * 4, head_dim * batch_size * 4, head_dim * batch_size * num_heads * 4, // nb query
            //     head_dim * 2, head_dim * kv_size * 2, head_dim * kv_size * num_heads * 2, // nb key value
            //     head_dim, num_heads, batch_size, 1);

            flash_attn_ext_f16<128, nqpb, ncpw><<<blocks_num, block_dim, shmem_f_, stream>>>(
                (const char*)d_query, (const char*)d_key, (const char*)d_value, (const char*)d_padded_mask, d_qkv, scale,
                head_dim, batch_size, num_heads, 1, // query
                head_dim, kv_size, num_heads, 1, // key value
                PADD(batch_size, 32), kv_size * 2, // mask
                head_dim * 4, head_dim * batch_size * 4, head_dim * batch_size * num_heads * 4, // nb query
                head_dim * 2, head_dim * kv_size * 2, head_dim * kv_size * num_heads * 2, // nb key value
                head_dim, num_heads, batch_size, 1);
        } else if(batch_size == 1) {
            const int num_warps = 8;
            constexpr int kv_per_block = 256;

            // assert(kv_size % kv_per_block == 0);
            dim3 grid_dim(kv_size / kv_per_block, num_heads, 1);
            dim3 block_dim(WARP_SIZE, num_warps, 1);

            int shmem =
                head_dim*2*sizeof(half) /* query buffer */ +
                (kv_per_block + 2)*sizeof(float) /* scores buffer */ +
                num_warps * (256 + 2) * sizeof(float) /* tensor core result buffer per warp */;

            int reduce_block = ((grid_dim.x + WMMA_M - 1) / WMMA_M) * WMMA_N;
            flash_attn_row<128, num_warps, 2, kv_per_block><<<grid_dim, block_dim, shmem, stream>>>(d_query, d_key, d_value, d_padded_mask, kv_size, scale, reduce_block, head_dim*kv_size);
            fa_reduce<128, num_warps><<<num_heads, block_dim, shmem, stream>>>(d_key, d_qkv, kv_size, kv_size / kv_per_block, reduce_block);
        }

        cudaMemcpyAsync(qkv_cuda, d_qkv, head_dim * batch_size * num_heads * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        print_array("Real Reference", (float*)tensor_qkv_ref->data, num_heads > 4 ? 4 : 1, 16, head_dim);
        print_array("CPU Compute", qkv_cpu, num_heads > 4 ? 4 : 1, 16, head_dim);
        print_array("CUDA Compute", qkv_cuda, num_heads > 4 ? 4 : 1, 16, head_dim);

        float max_diff = 0.0f;
        int head_idx = -1, batch_idx = -1, dim_idx = -1;
        int diff_count = 0;

        for(int h = 0; h < num_heads; h++) {
            for(int b = 0; b < batch_size; b++) {
                for(int i = 0; i < head_dim; i++) {
                    int index = h*batch_size*head_dim + b*head_dim + i;
                    if(fabs(qkv_cpu[index] - qkv_cuda[index]) > max_diff) {
                        max_diff = fabs(qkv_cpu[index] - qkv_cuda[index]);
                        head_idx = h;
                        batch_idx = b;
                        dim_idx = i;
                        diff_count++;
                    }
                }
            }
        }

        int diff_index = head_idx*batch_size*head_dim + batch_idx*head_dim + dim_idx;
        if(head_idx != -1 && batch_idx != -1 && dim_idx != -1) {
            printf("CPU (%.4f) CUDA(%.4f) diff: %.4f %d - head = %d, batch = %d, dim = %d\n", qkv_cpu[diff_index], qkv_cuda[diff_index], max_diff, diff_count, head_idx, batch_idx, dim_idx);
        } else {
            printf("No difference CPU - CUDA\n");
        }
        max_diff = 0.0f;
        diff_count = 0;
        // compare cpu computed with reference
        for(int h = 0; h < num_heads; h++) {
            for(int b = 0; b < batch_size; b++) {
                for(int i = 0; i < head_dim; i++) {
                    int index = h*batch_size*head_dim + b*head_dim + i;
                    if(fabs(((float*)tensor_qkv_ref->data)[index] - qkv_cpu[index]) > max_diff) {
                        max_diff = fabs(((float*)tensor_qkv_ref->data)[index] - qkv_cpu[index]);
                        head_idx = h;
                        batch_idx = b;
                        dim_idx = i;
                        diff_count++;
                    }
                }
            }
        }

        diff_index = head_idx*batch_size*head_dim + batch_idx*head_dim + dim_idx;
        if(head_idx != -1 && batch_idx != -1 && dim_idx != -1) {
            printf("REF (%.4f) CPU(%.4f) diff: %.4f %d - head = %d, batch = %d, dim = %d\n", ((float*)tensor_qkv_ref->data)[diff_index], qkv_cpu[diff_index], max_diff, diff_count, head_idx, batch_idx, dim_idx);
        } else {
            printf("No difference REF - CPU\n");
        }
        max_diff = 0.0f;
        diff_count = 0;
        // compare reference with cuda computed
        for(int h = 0; h < num_heads; h++) {
            for(int b = 0; b < batch_size; b++) {
                for(int i = 0; i < head_dim; i++) {
                    int index = h*batch_size*head_dim + b*head_dim + i;
                    if(fabs(((float*)tensor_qkv_ref->data)[index] - qkv_cuda[index]) > max_diff) {
                        max_diff = fabs(((float*)tensor_qkv_ref->data)[index] - qkv_cuda[index]);
                        head_idx = h;
                        batch_idx = b;
                        dim_idx = i;
                        diff_count++;
                    }
                }
            }
        }

        diff_index = head_idx*batch_size*head_dim + batch_idx*head_dim + dim_idx;
        if(head_idx != -1 && batch_idx != -1 && dim_idx != -1) {
            printf("REF (%.4f) CUDA(%.4f) diff: %.4f %d - head = %d, batch = %d, dim = %d\n", ((float*)tensor_qkv_ref->data)[diff_index], qkv_cuda[diff_index], max_diff, diff_count, head_idx, batch_idx, dim_idx);
        } else {
            printf("No difference REF - CUDA\n");
        }
    }
}

int main(int argc,const char* argv[]) {
    kernel_test(argc, argv);
    // test_llama();
    return 0;
}