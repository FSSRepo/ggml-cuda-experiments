#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include "cuda_info.h"
#include "utils.h"
#include "tensor-mma.h"
#include "flash-llama.h"
#include "flash_row_float.h"
#include "kernel_test.h"

#define PADD(x, n) (((x) + (n) - 1) & ~((n) - 1))

__global__ void mem_copy(const float* src, float* dst, int hdim) {
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    extern __shared__ float shared[];

    for(int off = 0; off < hdim; off += 80) {
        // shared[threadIdx.y*warpSize + threadIdx.x] = src[hdim*blockIdx.y + off + threadIdx.y*warpSize + threadIdx.x];
        // __syncthreads();
        cooperative_groups::memcpy_async(block, shared + off, src + off + hdim*blockIdx.y, sizeof(float) * 80);
    }
    cooperative_groups::wait(block);
    if((threadIdx.y*warpSize + threadIdx.x) < 80) {
        dst[128*blockIdx.y + threadIdx.y*warpSize + threadIdx.x] = shared[threadIdx.y*warpSize + threadIdx.x];
    }
}

void test_memasync() {
    int head_dim = 80, num_heads = 32;
    int head_dim_dest = 128;
    float* query =  (float*)malloc(head_dim           * num_heads * sizeof(float)); // assume batch size 1
    float* qkv_cuda = (float*)malloc(head_dim_dest         * num_heads * sizeof(float)); // assume batch size 1
    random(query, head_dim * num_heads);
    print_array("Source Q", query, 4, 16, head_dim);
    float *d_query, *d_qkv;
    cudaMalloc((void **)&d_query,  head_dim        * num_heads * sizeof(float));
    cudaMalloc((void **)&d_qkv,  head_dim_dest        * num_heads * sizeof(float));
    cudaMemset((void **)&d_qkv, 0,  head_dim_dest        * num_heads * sizeof(float));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_query, query, head_dim           * num_heads * sizeof(float), cudaMemcpyHostToDevice, stream);
    constexpr int num_warps = 4;
    int shmem = head_dim_dest*sizeof(float);
    dim3 grid_dim(1, num_heads, 1);
    dim3 block_dim(WARP_SIZE, num_warps, 1);
    cudaFuncSetAttribute(mem_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    mem_copy<<<grid_dim, block_dim, shmem, stream>>>(d_query, d_qkv, head_dim);
    cudaMemcpyAsync(qkv_cuda, d_qkv, head_dim_dest * num_heads * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    print_array("Dest Q", qkv_cuda, 4, 16, head_dim_dest);

    for(int h = 0; h < num_heads; h++) {
        for(int i = 0; i < head_dim; i++) {
            if(query[h * head_dim + i] != qkv_cuda[head_dim_dest*h + i]) {
                printf("error index %d %d\n", i, h);
                break;
            }
        }
    }
}

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
                ((half*)tensor_k->data)  + ((h % num_heads) * head_dim * kv_size),
                ((half*)tensor_mask->data),
                scores + (h * kv_size * batch_size),
                batch_size, kv_size, head_dim, scale, true);
            softmax(scores + h * kv_size * batch_size, kv_size, batch_size, h);
        }

        for(int h = 0; h < num_heads; h++) {
            mulmat_cpu(scores + (h * kv_size * batch_size),
                ((half*)tensor_v->data) + (h * head_dim*kv_size), nullptr,
                qkv_cuda + h * head_dim * batch_size, batch_size, head_dim, kv_size, 1.0f, true);
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

        half *d_query_fp16, *d_key, *d_value, *d_padded_mask;
        float *d_qkv, *d_query;

        cudaMalloc((void **)&d_query,   head_dim * batch_size * num_heads * sizeof(float));
        cudaMalloc((void **)&d_query_fp16,   head_dim * batch_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_key,     head_dim * kv_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_value,   head_dim * kv_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_padded_mask, PADD(batch_size, 32) * kv_size * sizeof(half));
        cudaMalloc((void **)&d_qkv,     head_dim * batch_size * num_heads * sizeof(float));

        // transpose value
        for(int h = 0; h < num_heads; h++) {
            for(int c = 0; c < head_dim; c++) {
                for(int r = 0; r < kv_size; r++) {
                    value_T[h * kv_size * head_dim + r * head_dim + c] = ((half*)tensor_v->data)[h * kv_size * head_dim + c*kv_size + r];
                }
            }
        }

        // for(int h = 0; h < num_heads; h++) {
        //     for(int c = 0; c < head_dim; c++) {
        //         for(int r = 0; r < kv_size; r++) {
        //             value_T[h * kv_size * head_dim + c*kv_size + r] = ((half*)tensor_v->data)[h * kv_size * head_dim + r * head_dim + c];
        //         }
        //     }
        // }

        half* q_f16 = (half*)malloc(head_dim * batch_size * num_heads * sizeof(half));
        for(int h = 0; h < head_dim * batch_size * num_heads; h++) {
            q_f16[h] = __float2half(((float*)tensor_q->data)[h]);
        }

        cudaMemcpyAsync(d_query, tensor_q->data,   head_dim * batch_size * num_heads * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_query_fp16, q_f16,   head_dim * batch_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_key,   tensor_k->data,   head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_value, tensor_v->data,   head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        // cudaMemcpyAsync(d_value, value_T,          head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_padded_mask,  tensor_mask->data,  PADD(batch_size, 32) * kv_size * sizeof(half), cudaMemcpyHostToDevice, stream);

        fill_buffer(qkv_cuda, 0.0f, head_dim * batch_size * num_heads);

        // launch llama.cpp implementation
        if(false) {
            constexpr int nqpb = 16;
            constexpr int ncpw = 128;

            const int nwarps = 2;

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
                head_dim * num_heads * 4, head_dim * 4, head_dim * num_heads * 4, // nb query
                head_dim * num_heads * 2, head_dim * 2, head_dim * num_heads * 2, // nb key value
                // head_dim * 2, head_dim * kv_size * 2, head_dim * kv_size * num_heads * 2, // nb key value
                head_dim, num_heads, batch_size, 1);
        } else if(batch_size == 1) {
            if(false) {
                const int num_warps = 2;
                constexpr int kv_per_block = 256;

                // assert(kv_size % kv_per_block == 0);
                dim3 grid_dim(kv_size / kv_per_block, num_heads, 1);
                dim3 block_dim(WARP_SIZE, num_warps, 1);

                int shmem =
                    head_dim*2*sizeof(half) /* query buffer */ +
                    (kv_per_block + 2)*sizeof(float) /* scores buffer */ +
                    num_warps * (256 + 2) * sizeof(float) /* tensor core result buffer per warp */;
                int num_kv_blocks = kv_size / kv_per_block;
                half* d_temporal;
                cudaMalloc((void **)&d_temporal,  ((num_kv_blocks * head_dim) + num_kv_blocks*2) * num_heads * sizeof(half));

                flash_attn_row<128, num_warps, 2, kv_per_block><<<grid_dim, block_dim, shmem, stream>>>(d_query, d_key, d_value, d_padded_mask, d_temporal, kv_size, scale, head_dim*kv_size, 1);
                fa_reduce<128, num_warps><<<num_heads, block_dim, shmem, stream>>>(d_temporal, d_qkv, kv_size, kv_size / kv_per_block, 1);
            } else {
                printf("FLASH DECODING\n");
                const int num_warps = 1;
                constexpr int kv_per_block = 256;
                const int num_kv_blocks = kv_size / kv_per_block;

                dim3 grid_dim(num_kv_blocks, num_heads, 1);
                dim3 block_dim(WARP_SIZE, num_warps, 1);
                const int tensor_elements = 256;

                int shmem = tensor_elements*sizeof(half) + // query
                    kv_per_block*head_dim*sizeof(half) + // kv size
                    num_warps*2*sizeof(half) + // softmax
                    num_warps*tensor_elements*sizeof(float); // query

                printf("Shared memory: %.2f KB\n", shmem / 1024.0f);

                cudaFuncSetAttribute(flash_attn_row_fast<128, num_warps, kv_per_block>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);

                half* d_temporal;
                cudaMalloc((void **)&d_temporal,  ((num_kv_blocks * head_dim) + num_kv_blocks*2) * num_heads * sizeof(half));

                // flash_attn_row_fast<128, num_warps, kv_per_block><<<grid_dim, block_dim, shmem, stream>>>(
                //     d_query_fp16, d_key, d_value, d_padded_mask, d_temporal, scale, kv_size, ((num_kv_blocks * head_dim) + num_kv_blocks*2), 1, 
                //     head_dim*num_heads, head_dim);
                fa_reduce<128, num_warps><<<num_heads, block_dim, shmem, stream>>>(d_temporal, d_qkv, kv_size, kv_size / kv_per_block, 1);
            }
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
    // test_memasync();
    // kernel_test(argc, argv);
    test_llama();
    return 0;
}