#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct dimen {
        int x, y, z;
        dimen(int x_, int y_, int z_) {
                x = x_;
                y = y_;
                z = z_;
        }
        
        int ne() {
                return x * y * z;
        }
};

#define WARP_SIZE 32

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(__shfl_xor_sync(0xffffffff, x, mask, 32), x);
    }
    return x;
}


#define CUDA_FLASH_ATTENTION_BLOCK_SIZE 256

template<int block_size>
static __global__ void flash_attn_f32(const float* q, const float* k,const float* v, float* dst, float kq_scale,
        int d_head, int seq_len, int num_heads, int head_size) {
        const int head = blockIdx.x / seq_len;
        const int s = blockIdx.x % seq_len;
        const int tid = threadIdx.x;

        extern __shared__  char work_data[];
        float* S = (float*)work_data; // theorical sequent length: 12848, due memory per block limit
        float* warp_data = (float*)(work_data + seq_len * sizeof(float));

        // QK^T
        for(int is = tid; is < seq_len; is += block_size) {
                S[is] = 0.0f;
                int key_offset = is * d_head + head * head_size;
                int query_offset = s * d_head + head * head_size;
                for(int d = 0; d < d_head; d++) {
                        S[is] += k[key_offset + d] * q[query_offset + d];
                }
                S[is] *= kq_scale;
        }

        __syncthreads();

        float max_val = -INFINITY;
        // get the max
        for(int is = tid; is < seq_len; is += block_size) {
                max_val = fmaxf(max_val , S[is]);
        }

        max_val = warp_reduce_max(max_val);
        { // get max from all threads
            int warp_id = threadIdx.x / WARP_SIZE;
            int lane_id = threadIdx.x % WARP_SIZE;
            if (lane_id == 0) {
                warp_data[warp_id] = max_val;
            }
            __syncthreads();
            max_val = warp_data[lane_id];
            max_val = warp_reduce_max(max_val);
        }

        // softmax(QK^T)
        float sum = 0.0f;
        for(int is = tid; is < seq_len;is += block_size) {
                const float val = expf(S[is] - max_val);
                S[is] = val;
                sum += val;
        }

        sum = warp_reduce_sum(sum);
        { // sum partials
            int warp_id = threadIdx.x / WARP_SIZE;
            int lane_id = threadIdx.x % WARP_SIZE;
            if (lane_id == 0) {
                warp_data[warp_id] = sum;
            }
            __syncthreads();
            sum = warp_data[lane_id];
            sum = warp_reduce_sum(sum);
        }

        float inv_sum = 1.0f / sum;
        for(int is = tid; is < seq_len; is += block_size) {
                S[is] *= inv_sum;
        }

        __syncthreads();
        // softmax(QK^T)V
        for (int d = tid; d < d_head; d += block_size) {
                int dst_index = d + s * d_head + head * head_size;
                int value_offset = d * seq_len +   head * head_size;
                dst[dst_index] = 0.0f;
                for(int ic = 0; ic < seq_len; ic++) {
                        dst[dst_index] += v[value_offset + ic] * S[ic];
                }
        }
}

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n");
      break;
      }
    return cores;
}

int main() {
        cudaError_t cudaStatus = cudaSetDevice(0); // Seleccione el dispositivo GPU (0 en este caso)

        if (cudaStatus != cudaSuccess)
        {
                fprintf(stderr, "Error al seleccionar el dispositivo GPU: %s\n", cudaGetErrorString(cudaStatus));
                return 1;
        }

        // Obtener información de la GPU
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0)
        {
                fprintf(stderr, "No CUDA Devices found.\n");
                return 1;
        }

        printf("Device Count: %d\n", deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, i);
                printf("\nGPU Information %d:\n", i);
                printf("Name: %s\n", deviceProp.name);
                printf("Architecture: %d.%d\n", deviceProp.major, deviceProp.minor);
                printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
                printf("L2 Cache size: %d bytes\n", deviceProp.l2CacheSize);
                printf("CUDA Cores: %d\n", getSPcores(deviceProp));
                printf("VRAM: %lu MB\n", (unsigned long)(deviceProp.totalGlobalMem / 1024 / 1024));
                printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
                printf("Max Blocks Per MP: %d\n", deviceProp.maxBlocksPerMultiProcessor);
                printf("Max Threads Per MP: %d\n", deviceProp.maxThreadsPerMultiProcessor);
                printf("Max Shared Memory Per Block: %zu\n", deviceProp.sharedMemPerBlock);
                printf("Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
                printf("Max Block Size: (%d, %d, %d)\n\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        }

        int d_head = 3;
        int seq_len = 4;
        int num_heads = 2;
        dimen query_dim(d_head, seq_len, num_heads);

        float Query[24] = { // [3, 4, 2]
                // z0
                2, 4, 2,
                4, 2, 1,
                4, 1, 3,
                4, 2, 2,

                // z1
                2, 1, 1,
                4, 2, 1,
                1, 1, 3,
                4, 2, 1
        };

        dimen key_dim(d_head, seq_len, num_heads);
        float Key[24] = { // [3, 4, 2]
                // z0
                2, 4, 2,
                4, 2, 1,
                4, 2, 3,
                1, 2, 1,

                // z1
                3, 1, 3,
                4, 2, 1,
                1, 1, 2,
                4, 3, 1
        };

        dimen value_dim(seq_len, d_head, num_heads); // transposed value
        float Value[24] = { // [4, 3, 2]
                // z0
                2, 4, 2, 1,
                2, 1, 4, 2,
                1, 4, 2, 3,

                // z1
                1, 4, 2, 1,
                2, 1, 1, 2,
                1, 4, 3, 3,
        };

        float *KQV_result;

        float *d_query;
        float *d_key;
        float *d_value;
        float *d_kqv_result;

        dimen kqv_res_dim(d_head, seq_len, num_heads);

        KQV_result = (float *)malloc(kqv_res_dim.ne() * sizeof(float));

        cudaMalloc((void **)&d_query, query_dim.ne() * sizeof(float));
        cudaMalloc((void **)&d_key,   key_dim.ne() * sizeof(float));
        cudaMalloc((void **)&d_value, value_dim.ne() * sizeof(float));
        cudaMalloc((void **)&d_kqv_result, kqv_res_dim.ne() * sizeof(float));

        // copy CPU data to GPU memory blocks
        cudaMemcpy(d_query, Query, query_dim.ne() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_key, Key,     key_dim.ne() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, Value, value_dim.ne() * sizeof(float), cudaMemcpyHostToDevice);

        float kq_scale = 1.0f / sqrtf((float)d_head);

        int num_blocks = seq_len * num_heads;
        int smem_size = seq_len * sizeof(float) + WARP_SIZE * sizeof(float);

        flash_attn_f32<CUDA_FLASH_ATTENTION_BLOCK_SIZE> <<<num_blocks, CUDA_FLASH_ATTENTION_BLOCK_SIZE, smem_size>>> (
            d_query, d_key, d_value, d_kqv_result, // pointers
            kq_scale, // scale
            d_head, seq_len, num_heads, d_head * seq_len);

        // transfer data from device to host
        cudaMemcpy(KQV_result, d_kqv_result, kqv_res_dim.ne() * sizeof(float), cudaMemcpyDeviceToHost);

        /*

        * Expected values
        2.0457 2.4446 1.3050
        2.4594 3.2287 2.4192
        2.0603 3.8987 2.0551
        2.1756 3.6809 2.1481
        1.8984 1.6943 2.9636
        1.7022 1.7658 3.1875
        1.2656 1.8836 1.5731
        1.7022 1.7658 3.1875

        */

        int N = 3;

        for(int i = 0; i < 24; i ++) {
                if(i > 0 && (i % N == 0)) {
                printf("\n");
                }
                printf("%2.4f ", KQV_result[i]);
        }

        // clean up device memory
        cudaFree(d_query);
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_kqv_result);

        free(Query);
        free(Key);
        free(Value);
        free(KQV_result);
        return 0;
}
