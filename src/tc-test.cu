#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> MatrixA{};
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::col_major> MatrixB{};
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                          Accum{};

__global__ void flash_attn(const half* query, const half* key, float* C, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc) {
    const uint32_t warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const uint32_t warp_n = blockIdx.y * blockDim.y + threadIdx.y;

    // set acumulator to 0
    nvcuda::wmma::fill_fragment(Accum, 1.0f);

    // Loop over K.
    for (uint32_t ki = 0; ki < k; ki += WMMA_K)
    {
        // swap if A is transpose
        const uint32_t a_row_idx = warp_m * WMMA_M;
        const uint32_t a_col_idx = ki;

        // swap if B is transpose
        const uint32_t b_row_idx = ki;
        const uint32_t b_col_idx = warp_n * WMMA_N;

        // Bounds checking
        if (a_row_idx < m &&
            a_col_idx < k &&
            b_row_idx < k &&
            b_col_idx < n)
        {
            const half * mm_a_ptr = query + a_row_idx + a_col_idx * lda;
            const half * mm_b_ptr = key + b_row_idx + b_col_idx * ldb;

            // Load the mma matrix inputs.
            nvcuda::wmma::load_matrix_sync(MatrixA, mm_a_ptr, lda);
            nvcuda::wmma::load_matrix_sync(MatrixB, mm_b_ptr, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(Accum, MatrixA, MatrixB, Accum);
        }
    }

    const uint32_t c_row_idx = warp_m * WMMA_M;
    const uint32_t c_col_idx = warp_n * WMMA_N;

    if (c_row_idx < m && c_col_idx < n)
    {
        float* mm_c_ptr = C + c_row_idx + c_col_idx * ldc;
        // Store the output
        nvcuda::wmma::store_matrix_sync(mm_c_ptr, Accum, ldc, nvcuda::wmma::mem_col_major);
    }
}

void mulmat_cpu(const float* A, const float* B, float* C, uint32_t m, uint32_t n, uint32_t k, bool B_transposed = false) {
    for (uint32_t ni = 0; ni < n; ++ni)
    {
        for (uint32_t mi = 0; mi < m; ++mi)
        {
            float accum = 0.0f;
            for (uint32_t ki{0}; ki < k; ++ki)
            {
                accum += A[ki * m + mi] * B[B_transposed ? (ki * n + ni) : ni * k + ki];
            }
            C[ni * m + mi] = accum;
        }
    }
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
            printf("VRAM: %lu MB\n", (unsigned long)(deviceProp.totalGlobalMem / 1024 / 1024));
            printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
            printf("Max Register Per Block: %d\n", deviceProp.regsPerBlock);
            printf("Max Shared Memory Per Block: %zu\n", deviceProp.sharedMemPerBlock);
            printf("Max Blocks Per MP: %d\n", deviceProp.maxBlocksPerMultiProcessor);
            printf("Max Threads Per MP: %d\n", deviceProp.maxThreadsPerMultiProcessor);
            printf("Max Register Per MP: %d\n", deviceProp.regsPerMultiprocessor);
            printf("Max Shared Memory Per MP: %d\n", deviceProp.sharedMemPerMultiprocessor);
            printf("Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
            printf("Max Block Size: (%d, %d, %d)\n\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }

    float Query[8 * 16] = { // [3, 4]
        // z0
        2, 4, 2, 4, 2, 1, 4, 1, 3, 4, 2, 2, 3, 4, 5, 2,
        2, 4, 2, 4, 2, 1, 4, 1, 3, 4, 2, 2, 3, 4, 5, 2,
        2, 4, 2, 4, 2, 1, 4, 1, 3, 4, 2, 2, 3, 4, 5, 2,
        2, 4, 2, 4, 2, 1, 4, 1, 3, 4, 2, 2, 3, 4, 5, 2,
        2, 4, 2, 4, 2, 1, 4, 1, 3, 4, 2, 2, 3, 4, 5, 2,
        2, 4, 2, 3, 2, 1, 4, 1, 3, 4, 2, 2, 3, 4, 5, 2,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 4, 2.3f, 2, 3, 4, 5, 4,
        2, 4, 2, 4, 2, 1, 4, 1, 3, 4, 2, 2, 3, 4, 5, 2,
    };

    float Key[32 * 16] = {
        // z0
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 1, 3, 1, 2, 1, 1.1f, 4, 5, 4,
        2, 4, 2, 2, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 5, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 1, 1, 3, 4, 5, 2,
        2, 4, 1, 3, 2, 1, 4, 1, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 2, 2, 3, 1, 1, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 7, 4, 5, 3,
        2, 4, 2, 1, 2, 1, 4, 1, 3, 1, 2, 3, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 3, 1, 3, 4, 5, 3,
        2, 4, 1, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 5, 2, 1, 4, 2, 3, 1, 2, 1, 4, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 1, 3, 1, 2, 1, 1.1f, 4, 5, 4,
        2, 4, 2, 2, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 5, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 1, 1, 3, 4, 5, 2,
        2, 4, 1, 3, 2, 1, 4, 1, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 2, 2, 3, 1, 1, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 7, 4, 5, 3,
        2, 4, 2, 1, 2, 1, 4, 1, 3, 1, 2, 3, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 3, 1, 3, 4, 5, 3,
        2, 4, 1, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 5, 2, 1, 4, 2, 3, 1, 2, 1, 4, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3,
        2, 4, 2, 4, 2, 1, 4, 2, 3, 1, 2, 1, 3, 4, 5, 3
    };

    int M = 8, N = 32, K = 16;

    float* result = (float*)malloc(M * N * sizeof(float));
    mulmat_cpu(Query, Key, result, M, N, K);

    half *query_f16 = (half*)malloc(M * K * sizeof(half));
    half *key_f16 = (half*)malloc(K * N * sizeof(half));

    for(int i = 0; i < M * K; i ++) {
        query_f16[i] = __float2half(Query[i]);
    }

    for(int i = 0; i < K * N; i ++) {
        key_f16[i] = __float2half(Key[i]);
    }

    float *qk_result;

    half *d_query;
    half *d_key;
    float *d_qk_result;

    qk_result = (float *)malloc(M * N * sizeof(float));

    cudaMalloc((void **)&d_query, M * K * sizeof(half));
    cudaMalloc((void **)&d_key, N * K * sizeof(half));
    cudaMalloc((void **)&d_qk_result, M * N * sizeof(float));

    // copy CPU data to GPU memory blocks
    cudaMemcpy(d_query, query_f16, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key,   key_f16,   N * K * sizeof(half), cudaMemcpyHostToDevice);

    int num_warps_x = 4;
    int num_warps_y = 4;

    dim3 grid_dim((M + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x),
                    (N + WMMA_N * num_warps_y - 1) /   (WMMA_N * num_warps_y));
    dim3 block_dim(num_warps_x * WARP_SIZE, num_warps_y, 1);

    printf("grid: [%d, %d], block: [%d, %d]\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    flash_attn<<<grid_dim, block_dim>>>(
        d_query, d_key, d_qk_result,
        M, N, K,
        K, K, M);

    // transfer data from device to host
    cudaMemcpy(qk_result, d_qk_result, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // mm_cpu(Query, Key, qk_result, M, N, K, M, K, M);

    for(int i = 0; i < M * N; i ++) {
        if(i > 0 && (i % M == 0)) {
            printf("\n");
        }
        printf("%2.2f ", qk_result[i]);
    }

    // clean up device memory
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_qk_result);

    free(Query);
    free(Key);
    free(qk_result);
    return 0;
}