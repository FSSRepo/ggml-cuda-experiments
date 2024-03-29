#ifndef __CUDAINFO__
#define __CUDAINFO__
void print_cuda_info() {
    cudaError_t cudaStatus = cudaSetDevice(0); // Seleccione el dispositivo GPU (0 en este caso)

    if (cudaStatus != cudaSuccess)
    {
            fprintf(stderr, "Error al seleccionar el dispositivo GPU: %s\n", cudaGetErrorString(cudaStatus));
            return;
    }

    // Obtener información de la GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
            fprintf(stderr, "No CUDA Devices found.\n");
            return;
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
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(__shfl_xor_sync(0xffffffff, x, mask, 32), x);
    }
    return x;
}


static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, mask, 32));
    }
    return a;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = __hmax2(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ half warp_reduce_max(half x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = __hmax(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

#endif // __CUDAINFO__