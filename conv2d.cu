#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemm_fp16_fp32(half  *x, half  *y, float *dst, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += __half2float(x[row * M + i]) * __half2float(y[col * M + i]);
        }
        dst[row * K + col] = sum;
    }
}

__global__ void img2col_fp32_fp16(const float* x,half* dst, int nb12, int nb13, int IW,int IH,int CHW,int s0,int s1,int p0,int p1,int d0,int d1) {
    int iiw = blockIdx.z * s0 + threadIdx.z * d0 - p0;
	int iih = blockIdx.y * s1 + threadIdx.y * d1 - p1;
    __syncthreads();
    if (!(iih < 0 || iih >= IH || iiw < 0 || iiw >= IW)) {
        int offset_dst = (threadIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z) * CHW;
        int offset_src = threadIdx.x * nb13 +  blockIdx.x * nb12;
        dst[offset_dst + (blockIdx.x * (blockDim.y * blockDim.z) + threadIdx.y * blockDim.z + threadIdx.z)] = __float2half(x[offset_src + iih * IW + iiw]);
    }
}

static int calc_conv_output_size(int ins, int ks, int s, int p, int d) {
        return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

void conv2d_stage0(float* src, half* dest,
    int OC, int OH,
    int IW, int IH,
    int OW, int IC,
    int KH, int KW, int N,
    int s0,int s1,int p0,int p1,int d0,int d1) {
        int nb11 = IW;
        int nb12 = nb11 * IH; // nb[1] * ne[1]
        int nb13 = nb12 * IC; // nb[2] * ne[2]
        int CHW = IC * KH * KW;
        dim3 gridDim(IC, OH, OW);
        dim3 blockDim(N, KH, KW);
        img2col_fp32_fp16<<<gridDim, blockDim>>>(src, dest, nb12, nb13, IW, IH, CHW, s0, s1, p0, p1, d0, d1);
}

void conv2d_stage1(half* a, half* b, float* output, int OC, int OH, int OW,int IC, int KH, int KW, int N) {
        int m = OC;
        int n = OH * OW;
        int k = IC * KH * KW;

        for(int i = 0; i < N; i++) {
                dim3 blockDim(16, 16);
                dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
                gemm_fp16_fp32<<<gridDim, blockDim>>>(a, b + i * m * k, output + i * m * n, m, k, n);
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
                printf("VRAM: %lu MB\n", (unsigned long)(deviceProp.totalGlobalMem / 1024 / 1024));
                printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
                printf("Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
                printf("Max Block Size: (%d, %d, %d)\n\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        }

        // benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        // variable initialization
        int KW = 3, KH = 3, IC = 640, OC = 640;
        int IW = 32, IH = 48, /* IC = 640 */ N = 1;

        int s0 = 1;
        int s1 = 1;
        int p0 = 1;
        int p1 = 1;
        int d0 = 1;
        int d1 = 1;

        half *ha;
        float *b;
        float *result_stage1;

        half *d_ha, *d_stage0; // result of stage 0
        float *d_b,* d_stage1; // result of stage 1

        // memory allocation
        ha = (half *)malloc(KW * KH * IC * OC * sizeof(half));

        for (size_t i = 0; i < (KW * KH * IC * OC); i++) {
                ha[i] = __float2half(1);
        }

        b = (float *)malloc(IW * IH * IC * N * sizeof(float));
        for (size_t i = 0; i < (IW * IH * IC * N); i++) {
                b[i] = 1.0f;
        }

        // conv 2d stage 0
        int OH = calc_conv_output_size(IH, KH, s1, p1, d1);
        int OW = calc_conv_output_size(IW, KW, s0, p0, d0);

        result_stage1 = (float *)malloc(OW * OH * OC * N * sizeof(float));

        cudaMalloc((void **)&d_ha, KW * KH * IC * OC * sizeof(half));
        cudaMalloc((void **)&d_b, IW * IH * IC * N * sizeof(float));
        cudaMalloc((void **)&d_stage0, ((KW * KH * IC) * OH * OW * N) * sizeof(half));
        cudaMalloc((void **)&d_stage1, OW * OH * OC * N * sizeof(float));

        // copy CPU data to GPU memory blocks
        cudaMemcpy(d_ha, ha, KW * KH * IC * OC * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, IW * IH * IC * N * sizeof(float), cudaMemcpyHostToDevice);

        // execute con2d kernels
        cudaEventRecord(start);
        conv2d_stage0(d_b, d_stage0, OC, OH, IW, IH, OW, IC, KH, KW, N, s0, s1, p0, p1, d0, d1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float stage0_time = 0.0f;
        cudaEventElapsedTime(&stage0_time, start, stop);
        printf("conv2d stage 0 Elapsed time: %.2f ms\n", stage0_time);

        // reset timer
        cudaEventRecord(start);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventRecord(start);
        conv2d_stage1(d_ha, d_stage0, d_stage1, OC, OH, OW, IC, KH, KW, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float stage1_time = 0.0f;
        cudaEventElapsedTime(&stage1_time, start, stop);
        printf("conv2d stage 1 Elapsed time: %.2f ms\n", stage1_time);

        // transfer data from device to host
        cudaMemcpy(result_stage1, d_stage1, OW * OH * OC * N * sizeof(float), cudaMemcpyDeviceToHost);

        // print statements
        printf("conv2d:\n%.2f %.2f %.2f %.2f\n%.2f %.2f %.2f %.2f\n",
                result_stage1[0], result_stage1[1], result_stage1[2],
                result_stage1[3], result_stage1[4], result_stage1[5],
                result_stage1[6], result_stage1[7]);

        // clean up device memory
        cudaFree(d_ha);
        cudaFree(d_stage0);
        cudaFree(d_b);
        cudaFree(d_stage1);

        free(ha);
        free(b);
        free(result_stage1);
        return 0;
}