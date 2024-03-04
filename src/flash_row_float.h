#ifndef __FLASH_ROW_F32__
#define __FLASH_ROW_F32__

#define FA_KV_BLOCK_256

template<int head_dim, int num_warps, int kv_tensor, int kv_block>
__global__ void flash_attn_row(const half* query, half* key /* reuse key buffer for partials result */,
    const half* value, const half* mask, int kv_size, float scale, int reduce_block, int head_stride) {
    const int lane_index = threadIdx.x;
    const int warp_index = threadIdx.y;

    const int warp_data_size = (head_dim*kv_tensor + 2);

    extern __shared__ char shmem[];
    half2* squery2      = (half2*)shmem; // load query buffer
    half * squery       = (half *)shmem; // probabilities buffer after online softmax
    float* sscores      = (float*)(shmem + head_dim*kv_tensor*sizeof(half)); // scores buffer after QK^T
    float* warp_buffer  = (float*)(shmem + head_dim*kv_tensor*sizeof(half) + (kv_block + 2)*sizeof(float) + (warp_index*warp_data_size*sizeof(float)));
#ifndef FA_KV_BLOCK_256
    half*  warp_buffer_half = (half*)warp_buffer;
#endif
    const int HD2 = head_dim / 2;

    // load query with 128x2 shape (repeat row twice)
    const half2* query_ = (const half2*)(query + head_dim*blockIdx.y); // shift as head
#pragma unroll
    for (int j = 0; j < kv_tensor; j += num_warps) {
        const int q_off = j + warp_index;
        if (q_off >= kv_tensor) {
            break;
        }

#pragma unroll
        for (int i = 0; i < HD2; i += WARP_SIZE) {
            const int h_offset = i + lane_index;
            if (h_offset >= HD2) {
                break;
            }
            squery2[q_off*HD2 + h_offset] = query_[h_offset];
        }
    }

    __syncthreads();

    {   // QK^T
        MatrixA query_m;
        nvcuda::wmma::load_matrix_sync(query_m, squery, 16);
        MatrixBT key_m;
        Accum kq_m;

        const int kv_per_warp = kv_block / num_warps;
        const int sum_diag = WMMA_K / kv_tensor;
        // assert(kv_per_warp % kv_tensor == 0);

        const int kvi = warp_index*kv_per_warp;

#pragma unroll
        for (int kv = 0; kv < kv_per_warp; kv += kv_tensor) {
            nvcuda::wmma::load_matrix_sync(key_m, key + head_stride*blockIdx.y + (blockIdx.x*kv_block + kvi + kv)*head_dim, 16);
            nvcuda::wmma::fill_fragment(kq_m, 0.0f);
            nvcuda::wmma::mma_sync(kq_m, query_m, key_m, kq_m);
            nvcuda::wmma::store_matrix_sync(warp_buffer, kq_m, 16, nvcuda::wmma::mem_row_major);

            // sum diagonal
            if (lane_index < kv_tensor) {
                float seq = 0.0f;
                const int seq_idx = kvi + kv + lane_index;
#pragma unroll
                for (int d0 = 0; d0 < sum_diag; d0++) {
                    const int diag_idx = d0 + lane_index * sum_diag;
                    seq += warp_buffer[diag_idx*WMMA_M + diag_idx]; // sum diagonal
                }

                // store sequence result
                sscores[seq_idx] = seq*scale + __half2float(mask[blockIdx.x*kv_block + seq_idx]); // save as float for softmax
            }
        }

        __syncthreads();
    }

    // perform online softmax
    {
        const int kv_per_warp = kv_block / num_warps;
        float M = -INFINITY;

        const int kvi = warp_index*kv_per_warp;

        for (int kv = lane_index*kv_tensor; kv < kv_per_warp; kv += WARP_SIZE*kv_tensor) {
            M = fmaxf(M, fmaxf(sscores[kvi + kv], sscores[kvi + kv + 1]));
        }

        M = warp_reduce_max(M);

        float S = 0.0f;

        for (int kv = lane_index*kv_tensor; kv < kv_per_warp; kv += WARP_SIZE*kv_tensor) {
            S += expf(sscores[kvi + kv] - M);
            S += expf(sscores[kvi + kv + 1] - M);
        }

        S = warp_reduce_sum(S);

        if(lane_index == 0) {
            warp_buffer[0] = M;
            warp_buffer[1] = S;
            // printf("warp index: %d, M= %.4f, S= %.4f\n", warp_index, M, S);
        }

        __syncthreads();

        // reduce warps
        if(warp_index == 0 && lane_index == 0) {
            float M0 = warp_buffer[0];
            float S0 = warp_buffer[1];

            for(int w = 1; w < num_warps; w++) {
                float M1 = warp_buffer[w * warp_data_size];
                float S1 = warp_buffer[w * warp_data_size + 1];

                float M = fmaxf(M0, M1);

                float ms0 = expf(M0 - M);
                float ms1 = expf(M1 - M);

                S0 = S0*ms0 + S1*ms1;
                M0 = M;
            }

            // printf("block M = %.4f, S= %.4f\n", M0, S0);

            // real softmax M and S for this block
            sscores[kv_block] = M0;
            sscores[kv_block + 1] = S0;
        }

        __syncthreads();

        const int tensor_elements = WMMA_M * WMMA_N;

        /*

            [S0, S1, S2,
            S0, S1, S2,
            S0, S1, S2]

        */

        // reuse shared memory padding
        M = sscores[kv_block];
        // S = sscores[kv_block + 1];

        const int te_per_warp = tensor_elements / num_warps;

        const int si = warp_index*te_per_warp;

#pragma unroll
        for (int t0 = 0; t0 < te_per_warp; t0 += WARP_SIZE) {
            const int tei = t0 + lane_index;
            if(tei >= te_per_warp) {
                break;
            }

            const int sq_offset = si + tei;
            squery[sq_offset] = __float2half(expf(sscores[sq_offset % kv_block] - M));
        }

        __syncthreads();
    }

#ifdef FA_KV_BLOCK_256
    {  // QK^TV
        MatrixA qk_m;
        nvcuda::wmma::load_matrix_sync(qk_m, squery, 16);
        MatrixBT value_m;
        Accum qkv_m;

        const int reduce_exccedent = reduce_block - gridDim.x;
#pragma unroll
        for(int h0 = 0; h0 < head_dim; h0 += num_warps) {
            const int hi = h0 + warp_index;
            if(hi >= head_dim) {
                break;
            }

            const int output_offset = blockIdx.y * head_stride + hi * reduce_block;

            // `value` need to be transposed
            nvcuda::wmma::load_matrix_sync(value_m, value + hi * kv_size + blockIdx.x*kv_block + blockIdx.y * head_stride, 16);
            nvcuda::wmma::fill_fragment(qkv_m, 0.0f);
            nvcuda::wmma::mma_sync(qkv_m, qk_m, value_m, qkv_m);
            nvcuda::wmma::store_matrix_sync(warp_buffer, qkv_m, 16, nvcuda::wmma::mem_row_major);

            // sum diagonal
            if (lane_index == 0) {
                float hdim = 0.0f;

                for (int d = 0; d < WMMA_K; d++) {
                    hdim += warp_buffer[d*WMMA_M + d]; // sum diagonal
                }

                // float hdim2 = 0.0f;
                // for (int d = 0; d < WMMA_K; d++) {
                //     if(d < 8) {
                //         hdim += warp_buffer[d*WMMA_M + d];
                //     } else {
                //         hdim2 += warp_buffer[d*WMMA_M + d];
                //     }
                // }

                // printf("warp 0 dim %d: %.4f\nwarp 1 dim %d: %.4f\n", hi, hdim, hi, hdim2);
                // float real_dim = hdim*__half2float(query[0]) + hdim2*__half2float(query[1]);
                // printf("real dim %d = %.4f, S=%.4f\n", hi, real_dim, sscores[kv_block + 1]);

                // assume the key has been processed by blocks launched per head
                key[output_offset + blockIdx.x] = __float2half(hdim);
                key[blockIdx.y * head_stride + head_dim*reduce_block + blockIdx.x*2] = __float2half(sscores[kv_block]); // max of this kv block
                key[blockIdx.y * head_stride + head_dim*reduce_block + blockIdx.x*2 + 1] = __float2half(sscores[kv_block + 1]); // sum of this kv block

                if(blockIdx.x == 0) { // just the first block will do this
                    for(int i = 0; i < reduce_exccedent; i ++) {
                        // this is a padding to perform a matrix multiplication without incorrect values
                        key[output_offset + gridDim.x + i] = __float2half(0.0f);
                    }
                }
            }
        }
    }
#else

    { // QK^TV
        MatrixA qk_m;
        nvcuda::wmma::load_matrix_sync(qk_m, squery, 16);
        MatrixB value_m;
        Accum qkv_m;

        const int hd_per_tensor = tensor_elements / kv_block; // head dims processed per tensor core
        const int hd_per_warp = head_dim / num_warps; // head dim split processed per warp
        // assert(hd_per_warp % hd_per_tensor == 0);

        // if(lane_index == 0) {
        //     printf("lidx = %d, widx= %d, hd per tensor= %d, hd per warp = %d\n", lane_index, warp_index, hd_per_tensor, hd_per_warp);
        // }
        const int next_hd_offset = WMMA_N / hd_per_tensor;
        const int sum_diag = WMMA_K / hd_per_tensor;
        const int reduce_exccedent = reduce_block - gridDim.x;

        for (int hdi = warp_index*hd_per_warp; hdi < head_dim; hdi += num_warps*hd_per_warp) {
            // create value matrix in warp buffer KxN
            /*
                head dim (cols)
                [S01, S11, S21, - seq dim (rows)
                S02, S12,  S22,
                S03, S13,  S23]
            */
            for (int hdw = 0; hdw < hd_per_warp; hdw += hd_per_tensor) {
                const int output_offset = blockIdx.y * head_stride + (hdi + hdw + lane_index) * reduce_block;
                if(lane_index < WMMA_N) {
                    for(int r = 0; r < WMMA_K; r++) { // tensor matrix rows
                        // if(lane_index == 15 && r == 15) {
                        //     printf("vidx= %d, vlen= %d, hd_per=%d\n", ((kv_block_offset + r + (c % next_hd_offset)*WMMA_K) * head_dim + (hdi + hdw + c/next_hd_offset)), kv_size*head_dim, hdw);
                        // }
                        warp_buffer_half[r*WMMA_N + lane_index] = value[
                            head_stride*blockIdx.y + // shift head
                            (blockIdx.x*kv_block + r + (lane_index % next_hd_offset)*WMMA_K) * head_dim + // shift sequence
                            (hdi + hdw + lane_index/next_hd_offset)];
                    }
                }

                // perform QK^TV
                nvcuda::wmma::load_matrix_sync(value_m, warp_buffer_half, 16);
                nvcuda::wmma::fill_fragment(qkv_m, 0.0f);
                nvcuda::wmma::mma_sync(qkv_m, qk_m, value_m, qkv_m);
                nvcuda::wmma::store_matrix_sync(warp_buffer, qkv_m, 16, nvcuda::wmma::mem_row_major);

                // sum diagonal
                if (lane_index < hd_per_tensor) {
                    float hdim = 0.0f;
#pragma unroll
                    for (int d0 = 0; d0 < sum_diag; d0++) {
                        const int diag_idx = d0 + lane_index * sum_diag;
                        hdim += warp_buffer[diag_idx*WMMA_M + diag_idx]; // sum diagonal
                    }

                    // assume the key has been processed by blocks launched per head
                    key[output_offset + blockIdx.x] = __float2half(hdim);

                    if(blockIdx.x == 0) { // just the first block will do this
                        for(int i = 0; i < reduce_exccedent; i ++) {
                            // this is a padding to perform a matrix multiplication without incorrect values
                            key[output_offset + gridDim.x + i] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
    }
#endif
}

template<int head_dim, int num_warps>
__global__ void fa_reduce(const half* red_buf, float* qkv, int kv_size, int num_kv_blocks, int reduce_block) {
    const int lane_index = threadIdx.x;
    const int warp_index = threadIdx.y;

    const int tensor_elements = WMMA_M*WMMA_N;
    //const int hi_per_tensor = tensor_elements / reduce_block;
    const int head_offset = head_dim * kv_size * blockIdx.x;

    extern __shared__ char shmem[];
    half * sscale = (half *)shmem;
    float* sf_lse = (float*)(shmem + tensor_elements*sizeof(half));
    // float* warp_buffer = (float*)(shmem + tensor_elements*sizeof(half) + (kv_size/reduce_block + 2) * sizeof(float) + warp_index*(tensor_elements + 2)*sizeof(float));

    // make scale 1.0 diagonal
    if(warp_index == 0 && lane_index == 0) {
        const int softmax_lse_offset = head_offset + head_dim*reduce_block;
        float M0 = __half2float(red_buf[softmax_lse_offset]);
        float S0 = __half2float(red_buf[softmax_lse_offset + 1]);

        for(int i = 1; i < num_kv_blocks; i++) {
            float M1 = __half2float(red_buf[softmax_lse_offset + i*2]);
            float S1 = __half2float(red_buf[softmax_lse_offset + i*2 + 1]);

            float M = fmaxf(M0, M1);

            float ms0 = expf(M0 - M);
            float ms1 = expf(M1 - M);

            S0 = S0*ms0 + S1*ms1;
            M0 = M;

            sscale[i*2    ] = __float2half(ms0);
            sscale[i*2 + 1] = __float2half(ms1);
        }

        sf_lse[0] = S0;

        // S0 is all sequence softmax denominator
        // printf("CUDA S: %.4f M: %.4f\n", S0, M0);
    }

    __syncthreads();

    const int hd_per_warp = head_dim / num_warps;

    // reduce kv blocks (very slow!!)
    for(int hi = warp_index*hd_per_warp; hi < head_dim; hi += num_warps*hd_per_warp) {
        for(int hdi = lane_index; hdi < hd_per_warp; hdi += WARP_SIZE) {
            float hdim = __half2float(red_buf[head_offset + (hi + hdi) * reduce_block]);
            for(int kv = 1; kv < num_kv_blocks; kv++) {
                hdim = hdim*__half2float(sscale[kv*2]) + __half2float(red_buf[head_offset + (hi + hdi) * reduce_block + kv]) * __half2float(sscale[kv*2 + 1]);
            }
            qkv[blockIdx.x * head_dim + hi + lane_index] = hdim / sf_lse[0];
        }
    }

    // for(int j = 1 + warp_index; j < hi_per_tensor; j += num_warps) {
    //     for(int i = lane_index; i < reduce_block; i += WARP_SIZE) {
    //         sscale[j * reduce_block + i] = sscale[i];
    //     }
    // }

//     MatrixA scale;
//     MatrixBT partials;
//     nvcuda::wmma::load_matrix_sync(scale, sscale, 16);
//     Accum qkv_m;

//     const int sum_diag = WMMA_K / hi_per_tensor;

// #pragma unroll
//     for (int h0 = 0; h0 < head_dim; h0 += num_warps*hi_per_tensor) {
//         const int hi = h0 + warp_index*hi_per_tensor;
//         nvcuda::wmma::load_matrix_sync(partials, red_buf + (head_offset + hi * reduce_block), 16);
//         nvcuda::wmma::fill_fragment(qkv_m, 0.0f);
//         nvcuda::wmma::mma_sync(qkv_m, scale, partials, qkv_m);
//         nvcuda::wmma::store_matrix_sync(warp_buffer, qkv_m, 16, nvcuda::wmma::mem_row_major);

//         // sum diagonal
//         if (lane_index < hi_per_tensor) {
//             float hdim = 0.0f;

//             for (int d = 0; d < sum_diag; d++) {
//                 const int diag_idx = lane_index * sum_diag + d;
//                 hdim += warp_buffer[diag_idx*WMMA_M + diag_idx]; // sum diagonal
//             }

//             qkv[blockIdx.x * head_dim + hi + lane_index] = hdim / sf_lse[0];
//         }
//     }
}
#endif