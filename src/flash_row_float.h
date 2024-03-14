#ifndef __FLASH_ROW_F32__
#define __FLASH_ROW_F32__

template<int head_dim, int num_warps, int kv_tensor, int kv_block>
__global__ void flash_attn_row(const float* query, half* key /* reuse key buffer for partials result */,
    const half* value, const half* mask, half* tmp, int kv_size, float scale, int head_stride, int r_kv_heads) {
    const int lane_index = threadIdx.x;
    const int warp_index = threadIdx.y;

    const int warp_data_size = (head_dim*kv_tensor + 2);

    extern __shared__ char shmem[];
    half2* squery2      = (half2*)shmem; // load query buffer
    half * squery       = (half *)shmem; // probabilities buffer after online softmax
    half * softmax_lse  = (half *)(shmem + head_dim*kv_tensor*sizeof(half));
    half * warp_buffer  = (half *)(shmem + head_dim*kv_tensor*sizeof(half) + 2*sizeof(half) + (warp_index*warp_data_size*sizeof(half)));

    const int HD2 = head_dim / 2;
    const int kv_head_offset = (blockIdx.y / r_kv_heads) * head_stride;

    // load query with 128x2 shape (repeat row twice)
    const float2* query_ = (const float2*)(query + head_dim*blockIdx.y); // shift as head
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
            squery2[q_off*HD2 + h_offset] = __float22half2_rn(query_[h_offset]);
        }
    }

    __syncthreads();

    const int kv_per_warp = kv_block / num_warps;
    const int KPW2 = kv_per_warp/2;
    const int kvi = warp_index*kv_per_warp;
    const int KVI2 = kvi/2;

    {   // QK^T
        MatrixA query_m;
        nvcuda::wmma::load_matrix_sync(query_m, squery, 16);
        MatrixBT key_m;
        AccumH kq_m;
        half scale_ = __float2half(scale);

        const int sum_diag = WMMA_K / kv_tensor;

#pragma unroll
        for (int kv = 0; kv < kv_per_warp; kv += kv_tensor) {
            nvcuda::wmma::load_matrix_sync(key_m, key + (blockIdx.x*kv_block + kvi + kv)*head_dim + kv_head_offset, 16);
            nvcuda::wmma::fill_fragment(kq_m, 0.0f);
            nvcuda::wmma::mma_sync(kq_m, query_m, key_m, kq_m);
            nvcuda::wmma::store_matrix_sync(warp_buffer, kq_m, 16, nvcuda::wmma::mem_row_major);

            // sum diagonal
            if (lane_index < kv_tensor) {
                half seq = __half2float(0.0f);
                const int seq_idx = kvi + kv + lane_index;
#pragma unroll
                for (int d0 = 0; d0 < sum_diag; d0++) {
                    const int diag_idx = d0 + lane_index * sum_diag;
                    seq += warp_buffer[diag_idx*WMMA_M + diag_idx]; // sum diagonal
                }

                // store sequence result
                squery[seq_idx] = seq*scale_ + mask[blockIdx.x*kv_block + seq_idx]; // save as float for softmax
            }
        }

        __syncthreads();
    }

    // perform online softmax
    {
        half M = __float2half(-INFINITY);

        const int kvi = warp_index*kv_per_warp;

        for (int kv = lane_index; kv < kv_per_warp; kv += WARP_SIZE) {
            M = __hmax(M, squery[kvi + kv]);
        }

        M = warp_reduce_max(M);
        half2 M2 = make_half2(M, M);
        half2 S = make_half2(0.0, 0.0);

        if(__hisinf(M) != -1) {
            for (int kv = lane_index; kv < KPW2; kv += WARP_SIZE) {
                S += h2exp(squery2[KVI2 + kv] - M2);
            }
        }
        // else if(lane_index == 0 && blockIdx.y == 0) {
        //     printf("ignore: %d -> %d\n", kvi, kvi + kv_per_warp);
        // }

        S = warp_reduce_sum(S);

        if(lane_index == 0) {
            warp_buffer[0] = M;
            warp_buffer[1] = S.x + S.y;
            // printf("warp index: %d, M= %.4f, S= %.4f\n", warp_index, M, S);
        }

        __syncthreads();

        // reduce warps
        if(warp_index == 0 && lane_index == 0) {
            half M0 = warp_buffer[0];
            half S0 = warp_buffer[1];

            for(int w = 1; w < num_warps; w++) {
                half M1 = warp_buffer[w * warp_data_size];
                half S1 = warp_buffer[w * warp_data_size + 1];

                half M_ = __hmax(M0, M1);

                half ms0 = hexp(M0 - M_);
                half ms1 = hexp(M1 - M_);

                S0 = S0*ms0 + S1*ms1;
                M0 = M_;
            }

            // real softmax M and S for this block
            softmax_lse[0] = M0;
            softmax_lse[1] = S0;
        }

        __syncthreads();

        M = softmax_lse[0];
        M2 = make_half2(M, M);

#pragma unroll
        for (int k0 = 0; k0 < KPW2; k0 += WARP_SIZE) {
            const int kv = k0 + lane_index;
            if(kv >= KPW2) {
                break;
            }
            const int sq_offset = KVI2 + kv;
            squery2[sq_offset] = h2exp(squery2[KVI2 + kv] - M2);
        }

        __syncthreads();
    }

    {  // QK^TV
        MatrixA qk_m;
        nvcuda::wmma::load_matrix_sync(qk_m, squery, 16);
        MatrixBT value_m;
        AccumH qkv_m;

        // const int qkv_block_size = gridDim.x * head_dim + gridDim.x * 2;
        // const int qkv_head_offset = kv_head_offset + (blockIdx.y % r_kv_heads) * qkv_block_size;

        const int qkv_block_size = gridDim.x * head_dim + gridDim.x * 2;
        const int qkv_head_offset = blockIdx.y * qkv_block_size;

#pragma unroll
        for(int h0 = 0; h0 < head_dim; h0 += num_warps) {
            const int hi = h0 + warp_index;
            if(hi >= head_dim) {
                break;
            }

            const int output_offset = qkv_head_offset + hi * gridDim.x;

            // `value` need to be transposed
            nvcuda::wmma::load_matrix_sync(value_m, value + hi * kv_size + blockIdx.x*kv_block + kv_head_offset, 16);
            nvcuda::wmma::fill_fragment(qkv_m, 0.0f);
            nvcuda::wmma::mma_sync(qkv_m, qk_m, value_m, qkv_m);
            nvcuda::wmma::store_matrix_sync(warp_buffer, qkv_m, 16, nvcuda::wmma::mem_row_major);

            // sum diagonal
            if (lane_index == 0) {
                half hdim = __float2half(0.0f);

                for (int d = 0; d < WMMA_K; d++) {
                    hdim += warp_buffer[d*WMMA_M + d]; // sum diagonal
                }

                // assume the key has been processed by blocks launched per head
                tmp[output_offset + blockIdx.x] = hdim;
            }
        }

        if(warp_index == 0 && lane_index == 0) {
            tmp[qkv_head_offset + gridDim.x * head_dim + blockIdx.x*2] = softmax_lse[0]; // max of this kv block
            tmp[qkv_head_offset + gridDim.x * head_dim + blockIdx.x*2 + 1] = softmax_lse[1]; // sum of this kv block
        }
    }
}

// too much instructions and barriers
template<int head_dim, int num_warps, int kv_block> // kv_block should be tensor elements
__global__ void flash_attn_row_fast(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    const half* __restrict__ mask,
    half* __restrict__ qkv_tmp,
    const float scale, const int kv_size, const int qkv_partial_size, int r_kv_heads, int kv_stride, int head_stride) {
    const int lane_index = threadIdx.x;
    const int warp_index = threadIdx.y;

    cooperative_groups::thread_block blk = cooperative_groups::this_thread_block();

    extern __shared__ char shmem[];
    half * sh_query =       (half *)shmem; // Query half[kv_block]
    half2* sh_query2 =      (half2*)shmem;
    half * warp_buffer =    (half *)(shmem + kv_block*sizeof(half) + warp_index*kv_block*sizeof(half)); // warp_buffer float[256]
    half * sh_kv_buffer =   (half *)(shmem + kv_block*sizeof(half) + num_warps*kv_block*sizeof(half)); // Key half[kv_block][head_dim] | Value half[head_dim][kv_block]
    half * sh_softmax =     (half *)(shmem + kv_block*sizeof(half) + num_warps*kv_block*sizeof(half) + head_dim*kv_block*sizeof(half)); // softmax half[num_warps*2]

    // load query in shared memory
    const int query_per_tensor = kv_block / head_dim;
    const int kv_head_index = blockIdx.y / r_kv_heads;
    const int HD2 = head_dim / 2;

#pragma unroll
    for (int qo = 0; qo < query_per_tensor; qo++) {
        if(qo == 0) { // first read from global memory
            cooperative_groups::memcpy_async(blk, sh_query, query + blockIdx.y*head_dim, sizeof(half) * head_dim);
            cooperative_groups::wait(blk);
        } else { // copy from shared memory
            for(int i = (threadIdx.y*WARP_SIZE + threadIdx.x); i < HD2; i += num_warps*WARP_SIZE) {
                sh_query2[qo*HD2 + i] = sh_query2[i];
            }
        }
    }
    __syncthreads();

    // load key in shared memory
    {
        for (int kv = 0; kv < kv_block; kv += query_per_tensor) {
            for (int qo = 0; qo < query_per_tensor; qo++) {
                const int key_index = blockIdx.x*kv_block + kv + qo;
                cooperative_groups::memcpy_async(blk,
                    sh_kv_buffer + key_index*head_dim,
                    key + key_index*kv_stride + kv_head_index*head_stride, sizeof(half) * head_dim);
            }
        }
        cooperative_groups::wait(blk);
    }

    const int kv_per_warp = kv_block / num_warps;
    const int kvi = warp_index*kv_per_warp;
    const int KPW2 = kv_per_warp/2;
    const int KVI2 = kvi/2;

    // perform QK^T*scale + mask and get max for softmax
    {
        MatrixA  qm;
        nvcuda::wmma::load_matrix_sync(qm, sh_query, 16);
        MatrixBT km;
        AccumH   kqm;
        half M = __float2half(-INFINITE);
        half scale_ = __float2half(scale);

        const int num_diag = WMMA_K / query_per_tensor;
        // half* warp_tmp_buffer = sh_kv_buffer + kvi*head_dim; // save results from tensor cores

        for (int kv = 0; kv < kv_per_warp; kv += query_per_tensor) {
            nvcuda::wmma::load_matrix_sync(km, sh_kv_buffer + (kvi + kv)*head_dim, 16);
            nvcuda::wmma::fill_fragment(kqm, 0.0f);
            nvcuda::wmma::mma_sync(kqm, qm, km, kqm);
            nvcuda::wmma::store_matrix_sync(warp_buffer, kqm, 16, nvcuda::wmma::mem_row_major);

            // sum diagonal
            if (lane_index < query_per_tensor) {
                // TODO: make this half type
                half seq = __half2float(0.0f);
                const int seq_idx = kvi + kv + lane_index;
#pragma unroll
                for (int d0 = 0; d0 < num_diag; d0++) {
                    const int diag_idx = d0 + lane_index * num_diag;
                    seq += warp_buffer[diag_idx*WMMA_M + diag_idx]; // sum diagonal
                }

                seq = seq*scale_ + mask[blockIdx.x*kv_block + seq_idx];

                // store sequence result
                sh_query[seq_idx] = seq; // save as float for softmax
                M = __hmax(M, seq);
            }
        }

        M = warp_reduce_max(M);
        if(lane_index == 0) {
            sh_softmax[warp_index*2] = M;
        }
    }
    __syncthreads();

    {
        half2 S = make_half2(0.0, 0.0);
        half M = sh_softmax[warp_index*2];

        if(__hisinf(M) != -1) {
            half2 M2 = make_half2(M, M);
            for (int kv = lane_index; kv < KPW2; kv += WARP_SIZE) {
                S += h2exp(sh_query2[KVI2 + kv] - M2);
            }
        }

        S = warp_reduce_sum(S);

        if(lane_index == 0) {
            sh_softmax[warp_index*2 + 1] = S.x + S.y;
        }
        __syncthreads();
    }

    if(warp_index == 0 && lane_index == 0) {
        half M0 = sh_softmax[0];
        half S0 = sh_softmax[1];

        for(int w = 1; w < num_warps; w++) {
            half M1 = sh_softmax[w*2];
            half S1 = sh_softmax[w*2 + 1];

            half M = __hmax(M0, M1);

            half ms0 = hexp(M0 - M);
            half ms1 = hexp(M1 - M);

            S0 = S0*ms0 + S1*ms1;
            M0 = M;
        }

        // real softmax M and S for this block
        sh_softmax[0] = M0;
        sh_softmax[1] = S0;
    }
    __syncthreads();

    {
        half M = sh_softmax[0];
        half2 M2 = make_half2(M, M);
#pragma unroll
        for (int k0 = 0; k0 < KPW2; k0 += WARP_SIZE) {
            const int kv = k0 + lane_index;
            if(kv >= kv_per_warp) {
                break;
            }
            const int kv_offset = KVI2 + kv;
            sh_query2[kv_offset] = h2exp(sh_query2[kv_offset] - M2);
        }
        __syncthreads();
    }

    // load values in shared memory (no contiguous) (no coalesing acceses!!)
    // for (int kv = warp_index; kv < kv_block; kv += num_warps) {
    //     const int kv_offset = (blockIdx.x*kv_block + kv)*kv_stride + kv_head_index*head_stride;
    //     for (int hdim = lane_index; hdim < head_dim; hdim += WARP_SIZE) {
    //         sh_kv_buffer[hdim*kv_block + kv] = value[kv_offset + hdim];
    //     }
    // }

    // coalesing shared and global access (requires value transposed and contigous)
    for (int hdim = 0; hdim < head_dim; hdim ++) {
        cooperative_groups::memcpy_async(blk,
                    sh_kv_buffer + hdim*kv_block,
                    value + (blockIdx.x*kv_block + hdim * kv_size + kv_head_index*kv_size*head_dim), sizeof(half) * kv_block);
    }
    cooperative_groups::wait(blk);

    // perform softmax(QK^T)V
    {
        MatrixA  kqm;
        nvcuda::wmma::load_matrix_sync(kqm, sh_query, 16);
        MatrixBT vm;
        AccumH   qkvm;

        const int qkv_head_offset = blockIdx.y * qkv_partial_size;

#pragma unroll
        for(int h0 = 0; h0 < head_dim; h0 += num_warps) {
            const int hi = h0 + warp_index;
            if(hi >= head_dim) {
                break;
            }

            const int output_offset = qkv_head_offset + hi * gridDim.x;

            nvcuda::wmma::load_matrix_sync(vm, sh_kv_buffer + hi * kv_block, 16);
            nvcuda::wmma::fill_fragment(qkvm, 0.0f);
            nvcuda::wmma::mma_sync(qkvm, kqm, vm, qkvm);
            nvcuda::wmma::store_matrix_sync(warp_buffer, qkvm, 16, nvcuda::wmma::mem_row_major);

            if (lane_index == 0) {
                half hdim = __float2half(0.0f);
                for (int d = 0; d < WMMA_K; d++) {
                    hdim += warp_buffer[d*WMMA_M + d]; // sum diagonal
                }
                qkv_tmp[output_offset + blockIdx.x] = hdim;
            }
        }

        if(warp_index == 0 && lane_index == 0) {
            qkv_tmp[qkv_head_offset + gridDim.x * head_dim + blockIdx.x*2] = sh_softmax[0]; // max of this kv block
            qkv_tmp[qkv_head_offset + gridDim.x * head_dim + blockIdx.x*2 + 1] = sh_softmax[1]; // sum of this kv block
        }
    }
}

template<int head_dim, int num_warps>
__global__ void fa_reduce(const half* partial_qkv, float* qkv, int kv_size, int num_kv_blocks, int r_kv_heads) {
    const int lane_index = threadIdx.x;
    const int warp_index = threadIdx.y;

    // const int qkv_partial_offset =
    //     (blockIdx.x / r_kv_heads) * kv_size * head_dim /* key tensor data */ +
    //     (blockIdx.x % r_kv_heads) * (num_kv_blocks * head_dim + num_kv_blocks*2);

    const int qkv_partial_offset = blockIdx.x * (num_kv_blocks * head_dim + num_kv_blocks*2);

    extern __shared__ char shmem[];
    half* softmax_lse = (half *)shmem;

    if(warp_index == 0 && lane_index == 0) {
        const int softmax_lse_offset = qkv_partial_offset + num_kv_blocks * head_dim;
        half M0 = partial_qkv[softmax_lse_offset];
        half S0 = partial_qkv[softmax_lse_offset + 1];

        for(int i = 1; i < num_kv_blocks; i++) {
            half M1 = partial_qkv[softmax_lse_offset + i*2];
            half S1 = partial_qkv[softmax_lse_offset + i*2 + 1];

            half M = __hmax(M0, M1);

            half ms0 = hexp(M0 - M);
            half ms1 = hexp(M1 - M);

            S0 = S0*ms0 + S1*ms1;
            M0 = M;

            softmax_lse[i*2    ] = ms0;
            softmax_lse[i*2 + 1] = ms1;
        }

        softmax_lse[0] = S0;

        // S0 is all sequence softmax denominator
        // printf("%d CUDA =  M: %.4f S: %.4f\n", blockIdx.x, M0, S0);
    }

    __syncthreads();

    const int hd_per_warp = head_dim / num_warps;

    // reduce kv blocks (very slow!!)
    for(int hi = warp_index*hd_per_warp; hi < head_dim; hi += num_warps*hd_per_warp) {
        for(int hdi = lane_index; hdi < hd_per_warp; hdi += WARP_SIZE) {
            const int hdim_index = hi + hdi;
            const int qkv_index = qkv_partial_offset + hdim_index * num_kv_blocks;
            half hdim = partial_qkv[qkv_index];
            for(int kv = 1; kv < num_kv_blocks; kv++) {
                hdim = hdim * softmax_lse[kv*2] +partial_qkv[qkv_index + kv] * softmax_lse[kv * 2 + 1];
            }
            qkv[blockIdx.x * head_dim + hdim_index] = __half2float(hdim / softmax_lse[0]);
        }
    }
}
#endif