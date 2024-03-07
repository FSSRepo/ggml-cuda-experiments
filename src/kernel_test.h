void kernel_test(int argc, const char* argv[]) {
    int kv_size = 256, num_warps = 8;
    bool paralell_kv = true;

    if(argc > 1) {
        for(int i = 1; i < argc; i++) {
            if(strcmp(argv[i], "--no-kv-parallel") == 0) {
                paralell_kv = false;
            } else if(strcmp(argv[i], "--n-warps") == 0) {
                if(++i >= argc) {
                    return;
                }
                num_warps = atoi(argv[i]);
            }  else if(strcmp(argv[i], "--kv-size") == 0) {
                if(++i >= argc) {
                    return;
                }
                kv_size = std::max(256, atoi(argv[i]));
            }
        }
    }

    print_cuda_info();

    int head_dim = 128, num_heads = 32;
    float scale = 1.0f / sqrtf((float)head_dim);
    // allocate memory

    // input buffers
    float* query =  (float*)malloc(head_dim           * num_heads * sizeof(float)); // assume batch size 1
    float* key =    (float*)malloc(head_dim * kv_size * num_heads * sizeof(float));
    float* value =  (float*)malloc(head_dim * kv_size * num_heads * sizeof(float));
    float* mask =   (float*)malloc(kv_size * sizeof(float));

    // output buffers
    float* qkv =    (float*)malloc(head_dim           * num_heads * sizeof(float)); // assume batch size 1
    float* qkv_cuda = (float*)malloc(head_dim           * num_heads * sizeof(float)); // assume batch size 1
    float* scores = (float*)malloc(           kv_size * num_heads * sizeof(float)); // QK^T

    // fill buffers
    fill_buffer(qkv, 0.0f, head_dim * num_heads);
    fill_buffer(scores, 0.0f, kv_size * num_heads);

    random(query, head_dim * num_heads);
    random(key,   head_dim * kv_size * num_heads);
    random(value, head_dim * kv_size * num_heads);
    random(mask,  kv_size);

    if(true) {
        // cpu cmputation
        for(int h = 0; h < num_heads; h++) {
            mulmat_cpu(query + h*head_dim, key + (h * head_dim*kv_size), mask, scores + h*kv_size, 1, kv_size, head_dim, scale, true);
            softmax(scores + h*kv_size, kv_size, 1);
        }

        // print_array("Scores", scores, kv_size, 8);

        for(int h = 0; h < num_heads; h++) {
            mulmat_cpu(scores + h*kv_size, value + (h * head_dim*kv_size), nullptr, qkv + h*head_dim, 1, head_dim, kv_size, 1.0f);
        }

        print_array("Reference", qkv, 2, 16, head_dim);

        fill_buffer(qkv_cuda, 0.0f, head_dim * num_heads);
    }

    if(true) {
        // cuda cumputation
        half * key_f16 =     (half*)malloc(head_dim * kv_size * num_heads * sizeof(half));
        half * value_f16 =   (half*)malloc(head_dim * kv_size * num_heads * sizeof(half));
        half * value_f16_nT =   (half*)malloc(head_dim * kv_size * num_heads * sizeof(half));
        half * mask_f16 =    (half*)malloc(kv_size * sizeof(half));
        half * mask_f16_padded = (half*)malloc(kv_size * 32 * sizeof(half));

        for(int b = 0; b < 32; b ++) {
            for(int i = 0; i < kv_size; i ++) {
                if(b == 0) {
                    mask_f16[i] = __float2half(mask[i]);
                    mask_f16_padded[i] = __float2half(mask[i]);
                } else {
                    mask_f16_padded[b*kv_size + i] =  __float2half(0.0f);
                }
            }
        }

        for(int i = 0; i < head_dim * kv_size * num_heads; i ++) {
            key_f16[i] = __float2half(key[i]);
#ifndef FA_KV_BLOCK_256
            value_f16[i] = __float2half(value[i]);
#else
            value_f16_nT[i] = __float2half(value[i]);
#endif
        }

#ifdef FA_KV_BLOCK_256
        // transpose value
        for(int h = 0; h < num_heads; h++) {
            for(int c = 0; c < head_dim; c++) {
                for(int r = 0; r < kv_size; r++) {
                    value_f16[h*kv_size*head_dim + c*kv_size + r] = __float2half(value[h*kv_size*head_dim + r*head_dim + c]);
                }
            }
        }
#endif

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        half *d_key, *d_value, *d_value_nT, *d_mask, *d_padded_mask;
        float *d_query, *d_score, *d_qkv;

        cudaMalloc((void **)&d_query,  head_dim        * num_heads * sizeof(float));

        cudaMalloc((void **)&d_key,     head_dim * kv_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_value,   head_dim * kv_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_value_nT,  head_dim * kv_size * num_heads * sizeof(half));
        cudaMalloc((void **)&d_mask,    kv_size * sizeof(half));
        cudaMalloc((void **)&d_padded_mask,  16 *  kv_size * sizeof(half));

        cudaMalloc((void **)&d_score,              kv_size * num_heads * sizeof(float));
        cudaMalloc((void **)&d_qkv,     head_dim           * num_heads * sizeof(float));

        // copy CPU data to GPU memory blocks
        cudaMemcpyAsync(d_query, query, head_dim           * num_heads * sizeof(float), cudaMemcpyHostToDevice, stream);
        
        cudaMemcpyAsync(d_key,   key_f16,   head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_value, value_f16, head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_value_nT, value_f16_nT, head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mask,  mask_f16,  kv_size * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_padded_mask,  mask_f16_padded, 16 * kv_size * sizeof(half), cudaMemcpyHostToDevice, stream);

        constexpr int kv_per_block = 256;

        // assert(kv_size % kv_per_block == 0);
        dim3 grid_dim(kv_size / kv_per_block, num_heads, 1);
        dim3 block_dim(WARP_SIZE, num_warps, 1);

        int shmem =
            head_dim*2*sizeof(half) /* query buffer */ +
            (kv_per_block + 2)*sizeof(float) /* scores buffer */ +
            num_warps * (256 + 2) * sizeof(float) /* tensor core result buffer per warp */;

        for(int i = 0; i < head_dim * kv_size * num_heads; i ++) {
            key_f16[i] = __float2half(0.0f);
        }

        int reduce_block = ((grid_dim.x + WMMA_M - 1) / WMMA_M) * WMMA_N;

        cudaStreamSynchronize(stream);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);

        if(paralell_kv) {
            switch (num_warps)
            {
            case 8:
                flash_attn_row<128, 8, 2, kv_per_block><<<grid_dim, block_dim, shmem, stream>>>(d_query, d_key, d_value, d_mask, kv_size, scale, reduce_block, head_dim*kv_size);
                fa_reduce<128, 8><<<num_heads, block_dim, shmem, stream>>>(d_key, d_qkv, kv_size, kv_size / kv_per_block, reduce_block);
                break;
            case 4:
                flash_attn_row<128, 4, 2, kv_per_block><<<grid_dim, block_dim, shmem, stream>>>(d_query, d_key, d_value, d_mask, kv_size, scale, reduce_block, head_dim*kv_size);
                fa_reduce<128, 4><<<num_heads, block_dim, shmem, stream>>>(d_key, d_qkv, kv_size, kv_size / kv_per_block, reduce_block);
                break;
            case 2:
                flash_attn_row<128, 2, 2, kv_per_block><<<grid_dim, block_dim, shmem, stream>>>(d_query, d_key, d_value, d_mask, kv_size, scale, reduce_block, head_dim*kv_size);
                fa_reduce<128, 2><<<num_heads, block_dim, shmem, stream>>>(d_key, d_qkv, kv_size, kv_size / kv_per_block, reduce_block);
                break;
            default:
                printf("invalid num_warps, should be 2, 4, 8\n");
                break;
            }
        } else {
            // launch llama.cpp implementation
            const int nwarps = 8;
            constexpr int nqpb = 16;
            constexpr int ncpw = 128;

            dim3 blocks_num(1, num_heads, 1);
            dim3 block_dim(32, nwarps, 1);

            const size_t shmem_f_ = 16*(head_dim + nwarps*(ncpw + nqpb))*(sizeof(float)/2);

            flash_attn_ext_f16<128, nqpb, ncpw><<<blocks_num, block_dim, shmem_f_, stream>>>(
                (const char*)d_query, (const char*)d_key, (const char*)d_value_nT, (const char*)d_padded_mask, d_qkv, scale,
                head_dim, 1, num_heads, 1, head_dim, kv_size, num_heads, 1, kv_size, 16*2,
                head_dim*4, head_dim*4, head_dim*num_heads*4,
                head_dim*2, head_dim*kv_size*2, head_dim*kv_size*num_heads*2,
                head_dim, num_heads, 1, 1);
        }

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float millis = 0.0f;
        cudaEventElapsedTime(&millis, start, stop);

        // transfer data from device to host
        cudaMemcpyAsync(qkv_cuda, d_qkv, head_dim           * num_heads * sizeof(float), cudaMemcpyDeviceToHost, stream);
        // cudaMemcpyAsync(key_f16, d_key, head_dim * kv_size * num_heads * sizeof(half), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        print_array(paralell_kv ? "Parallel KV CUDA" : "No paralell KV CUDA", qkv_cuda, 2, 16, head_dim);

        float max_diff = 0.0f;
        int head_idx = 0, dim_idx = 0;

        for(int h = 0; h < num_heads; h++) {
            for(int i = 0; i < head_dim; i++) {
                if(fabs(qkv[h*head_dim + i] - qkv_cuda[h*head_dim + i]) > max_diff) {
                    max_diff = fabs(qkv[h*head_dim + i] - qkv_cuda[h*head_dim + i]);
                    head_idx = h;
                    dim_idx = i;
                }
            }
        }

        printf("\ncuda time: %.4f ms\n", millis);

        if(paralell_kv) {
            printf("Shared memory: %.2f KB\n", shmem/1024.0f);
        }

        printf("R (%.4f) CUDA(%.4f) diff: %.4f - head = %d, dim = %d\n", qkv[head_idx*head_dim + dim_idx], qkv_cuda[head_idx*head_dim + dim_idx], max_diff, head_idx, dim_idx);

        // clean up device memory
        cudaFree(d_query);
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_qkv);
        cudaFree(d_score);
    }

    free(query);
    free(key);
    free(value);
    free(qkv);
    free(qkv_cuda);
    free(scores);
}