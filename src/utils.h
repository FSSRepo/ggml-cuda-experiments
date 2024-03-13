#ifndef __UTILS__
#define __UTILS__

// A[M, K] B[K, N] = C[M, N]
void mulmat_cpu(const float* A, const float* B, const float* mask, float* C, uint32_t M, uint32_t N, uint32_t K, float scale, bool B_transposed = false) {
    for(int c = 0; c < N; c++) {
        for(int r = 0; r < M; r++) {
            float acc = 0.0f;
            for(int k = 0; k < K; k++) {
                    acc += __half2float(__float2half(A[r*K + k])) *
                        __half2float(__float2half(B[B_transposed ? (c*K + k) : k*N + c]));
            }
            C[r*N + c] = acc*scale + (mask != nullptr ? mask[c] : 0.0f);
        }
    }
}

void mulmat_cpu(const float* A, const half* B, const half* mask, float* C, uint32_t M, uint32_t N, uint32_t K, float scale, bool B_transposed = false) {
    for(int c = 0; c < N; c++) {
        for(int r = 0; r < M; r++) {
            float acc = 0.0f;
            for(int k = 0; k < K; k++) {
                    acc += A[r*K + k] * __half2float(B[B_transposed ? (c*K + k) : k*N + c]);
            }
            C[r*N + c] = acc*scale + (mask != nullptr ? __half2float(mask[r * N + c]) : 0.0f);
        }
    }
}

void softmax(float* scores, int kv_size, int batch_size, int head) {
    for(int b = 0; b < batch_size; b++) {
        float M = -INFINITY;
        float S = 0.0f;

        for(int i = 0; i < kv_size;i++) {
            float s = scores[b*kv_size + i];
            if(s > M) {
                S = 1.0f + S*expf(M - s);
                M = s;
            } else {
                S += expf(s - M);
            }
        }

        for(int i = 0; i < kv_size;i++) {
            scores[b*kv_size + i] = expf(scores[b*kv_size + i] - M) / S;
        }
    }
}

void fill_buffer(float* arr, float val, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        arr[i] = 0.0f;
    }
}

void random(float* arr, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        arr[i] =  1.0f - (rand() * 1.0f / RAND_MAX) * 2.0f;
    }
}

void print_array(const char* name, float* arr, int cols, int count) {
    printf("---------------- %s ------------------\n", name);
    for(int i = 0; i < count; i ++) {
        if(i > 0 && (i % cols == 0)) {
            printf("\n");
        }
        printf("%0.4ff, ", arr[i]);
    }
}

void print_array_off(const char* name, float* arr, int cols, int count, int offset) {
    printf("---------------- %s ------------------\n", name);
    for(int i = offset; i < (offset + count); i ++) {
        if(i > 0 && (i % cols == 0)) {
            printf("\n");
        }
        printf("%0.4ff, ", arr[i]);
    }
}

void print_array(const char* name, half* arr, int row_count, int cols_count, int row_size) {
    printf("---------------- %s ------------------\n", name);
    for(int r = 0;r < row_count; r ++) {
        for(int c = 0;c < cols_count; c ++) {
            printf("%0.5ff, ", __half2float(arr[r * row_size + c]));
        }
        printf("\n");
    }
}

void print_array(const char* name, float* arr, int row_count, int cols_count, int row_size) {
    printf("---------------- %s ------------------\n", name);
    for(int r = 0;r < row_count; r ++) {
        for(int c = 0;c < cols_count; c ++) {
            printf("%0.5ff, ",arr[r * row_size + c]);
        }
        printf("\n");
    }
    printf("\n");
}

struct tensor {
    void* data;
    int type;
    char name[20];
};

tensor* load_tensor_from_file(const char* path, bool query = false) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    int32_t n_dims;
    int32_t length;
    int32_t ttype;
    fread(&n_dims, 1, sizeof(n_dims), f);
    fread(&ttype, 1, sizeof(ttype), f);
    int32_t nelements = 1;
    int32_t ne[4]     = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        fread(&ne[i], 1, sizeof(ne[i]), f);
        nelements *= ne[i];
    }
    fread(&length, 1, sizeof(length), f);
    tensor* te = new tensor();
    fread(te->name, 1, length, f);
    te->name[length] = '\0';
    printf("Tensor: %15s (%s) [", te->name, ttype == 1 ? "f16": "f32");
    for (int i = 0; i < n_dims; ++i) {
        printf("%d%s", ne[i], i < n_dims - 1 ? ", ": "]\n");
    }
    int data_size = nelements * (ttype == 1 ? 2 : 4);
    te->data = malloc(data_size);
    fread(te->data, 1, data_size, f);
    // for(int r = 0;r < (ne[1] > 1 ? 2 : 1); r ++) {
    //     for(int c = 0;c < 4; c ++) {
    //         if(ttype == 0) {
    //             printf("%0.5ff, ",((float*)te->data)[r * ne[0] + c]);
    //         } else if(ttype == 1) {
    //             printf("%0.5ff, ", __half2float(((half*)te->data)[r * ne[0] + c]));
    //         }
    //     }
    //     printf("\n");
    // }
    printf("\n");
    fclose(f);
    return te;
}

#endif // __UTILS__