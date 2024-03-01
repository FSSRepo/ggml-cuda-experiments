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

void softmax(float* scores, int kv_size) {
    float M = -INFINITY;
    float S = 0.0f;

    for(int i = 0; i < kv_size;i++) {
        float s = scores[i];
        if(s > M) {
            S = 1.0f + S*expf(M - s);
            M = s;
        } else {
            S += expf(s - M);
        }
    }

    printf("M= %.4f, S=%.4f\n", M, S);

    for(int i = 0; i < kv_size;i++) {
        scores[i] = expf(scores[i] - M) / S;
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
}

#endif // __UTILS__