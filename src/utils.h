#ifndef __UTILS__
#define __UTILS__

// A[M, K] B[K, N] = C[M, N]
void mulmat_cpu(const float* A, const float* B, float* C, uint32_t M, uint32_t N, uint32_t K, bool B_transposed = false) {
    for(int c = 0; c < N; c++) {
        for(int r = 0; r < M; r++) {
            float acc = 0.0f;
            for(int k = 0; k < K; k++) {
                    acc += A[r*K + k] * B[B_transposed ? (c*K + k) : k*N + c];
            }
            C[r*N + c] = acc;
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
    printf("\n");
}

void print_array(const char* name, half* arr, int row_count, int cols_count, int row_size) {
    printf("---------------- %s ------------------\n", name);
    for(int r = 0;r < row_count; r ++) {
        for(int c = 0;c < cols_count; c ++) {
            printf("%0.4ff, ", __half2float(arr[r * row_size + c]));
        }
        printf("\n");
    }
    printf("\n");
}

void print_array(const char* name, float* arr, int row_count, int cols_count, int row_size) {
    printf("---------------- %s ------------------\n", name);
    for(int r = 0;r < row_count; r ++) {
        for(int c = 0;c < cols_count; c ++) {
            printf("%0.4ff, ",arr[r * row_size + c]);
        }
        printf("\n");
    }
    printf("\n");
}

#endif // __UTILS__