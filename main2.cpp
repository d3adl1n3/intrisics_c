#include <iostream>
#include <arm_neon.h>
#include <time.h>

using namespace std;

float** get_I_matrix(int N) {
    float **arr;
    arr = (float **)calloc(N, sizeof(*arr));
    for (int i = 0; i < N; ++i)
        arr[i] = (float*)calloc(N, sizeof(*arr[i]));
       
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            arr[i][j] = 1;
        }
    }
    return arr;
}


float** get_rand_matrix(int N) {
    float **arr;
    arr = (float **)calloc(N, sizeof(*arr));
    for (int i = 0; i < N; ++i)
        arr[i] = (float*)calloc(N, sizeof(*arr[i]));
       
    float r = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            r = (float) (rand() % 10000);
            arr[i][j] = r;
        }
    }
    return arr;
}

float** get_10_matrix(int N) {
    float **arr;
    arr = (float **)calloc(N, sizeof(*arr));
    for (int i = 0; i < N; ++i)
        arr[i] = (float*)calloc(N, sizeof(*arr[i]));
       
    float r = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            r = (float) (rand() % 10000);
            arr[i][j] = r;
        }
    }
    return arr;
}


void add_arrays_intrins(int* a, int* b, int* target, int size) {
    for(int i=0; i<size; i+=4) {
        /* Load data into NEON register */
        int32x4_t av = vld1q_s32(&(a[i]));
        int32x4_t bv = vld1q_s32(&(b[i]));

        /* Perform the addition */
        int32x4_t targetv = vaddq_s32(av, bv);

        /* Store the result */
        vst1q_s32(&(target[i]), targetv);
    }
}

void add_matrix_intrins(float** a, float** b, float** target, int size) {
    for(int j=0; j<size; j++) {
        for(int i=0; i<size; i+=4) {
            /* Load data into NEON register */
            float32x4_t av = vld1q_f32(&(a[j][i]));
            float32x4_t bv = vld1q_f32(&(b[j][i]));

            /* Perform the addition */
            float32x4_t targetv = vaddq_f32(av, bv);

            /* Store the result */
            vst1q_f32(&(target[i][j]), targetv);
        }
    }
}

void add_matrix(float** a, float** b, float** target, int size) {
    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            int av = a[j][i];
            int bv = b[j][i];
            int c = av + bv;
            target[i][j] = c;
        }
    }
}

void sub_matrix(float** a, float** b, float** target, int size) {
    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            int av = a[j][i];
            int bv = b[j][i];
            int c = av - bv;
            target[i][j] = c;
        }
    }
}

void sub_matrix_intrins(float** a, float** b, float** target, int size) {
    for(int j=0; j<size; j++) {
        for(int i=0; i<size; i+=4) {
            float32x4_t av = vld1q_f32(&(a[j][i]));
            float32x4_t bv = vld1q_f32(&(b[j][i]));
            float32x4_t targetv = vsubq_f32(av, bv);
            vst1q_f32(&(target[i][j]), targetv);
        }
    }
}

void div_matrix(float** a, float** b, float** target, int size) {
    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            int av = a[j][i];
            int bv = b[j][i];
            int c = av / bv;
            target[i][j] = c;
        }
    }
}
void div_matrix_intrins(float** a, float** b, float** target, int size) {
    for(int j=0; j<size; j++) {
        for(int i=0; i<size; i+=4) {
            float32x4_t av = vld1q_f32(&(a[j][i]));
            float32x4_t bv = vld1q_f32(&(b[j][i]));
            float32x4_t targetv = vdivq_f32(av, bv);
            vst1q_f32(&(target[i][j]), targetv);
        }
    }
}

void mul_matrix(float** a, float** b, float** target, int size) {
    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            int av = a[j][i];
            int bv = b[j][i];
            int c = av * bv;
            target[i][j] = c;
        }
    }
}

void mul_matrix_intrins(float** a, float** b, float** target, int size) {
    for(int j=0; j<size; j++) {
        for(int i=0; i<size; i+=4) {
            float32x4_t av = vld1q_f32(&(a[j][i]));
            float32x4_t bv = vld1q_f32(&(b[j][i]));
            float32x4_t targetv = vmulq_f32(av, bv);
            vst1q_f32(&(target[i][j]), targetv);
        }
    }
}

void transpose_matrix(float** a, float** target, int size) {
    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            target[i][j] = a[j][i];
        }
    }
}

float find_max_row_sum(float** a, int size) {
    float res = 0;
    for (int j=0; j<size; j++) {
        float sum = 0;
        for (int i=0; i<size; i++) {
            sum += a[j][i];
        }
        if ((j == 0) || (res < sum)) {
            res = sum;
        }
    }
    return res;
}

float find_max_column_sum(float** a, int size) {
    float res = 0;
    for (int j=0; j<size; j++) {
        float sum = 0;
        for (int i=0; i<size; i++) {
            sum += a[i][j];
        }
        if ((j == 0) || (res < sum)) {
            res = sum;
        }
    }
    return res;
}

void calculate_B_matrix(float** a, float** b, int size) {
    b = get_I_matrix(size);
    transpose_matrix(a, b, size);
    
    float maxColumn, maxRow;
    maxColumn = find_max_column_sum(a, size);
    maxRow = find_max_row_sum(a, size);
    
    float buf = maxColumn * maxRow;
    
    for(int j=0; j<size; j++) {
        for(int i=0; i<size; i++) {
            b[j][i] /= buf;
        }
    }
}

void calculate_B_matrix_intrins(float** a, float** b, int size) {
    b = get_I_matrix(size);
    transpose_matrix(a, b, size);
    
    float maxColumn, maxRow;
    maxColumn = find_max_column_sum(a, size);
    maxRow = find_max_row_sum(a, size);
    
    float buf = maxColumn * maxRow;
    float32x4_t bv = vld1q_f32(&buf);
    
    for(int j=0; j<size; j++) {
        for(int i=0; i<size; i+=4) {
            float32x4_t av = vld1q_f32(&(b[j][i]));
            float32x4_t targetv = vdivq_f32(av, bv);
            vst1q_f32(&(b[i][j]), targetv);
        }
    }
}

int main(int argc, const char * argv[]) {
    int loops = 1000;
    int N = 3;
    struct timespec start, end;
    
    float **A, **B, **buf, **R, **I, **res, **mult;
    A = get_rand_matrix(N);
    
    printf("A:\n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%f, ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    B = get_I_matrix(N);
    buf = get_I_matrix(N);
    R = get_I_matrix(N);
    res = get_I_matrix(N);
    
    calculate_B_matrix(A, B, N);
    I = get_I_matrix(N);
    mul_matrix(A, B, buf, N);
    sub_matrix(I, buf, R, N);
    
    mult = get_I_matrix(N);
    for (int i=0; i<loops; i++) {
        add_matrix(res, mult, res, N);
        mul_matrix(mult, R, mult, N);
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken to calculate without intrinsics is %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    
    printf("Res\n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%f, ", res[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < N; ++i) {
        free(B[i]);
        free(buf[i]);
        free(R[i]);
        free(I[i]);
    }
    free(B);
    free(buf);
    free(R);
    free(I);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    B = get_I_matrix(N);
    buf = get_I_matrix(N);
    R = get_I_matrix(N);
    res = get_I_matrix(N);
    
    calculate_B_matrix_intrins(A, B, N);
    I = get_I_matrix(N);
    mul_matrix_intrins(A, B, buf, N);
    sub_matrix_intrins(I, buf, R, N);
    
    mult = get_I_matrix(N);
    for (int i=0; i<loops; i++) {
        add_matrix_intrins(res, mult, res, N);
        mul_matrix_intrins(mult, R, mult, N);
    }
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken to calculate with intrinsics is %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    printf("Res\n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%f, ", res[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    for (int i = 0; i < N; ++i) {
        free(A[i]);
        free(B[i]);
        free(buf[i]);
        free(R[i]);
        free(I[i]);
    }
    
    free(A);
    free(B);
    free(buf);
    free(R);
    free(I);
    return 0;
}

