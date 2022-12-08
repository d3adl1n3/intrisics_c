#include <iostream>
#include <arm_neon.h>
#include <time.h>

using namespace std;

float rand(int min, int max) {
    return 1;
}

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
       
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            arr[i][j] = rand(0, 256);
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
            int32x4_t targetv = vaddq_f32(av, bv);

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
            /* Load data into NEON register */
            float32x4_t av = vld1q_f32(&(a[j][i]));
            float32x4_t bv = vld1q_f32(&(b[j][i]));

            /* Perform the addition */
            int32x4_t targetv = vsubq_f32(av, bv);

            /* Store the result */
            vst1q_f32(&(target[i][j]), targetv);
        }
    }
}

void div_matrix() {}
void div_matrix_intrins() {}

void mult_matrix() {}
void mult_matrix_intrins() {}

void calculate_T_matrix() {}
void calculate_T_matrix_intrins() {}

void calculate_B_matrix() {}
void calculate_B_matrix_intrins() {}

void calculate_R_matrix() {}
void calculate_R_matrix_intrins() {}


int main(int argc, const char * argv[]) {
    int N = 10000;
    struct timespec start, end;
    
    float **a;
    a = get_rand_matrix(N);
    a = get_rand_matrix(N);
    float **b;
    b = get_rand_matrix(N);
    float **res;
    res = get_rand_matrix(N);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    add_matrix(a, b, res, N);
    sub_matrix(a, b, res, N);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken to calculate with intrinsics is %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    printf("%f\n", res[0][0]);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    add_matrix_intrins(a, b, res, N);
    sub_matrix_intrins(a, b, res, N);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken to calculate with intrinsics is %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    printf("%f\n", res[0][0]);
    
    for (int i = 0; i < N; ++i) {
        free(a[i]);
        free(b[i]);
        free(res[i]);
    }
    
    free(a);
    free(b);
    free(res);
    return 0;
}

