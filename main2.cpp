#include <iostream>
#include <arm_neon.h>
#include <time.h>

using namespace std;

int rand(int min, int max) {
    return 1;
}

int** getSquareRandMatrix(int N) {
    int **arr;
    arr = (int **)calloc(N, sizeof(*arr));
    for (int i = 0; i < N; ++i)
        arr[i] = (int*)calloc(N, sizeof(*arr[i]));
       
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

void add_matrix_intrins(int** a, int** b, int** target, int size) {
    for(int j=0; j<size; j++) {
        for(int i=0; i<size; i+=4) {
            /* Load data into NEON register */
            int32x4_t av = vld1q_s32(&(a[j][i]));
            int32x4_t bv = vld1q_s32(&(b[j][i]));

            /* Perform the addition */
            int32x4_t targetv = vaddq_s32(av, bv);

            /* Store the result */
            vst1q_s32(&(target[i][j]), targetv);
        }
    }
}

void add_matrix(int** a, int** b, int** target, int size) {
    for (int j=0; j<size; j++) {
        for (int i=0; i<size; i++) {
            int av = a[j][i];
            int bv = b[j][i];
            int c = av + bv;
            target[i][j] = c;
        }
    }
}

int main(int argc, const char * argv[]) {
    int N = 10000;
    struct timespec start, end;
    
    int **a;
    a = getSquareRandMatrix(N);
    a = getSquareRandMatrix(N);
    int **b;
    b = getSquareRandMatrix(N);
    int **res;
    res = getSquareRandMatrix(N);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    add_matrix(a, b, res, N);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken to calculate with intrinsics is %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    printf("%d\n", res[0][0]);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    add_matrix_intrins(a, b, res, N);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken to calculate with intrinsics is %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    printf("%d\n", res[0][0]);
    
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

