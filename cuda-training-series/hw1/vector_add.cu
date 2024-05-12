#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
    
const int N = 4096;
const int thread_size = 256;  // CUDA maximum is 1024
__global__ void add_vector(int* a, int* b, int* c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}


void random_int(int* vector, int size){
    for (int i = 0; i < size; i++){
        vector[i] = rand() % 10;
    }
}
int main(){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    srand(time(NULL));
    a = new int[N];
    b = new int[N];
    c = new int[N];

    random_int(a, N);
    random_int(b, N);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");


    add_vector<<<(N + thread_size - 1) / thread_size, thread_size>>>(d_a, d_b, d_c);
    cudaCheckErrors("kernel launch failure");
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("A,   B,   C\n");
    for (int i = 0; i < N; i++){
        printf("%d    %d    %d\n", a[i], b[i], c[i]);
    }
    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);
    free(c);
    cudaFree(d_c);
    return 0;
}