#include <stdio.h>

__global__ void hello(){
    printf("Hello from block: %u, thread:%u\n", blockIdx.x, threadIdx.x);
}

int main(){
    int m = 2;
    int n = 4;
    hello<<<n, m>>>();
    cudaDeviceSynchronize();
    return 0;
}