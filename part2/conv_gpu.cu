#include <cuda_runtime.h>

__global__ void conv2d_gpu(
    unsigned int *image,
    unsigned int *kernel,
    unsigned int *output,
    int M, int N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = N / 2;

    if (x >= offset && x < M - offset &&
        y >= offset && y < M - offset) {

        unsigned int sum = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int xi = x + i - offset;
                int yj = y + j - offset;
                sum += image[xi * M + yj] * kernel[i * N + j];
            }
        }
        output[x * M + y] = sum;
    }
}


