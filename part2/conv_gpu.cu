#include <cuda_runtime.h>
#include <stdio.h>

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

//simple CUDA error checker
static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        // Print error to help debugging
        printf("[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(err));
    }
}

// This is what Python ctypes will call
// Windows export: __declspec(dllexport)
// Prevent name mangling: extern "C"
extern "C" __declspec(dllexport)
void gpu_convolution(
    unsigned int *image,
    unsigned int *kernel,
    unsigned int *output,
    int M, int N
) {
    unsigned int *d_image = nullptr;
    unsigned int *d_kernel = nullptr;
    unsigned int *d_output = nullptr;

    size_t img_bytes = (size_t)M * (size_t)M * sizeof(unsigned int);
    size_t ker_bytes = (size_t)N * (size_t)N * sizeof(unsigned int);

    cuda_check(cudaMalloc((void**)&d_image, img_bytes), "cudaMalloc d_image");
    cuda_check(cudaMalloc((void**)&d_kernel, ker_bytes), "cudaMalloc d_kernel");
    cuda_check(cudaMalloc((void**)&d_output, img_bytes), "cudaMalloc d_output");

    cuda_check(cudaMemcpy(d_image, image, img_bytes, cudaMemcpyHostToDevice), "cudaMemcpy image H2D");
    cuda_check(cudaMemcpy(d_kernel, kernel, ker_bytes, cudaMemcpyHostToDevice), "cudaMemcpy kernel H2D");

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    conv2d_gpu<<<grid, block>>>(d_image, d_kernel, d_output, M, N);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cuda_check(cudaMemcpy(output, d_output, img_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output D2H");

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
