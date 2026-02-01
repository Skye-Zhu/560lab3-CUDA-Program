#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
  #define DLL_EXPORT extern "C" __declspec(dllexport)
#else
  #define DLL_EXPORT extern "C"
#endif

__global__ void conv2d_gpu_i32(
    const int32_t *image,
    const int32_t *kernel,
    int32_t *output,
    int M, int N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // row
    int y = blockIdx.y * blockDim.y + threadIdx.y; // col

    int offset = N / 2;

    if (x >= offset && x < M - offset &&
        y >= offset && y < M - offset) {

        int32_t sum = 0;
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

// simple CUDA error checker
static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(err));
    }
}

// Python ctypes will call this
DLL_EXPORT void gpu_convolution_i32(
    const int32_t *image,
    const int32_t *kernel,
    int32_t *output,
    int M, int N
) {
    int32_t *d_image = nullptr;
    int32_t *d_kernel = nullptr;
    int32_t *d_output = nullptr;

    size_t img_bytes = (size_t)M * (size_t)M * sizeof(int32_t);
    size_t ker_bytes = (size_t)N * (size_t)N * sizeof(int32_t);

    cuda_check(cudaMalloc((void**)&d_image, img_bytes), "cudaMalloc d_image");
    cuda_check(cudaMalloc((void**)&d_kernel, ker_bytes), "cudaMalloc d_kernel");
    cuda_check(cudaMalloc((void**)&d_output, img_bytes), "cudaMalloc d_output");

    cuda_check(cudaMemcpy(d_image, image, img_bytes, cudaMemcpyHostToDevice), "cudaMemcpy image H2D");
    cuda_check(cudaMemcpy(d_kernel, kernel, ker_bytes, cudaMemcpyHostToDevice), "cudaMemcpy kernel H2D");

    // Optional: clear output
    cuda_check(cudaMemset(d_output, 0, img_bytes), "cudaMemset d_output");

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    conv2d_gpu_i32<<<grid, block>>>(d_image, d_kernel, d_output, M, N);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cuda_check(cudaMemcpy(output, d_output, img_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output D2H");

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}