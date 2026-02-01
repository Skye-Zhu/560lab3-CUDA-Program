
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

__global__
void conv_kernel_naive(const uint32_t *d_in, uint32_t *d_out, int M, int N, const float *d_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= M) return;
    int pad = N/2;
    float acc = 0.0f;
    for (int ky=0; ky<N; ++ky) {
        int iy = y + ky - pad;
        for (int kx=0; kx<N; ++kx) {
            int ix = x + kx - pad;
            uint32_t val = 0;
            if (ix >= 0 && ix < M && iy >= 0 && iy < M) {
                val = d_in[iy * M + ix];
            }
            acc += ((float)val) * d_kernel[ky * N + kx];
        }
    }
    if (acc < 0.0f) acc = 0.0f;
    if (acc > (float)UINT32_MAX) acc = (float)UINT32_MAX;
    d_out[y*M + x] = (uint32_t)(acc + 0.5f);
}

extern "C" int conv_gpu_naive_launch(const uint32_t *h_in, uint32_t *h_out, int M, int N, const float *h_kernel) {
    size_t bytes = sizeof(uint32_t) * (size_t)M * M;
    uint32_t *d_in=nullptr, *d_out=nullptr;
    float *d_kernel=nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_kernel, sizeof(float)*N*N);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((M+block.x-1)/block.x, (M+block.y-1)/block.y);
    conv_kernel_naive<<<grid, block>>>(d_in, d_out, M, N, d_kernel);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
    return 0;
}