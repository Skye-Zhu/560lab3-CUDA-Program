
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define TILE_DIM 16  


__global__
void conv_kernel_tiled(const uint32_t *d_in, uint32_t *d_out, int M, int N, const float *d_kernel) {
    __shared__ uint32_t tile[TILE_DIM + 32][TILE_DIM + 32]; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    int x = bx + tx;
    int y = by + ty;

    int pad = N/2;
    int shared_size = TILE_DIM + 2*pad;

    for (int sy = ty; sy < shared_size; sy += blockDim.y) {
        for (int sx = tx; sx < shared_size; sx += blockDim.x) {
            int img_x = bx + sx - pad;
            int img_y = by + sy - pad;
            uint32_t v = 0;
            if (img_x >=0 && img_x < M && img_y >=0 && img_y < M)
                v = d_in[img_y * M + img_x];
            tile[sy][sx] = v;
        }
    }
    __syncthreads();

    if (x < M && y < M) {
        float acc = 0.0f;
        for (int ky=0; ky<N; ++ky) {
            for (int kx=0; kx<N; ++kx) {
                int sx = tx + kx;
                int sy = ty + ky;
                uint32_t v = tile[sy][sx];
                acc += ((float)v) * d_kernel[ky * N + kx];
            }
        }
        if (acc < 0.0f) acc = 0.0f;
        if (acc > (float)UINT32_MAX) acc = (float)UINT32_MAX;
        d_out[y*M + x] = (uint32_t)(acc + 0.5f);
    }
}

extern "C" int conv_gpu_tiled_launch(const uint32_t *h_in, uint32_t *h_out, int M, int N, const float *h_kernel) {
    size_t bytes = sizeof(uint32_t) * (size_t)M * M;
    uint32_t *d_in=nullptr, *d_out=nullptr;
    float *d_kernel=nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_kernel, sizeof(float)*N*N);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    conv_kernel_tiled<<<grid, block>>>(d_in, d_out, M, N, d_kernel);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_kernel);
    return 0;
}
