#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifdef _WIN32
  #define DLL_EXPORT extern "C" __declspec(dllexport)
#else
  #define DLL_EXPORT extern "C"
#endif

//Error checking helpers
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,              \
              cudaGetErrorString(err));                                          \
      fflush(stderr);                                                            \
      return;                                                                    \
    }                                                                            \
  } while (0)

//Tiled GEMM kernel (row-major)
// C = A * B
// A, B, C are row-major N x N arrays
#ifndef TILE
#define TILE 16
#endif

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;

  // Loop over tiles
  for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
    int a_col = t * TILE + threadIdx.x;  // A: (row, a_col)
    int b_row = t * TILE + threadIdx.y;  // B: (b_row, col)

    // Load tile from global to shared (with boundary checks)
    As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    // Compute partial dot product for this tile
    #pragma unroll
    for (int k = 0; k < TILE; k++) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < N && col < N) {
    C[row * N + col] = acc;
  }
}

//DLL exported function
// Python/ctypes will use this.
// Inputs: h_A, h_B are host pointers to float32 arrays length N*N (row-major)
// Output: h_C host pointer to float32 array length N*N (row-major)
DLL_EXPORT void gpu_matrix_multiply(const float* h_A,
                                    const float* h_B,
                                    float* h_C,
                                    int N) {
  if (!h_A || !h_B || !h_C || N <= 0) return;

  size_t bytes = (size_t)N * (size_t)N * sizeof(float);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);

  // Make sure kernel finished and check launch errors
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}