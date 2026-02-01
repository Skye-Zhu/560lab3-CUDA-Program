// src/matrix_gpu_tiled.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)


__global__ void matmul_tiled_kernel(const float *A,
                                    const float *B,
                                    float *C,
                                    int N) {

  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_WIDTH + ty;
  int col = blockIdx.x * TILE_WIDTH + tx;

  float value = 0.0f;


  for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {


    if (row < N && (m * TILE_WIDTH + tx) < N)
      As[ty][tx] = A[row * N + m * TILE_WIDTH + tx];
    else
      As[ty][tx] = 0.0f;


    if (col < N && (m * TILE_WIDTH + ty) < N)
      Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
    else
      Bs[ty][tx] = 0.0f;


    __syncthreads();


    for (int k = 0; k < TILE_WIDTH; k++) {
      value += As[ty][k] * Bs[k][tx];
    }


    __syncthreads();
  }

  if (row < N && col < N)
    C[row * N + col] = value;
}


static void fill_random(float *X, int n) {
  for (int i = 0; i < n; i++)
    X[i] = (float)(rand() % 100) / 100.0f;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  size_t bytes = (size_t)N * (size_t)N * sizeof(float);

  float *h_A = (float*)malloc(bytes);
  float *h_B = (float*)malloc(bytes);
  float *h_C = (float*)malloc(bytes);

  srand(0);
  fill_random(h_A, N*N);
  fill_random(h_B, N*N);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
            (N + TILE_WIDTH - 1) / TILE_WIDTH);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  printf("impl=CUDA_TILED,N=%d,time_ms=%.3f\n", N, ms);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A); free(h_B); free(h_C);

  return 0;
}