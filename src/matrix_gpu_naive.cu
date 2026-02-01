// src/matrix_gpu_naive.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)


__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int N) {


  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 

  if (row < N && col < N) {
    float sum = 0.0f;


    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum; 
  }
}

static void fill_random(float *X, int n) {
  for (int i = 0; i < n; i++) X[i] = (float)(rand() % 100) / 100.0f;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  size_t bytes = (size_t)N * (size_t)N * sizeof(float);


  float *h_A = (float*)malloc(bytes);
  float *h_B = (float*)malloc(bytes);
  float *h_C = (float*)malloc(bytes);
  if (!h_A || !h_B || !h_C) { fprintf(stderr, "malloc failed\n"); return 1; }

  srand(0);
  fill_random(h_A, N*N);
  fill_random(h_B, N*N);


  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));


  //HostToDevice: h_A/h_B -> d_A/d_B
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));


  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (N + block.y - 1) / block.y);


  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);  
  CUDA_CHECK(cudaEventRecord(stop));


  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));


  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));


  printf("impl=CUDA_NAIVE,N=%d,time_ms=%.3f\n", N, ms);


  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A); free(h_B); free(h_C);

  return 0;
}