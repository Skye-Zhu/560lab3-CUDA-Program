// src/matrix_cublas.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// --- CUDA 错误检查（你先当模板用，不用深究）
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t st = (call);                                              \
    if (st != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, \
              (int)st);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// CPU helper：填充随机数
static void fill_random(float *X, int n) {
  for (int i = 0; i < n; i++) X[i] = (float)(rand() % 100) / 100.0f;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  size_t bytes = (size_t)N * (size_t)N * sizeof(float);

  // 1) 在 CPU 上分配并生成 A, B
  float *h_A = (float*)malloc(bytes);
  float *h_B = (float*)malloc(bytes);
  float *h_C = (float*)malloc(bytes);
  if (!h_A || !h_B || !h_C) { fprintf(stderr, "malloc failed\n"); return 1; }

  srand(0);
  fill_random(h_A, N*N);
  fill_random(h_B, N*N);

  // 2) 在 GPU 上分配 d_A, d_B, d_C
  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

  // 3) 把 A, B 从 CPU 拷贝到 GPU
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  // 4) 创建 cuBLAS handle（相当于“打开库的上下文”）
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // 5) cuBLAS 的矩阵乘法参数
  //    C = alpha * A * B + beta * C
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // 6) 计时（只计 cuBLAS 乘法本身）
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  // 7) 调用 cublasSgemm
  //    ⚠️ 注意：cuBLAS 默认使用“列主序”（column-major）
  //    我们的数组是按 C 语言常见的“行主序”（row-major）填的
  //    解决方案：交换 A/B 的顺序，等价实现 C(row-major) = A*B
  //    也就是这里用：B 和 A 交换传入
  CUBLAS_CHECK(
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,   // 注意：这里先传 B
                d_A, N,   // 再传 A
                &beta,
                d_C, N)
  );

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // 8) 把结果拷回 CPU（可选，但一般做以保证流程完整）
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 输出统一格式（便于你填表/画图）
  printf("impl=CUBLAS,N=%d,time_ms=%.3f\n", N, ms);

  // 9) 清理资源
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A); free(h_B); free(h_C);

  return 0;
}