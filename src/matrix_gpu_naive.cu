// src/matrix_gpu_naive.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// -------------------- 你现在可以先不深究：
// CUDA 的错误检查宏：让我们在云上跑的时候一旦出错能立刻知道原因。
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// -------------------- CUDA Kernel（GPU 上并行跑的函数）
// 每个 thread 负责算 C 的一个元素 C[row, col]
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C, int N) {
  // blockIdx / threadIdx 是 CUDA 内置变量：告诉你“我是谁”
  // 这里我们用 2D 网格和 2D block，把 thread 映射到矩阵坐标 (row, col)

  int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前 thread 对应的行号
  int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前 thread 对应的列号

  // 防止越界：当 N 不是 block 大小的整倍数时，网格会“多出一些 thread”
  if (row < N && col < N) {
    float sum = 0.0f;

    // 计算 C[row, col] = Σ A[row, k] * B[k, col]
    // 这就是最朴素的矩阵乘法：一个元素要做 N 次乘加
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum; // 写回结果
  }
}

// -------------------- 工具函数：填充随机数（CPU 上跑）
static void fill_random(float *X, int n) {
  for (int i = 0; i < n; i++) X[i] = (float)(rand() % 100) / 100.0f;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  size_t bytes = (size_t)N * (size_t)N * sizeof(float);

  // -------------------- 1) 在 CPU（host）上分配内存
  float *h_A = (float*)malloc(bytes);
  float *h_B = (float*)malloc(bytes);
  float *h_C = (float*)malloc(bytes);
  if (!h_A || !h_B || !h_C) { fprintf(stderr, "malloc failed\n"); return 1; }

  srand(0);
  fill_random(h_A, N*N);
  fill_random(h_B, N*N);

  // -------------------- 2) 在 GPU（device）上分配内存
  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

  // -------------------- 3) 把数据从 CPU 拷贝到 GPU
  // HostToDevice: h_A/h_B -> d_A/d_B
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  // -------------------- 4) 配置线程布局（block / grid）
  // blockDim: 每个 block 里有多少个 thread（这里 16x16）
  // gridDim: 需要多少个 block 才能覆盖 N x N 的输出矩阵
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (N + block.y - 1) / block.y);

  // -------------------- 5) 计时：用 CUDA events 测“GPU kernel + 同步”的耗时（毫秒）
  // 你现在不用管 events 的细节，云上跑出来一个 ms 就够用写表格/画图
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);  // 启动 kernel（异步）
  CUDA_CHECK(cudaEventRecord(stop));

  // 必须同步：否则你计到的可能只是“发射 kernel 的时间”，不是真正执行时间
  CUDA_CHECK(cudaEventSynchronize(stop));

  // 把 start~stop 的时间差取出来（ms）
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // -------------------- 6) 把结果从 GPU 拷回 CPU（可选但一般做，保证流程完整）
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 输出统一格式（便于你后面做 timing.csv）
  printf("impl=CUDA_NAIVE,N=%d,time_ms=%.3f\n", N, ms);

  // -------------------- 7) 清理资源（你现在也不用深究，照做就行）
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A); free(h_B); free(h_C);

  return 0;
}