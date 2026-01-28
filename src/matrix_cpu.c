#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void fill_random(float *X, int n) {
  for (int i = 0; i < n; i++) X[i] = (float)(rand() % 100) / 100.0f;
}

void matmul_cpu(const float *A, const float *B, float *C, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) sum += A[i*N + k] * B[k*N + j];
      C[i*N + j] = sum;
    }
  }
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  size_t bytes = (size_t)N * (size_t)N * sizeof(float);

  float *A = (float*)malloc(bytes);
  float *B = (float*)malloc(bytes);
  float *C = (float*)malloc(bytes);
  if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 1; }

  srand(0);
  fill_random(A, N*N);
  fill_random(B, N*N);

  clock_t start = clock();
  matmul_cpu(A, B, C, N);
  clock_t end = clock();

  double sec = (double)(end - start) / CLOCKS_PER_SEC;

  //防止编译器优化掉计算
  double checksum = 0.0;
  for (int i = 0; i < N * N; i++) {
    checksum += C[i];
  }

  printf("impl=CPU,N=%d,time_sec=%.6f,checksum=%f\n", N, sec, checksum);

  free(A); free(B); free(C);
  return 0;
}