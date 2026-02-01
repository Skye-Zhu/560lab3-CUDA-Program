#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <string>

#ifndef TILE
#define TILE 16
#endif

#define CUDA_CHECK(call) do {                                   \
  cudaError_t err = (call);                                     \
  if (err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));       \
    exit(1);                                                    \
  }                                                             \
} while (0)

static inline long long checksum_i32(const int32_t* a, int n) {
  long long s = 0;
  for (int i = 0; i < n; i++) s += (long long)a[i];
  return s;
}

static uint8_t* read_pgm_p5(const char* path, int* w, int* h) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "Failed to open %s\n", path); return nullptr; }

  char magic[3] = {0};
  if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
    fprintf(stderr, "Not a P5 PGM: %s\n", path);
    fclose(f);
    return nullptr;
  }


  int c = fgetc(f);
  while (c == '#') {
    while (c != '\n' && c != EOF) c = fgetc(f);
    c = fgetc(f);
  }
  ungetc(c, f);

  int width = 0, height = 0, maxv = 0;
  if (fscanf(f, "%d %d %d", &width, &height, &maxv) != 3) {
    fprintf(stderr, "Bad PGM header: %s\n", path);
    fclose(f);
    return nullptr;
  }
  if (maxv != 255) {
    fprintf(stderr, "PGM maxval must be 255 (got %d): %s\n", maxv, path);
    fclose(f);
    return nullptr;
  }

  // consume single whitespace after header
  fgetc(f);

  size_t bytes = (size_t)width * (size_t)height;
  uint8_t* data = (uint8_t*)malloc(bytes);
  if (!data) { fclose(f); return nullptr; }

  if (fread(data, 1, bytes, f) != bytes) {
    fprintf(stderr, "Failed to read pixel data: %s\n", path);
    free(data);
    fclose(f);
    return nullptr;
  }

  fclose(f);
  *w = width; *h = height;
  return data;
}

static bool write_pgm_p5(const char* path, const uint8_t* data, int w, int h) {
  FILE* f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "Failed to write %s\n", path); return false; }
  fprintf(f, "P5\n%d %d\n255\n", w, h);
  size_t bytes = (size_t)w * (size_t)h;
  fwrite(data, 1, bytes, f);
  fclose(f);
  return true;
}


static void visualize_i32_to_u8(const int32_t* in, uint8_t* out, int M,
                                const char* mode, int div_for_blur) {
  if (strcmp(mode, "linear") == 0) {
    for (int i = 0; i < M*M; i++) {
      int v = in[i];
      if (v < 0) v = 0;
      if (v > 255) v = 255;
      out[i] = (uint8_t)v;
    }
    return;
  }

  if (strcmp(mode, "blur_div") == 0) {
    for (int i = 0; i < M*M; i++) {
      int v = in[i] / div_for_blur;
      if (v < 0) v = 0;
      if (v > 255) v = 255;
      out[i] = (uint8_t)v;
    }
    return;
  }

  // edge
  int maxv = 0;
  for (int i = 0; i < M*M; i++) {
    int v = in[i];
    if (v < 0) v = -v;
    if (v > maxv) maxv = v;
  }
  if (maxv == 0) maxv = 1;
  for (int i = 0; i < M*M; i++) {
    int v = in[i];
    if (v < 0) v = -v;
    int scaled = (v * 255) / maxv;
    if (scaled < 0) scaled = 0;
    if (scaled > 255) scaled = 255;
    out[i] = (uint8_t)scaled;
  }
}


static void conv2d_cpu_i32(const int32_t* image, const int32_t* kernel, int32_t* output, int M, int N) {
  int offset = N / 2;
  // zero init
  for (int i = 0; i < M*M; i++) output[i] = 0;

  for (int i = offset; i < M - offset; i++) {
    for (int j = offset; j < M - offset; j++) {
      int32_t sum = 0;
      for (int ki = 0; ki < N; ki++) {
        for (int kj = 0; kj < N; kj++) {
          int ii = i + ki - offset;
          int jj = j + kj - offset;
          sum += image[ii * M + jj] * kernel[ki * N + kj];
        }
      }
      output[i * M + j] = sum;
    }
  }
}


__global__ void conv2d_gpu_i32(const int32_t* image, const int32_t* kernel, int32_t* output, int M, int N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // row
  int y = blockIdx.y * blockDim.y + threadIdx.y; // col
  int offset = N / 2;

  if (x >= offset && x < M - offset && y >= offset && y < M - offset) {
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

static float run_gpu_once(const int32_t* h_img, const int32_t* h_ker, int32_t* h_out, int M, int N) {
  int32_t *d_img=nullptr, *d_ker=nullptr, *d_out=nullptr;
  size_t img_bytes = (size_t)M*(size_t)M*sizeof(int32_t);
  size_t ker_bytes = (size_t)N*(size_t)N*sizeof(int32_t);

  CUDA_CHECK(cudaMalloc((void**)&d_img, img_bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_ker, ker_bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_out, img_bytes));

  CUDA_CHECK(cudaMemcpy(d_img, h_img, img_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ker, h_ker, ker_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, img_bytes));

  dim3 block(16,16);
  dim3 grid((M + block.x - 1)/block.x, (M + block.y - 1)/block.y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  conv2d_gpu_i32<<<grid, block>>>(d_img, d_ker, d_out, M, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, img_bytes, cudaMemcpyDeviceToHost));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_img);
  cudaFree(d_ker);
  cudaFree(d_out);
  return ms / 1000.0f; // seconds
}


static std::vector<int32_t> kernel_blur_3() {
  // sum=9
  return {1,1,1, 1,1,1, 1,1,1};
}
static std::vector<int32_t> kernel_sharpen_3() {
  return {0,-1,0, -1,5,-1, 0,-1,0};
}
static std::vector<int32_t> kernel_sobelx_3() {
  return {-1,0,1, -2,0,2, -1,0,1};
}
static std::vector<int32_t> make_edge_kernel_N(int N) {
  if (N == 3) return kernel_sobelx_3();
  std::vector<int32_t> k(N*N, 0);
  int mid = N/2;
  for (int r = 0; r < N; r++) {
    k[r*N + 0] = -1;
    k[r*N + (N-1)] = 1;
  }
  k[mid*N + 0] = -2;
  k[mid*N + (N-1)] = 2;
  return k;
}

int main(int argc, char** argv) {
  const int Ms[3] = {512, 1024, 2048};
  const int Ns[3] = {3, 5, 7};

  // Determine base directory and input name pattern
  std::string base_dir = ".";
  if (argc >= 2) {
    std::string p = argv[1];
    size_t pos = p.find_last_of("/\\");
    if (pos != std::string::npos) base_dir = p.substr(0, pos);
  }

#ifdef _WIN32
  std::string out_dir = base_dir + "\\out_purecuda";
  std::string csv_path = base_dir + "\\perf_conv_purecuda.csv";
#else
  std::string out_dir = base_dir + "/out_purecuda";
  std::string csv_path = base_dir + "/perf_conv_purecuda.csv";
#endif
#ifdef _WIN32
  system((std::string("if not exist \"") + out_dir + "\" mkdir \"" + out_dir + "\"").c_str());
#else
  system((std::string("mkdir -p \"") + out_dir + "\"").c_str());
#endif

  // CSV header
  FILE* csv = fopen(csv_path.c_str(), "w");
  if (!csv) {
    fprintf(stderr, "Failed to write CSV: %s\n", csv_path.c_str());
    return 1;
  }
  fprintf(csv, "M,N,cpu_time_sec,gpu_time_sec,speedup,cpu_checksum,gpu_checksum\n");

  {
#ifdef _WIN32
    std::string in_demo = base_dir + "\\input_M1024.pgm";
#else
    std::string in_demo = base_dir + "/input_M1024.pgm";
#endif
    int w=0,h=0;
    uint8_t* in_u8 = read_pgm_p5(in_demo.c_str(), &w, &h);
    if (in_u8 && w==1024 && h==1024) {
      int M = 1024, N = 3;
      std::vector<int32_t> img_i32(M*M);
      for (int i = 0; i < M*M; i++) img_i32[i] = (int32_t)in_u8[i];

      std::vector<int32_t> out_gpu(M*M);

      struct Demo { const char* name; std::vector<int32_t> ker; const char* mode; int div; };
      Demo demos[3] = {
        {"blur", kernel_blur_3(), "blur_div", 9},
        {"sharpen", kernel_sharpen_3(), "linear", 1},
        {"edge", kernel_sobelx_3(), "edge", 1},
      };

      for (auto &d : demos) {
        // warmup once
        run_gpu_once(img_i32.data(), d.ker.data(), out_gpu.data(), M, N);

        float t = run_gpu_once(img_i32.data(), d.ker.data(), out_gpu.data(), M, N);

        std::vector<uint8_t> out_u8(M*M);
        visualize_i32_to_u8(out_gpu.data(), out_u8.data(), M, d.mode, d.div);

#ifdef _WIN32
        std::string out_path = out_dir + "\\" + std::string(d.name) + "_gpu_M1024_N3.pgm";
#else
        std::string out_path = out_dir + "/" + std::string(d.name) + "_gpu_M1024_N3.pgm";
#endif
        write_pgm_p5(out_path.c_str(), out_u8.data(), M, M);
        printf("[DEMO] %s saved: %s (gpu_time=%.6fs)\n", d.name, out_path.c_str(), t);
      }
      free(in_u8);
    } else {
      printf("[DEMO] Skipped: cannot find or wrong size input_M1024.pgm (need 1024x1024)\n");
      if (in_u8) free(in_u8);
    }
  }

  for (int mi = 0; mi < 3; mi++) {
    int M = Ms[mi];

#ifdef _WIN32
    std::string in_path = base_dir + "\\input_M" + std::to_string(M) + ".pgm";
#else
    std::string in_path = base_dir + "/input_M" + std::to_string(M) + ".pgm";
#endif

    int w=0,h=0;
    uint8_t* in_u8 = read_pgm_p5(in_path.c_str(), &w, &h);
    if (!in_u8 || w != M || h != M) {
      printf("[WARN] Missing or wrong size: %s (need %dx%d)\n", in_path.c_str(), M, M);
      if (in_u8) free(in_u8);
      continue;
    }

    std::vector<int32_t> img_i32(M*M);
    for (int i = 0; i < M*M; i++) img_i32[i] = (int32_t)in_u8[i];

    std::vector<int32_t> out_cpu(M*M);
    std::vector<int32_t> out_gpu(M*M);

    for (int ni = 0; ni < 3; ni++) {
      int N = Ns[ni];
      auto ker = make_edge_kernel_N(N);

      // CPU timing
      auto t0 = std::chrono::high_resolution_clock::now();
      conv2d_cpu_i32(img_i32.data(), ker.data(), out_cpu.data(), M, N);
      auto t1 = std::chrono::high_resolution_clock::now();
      double cpu_sec = std::chrono::duration<double>(t1 - t0).count();

      // GPU warmup + timing
      run_gpu_once(img_i32.data(), ker.data(), out_gpu.data(), M, N); // warmup
      float gpu_sec = run_gpu_once(img_i32.data(), ker.data(), out_gpu.data(), M, N);

      long long csum_cpu = checksum_i32(out_cpu.data(), M*M);
      long long csum_gpu = checksum_i32(out_gpu.data(), M*M);
      double speedup = (gpu_sec > 0.0) ? (cpu_sec / (double)gpu_sec) : 0.0;

      printf("M=%d N=%d | CPU=%.6fs GPU=%.6fs speedup=%.2fx | cpu_sum=%lld gpu_sum=%lld\n",
             M, N, cpu_sec, gpu_sec, speedup, csum_cpu, csum_gpu);

      fprintf(csv, "%d,%d,%.9f,%.9f,%.6f,%lld,%lld\n",
              M, N, cpu_sec, (double)gpu_sec, speedup, csum_cpu, csum_gpu);
    }

    free(in_u8);
  }

  fclose(csv);
  printf("Saved CSV: %s\n", csv_path.c_str());
  printf("Output demo images folder: %s\n", out_dir.c_str());
  printf("Done.\n");
  return 0;
}