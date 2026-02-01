#include <stdint.h>

#ifdef _WIN32
  #ifdef __cplusplus
    #define DLL_EXPORT extern "C" __declspec(dllexport)
  #else
    #define DLL_EXPORT __declspec(dllexport)
  #endif
#else
  #ifdef __cplusplus
    #define DLL_EXPORT extern "C"
  #else
    #define DLL_EXPORT
  #endif
#endif



void conv2d_cpu_i32(
    const int32_t *image,
    const int32_t *kernel,
    int32_t *output,
    int M, int N
) {
    int offset = N / 2;

    // initialize output to 0
    for (int i = 0; i < M * M; i++) output[i] = 0;

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

DLL_EXPORT void cpu_convolution_i32(
    const int32_t *image,
    const int32_t *kernel,
    int32_t *output,
    int M, int N
) {
    conv2d_cpu_i32(image, kernel, output, M, N);
}
