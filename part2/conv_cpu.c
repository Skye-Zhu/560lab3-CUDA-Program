#include <stdio.h>
#include <stdlib.h>

void conv2d_cpu(
    unsigned int *image,
    unsigned int *kernel,
    unsigned int *output,
    int M, int N
) {
    int offset = N / 2;

    for (int i = offset; i < M - offset; i++) {
        for (int j = offset; j < M - offset; j++) {
            unsigned int sum = 0;

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
