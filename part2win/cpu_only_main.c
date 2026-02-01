#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

void conv2d_cpu_i32(
    const int32_t *image,
    const int32_t *kernel,
    int32_t *output,
    int M, int N
);

static long long checksum(const int32_t* a, int n){
    long long s = 0;
    for(int i = 0; i < n; i++) s += a[i];
    return s;
}

int main(void){
    int M = 1024;
    int N = 3;

    int32_t *image  = (int32_t*)malloc((size_t)M*M*sizeof(int32_t));
    int32_t *output = (int32_t*)malloc((size_t)M*M*sizeof(int32_t));

    int32_t kernel[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    for(int i = 0; i < M*M; i++)
        image[i] = i % 256;

    clock_t t0 = clock();
    conv2d_cpu_i32(image, kernel, output, M, N);
    clock_t t1 = clock();

    double sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("CPU convolution time: %.6f sec\n", sec);
    printf("Checksum: %lld\n", checksum(output, M*M));

    free(image);
    free(output);
    return 0;
}
