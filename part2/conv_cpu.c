
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

void conv_cpu_uint32(
    const uint32_t *in, uint32_t *out,
    int M, int N, const float *kernel)
{
    int pad = N / 2;
    for (int y = 0; y < M; ++y) {
        for (int x = 0; x < M; ++x) {
            float acc = 0.0f;
            for (int ky = 0; ky < N; ++ky) {
                int iy = y + ky - pad;
                for (int kx = 0; kx < N; ++kx) {
                    int ix = x + kx - pad;
                    uint32_t val = 0;
                    if (ix >= 0 && ix < M && iy >= 0 && iy < M) {
                        val = in[iy * M + ix];
                    } 
                    acc += ((float)val) * kernel[ky * N + kx];
                }
            }
            if (acc < 0.0f) acc = 0.0f;
            if (acc > (float)UINT32_MAX) acc = (float)UINT32_MAX;
            out[y * M + x] = (uint32_t)(acc + 0.5f);
        }
    }
}


int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s M N\n", argv[0]);
        printf("Example: %s 1024 3\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int img_size = M*M;
    uint32_t *in = (uint32_t*)malloc(sizeof(uint32_t)*img_size);
    uint32_t *out = (uint32_t*)malloc(sizeof(uint32_t)*img_size);
    float *kernel = (float*)malloc(sizeof(float)*N*N);

    srand(0);
    for (int i=0;i<img_size;i++) in[i] = rand()%256;

    float s = 1.0f/(N*N);
    for (int i=0;i<N*N;i++) kernel[i] = s;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    conv_cpu_uint32(in, out, M, N, kernel);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
    FILE *fp = fopen("cpu_results.csv", "a");
    fprintf(fp, "cpu,%d,%d,%.6f\n", M, N, elapsed);
    fclose(fp);
    printf("M=%d N=%d CPU time: %.6f s\n", M, N, elapsed);



    if (M<=8) {
        for (int y=0;y<M;y++){
            for (int x=0;x<M;x++){
                printf("%u ", out[y*M+x]);
            }
            printf("\n");
        }
    }

    free(in); free(out); free(kernel);
    return 0;
}
