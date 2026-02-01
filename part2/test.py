import numpy as np
import time
from PIL import Image
from wrapper import conv_gpu_naive_py
import csv

def gen_image(M, maxval=255, seed=0):
    np.random.seed(seed)
    return np.random.randint(0, maxval + 1, size=(M, M), dtype=np.uint32)

def mean_kernel(N):
    return np.ones((N, N), dtype=np.float32) / (N * N)

def conv_cpu_py(img, kernel):
    M = img.shape[0]
    N = kernel.shape[0]
    pad = N // 2
    out = np.zeros_like(img, dtype=np.uint32)
    for y in range(M):
        for x in range(M):
            acc = 0.0
            for ky in range(N):
                for kx in range(N):
                    iy = y + ky - pad
                    ix = x + kx - pad
                    if 0 <= iy < M and 0 <= ix < M:
                        acc += img[iy, ix] * kernel[ky, kx]
            out[y, x] = max(0, int(acc))
    return out

def run_tests():
    Ms = [512, 1024, 2048]
    Ns = [3, 5, 11]
    results = []

    for M in Ms:
        img = gen_image(M)
        for N in Ns:
            kern = mean_kernel(N)
            t0 = time.time()
            _ = conv_gpu_naive_py(img, kern)
            t1 = time.time()
            gpu_time = t1 - t0
            print(f"M={M}, N={N}, GPU naive time: {gpu_time:.6f}s")
            results.append(("gpu_naive", M, N, gpu_time))

    with open("perf_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "M", "N", "time_s"])
        writer.writerows(results)

    print("Performance results saved to perf_results.csv")

if __name__ == "__main__":
    run_tests()
