import os
import sys
import csv
import time
import ctypes
import numpy as np
from PIL import Image


DEFAULT_INPUT = "input.png"

# Performance sweep 
MS = [512, 1024, 2048]
NS = [3, 5, 7]


DEMO_M = 1024
DEMO_N = 3

GPU_WARMUP = 1
GPU_REPEAT = 3  # average over repeats



def load_dll():
    here = os.path.dirname(os.path.abspath(__file__))
    dll_path = os.path.join(here, "libmatrix.dll")
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"Cannot find libmatrix.dll at: {dll_path}")

    lib = ctypes.CDLL(dll_path)

    # CPU function
    lib.cpu_convolution_i32.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.cpu_convolution_i32.restype = None

    # GPU function
    lib.gpu_convolution_i32.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int
    ]
    lib.gpu_convolution_i32.restype = None

    return lib



def load_grayscale_i32(path: str, M: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((M, M))
    return np.array(img, dtype=np.int32)


def save_i32_as_png(arr_i32: np.ndarray, out_path: str, mode: str):
    x = arr_i32.astype(np.int32)

    if mode == "linear":
        x = np.clip(x, 0, 255).astype(np.uint8)

    elif mode == "edge":
        x = np.abs(x)
        mx = int(x.max()) if x.size else 0
        if mx > 0:
            x = (x * 255) // mx
        x = np.clip(x, 0, 255).astype(np.uint8)

    elif mode.startswith("blur_div:"):
        div = int(mode.split(":")[1])
        x = x // div
        x = np.clip(x, 0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Unknown save mode: {mode}")

    Image.fromarray(x, mode="L").save(out_path)


def kernel_blur_3x3() -> np.ndarray:
    # Sum to 9 -> we will divide in visualization to keep it in 0..255
    return np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]], dtype=np.int32)


def kernel_sharpen_3x3() -> np.ndarray:
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]], dtype=np.int32)


def kernel_sobel_x_3x3() -> np.ndarray:
    # Edge detection (Sobel X)
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.int32)


def make_edge_kernel_N(N: int) -> np.ndarray:
    if N == 3:
        return kernel_sobel_x_3x3()

    k = np.zeros((N, N), dtype=np.int32)
    mid = N // 2

    # left column negative, right column positive
    k[:, 0] = -1
    k[:, -1] = 1

    # emphasize center row a bit
    k[mid, 0] = -2
    k[mid, -1] = 2
    return k



def run_cpu(lib, image_i32: np.ndarray, kernel_i32: np.ndarray, M: int, N: int):
    img1d = np.ascontiguousarray(image_i32.reshape(-1), dtype=np.int32)
    ker1d = np.ascontiguousarray(kernel_i32.reshape(-1), dtype=np.int32)
    out1d = np.zeros(M * M, dtype=np.int32)

    t0 = time.perf_counter()
    lib.cpu_convolution_i32(img1d, ker1d, out1d, M, N)
    t1 = time.perf_counter()

    return out1d.reshape(M, M), (t1 - t0), int(out1d.sum())


def run_gpu(lib, image_i32: np.ndarray, kernel_i32: np.ndarray, M: int, N: int,
            warmup: int = GPU_WARMUP, repeat: int = GPU_REPEAT):
    img1d = np.ascontiguousarray(image_i32.reshape(-1), dtype=np.int32)
    ker1d = np.ascontiguousarray(kernel_i32.reshape(-1), dtype=np.int32)
    out1d = np.zeros(M * M, dtype=np.int32)

    # warmup
    for _ in range(max(warmup, 0)):
        lib.gpu_convolution_i32(img1d, ker1d, out1d, M, N)

    t0 = time.perf_counter()
    for _ in range(max(repeat, 1)):
        lib.gpu_convolution_i32(img1d, ker1d, out1d, M, N)
    t1 = time.perf_counter()

    avg = (t1 - t0) / max(repeat, 1)
    return out1d.reshape(M, M), avg, int(out1d.sum())



def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # Input path
    img_path = os.path.join(here, DEFAULT_INPUT)
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        if not os.path.isabs(img_path):
            img_path = os.path.join(here, img_path)

    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Cannot find input image: {img_path}\n"
            f"Put an image named '{DEFAULT_INPUT}' in the same folder, or run:\n"
            f"  python call_images.py your_image.png"
        )

    # Output folder
    out_dir = os.path.join(here, "out_images")
    os.makedirs(out_dir, exist_ok=True)

    # Load DLL
    lib = load_dll()

 
    print("Demo images (blur/sharpen/edge)")
    image_demo = load_grayscale_i32(img_path, DEMO_M)
    Image.fromarray(image_demo.astype(np.uint8), mode="L").save(
        os.path.join(out_dir, f"input_M{DEMO_M}.png")
    )

    demo_filters = [
        ("blur", kernel_blur_3x3(), "blur_div:9"),
        ("sharpen", kernel_sharpen_3x3(), "linear"),
        ("edge", kernel_sobel_x_3x3(), "edge"),
    ]

    for name, ker, save_mode in demo_filters:
        out_gpu, t_gpu, chk_gpu = run_gpu(lib, image_demo, ker, DEMO_M, DEMO_N, warmup=1, repeat=3)
        out_path = os.path.join(out_dir, f"{name}_gpu_M{DEMO_M}_N{DEMO_N}.png")
        save_i32_as_png(out_gpu, out_path, save_mode)
        print(f"[GPU] {name}: avg_time={t_gpu:.6f}s checksum={chk_gpu} saved={out_path}")


    print("\nPerformance sweep (CPU vs GPU) using EDGE filter")
    rows = []

    for M in MS:
        image = load_grayscale_i32(img_path, M)

        for N in NS:
            ker = make_edge_kernel_N(N)

            # CPU (single run)
            out_cpu, t_cpu, chk_cpu = run_cpu(lib, image, ker, M, N)

            # GPU (warmup + average)
            out_gpu, t_gpu, chk_gpu = run_gpu(lib, image, ker, M, N, warmup=1, repeat=3)

            speedup = (t_cpu / t_gpu) if t_gpu > 0 else float("inf")

            print(f"M={M} N={N} | CPU={t_cpu:.6f}s GPU={t_gpu:.6f}s speedup={speedup:.2f} "
                  f"| cpu_sum={chk_cpu} gpu_sum={chk_gpu}")

            rows.append({
                "M": M,
                "N": N,
                "cpu_time_sec": t_cpu,
                "gpu_time_sec": t_gpu,
                "speedup": speedup,
                "cpu_checksum": chk_cpu,
                "gpu_checksum": chk_gpu,
            })


    # Save CSV
    csv_path = os.path.join(here, "perf_conv.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nSaved performance CSV:", csv_path)
    print("Output images folder:", out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()