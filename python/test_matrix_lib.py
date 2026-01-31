import os
import ctypes
import time
import numpy as np

def load_dll():
    # project_root/python/test_matrix_lib.py -> project_root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dll_path = os.path.join(project_root, "matrix_lib.dll")

    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"matrix_lib.dll not found at: {dll_path}")

    # Use WinDLL (stdcall/cdecl differences usually OK here, but WinDLL is safer on Windows)
    lib = ctypes.WinDLL(dll_path)
    return lib, dll_path

def setup_signature(lib):
    # void gpu_matrix_multiply(const float* h_A, const float* h_B, float* h_C, int N)
    lib.gpu_matrix_multiply.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    lib.gpu_matrix_multiply.restype = None

def run_once(lib, A, B, C, N):
    # Ensure contiguous float32
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    C = np.ascontiguousarray(C, dtype=np.float32)

    lib.gpu_matrix_multiply(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N
    )
    return C

def checksum(C):
    return float(np.sum(C, dtype=np.float64))

def main():
    lib, dll_path = load_dll()
    setup_signature(lib)

    # Test sizes (keep small for correctness, larger for timing)
    sizes = [512, 1024, 2048]

    # Repeats for timing (GPU calls include H2D/D2H in the DLL, so repeats help stability)
    repeats = 5

    print(f"Loaded DLL: {dll_path}")

    # Create results folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "timing_part7.output")

    # Write header
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("implementation,N,time_ms,checksum\n")

        for N in sizes:
            # Allocate host matrices
            np.random.seed(0)
            A = np.random.rand(N, N).astype(np.float32)
            B = np.random.rand(N, N).astype(np.float32)
            C = np.zeros((N, N), dtype=np.float32)

            # ---- correctness check (spot-check vs numpy for small N only)
            if N <= 512:
                # compute reference on CPU (numpy)
                ref = A @ B
            else:
                ref = None

            # ---- warm-up
            run_once(lib, A, B, C, N)

            # ---- timing (average ms per call)
            t0 = time.perf_counter()
            for _ in range(repeats):
                run_once(lib, A, B, C, N)
            t1 = time.perf_counter()

            avg_ms = (t1 - t0) * 1000.0 / repeats
            cs = checksum(C)

            # correctness (relative error) for small N
            if ref is not None:
                # avoid divide by zero
                denom = np.linalg.norm(ref) + 1e-8
                rel_err = float(np.linalg.norm(ref - C) / denom)
                print(f"impl=LIB_TILED,N={N},time_ms={avg_ms:.3f},checksum={cs:.6f},rel_err={rel_err:.3e}")
            else:
                print(f"impl=LIB_TILED,N={N},time_ms={avg_ms:.3f},checksum={cs:.6f}")

            # write file
            f.write(f"LIB_TILED,{N},{avg_ms:.3f},{cs:.6f}\n")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()