import ctypes
import numpy as np
import time
lib = ctypes.CDLL('./libconv_cuda.so')

# assume functions extern "C" int conv_gpu_naive_launch(const uint32_t *h_in, uint32_t *h_out, int M, int N, const float *h_kernel);
lib.conv_gpu_naive_launch.restype = ctypes.c_int
lib.conv_gpu_naive_launch.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

def conv_gpu_naive_py(img: np.ndarray, kernel: np.ndarray):
    M = img.shape[0]
    N = kernel.shape[0]
    assert img.dtype == np.uint32
    assert kernel.dtype == np.float32
    out = np.zeros_like(img)
    lib.conv_gpu_naive_launch(img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                              ctypes.c_int(M), ctypes.c_int(N),
                              kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out