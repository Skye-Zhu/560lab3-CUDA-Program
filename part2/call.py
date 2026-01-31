import ctypes
import numpy as np
import time

lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C"),
    ctypes.c_int,
    ctypes.c_int
]

M = 1024
N = 3

image = np.random.randint(0, 255, (M, M), dtype=np.uint32)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.uint32)

output = np.zeros((M, M), dtype=np.uint32)

start = time.time()
lib.gpu_convolution(
    image.ravel(),
    kernel.ravel(),
    output.ravel(),
    M, N
)
end = time.time()

print("CUDA convolution time:", end - start)
