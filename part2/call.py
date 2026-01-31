import ctypes
import numpy as np
import time

lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),  
    np.ctypeslib.ndpointer(dtype=np.int32,  ndim=1, flags="C_CONTIGUOUS"),  
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),  
    ctypes.c_int,
    ctypes.c_int
]
lib.gpu_convolution.restype = None  

M = 1024
N = 3

image = np.random.randint(0, 255, (M, M), dtype=np.uint32)

kernel = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
], dtype=np.int32)   

output = np.zeros((M, M), dtype=np.uint32)

img_buf = np.ascontiguousarray(image.ravel(), dtype=np.uint32)
ker_buf = np.ascontiguousarray(kernel.ravel(), dtype=np.int32)
out_buf = np.ascontiguousarray(output.ravel(), dtype=np.uint32)


start = time.time()
lib.gpu_convolution(img_buf, ker_buf, out_buf, M, N)
end = time.time()

print("CUDA convolution time:", end - start)

output2d = out_buf.reshape(M, M)
