import numpy as np
import cupy as cp
import time

### Numpy and CPU
s = time.time()
x_cpu = np.ones((980,980,980))
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu = cp.ones((980,980,980))
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)


### Numpy and CPU
s = time.time()
x_cpu *= 5
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu *= 5
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)


### Numpy and CPU
s = time.time()
x_cpu *= 5
x_cpu *= x_cpu
x_cpu += x_cpu
e = time.time()
print(e - s)
### CuPy and GPU
s = time.time()
x_gpu *= 5
x_gpu *= x_gpu
x_gpu += x_gpu
cp.cuda.Stream.null.synchronize()
e = time.time()
print(e - s)