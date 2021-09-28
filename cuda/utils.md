### `atomicAdd` for `at::Half`

```cpp
// at::Half atomicAdd, requires CUDA >= 10 && arch >= 7.x
// ref: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh#L184
static inline  __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
  return atomicAdd(reinterpret_cast<__half*>(address), val);
}
```





### `atomicMax` for `at::Half, float, double`

```cpp
// atomicMax for __half, float, double (requires arch > 7.0 for ushort ver of atomicCAS)
// ref: https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
// ref: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-aware-mpi-example/src/Device.cu
static inline __device__ double atomicMax(double* address, double val)
{
  unsigned long long* address_as_ll = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ll, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ll, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

static inline __device__ float atomicMax(float* address, float val)
{
  int* address_as_i = reinterpret_cast<int*>(address);
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

static inline __device__ __half atomicMax(__half *address, __half val) {
  unsigned short* address_as_ushort = reinterpret_cast<unsigned short*>(address);
  unsigned short old = *address_as_ushort, assumed;
  do {
    assumed = old;
    // ref: https://github.com/NVIDIA/TensorRT/issues/1003
    // __hmax is only available at arch >= 800 ? even the doc says it should be available from arch >= 520.
    //old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(__hmax(val, __ushort_as_half(assumed))));
    old = atomicCAS(address_as_ushort, assumed, __half_as_ushort(__float2half(fmaxf(__half2float(val), __half2float(__ushort_as_half(assumed))))));
  } while (assumed != old);
  return __ushort_as_half(old);
}

static inline __device__ at::Half atomicMax(at::Half *address, at::Half val) {
  return atomicMax(reinterpret_cast<__half*>(address), val);
}
```

