### coalesced atomics

From [`nvdiffrast common.h`](https://github.com/NVlabs/nvdiffrast/blob/main/nvdiffrast/common/common.h)

TODO: how much will it improve performance???

```cpp
//------------------------------------------------------------------------
// Coalesced atomics. These are all done via macros.

#if __CUDA_ARCH__ >= 700 // Warp match instruction __match_any_sync() is only available on compute capability 7.x and higher

#define CA_TEMP       _ca_temp
#define CA_TEMP_PARAM float* CA_TEMP
#define CA_DECLARE_TEMP(threads_per_block) \
    __shared__ float CA_TEMP[(threads_per_block)]

#define CA_SET_GROUP_MASK(group, thread_mask)                   \
    bool   _ca_leader;                                          \
    float* _ca_ptr;                                             \
    do {                                                        \
        int tidx   = threadIdx.x + blockDim.x * threadIdx.y;    \
        int lane   = tidx & 31;                                 \
        int warp   = tidx >> 5;                                 \
        int tmask  = __match_any_sync((thread_mask), (group));  \
        int leader = __ffs(tmask) - 1;                          \
        _ca_leader = (leader == lane);                          \
        _ca_ptr    = &_ca_temp[((warp << 5) + leader)];         \
    } while(0)

#define CA_SET_GROUP(group) \
    CA_SET_GROUP_MASK((group), 0xffffffffu)

#define caAtomicAdd(ptr, value)         \
    do {                                \
        if (_ca_leader)                 \
            *_ca_ptr = 0.f;             \
        atomicAdd(_ca_ptr, (value));    \
        if (_ca_leader)                 \
            atomicAdd((ptr), *_ca_ptr); \
    } while(0)

#define caAtomicAddTexture(ptr, level, idx, value)  \
    do {                                            \
        CA_SET_GROUP((idx) ^ ((level) << 27));      \
        caAtomicAdd((ptr)+(idx), (value));          \
    } while(0)

//------------------------------------------------------------------------
// Disable atomic coalescing for compute capability lower than 7.x

#else // __CUDA_ARCH__ >= 700
#define CA_TEMP _ca_temp
#define CA_TEMP_PARAM float CA_TEMP
#define CA_DECLARE_TEMP(threads_per_block) CA_TEMP_PARAM
#define CA_SET_GROUP_MASK(group, thread_mask)
#define CA_SET_GROUP(group)
#define caAtomicAdd(ptr, value) atomicAdd((ptr), (value))
#define caAtomicAddTexture(ptr, level, idx, value) atomicAdd((ptr)+(idx), (value))
#endif // __CUDA_ARCH__ >= 700

//------------------------------------------------------------------------
```


### `atomicAdd` for `at::Half`

```cpp
// at::Half atomicAdd, requires CUDA >= 10 && arch >= 7.x
// ref: https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCAtomics.cuh#L184
// caveat: extremely slow, never use!
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

