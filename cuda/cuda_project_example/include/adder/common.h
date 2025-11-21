#include <iostream>
#include <cuda.h>

// common functions
namespace adder {

template<typename T>
__host__ __device__ T add(T x, T y) {
    return x + y;
}

}
