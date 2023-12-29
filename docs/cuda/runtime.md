# Runtime

### Memory transfer

```c
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = 10;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);

    // Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    float *h_C;
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}
```

### Error check

```cpp
#include <cuda_runtime.h>
#include <helper_cuda.h>

int main() {
    cudaError_t err = cudaSuccess;
    
    // functions
    err = cudaMalloc((void**)&d, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }
    
    // kernels
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    
    // simple way
	checkCudaErrors(cudaMalloc(.));
}
```


