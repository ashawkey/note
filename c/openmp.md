# openmp

[document](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5.0.pdf)

### Code example

```c++
#include <omp.h> // OpenMP header 
#include <stdio.h> 
#include <stdlib.h> 
  
int main(int argc, char* argv[]) 
{ 
    int nthreads, tid; 
    // Begin of parallel region 
    #pragma omp parallel private(nthreads, tid) 
    { 
        // Getting thread number 
        tid = omp_get_thread_num(); 
        printf("Welcome from thread = %d\n", tid); 
  
        if (tid == 0) { 
            // Only master thread does this 
            nthreads = omp_get_num_threads(); 
            printf("Number of threads = %d\n", 
                   nthreads); 
        } 
    } 
} 
```

### Compile with

```bash
gc++ main.c -fopenmp

# run with 4 threads:
OMP_NUM_THREADS=4 ./a.out
```

