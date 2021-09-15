# thrust 

### Vectors

Just like the `std::vector`.

```c
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>

int main(void) {
    thrust::host_vector<int> H(10, 0);
    thrust::device_vector<int> D = H; // support H <--> D copy (call cudaMemcpy inside)
    
    int s = H.size();
    D[0] = 0;
    int x = H[0];
    
    // functions
    thrust::fill(D.begin(), D.begin + 3, 1);
    thrust::sequence(H.begin, H.end()); // assign range [0, 1, ..., H.size() - 1]
    thrust::copy(H.begin(), H.end(), D.begin()); // copy all H to D
    
    // also support raw device array, but need to wrap it with device_ptr<T>
    int *arr;
    cudaMalloc((void**)&arr, 10 * sizeof(int));
    thrust::device_ptr<int> dev_arr(arr); // wrap!
    thrust::fill(dev_arr, dev_arr + 10, 1); 
    
    // also support unwrap.
    thrust::device_ptr<int> dev_ptr = thrust::device_malloc<int>(N);
    int* arr =  thrust::raw_pointer_cast(dev_arr); // unwrap 
    
    // support copy to STL
    vector<int> S(D.size());
    thrust::copy(D.begin(), D.end(), S.begin()); // cudaMemcpy inside
}
```



### Algorithms

Thrust algorithms will auto detect host/device and run the correct version.

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <iostream>

int main(void)
{
    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> X(10);
    thrust::device_vector<int> Y(10);
    thrust::device_vector<int> Z(10);

    // initialize X to 0,1,2,3, ....
    thrust::sequence(X.begin(), X.end());

    // compute Y = -X
    thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());

    // fill Z with twos
    thrust::fill(Z.begin(), Z.end(), 2);

    // compute Y = X mod 2
    thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());

    // replace all the ones in Y with tens
    thrust::replace(Y.begin(), Y.end(), 1, 10);
    
    // reduce
    int sum = thrust::reduce(D.begin(), D.end(), (int) 0, thrust::plus<int>());
	int sum = thrust::reduce(D.begin(), D.end(), (int) 0);
    int sum = thrust::reduce(D.begin(), D.end())
        
    // count the 1s
    thrust::device_vector<int> vec(5,0);
    vec[1] = 1;
    vec[3] = 1;
    vec[4] = 1;
    int result = thrust::count(vec.begin(), vec.end(), 1); // 3
    
    // prefix-sum (scan)
    int data[6] = {1, 0, 2, 2, 1, 3};
	thrust::inclusive_scan(data, data + 6, data); // in-place, data is now {1, 1, 3, 5, 6, 9}
    int data[6] = {1, 0, 2, 2, 1, 3};
    thrust::exclusive_scan(data, data + 6, data); // in-place, data is now {0, 1, 1, 3, 5, 6} (right shifted)
    
    // sort
    const int N = 6;
    int A[N] = {1, 4, 2, 8, 5, 7};
	thrust::sort(A, A + N); // in-place sort
    
    int keys[N] = {1, 4, 2, 8, 5, 7};
    char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};
    thrust::sort_by_key(keys, keys + N, values); // both keys and values are sorted in-place
    
    // print Y
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));
   
    return 0;    
}
```

`transform` is very useful for paralleled vector operations (at most two parameters), for example, we can implement our own function and parallel it:

```cpp
// custorm_func(x, y) = x + y * a;
struct custom_func {
    const float a;
    custom_func(float _a) : a(_a) {}
	__host__ __device__ float operator() (const float& x, const float& y) const {
        return x + y * a;
    }
};

// call with
thrust::transform(X.begin(), X.end(), Y.begin(), Y.end(), custom_func(a));

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__ T operator()(const T& x) const { 
            return x * x;
        }
};

// compute norm
float norm = std::sqrt(thrust::transform_reduce(d_x.begin(), d_x.end(), square<float>(), 0, thrust::plus<float>()));
```



### iterators

```cpp
#include <thrust/iterator/constant_iterator.h>
thrust::constant_iterator<int> first(10); // init value is 10
first[0]; // 10
first[100]; // 10
thrust.reduce(first, first + 3); // 30


#include <thrust/iterator/counting_iterator.h>
thrust::counting_iterator<int> first(10); // init value is 10, auto incremental
first[0]; // 10
first[1]; // 11
first[100]; // 110


#include <thrust/iterator/zip_iterator.h>
// initialize vectors
thrust::device_vector<int>  A(3);
thrust::device_vector<char> B(3);
A[0] = 10;  A[1] = 20;  A[2] = 30;
B[0] = 'x'; B[1] = 'y'; B[2] = 'z';
// create iterator (type omitted)
first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));
last  = thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end()));
first[0]   // returns tuple(10, 'x')
first[1]   // returns tuple(20, 'y')
first[2]   // returns tuple(30, 'z')
// maximum of [first, last)
thrust::tuple<int,char> init = first[0];
thrust::reduce(first, last, init, thrust::maximum<tuple<int,char>>()); // returns tuple(30, 'z')
```



