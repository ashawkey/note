# CuTe DSL

## Install

Only for Linux + Python 3.12 + CUDA 12.9

```bash
pip install nvidia-cutlass-dsl
```

```python
import cutlass
import cutlass.cute as cute
```



### Basics

Kernels (`@cute.kernel`) are CUDA kernels (like `__global__`) that runs on GPU devices.

```python
@cute.kernel
def kernel():
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    # Only the first thread (thread 0) prints the message
    if tidx == 0:
        cute.printf("Hello world")
```

To launch kernels, we need host function on CPU (`@cute.jit`) 

```python
@cute.jit
def hello_world():
    cute.printf("hello world")
    # Launch kernel
    kernel().launch(
        grid=(1, 1, 1),   # Single thread block
        block=(32, 1, 1)  # One warp (32 threads) per thread block
    )
```

The code can run in JIT or pre-compile modes:

```python
# directly call will run in JIT.
hello_world()

# or pre-compile
hello_world_compiled = cute.compile(hello_world)
hello_world_compiled()
```

### Print

The cute program will run at both **compile** and **runtime**. Python `print` will be called at compile time (and only know static values), while `cute.printf` will be called at runtime (know both static and dynamic values).

```python
@cute.jit
def print_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    """
    Demonstrates different printing methods in CuTe and how they handle static vs dynamic values.

    This example shows:
    1. How Python's `print` function works with static values at compile time but can't show dynamic values
    2. How `cute.printf` can display both static and dynamic values at runtime
    3. The difference between types in static vs dynamic contexts
    4. How layouts are represented in both printing methods

    Args:
        a: A dynamic Int32 value that will be determined at runtime
        b: A static (compile-time constant) integer value
    """
    # Use Python `print` to print static information
    print(">>>", b)  # => 2
    # `a` is dynamic value
    print(">>>", a)  # => ?

    # Use `cute.printf` to print dynamic information
    cute.printf(">?? {}", a)  # => 8
    cute.printf(">?? {}", b)  # => 2

    print(">>>", type(a))  # => <class 'cutlass.Int32'>
    print(">>>", type(b))  # => <class 'int'>

    layout = cute.make_layout((a, b))
    print(">>>", layout)            # => (?,2):(1,?)
    cute.printf(">?? {}", layout)   # => (8,2):(1,8)
```

Output:

```bash
### print_example(cutlass.Int32(8), 2)
# print at compile time
>>> 2 
>>> ?
>>> Int32
>>> <class 'int'>
>>> (?,2):(1,?)
# printf at runtime
>?? 8
>?? 2
>?? (8,2):(1,8)
```

