# Custom C++ & CUDA extensions


## Pitfalls

* always check the input tensor is `contiguous` !!!

  else `.data_ptr<float>()` will generate a mess and you never get the expected values.

* 


## Bindings

### Ahead of Time (setup)

```c
#include <pybind11/pybind11.h>
#include "src/add.hpp"

/*
 * macro PYBIND11_MODULE(module_name, m)
 *     module_name: should not be in quotes. import this name in python.
 *     m: pybind11::module interface
 * */
PYBIND11_MODULE(add_cpp, m) {
    m.def("forward", &Add_forward_cpu, "Add forward");
    m.def("backward", &Add_backward_cpu, "Add backward");

}
```

```python
import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

source_files = glob.glob('src/*.cpp') + ['bindings.cpp']

'''
name: import name in python, should be the same as in bindings.cpp
'''
setup(
    name = 'add_cpp',
    ext_modules = [
        CppExtension(
            name = 'add_cpp', 
            sources = source_files,
            extra_compile_args = {
                'cxx': ['-O3'],
            }
        ),
    ],
    cmdclass = {'build_ext': BuildExtension}
)
```

```python
import torch
from torch.autograd import Function
import torch.nn as nn

import add_cpp

# Function
class _add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return add_cpp.forward(x, y)

    @staticmethod
    def backward(ctx, gradOutput):
        gradX, gradY = add_cpp.backward(gradOutput)
        return gradX, gradY

add = _add.apply


# Module
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return add(x, y)
```

```python
from add import add, Add
import torch

'''
static setup, run first:
python setup.py install
'''

one = torch.ones(1)

adder = Add()
print(adder(one, one))

print(add(one, one))
```


### Just in Time (load)

Better use this.

```python
import os
import glob
from torch.utils.cpp_extension import load

source_files = glob.glob('src/*.cpp') + ['bindings.cpp']

'''
dynamic load of extensions.
    name: should be the same module_name as in bindings.cpp
'''
_backend = load(
    name='_backend',
    extra_cflags=['-O3', '-std=c++17'],
    sources=source_files,
)
```

```python
import torch
from torch.autograd import Function
import torch.nn as nn

from backend import _backend

'''
Encapsulation of raw methods.
pytorch has functional and Module level encapsulations.
'''

# Function
class _add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return _backend.forward(x, y)

    @staticmethod
    def backward(ctx, gradOutput):
        gradX, gradY = _backend.backward(gradOutput)
        return gradX, gradY

add = _add.apply

# Module
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return add(x, y)
```


## Torch C++ API

### Header

```c++
#include <ATen/ATen.h> // at::
#include <torch/extension.h> // torch::
```


### ATen

A Tensor Library for CPU & GPU. Not differentiable.

```c++
#include <ATen/ATen.h>

at::Tensor a = at::randn({2, 2});
bool gpu = at::cuda::is_available();
```

```c++
// shape
int x = a.size(idx);
auto sizes = a.sizes();

// access raw data
a.data_ptr<int>()

// property
a.device();
a.dtype();
```

```c++
at::cuda::getCurrentCUDAStream();
at::device(a.device()).dtype(at::ScalarType::Int));

```


### C++ Frontend

high level library providing CPU & GPU tensor support with automatic differentiation.

```c++
torch::Tensor x = torch::ones({2,3,4});
x.sizes();
x.size(0);
```

A full example:

```c++
#include <torch/torch.h>

struct Net : torch::nn::Module {
  // constructor
  Net()
      : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(320, 50),
        fc2(50, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(
        torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
  }

  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::FeatureDropout conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
};
```
