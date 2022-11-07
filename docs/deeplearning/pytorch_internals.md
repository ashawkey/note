## PyTorch Internals

A summary of [ezyang's blog](http://blog.ezyang.com/2019/05/pytorch-internals/).

### Tensor View and Storage

```python
## View (logical)
# shape: [2, 2], logical shape
# stride: [2, 1], the stride for each axis
# offset: 0, the offset of storage pointer
x = [[1, 2],
     [3, 4]]
 
## Storage (physical)
# always a 1D contiguous array.
x.storage = [1, 2, 3, 4]

## Slicing
y = x[1:] # [3, 4], shape = [2], stride = [1], offset = 2
z = x[:, 0] # [1, 3], shape = [2], stride = [2], offset = 0

## Stride and Index:
def index_to_pos(index, stride, offset=0):
    # index: logical, same dimension as stride
    # pos: physical, 1d value.
    # this works for both continuous and discoutinuous tensors!
    pos = offset
    for i, s in zip(index, stride):
        pos += i * s
    return pos

def strides_from_shape(shape):
    # assuming a continuous tensor!
    # example: shape [3, 4, 5] --> stride []
    layout = [1]
    offset = 1
    for s in reversed(shape[1:]): # [5, 4]
        layout.append(s * offset) # [1, 5, 20]
        offset = s * offset # [5, 20]
    return tuple(reversed(layout)) # [20, 5, 1]

def is_contiguous(stride):
    # continuous tensor <==> mono-decreasing stride && last stride is 1
    last = stride[0]
    for s in stride[1:]:
        if s > last: 
            return False
	    else:
            last = s
    if last > 1:
        return False
   	else;
		return True
```

* Each tensor always has a view-storage pair. 
* Multiple tensor views can share the same storage.

* Tensor trinity that decides its true implementation:

  * device: CPU, CUDA, XLA, ...

  * layout: Strided, Sparse, ...

  * dtype: int, float, ...



### Source Structure

```bash
torch/ # torch frontend
torch/csrc/ # c++ frontent, python bindings, autograd engine, JIT compiler
aten/ # a tensor library, most tensor operations.
c10/ # implementation of core abstractions.
```

Mostly we would want to extend the `aten` part for new operators.

First, you should register your op in some config file: [detailed workflow](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md)

Each op should be implemented in three versions:

```python
myop_out(..., out) # write to out
# usually the following two funcs can just call myop_out()
myop_(...) # inplace
myop() # return the output
```

Then you should write kernels as the real implementation.

You'll need to `dispatch` based on `dtype`, and implement each (or use template).



### Workflow efficiency

* Editing headers sparingly, since it may cause re-compilation of lots of files...
* [setup ccache](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#use-ccache)