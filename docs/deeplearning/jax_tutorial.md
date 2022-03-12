## [JAX tutorial](https://jax.readthedocs.io/en/latest/index.html)



### why

`jax` is more suitable for:

* functional programming lover.

* higher order optimization.
* data parallel, especially on TPU.



### install

```bash
# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
```



### basics

```python
import numpy as np

import jax
import jax.numpy as jnp

from jax import grad, jit
from jax import lax
from jax import random
```



### jax.numpy

`jax.numpy` mimics `numpy` API:

```python
# jax array: DeviceArray
arr = jnp.array([1, 2, 3], dtype=jnp.float32)
arr = jnp.asarray([1, 2, 3], dtype=jnp.float32)

arr.shape # python tuple
arr.size # prod of shape
arr.dtype # numpy dtype
arr.device() # it's a function! GpuDevice(id=0, process_index=0)

```

* DeviceArray are **immutable** (cannot be in-place modified)

  ```python
  arr[0] = 0 # TypeError
  
  # you must copy it and set new values by:
  # However, jit can be used avoid the copy.
  arr.at[0].set(0)
  ```



### jax.grad

`jax.grad` for gradient calculation:

```python
def mse_loss(x, y):
    return jnp.mean((x - y) ** 2)

x = jnp.asarray([1, 2, 3], dtype=jnp.float32)
y = jnp.asarray([0, 0, 0], dtype=jnp.float32)

# calculate loss
l = mse_loss(x, y)

# calculate gradient: dl/dx
mse_loss_dx = jax.grad(mse_loss) # default to use first argument
x_grad = mse_loss_dx(x, y)

# both (dl/dx, dl/dy)
mse_loss_dxdy = jax.grad(mse_loss, argnums=(0, 1))
x_grad, y_grad = mse_loss_dxdy(x, y)

# a convenient shortcut:
# jax.value_and_grad(f)(*xs) == (f(*xs), jax.grad(f)(*xs))
l, x_grad = jax.value_and_grad(mse_loss)(x, y)
```

* `jax.grad` only works on **scalar-output** functions!

  ```python
  def loss_aux(x, y):
      # return (loss, aux), aux may be any value used to debug.
      return mse_loss(x, y), x - y
  
  # then you should use `has_aux=True`:
  loss_dx = jax.grad(loss_aux, has_aux=True)
  x_grad = loss_dx(x, y)
  ```



### jax.jit

`jax.jit` for just-in-time compilation of functions.

It trace the function at the first time call, and generate `jaxpr` intermediate results, for faster later calls.

Therefore, it should only be used on **Pure functions**, i.e., functions with no side-effect, including:

* `print()`: it will only be valid at the first time call (tracing), and later calls will ignore the print. Therefore, it still can be used to debug.
* reading/writing global variables. NEVER do these!

```python
def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

selu_jit = jax.jit(selu)

x = jnp.arange(1000000)
selu(x) # slower.
selu_jit(x) # first time call, will trace the function. still, it is faster.
selu_jit(x) # should be much faster.
```

by default, jit tracers record at the level of **input shape**, i.e., all inputs with the same type should behave uniformly. Therefore, control flows that depend on **input value** will fail if transformed with jit:

```python
# cond
def f(x):
    return x if x > 0 else 0

f_jit = jax.jit(f)
f_jit(10) # ConcretizationTypeError!

# note: jax.grad can be used with python control flows
f_dx = jax.grad(f)
f_dx(10) # OK
```

There are two solutions:

* use control flow operators in `jax.lax`:

  ```python
  # for cond, we still need to know if x > 0 first... so this is not a solution.
  jax.lax.cond(True, lambda x: x, lambda x: 0, x)
  jax.lax.cond(False, lambda x: x, lambda x: 0, x)
  ```

* specify static arguments, and avoid jit on these args.

  ```python
  x = jnp.array([1., 2., 3.])
  
  def g(x, n):
      y = 0
  	for i in range(n):
          y = y + x[i]
      return y
  
  # this fails, because the flow depends on dynamic value n.
  jax.jit(g)(x, 3)
  
  # this works, but everytime n change, it will be re-compiled.
  jax.jit(g, static_argnums=(1,))(x, 3)
  
  # similarly for f (but this is meaningless)
  jax.jit(f, static_argnums=(0,))(10)
  
  def g2(x):
    y = 0.
    for i in range(x.shape[0]):
      y = y + x[i]
    return y
  
  # this works, because jit trace the array shape, so x.shape[0] is static.
  jax.jit(g2)(x)
  ```

Tips to use `jit`:

* Use on functions that will be called numerous times.
* if use `static_argnums`, the static argument should not change too frequently, otherwise the compilation overhead will be terrible.
* Do not jit function in loops, always jit the function before using it in a loop.



### vmap

`jax.vmap` for automatical vectorization (batch).

```python
# any function that operates on unbatched inputs
def func(x, y):
    return x @ y

# unbatched inputs
x = jnp.array([1., 2., 3.])
y = jnp.array([0., 0., 0.])

z = func(x, y)

# batched inputs
xs = jnp.stack([x, x], axis=0)
ys = jnp.stack([y, y], axis=0)

# automatically batched function
func_batch = jax.vmap(func)
zs = func_batch(xs, ys)

# specify the batch axis
func_batch_transposed = jax.vmap(func, in_axes=1, out_axes=1)
zst = func_batch_transposed(jnp.transpose(xs), jnp.transpose(ys))

# you can even specify different batch axes for different inputs/outputs
func_batch_mixed = jax.vmap(func, in_axes=(0, 1))
zsm = func_batch_mixed(xs, jnp.transpose(ys))

# or broadcast:
func_batch_broadcast = jax.vmap(func, in_axes=(0, None))
zsb = func_batch_broadcast(xs, y)
```

`jit`, `vmap`, `grad` can be combined in arbitrary orders and all works well.



### RNG

random number generator in jax is very different from numpy, to be reproducible, parallelizable, vectorizable.

```python
from jax import random

key = random.PRNGKey(42) # just a two-element array, [0, 42]

random.normal(key) # -0.1847
random.normal(key) # -0.1847, same since key is the same.

# split key for new value:
key, subkey = random.split(key)
random.normal(subkey) # 0.7692, different now.

# to split multiple subkeys (useful for parallel)
key, subkeys = random.split(key, num=10)
```



### Pytree

Any nested data structures.

> a pytree is a container of leaf elements and/or more pytrees. Containers include lists, tuples, and dicts. A leaf element is anything that’s not a pytree, e.g. an array.
>
> If nested, note that the container types do not need to match. 
>
> A single “leaf”, i.e. a non-container object, is also considered a pytree.

```python
example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

# Let's see how many leaves they have:
for pytree in example_trees:
  leaves = jax.tree_leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

'''
[1, 'a', <object object at 0x7fded60bb8c0>]   has 3 leaves: [1, 'a', <object object at 0x7fded60bb8c0>]
(1, (2, 3), ())                               has 3 leaves: [1, 2, 3]
[1, {'k1': 2, 'k2': (3, 4)}, 5]               has 5 leaves: [1, 2, 3, 4, 5]
{'a': 2, 'b': (2, 3)}                         has 3 leaves: [2, 2, 3]
DeviceArray([1, 2, 3], dtype=int32)           has 1 leaves: [DeviceArray([1, 2, 3], dtype=int32)]
'''    
```

e.g., Model parameters can be viewed as a pytree.

```python
tree = [
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]

# apply a function on pytree's leaves
jax.tree_map(lambda x: x*2, tree)

# apply a function on two (or more) pytrees' leaves
# note: the pytrees' structure must be the same!
jax.tree_multimap(lambda x, y: x + y, tree, tree)
```

To use custom data container (e.g., custom class) as leaves, you should register it with flatten & unflatten methods. Usually, you don't want to do this... Internal classes (e.g., `NamedTuple`) can handle most cases.

Tips:

* `None` is treated as an empty node, insteaad of a leaf:

  ```python
  jax.tree_leaves([None, None, None]) # []
  ```

* since array  `shape` are tuples, it will be treated as a node instead of a leaf.



### pmap

`jax.pmap` performs parallel evaluation. Different from `jax.vmap` which vectorizes on the same device, `jax.pmap` aims to map the function to **different devices** and run in parallel (data parallelism).

```python
# check how many devices we have
jax.devices()
n_devices = jax.local_device_count() 

x = jnp.array([1., 2., 3.])
y = jnp.array([0., 0., 0.])

# batched inputs
xs = jnp.stack([x, x], axis=0)
ys = jnp.stack([y, y], axis=0)

# call pmap
zs = jax.pmap(func)(xs, ys) # ShardedDeviceArray (on different devices)
```

communication between devices by collective ops (`jax.lax.p*`):

```python
def normalized_convolution(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  output = jnp.array(output)
  # axis_name is just an identifier to match between pmap and psum
  return output / jax.lax.psum(output, axis_name='p')

jax.pmap(normalized_convolution, axis_name='p')(xs, ws)
```

nesting `vmap` and `pmap`:

```python
# any order is OK.
jax.vmap(jax.pmap(f, axis_name='i'), axis_name='j')
```

An example:

```python
from typing import NamedTuple, Tuple
import functools

class Params(NamedTuple):
  weight: jnp.ndarray
  bias: jnp.ndarray


def init(rng) -> Params:
  """Returns the initial model params."""
  weights_key, bias_key = jax.random.split(rng)
  weight = jax.random.normal(weights_key, ())
  bias = jax.random.normal(bias_key, ())
  return Params(weight, bias)


def loss_fn(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
  """Computes the least squares error of the model's predictions on x against y."""
  pred = params.weight * xs + params.bias
  return jnp.mean((pred - ys) ** 2)

LEARNING_RATE = 0.005

# So far, the code is identical to the single-device case. Here's what's new:

# Remember that the `axis_name` is just an arbitrary string label used
# to later tell `jax.lax.pmean` which axis to reduce over. Here, we call it
# 'num_devices', but could have used anything, so long as `pmean` used the same.
@functools.partial(jax.pmap, axis_name='num_devices')
def update(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> Tuple[Params, jnp.ndarray]:
  """Performs one SGD update step on params using the given data."""

  # Compute the gradients on the given minibatch (individually on each device).
  loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

  # Combine the gradient across all devices (by taking their mean).
  grads = jax.lax.pmean(grads, axis_name='num_devices')

  # Also combine the loss. Unnecessary for the update, but useful for logging.
  loss = jax.lax.pmean(loss, axis_name='num_devices')

  # Each device performs its own update, but since we start with the same params
  # and synchronise gradients, the params stay in sync.
  new_params = jax.tree_multimap(
      lambda param, g: param - g * LEARNING_RATE, params, grads)

  return new_params, loss

# Generate true data from y = w*x + b + noise
true_w, true_b = 2, -1
xs = np.random.normal(size=(128, 1))
noise = 0.5 * np.random.normal(size=(128, 1))
ys = xs * true_w + true_b + noise

# Initialise parameters and replicate across devices.
params = init(jax.random.PRNGKey(123))
n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)

def split(arr):
  """Splits the first axis of `arr` evenly across the number of devices."""
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

# Reshape xs and ys for the pmapped `update()`.
x_split = split(xs)
y_split = split(ys)

def type_after_update(name, obj):
  print(f"after first `update()`, `{name}` is a", type(obj))

# Actual training loop.
for i in range(1000):

  # This is where the params and data gets communicated to devices:
  replicated_params, loss = update(replicated_params, x_split, y_split)

  if i % 100 == 0:
    # Note that loss is actually an array of shape [num_devices], with identical
    # entries, because each device returns its copy of the loss.
    # So, we take the first element to print it.
    print(f"Step {i:3d}, loss: {loss[0]:.3f}")


# Like the loss, the leaves of params have an extra leading dimension,
# so we take the params from the first device.
params = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))
```



Tips:

* `pmap` will call `jit` internally.





### misc

* synchronization for timing:

  ```python
  jnp.dot(a, b).block_until_ready()
  ```

* debugging `NaN`s:

  * run with `JAX_DEBUG_NANS=True`, or

  * add at the top of main:

    ```python
    from jax.config import config
    config.update("jax_debug_nans", True)
    ```

* use fp64:

  * run with `JAX_ENABLE_X64=True`, or

  * add at the top of main:

    ```python
    from jax.config import config
    config.update("jax_enable_x64", True)
    ```

    

  

