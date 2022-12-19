# numpy tricks

### automatic broadcast from front

```python
import numpy as np

a = np.array([1,2,3]) # [3,]
b = np.zeros((3,3)) # [3, 3]

# a [3,] --> [1, 3] (default to expand from front.)
print(a + b) # [3, 3]

'''
[[1. 2. 3.]
 [1. 2. 3.]
 [1. 2. 3.]]
'''

a = np.array([1,2,3]) # [3,]
b = np.zeros((3,5)) # [3, 5]

# auto expand failed
print(a + b) 

'''
ValueError: operands could not be broadcast together with shapes (3,) (3,5)
'''

# manual expand from back
print(a[:, None] + b)
'''
[[1. 1. 1. 1. 1.]
 [2. 2. 2. 2. 2.]
 [3. 3. 3. 3. 3.]]
'''
```



### View

Without touching underlying data, a `view` is the datatype and shape of the data.

```python
a = np.arange(6)
print(a.shape) # (6,)

b = a.view()
b = b.reshape(2, 3)
print(a.shape, b.shape) # (6,), (2, 3)

a[0] = -1
print(b[0, 0]) # also -1, since a and b shares data.

# check if two arrays are just views of the same data (maybe slow)
np.shares_memory(x[::2], x) # True
```

Pros:

* [time & space] do not need to copy at every time you reshape, slice, index, ...

Cons:

* [space] even if only a small slice is used, the pointed data maybe large and cannot be released. Use `.copy()` if you are sure the old data will not be used and should be garbage-collected!



### indexing & Slicing (IMPORTANT!)

We discuss the behavior of  `x[obj]`, which can be classified as:

#### Basic Indexing

Features:

* `obj` is:

  * `slice`

  * `integer`

  * a  `tuple` that contains `slice` or `integer`. 

    In the form of `x[(obj1, obj2, ...)]`, which equals `x[obj1, obj2, ...]`. 

    So `x[(1,2,3)]` equals `x[1,2,3]` and is a basic indexing.

  `...` or `None` can be interspersed.

* Never copy. Instead, always return a `view` of the original array.

Some tips:

* `slice` is in the form of `[i:j:k]`, and can be created by `np.s_[i:j:k]`.
* `...` (Ellipsis) is usually used in the form of `x[..., slice], x[..., slice, :]`.
* `None` or `np.newaxis` is used to expand the dimension.
* `x[i]` reduces the dim, but `x[[i]], x[i:i+1]` keeps the dim.
* `x[s1, s2]` equals `x[s1][s2]`.
* `x[s1] = v` is in-place modification (and support broadcast).
* `np.intp == np.int64` is the smallest data type sufficient to safely index any array; for advanced indexing it may be faster than other types.

#### Advanced Indexing

Features:

* `obj` is:
  * non-`tuple` sequence object (such as `list`)
  * `ndarray` of `integer` or `bool`
  * a `tuple` that contains the first two types of objects. 
* Always copy.

Tips:

* Different from `x[(1,2,3)] == x[1,2,3]`, `x[(1,2,3),]` equals `x[(1,2,3), :]` and is an advanced indexing.

  `x[1:4]` takes the same slice but is a basic indexing.

  `x[[1,2,3]]` is also an advanced indexing that equals `x[(1,2,3),]`

* Integer array indexing performs **index selection** (the output's shape is the same as each index array's shape):

  ```python
  result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
                             ..., ind_N[i_1, ..., i_M]]
  
  # example
  x = np.array([[1, 2], [3, 4], [5, 6]])
  x[[0, 1, 2], [0, 1, 0]] # array([1, 4, 5])
  ```

  **submatrix selection** can be achieved with broadcasting.

  ```python
  x = np.array([[ 0,  1,  2],
                [ 3,  4,  5],
                [ 6,  7,  8],
                [ 9, 10, 11]])
  rows = np.array([0, 3], dtype=np.intp) # intp == int64, specially used for indexing.
  cols = np.array([0, 2], dtype=np.intp)
  
  # element selection
  x[rows, cols] # array([0, 11])
  
  # submatrix selection
  x[rows[:, None], cols[None, :]] # array([[0, 2], [9, 11]])
  x[rows[:, None], cols] # array([[0, 2], [9, 11]]), just a broadcast shortcut of the above line 
  
  x[rows[None, :], cols[:, None]] # array([[0, 9], [2, 11]]), transposed version.
  x[rows, cols[:, None]] # array([[0, 9], [2, 11]]), shortcut
  
  # alias with np.ix_ 
  x[np.ix_(rows, cols)] # array([[0, 2], [9, 11]])
  np.ix_(rows, cols) # (array([[0], [3]], dtype=int64), array([[0, 2]], dtype=int64))
  ```

* bool array indexing performs **masked selection**. The index array should be able to broadcast to the source array.

  ```python
  x = np.random.randn(3, 3)
  mask = x > 0
  x[mask] # a usual case.
  x[mask.nonzero()] # the equal integer array indexing.
  ```

  **submatrix selection** can be achieved in a similar way.

  ```python
  x = np.array([[ 0,  1,  2],
                [ 3,  4,  5],
                [ 6,  7,  8],
                [ 9, 10, 11]])
  
  rows = np.array([True, False, False, True])
  cols = np.array([True, False, True])
  
  # just the same as the above example!
  
  # element selection
  x[rows, cols]
  
  # submatrix selection
  x[np.ix_(rows, cols)]
  x[rows.nonzero()[0][:, None], cols] # a little ugly
  
  ```

* For advanced assignments, there is in general **no guarantee for the iteration order**. This means that if an element is set more than once, it is not possible to predict the final result.

  ```python
  a = np.zeros(6)
  b = np.array([3, 2, 5, 2]) # 1d indices
  c = np.array([1, 1, 1, 1]) # values to add at these indices
  
  # extract by advanced indexing is always OK.
  print(a[b])
  
  # assign by advanced indexing is wrong is the indices duplicate.
  a[b] += c # array([0, 0, 1, 1, 0, 1]), DO NOT USE! a[2] is only added once.
  
  # the correct way to do this is to use ufunc.
  np.add.at(a, b, c) # array([0, 0, 2, 1, 0, 1]), as expected.
  ```

  



### assign value by a list of coordinates

```python
import numpy as np

img = np.zeros((5, 5))

# goal: assign vals to coords
coords = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
vals = np.array([1, 2, 3, 4])

# wrong:
img[coords]
'''
This takes a [4, 2, 5] matrix.
i.e., [(row0, row1), (row1, row2), ...]
'''

# correct:
img[tuple(coords.T)] = vals
'''
tuple(coords.T) is: (array([0, 1, 2, 3]), array([1, 2, 3, 4]))
'''
```



### slice

```python
import numpy as np

# use np.s_ to create reusable slices.
s_even = np.s_[::2]
s_odd = np.s_[1::2]

a = np.arange(10)
print(a[s_even]) # equals a[::2]
print(a[s_odd]) # equals a[1::2]
```



### meshgrid

```python
# default behaviour
np.meshgrid(np.arange(3), np.arange(4)) # or indexing='xy'
'''
[array([[0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]]),
 array([[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])]
'''

# specify indexing = `ij`
np.meshgrid(np.arange(3), np.arange(4), indexing='ij')
'''
[array([[0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2]]),
 array([[0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3]])]
'''

# note: the default behaviour of numpy (xy) is different from torch (ij) !!!
torch.meshgrid(torch.arange(3), torch.arange(4))
'''
(tensor([[0, 0, 0, 0],
         [1, 1, 1, 1],
         [2, 2, 2, 2]]),
 tensor([[0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3]]))
'''
```





### numpy add by index, with duplicated indices

In numpy >= 1.8, you can also use the `at` method of the addition 'universal function' ('ufunc'). As the [docs note](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html):

> For addition ufunc, this method is equivalent to a[indices] += b, **except that results are accumulated for elements that are indexed more than once.**

Example:

```python
a = np.zeros(6)
b = np.array([3, 2, 5, 2]) # 1d indices
c = np.array([1, 1, 1, 1]) # values to add at these indices

# extract
print(a[b])

# manipulate
a[b] += c          # array([0, 0, 1, 1, 0, 1]), DO NOT USE! a[2] is only added once.
np.add.at(a, b, c) # array([0, 0, 2, 1, 0, 1]), as expected.
```

Extending: the `np.ufunc.at` function family.

> Performs **unbuffered in place** operation on operand ‘a’ for elements specified by ‘indices’. 

```python
np.negative.at(np.array([1,2,3]), [0,2]) # [-1,2,-3]
```



Torch's equivalent:

```python
a = torch.zeros(6)
b = torch.LongTensor([3, 2, 5, 2]) # 1d indices
c = torch.FloatTensor([1, 1, 1, 1]) # values to add at these indices

# extract
print(a[b])

# manipulate
a[b] += c                              # ([0, 0, 1, 1, 0, 1]), DO NOT USE! a[2] is only added once.
a.index_put_((b,), c)                  # ([0, 0, 1, 1, 0, 1]), ditto, this is really 'put' (the original values will be overwriten!)
a.index_put_((b,), c, accumulate=True) # ([0, 0, 2, 1, 0, 1]), correctly accumulate on the original values.
```

Note: the name `index_put_` is tricky, `index_add_` is another different thing in torch, which is less flexible to achieve what we are doing.



### numpy add by 2D (or nD) index 

The desired operation:

```python
a = np.zeros((3, 3))
b = np.array([[0,0], [1,1], [2,2], [2,2]]) # [M, 2], 2d indices
c = np.array([1, 1, 1, 1]) # values to add

# extract
a[tuple(b.T)]

# manipulate
a[tuple(b.T)] += c # same, duplicated indices are added once. DO NOT USE!
np.add.at(a, tuple(b.T), c) # as expected.
```

Torch's equivalent:

```python
a = torch.zeros((3, 3))
b = torch.LongTensor([[0,0], [1,1], [2,2], [2,2]]) # [M, 2], 2d indices
c = torch.FloatTensor([1, 1, 1, 1]) # values to add

# extract
a[tuple(b.T)]

# manipulate
a[tuple(b.T)] += c # same, duplicated indices are added once. DO NOT USE!
a.index_put_(tuple(b.T), c, accumulate=True) # as expected.
```





### take v.s. take_along_axis

`np.take` is in fact simple indexing.

```python
### take: some common indices from array.
a = np.arange(15).reshape(3, 5) # data
'''
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])

'''

b = np.array([0, 2]) # indices to take

# take the [0, 2]-th row
np.take(a, b, axis=0)
a[b] # the same
'''
array([[ 0,  1,  2,  3,  4],
       [10, 11, 12, 13, 14]])
'''

# take the [0, 2]-th col
np.take(a, b, axis=1)
a[:, b]

'''
array([[ 0,  2],
       [ 5,  7],
       [10, 12]])
'''
```

However, sometimes we want to take one exact indexed element from each row (like the `torch.gather` operation). `np.take_along_axis` does this:

```python
a = np.arange(15).reshape(3, 5) # data
'''
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])

'''

b = np.array([0, 2, 4]) # aim: get [0, 7, 14]
np.take_along_axis(a, b[:, None], axis=1)
'''
array([[ 0],
       [ 7],
       [14]])
'''

c = np.array([0, 0, 0, 1, 1]) # aim: get [0, 2, 3, 8, 14]
np.take_along_axis(a, c[None, :], axis=0)
'''
array([[0, 1, 2, 8, 9]])
'''
```

