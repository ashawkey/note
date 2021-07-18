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

s_even = np.s_[::2]
s_odd = np.s_[1::2]

a = np.arange(10)

print(a[s_even])
print(a[s_odd])

'''
[0 2 4 6 8]
[1 3 5 7 9]
'''
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

