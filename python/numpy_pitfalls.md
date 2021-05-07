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





