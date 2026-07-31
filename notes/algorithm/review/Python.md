# Python 迁移笔记

## ACM 风格 I/O

Input:

```python
# read a line to string
l = input()

# to read space (include multiple spaces and tabs) separated int
ls = [int(x) for x in input().split()]
```

Output:

```python
x = 0.123
print(f'{x:.2f}')
print('{:.2f}'.format(x))

i = 6
print(f'{x:05d}')

s = 'hello'
print(f'{s}')

# ints separated by one space
ls = [1, 2, 3]
print(' '.join([str(x) for x in ls])) # "1 2 3"

# print end
print(x, end='') # do not append \n
```

## list

```python
l = []
l.append(x)
l += [x]
l.extend(l2)

# slicing
l[1:]
l[:-1]

# reverse
l[::-1]

# sort
sorted(l) # small to large
sorted(l, reverse=True) # large to small
l.sort()

# remove by index
del l[idx]
v = l.pop(idx)

# remove by value
l.remove(val) # ValueError if not found
l = [x for x in l if x != val]
```

* 在 for 循环中删除元素：支持不好，用列表推导式或 filter。

## set

```python
s = set()

s.add(x)

if x in s:
    print('YES')

s.remove(x) # ValueError if not found
s.discard(x) # pass if not found.
```

## dict

```python
d = {}

d['x'] = 1

del d['x']

for k, v in d.items():
    pass

for k in d:
    print(k, d[k])

if k in d:
    print(d[k])
```

## tricks

* 自动记忆化递归：

  ```python
  from functools import lru_cache

  @lru_cache(maxsize=None)
  def fib(n):
      if n < 2: return n
      return fib(n - 1) + fib(n - 2)
  ```
