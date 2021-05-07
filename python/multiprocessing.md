# multiprocessing



### basics

simple multi-threading:

```python
from multiprocessing import Pool

def f(x):
	pass

# call1
p = Pool(8)
p.map(f, [1, 2, 3])
p.close()
p.join()

# call2
with Pool(8) as p:
    p.map(f, [1, 2, 3])
```

with return values:

```python
def f(x): return x
res = p.map(f, [1, 2, 3])
# res: [1, 2, 3]
```



