# hash in python

Different classes implement different `__hash__()` method.

```python
# hash(immutable object) -> int

# primitives
hash(10) # 10
hash(10.0) # 10
hash(10.1) # 230584300921368586
hash('str') # 3512716393951003388

# tuples
hash((1)) # 1
hash((1,2)) # -3550055125485641917

# collision example
hash(10.1) == hash(230584300921368586) # True
```



### The exact algorithms

in `cpython` source, e.g.,

```c
static long string_hash(PyStringObject *a)
{
    register Py_ssize_t len;
    register unsigned char *p;
    register long x;

    if (a->ob_shash != -1)
        return a->ob_shash;
    len = Py_SIZE(a);
    p = (unsigned char *) a->ob_sval;
    x = *p << 7;
    while (--len >= 0)
        x = (1000003*x) ^ *p++;
    x ^= Py_SIZE(a);
    if (x == -1)
        x = -2;
    a->ob_shash = x;
    return x;
}
```

