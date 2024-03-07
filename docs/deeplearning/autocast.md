### autocast

Native functions in an `autocast` context will incur the `fp16` version.

But if you want a function to be always high precision `fp32`, you should wrap it with:

```python
@torch.cuda.amp.autocast(enabled=False)
def high_prec_func(x):
    ...
    return x

# call 
with torch.cuda.amp.autocast(enabled=True):
    x = high_prec_func(x)
    x = low_prec_func(x)
```

