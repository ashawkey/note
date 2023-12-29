## Monkey patch a method


To patch the `forward` of a `nn.Module`, **define a closure** that keeps temporary variables and returns your new `forward`:

```python
import torch.nn as nn

class A(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def forward(self):
        print(f'original forward of {self.name}')

a = A('a')
b = A('b')

for name, m in zip(['a', 'b'], [a, b]):

    def make_forward():
        # record the current name in closure !
        cur_name = name
        def _forward():
            print(f'patched forward of {cur_name}')
        return _forward

    m.forward = make_forward()

a()
b()
```

Output:

```
patched forward of a
patched forward of b
```


However, you cannot patch magic methods like `__call__` by this:

```python
class A:
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def __call__(self):
        print(f'original forward of {self.name}')

a = A('a')
b = A('b')

for name, m in zip(['a', 'b'], [a, b]):

    def make_call():
        # record the current name in closure !
        cur_name = name
        def _call():
            print(f'patched forward of {cur_name}')
        return _call

    m.__call__ = make_call()

a()
b()
```

Output:

```
original forward of a
original forward of b
```

This is because `__call__` is looked-up with respect to the class instead of instance, so we are still calling the original `__call__`.

We have to patch the class to make this work, and **cast instances to the derived class**:

```python
class A:
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def __call__(self):
        print(f'original forward of {self.name}')

a = A('a')
b = A('b')

# a derived class that redirect __call__ to our patched call
class B(A):
    def __call__(self):
        self.patched_call()

for name, m in zip(['a', 'b'], [a, b]):

    def make_call():
        # record the current name in closure !
        cur_name = name
        def _call():
            print(f'patched forward of {cur_name}')
            
        return _call

    m.__class__ = B # magic cast!
    m.patched_call = make_call()

a()
b()
```

Output:

```
patched forward of a
patched forward of b
```

