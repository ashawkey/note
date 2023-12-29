# inspect

### `vars()`

`vars(x)` equals `x.__dict__`, which returns a dict of the **current class**'s attributes & methods of  `x`. (both name and value)

```python
class Foo:
    def __init__(self):
        self.x = 1
    def call(self):
        print(self.x)

foo = Foo()
        
vars(Foo)
'''
mappingproxy({'__module__': '__main__',
              '__init__': <function __main__.Foo.__init__(self)>,
              'call': <function __main__.Foo.call(self)>,
              '__dict__': <attribute '__dict__' of 'Foo' objects>,
              '__weakref__': <attribute '__weakref__' of 'Foo' objects>,
              '__doc__': None})
'''

vars(foo)
'''
{'x': 1}
'''
```


### `dir()`

`dir(x)` returns the names of attributes & methods of all levels (both current class and base class).

```python
dir(Foo)
'''
['__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 'call']
'''

dir(foo)
'''
['__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 'call',
 'x']
'''
```


### inspect

```python
import inspect

for name, value in inspect.getmembers(foo):
    print(name, value)
    
# eqs
for name in dir(foo):
    print(name, getattr(foo, name))
```

