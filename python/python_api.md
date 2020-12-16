# python migration

### list

```python
# remove by index
del l[idx]
v = l.pop(idx)

# remove by value
l.remove(val) # ValueError if not found
l = [x in l if x != val]
```

* remove in for loop:

  Not well supported. Use list comprehension or filter.

  

### set

```python
s = set()
s.add(x)
s.remove(x) # ValueError if not found
s.discard(x) # pass if not found.
```



