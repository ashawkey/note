# os

### os.path

```python
# safe join of dirs & files
os.path.join(a, b)

# extract basename, dirname
os.path.basename(path) # a/b/c --> c
os.path.dirname(path) # a/b/c --> a/b
```


### os.walk

```python
# regular usage: find all files under path recursively
images = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endwith('.png'):
		    images.append(f)

    
# exclude directory while walking 
for root, dirs, files in os.walk(path):
    [dirs.remove(d) for d in list(dirs) if d in exclude_lists]
    ...
```

