# blender script

logs are printed to the terminal console, use `window --> toggle system console` to open it.

### auto import many objects

assume you have a folder with lots of obj models (each in a sub-folder):

```python
import bpy
import os

context = bpy.context
dir = r"C:\Users\hawke\Downloads\bear"

obj_dirs = os.listdir(dir)

for i, name in enumerate(obj_dirs):
    obj_dir = os.path.join(dir, name)
    files = os.listdir(obj_dir)
    for file in files:
        if not file.endswith('.obj'): continue
        path = os.path.join(obj_dir, file)
        
        print(f'[INFO] {i} process {file}')
        
        bpy.ops.import_scene.obj(filepath=path, filter_glob="*.obj;*.mtl;*.png") # also load png textures
        obj = context.selected_objects[0]

        # location (10 in a row)
        h, w = i % 10 - 5, i // 10 - 5
        obj.location = (h, w, 0)
```

### locate objects in current scene

```python
import bpy

# all objects 
bpy.data.objects

# access by index
bpy.data.objects[0]

# access by name
bpy.data.objects['Light']

# modify property
bpy.data.objects['Light'].location = (0, 0, 0)
```

