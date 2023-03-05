# blender script

### locate objects

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

