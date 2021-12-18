# MeshLab Tutorial

### Input format

* XYZ (.txt)

  ```
  x; y; z; r; g; b; ....
  x; y; z; r; g; b; ....
  ...
  ```

  (separator and format can be changed)

### Coloring

can only use the RGB to color.



### Codelet

```python
palette = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [192,192,192], [128, 0, 0], [128, 128, 0], [0, 128, 0], [128, 0, 128], [0, 128, 128], [0, 0, 128]]

def color_gradient(x, a=[255,0,0], b=[0,255,0]):
    # x in [-1, 1]
    return (np.array(a) + ((x+1)/2) * np.array(b)).astype(int)

def color_palette(x, colors=palette):
    return colors[x]


def save_xyz(path, arr, color='rgb'):
    with open(path, 'w') as f:
        # arr=[N, C], col=[x, y, z, r, g, b, tsdf, height, label]
        if color == 'rgb':
            for l in arr:
                f.write(';'.join([str(x) for x in l])+'\n')
        elif color == 'tsdf':
            for l in arr:
                f.write(';'.join([str(x) for x in l[:3]])+';'+';'.join([str(x) for x in color_gradient(l[6])])+'\n')
        elif color == 'label':
            for l in arr:
                f.write(';'.join([str(x) for x in l[:3]])+';'+';'.join([str(x) for x in color_palette(int(l[-1]))])+'\n')

```

