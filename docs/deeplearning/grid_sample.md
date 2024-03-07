## Coordinate system of `F.grid_sample`

### 2D case

First we use `meshgrid` to build grid coordinates:

```python
import torch
import torch.nn.functional as F

res = 2
dim = 2
device = torch.device("cpu")

line = torch.linspace(-1, 1, res, device=device)
""" [-1, 1] """

points_ij = torch.stack(torch.meshgrid([line]*dim, indexing="ij"), dim=-1)
"""
[-1, -1] [-1, 1]
[1,  -1] [1,  1]
"""

points_xy = torch.stack(torch.meshgrid([line]*dim, indexing="xy"), dim=-1)
"""
[-1, -1] [1, -1]
[-1,  1] [1,  1]
"""

# xy is transpose of ij
assert torch.allclose(points_ij[..., [1, 0]], points_xy)
```

Why `ij` and `xy` are **transposed**:

```bash
# coordinate system of a 2D image (i/x row, j/y col!)
O -------- x/j
|      
|  
|
y/i

# i.e.
-->
xy
ji
<--
```

Now we can try `grid_sample`:

```python
# simplest image
grid = torch.arange(res**dim).view([res]*dim).float()
"""
0 1
2 3
"""

# use ij
val_ij = F.grid_sample(grid.unsqueeze(0).unsqueeze(0), points_ij.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()
"""
0=[-1, -1] 2=[-1, 1]
1=[1,  -1] 3=[1,  1]
"""

# use xy (or ji)
val_xy = F.grid_sample(grid.unsqueeze(0).unsqueeze(0), points_xy.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()
"""
0=[-1, -1] 1=[1, -1]
2=[-1,  1] 3=[1,  1]
"""
```

We can see the internal coordinate system of `grid_sample` is:

```python
                 -       
                 |
      0=[-1,-1]--|---------1=[1,-1]
      |          |         |
      |          |         |
      |          |         |
 - ------------- O ------------ +x/j
      |          |         |
      |          |         |
      |          |         |
      2=[-1,1]---|---------3=[1,1]
                 |
                 +y/i    
```

Therefore, just use `ji` (i.e., `xy`) to make sure the output is aligned with input:

```python
assert torch.allclose(grid, val_xy)
```


### 3D case

```python
import torch
import torch.nn.functional as F

res = 2
dim = 3
device = torch.device("cpu")

line = torch.linspace(-1, 1, res, device=device) # [N]

points_ijk = torch.stack(torch.meshgrid([line]*dim, indexing="ij"), dim=-1) # [N, N, N, 3]
"""
        [1,-1,-1]---------[1,-1,1]
       /|                /|
     /  |              /  |
    [-1,-1,-1]--------[-1,-1,1]
    |   |             |   |  
    |   |             |   |
    |   [1,1,-1]------|---[1,1,1]
    |  /              |  /
    |/                |/
    [-1,1,-1]---------[-1,1,1]
"""

# NOTE: here the order is not xyz, but yzx !!!
points_yzx = torch.stack(torch.meshgrid([line]*dim, indexing="xy"), dim=-1) # [N, N, N, 3]
"""
        [-1,1,-1]---------[-1,1,1]
       /|                /|
     /  |              /  |
    [-1,-1,-1]--------[-1,-1,1]
    |   |             |   |  
    |   |             |   |
    |   [1,1,-1]------|---[1,1,1]
    |  /              |  /
    |/                |/
    [1,-1,-1]---------[1,-1,1]
"""
```

Relationship with `xyz` and `ijk`:

```bash
# coordinate system of a 3D volume
       z/i
     /
   /
 /  
O---------x/k
|   
|   
|
|
y/j

# this is because xy[z] and ij[k] notation are always reversed!
2D       3D
-->      --->
xy       xyz
ji       kji
<--      <---
```

Try `grid_sample`:

```python
# simplest volume
grid = torch.arange(res**dim).view([res]*dim).float()
"""
        4-----------5
       /|          /|
     /  |        /  |
    0-----------1   |
    |   |       |   |  
    |   6-------|---7
    |  /        |  /
    |/          |/
    2-----------3
"""

# use ijk
F.grid_sample(grid.unsqueeze(0).unsqueeze(0), points_ijk.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()
"""
        1=[1,-1,-1]-------5=[1,-1,1]
       /|                /|
     /  |              /  |
    0=[-1,-1,-1]------4=[-1,-1,1]
    |   |             |   |  
    |   |             |   |
    |   3=[1,1,-1]----|---7=[1,1,1]
    |  /              |  /
    |/                |/
    2=[-1,1,-1]-------6=[-1,1,1]
"""

# use yzx
F.grid_sample(grid.unsqueeze(0).unsqueeze(0), points_yzx.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()
"""
        2=[-1,1,-1]-------6=[-1,1,1]
       /|                /|
     /  |              /  |
    0=[-1,-1,-1]------4=[-1,-1,1]
    |   |             |   |  
    |   |             |   |
    |   3=[1,1,-1]----|---7=[1,1,1]
    |  /              |  /
    |/                |/
    1=[1,-1,-1]-------5=[1,-1,1]
"""
```

Internal coordinate system of `grid_sample` is:

```python
                       -           +z/i
                       |          /
                       |        /
             4=[-1,-1,1]----------5=[1,-1,1]
            /|         |     /   /|
          /  |         |   /   /  |
         0=[-1,-1,-1]---------1=[1,-1,-1]
         |   |         |/     |   |  
- -------|---|-------- O -----|---|------------ +x/k
         |   |        /|      |   |
         |   6=[-1,1,1]-------|---7=[1,1,1]
         |  /      /   |      |  /
         |/      /     |      |/
         2=[-1,1,-1]----------3=[1,1,-1]
              /        |
            /          |
          /            |
        -              +y/j
```

Notice that both `ijk` and `yzx` **cannot align the output with input**, and instead we need `kji` (or `xyz`):

```python
points_ijk = torch.stack(torch.meshgrid([line]*dim, indexing="ij"), dim=-1) # [N, N, N, 3]
points_kji = points_ijk[..., [2,1,0]]

val = F.grid_sample(grid.unsqueeze(0).unsqueeze(0), points_kji.unsqueeze(0), mode='bilinear', align_corners=True).squeeze()
assert torch.allclose(grid, val)
```

