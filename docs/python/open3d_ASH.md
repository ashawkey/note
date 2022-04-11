## ASH

### Tensor API

```python
### pytorch communication (share the same data blob)

import torch
import torch.utils.dlpack
import open3d.core as o3c

# torch --> open3d
th_a = torch.ones((5,)).cuda(0)
o3_a = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))

# open3d --> torch
o3_a = o3c.Tensor([1, 1, 1, 1, 1], device=o3c.Device("CUDA:0"))
th_a = torch.utils.dlpack.from_dlpack(o3_a.to_dlpack())
```



### Hash API

```python
import open3d.core as o3c
import numpy as np

capacity = 10 # initial estimated hashmap capacity, not a limit though.
device = o3c.Device('cpu:0')

# unordered_map<int, int>
hashmap = o3c.HashMap(capacity,
                      key_dtype=o3c.int64,
                      key_element_shape=(1,),
                      value_dtype=o3c.int64,
                      value_element_shape=(1,),
                      device=device)

### reserve space (set capacity)
hashmap.reserve(capacity)

### insert
keys = o3c.Tensor([[100], [200], [400], [800], [300], [200], [100]],
                  dtype=o3c.int64,
                  device=device)
vals = o3c.Tensor([[1], [2], [4], [8], [3], [2], [1]],
                  dtype=o3c.int64,
                  device=device)
buf_indices, masks = hashmap.insert(keys, vals)

# masks indicate each insertion succeeded / failed.
print(masks) # [True True True True True False False], last two keys are duplicated and ignored.

# buf_indices indicate the inserted position.
buf_indices = buf_indices[masks].to(o3c.int64) # also need to remove failed insertion.
print(buf_indices) # [0 1 3 4 2], row vec [5]

# retrieve keys/vals by buf_indices
buf_keys = hashmap.key_tensor()
buf_vals = hashmap.value_tensor()
inserted_keys = buf_keys[buf_indices] # [100, 200, 400, 800, 300] (col vec [5, 1])
inserted_vals = buf_vals[buf_indices] # [1, 2, 4, 8, 3]

### query
query_keys = o3c.Tensor([[1000], [100], [300], [200], [100], [0]],
                        dtype=o3c.int64,
                        device=device)

buf_indices, masks = hashmap.find(query_keys)

# remove failed query (non-exist keys)
valid_keys = query_keys[masks] # [100, 300, 200, 100]

# retrieve query results (only valid query)
buf_indices = buf_indices[masks].to(o3c.int64)
valid_vals = buf_vals[buf_indices] # [1, 3, 2, 1]

### get all valid (active) keys
all_buf_indices = hashmap.active_buf_indices().to(o3c.int64)
all_keys = buf_keys[all_buf_indices]
all_vals = buf_vals[all_buf_indices]

### erase
erase_keys = o3c.Tensor([[100], [1000], [100]], dtype=o3c.int64, device=device)
masks = hashmap.erase(erase_keys)

# check succeeded erasion
valid_erase_keys = erase_keys[masks] # [100,]

### activate (reserve keys first, in-place set value later)
keys = o3c.Tensor([[1000], [0]], dtype=o3c.int64, device=device)
buf_indices, masks = hashmap.activate(activate_keys) # different from insert, we don't need to provide vals here.
# instead, we can inplace update vals later.
buf_vals[buf_indices[masks].to(o3c.int64)] = o3c.Tensor([[10], [0]], dtype=o3c.int64, device=device)

```

For multi-valued hashmap:

```python
mhashmap = o3c.HashMap(capacity,
                       key_dtype=o3c.int32,
                       key_element_shape=(3,),
                       value_dtypes=(o3c.uint8, o3c.float32),
                       value_element_shapes=((3,), (1,)),
                       device=device)

n_buffers = len(mhashmap.value_tensors()) # 2 value tensors

# insert
voxel_coords = o3c.Tensor([[0, 1, 0], [-1, 2, 3], [3, 4, 1]],
                          dtype=o3c.int32,
                          device=device)

voxel_colors = o3c.Tensor([[0, 255, 0], [255, 255, 0], [255, 0, 0]],
                          dtype=o3c.uint8,
                          device=device)

voxel_weights = o3c.Tensor([[0.9], [0.1], [0.3]],
                           dtype=o3c.float32,
                           device=device)

mhashmap.insert(voxel_coords, (voxel_colors, voxel_weights))

# query
query_coords = o3c.Tensor([[0, 1, 0]], dtype=o3c.int32, device=device)
buf_indices, masks = mhashmap.find(query_coords)

valid_keys = query_coords[masks]
buf_indices = buf_indices[masks].to(o3c.int64)

valid_colors = mhashmap.value_tensor(0)[buf_indices]
valid_weights = mhashmap.value_tensor(1)[buf_indices]
```

For non-value hashmap, i.e., hashset:

```python
hashset = o3c.HashSet(capacity,
                      key_dtype=o3c.int64,
                      key_element_shape=(1,),
                      device=device)

# insert
keys = o3c.Tensor([1, 3, 5, 7, 5, 3, 1], dtype=o3c.int64, device=device).reshape((-1, 1))
hashset.insert(keys)

# retrieve
active_buf_indices = hashset.active_buf_indices().to(o3c.int64)
active_keys = hashset.key_tensor()[active_buf_indices]
```



