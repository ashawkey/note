# pytorch pitfalls

### `torch.gather(input, dim, index)`

* `assert input.dim == index.dim`

  Only allows `input.shape[dim] != index.shape[dim]` (but must be broadcast-able, the most usual case is `index.shape[dim] == 1`)

  For the other dims, we requires `input.shape[other_dim] >= index.shape[other_dim]`, and only use `input.shape[:index.shape[other_dim]]`.

* `output.shape == index.shape`

  ```python
  # 1d example
  out[i] = input[index[i]] # dim == 0
  # shortcut: `out = input[index]`
  
  # 2d example
  out[i][j] = input[index[i][j]][j] # dim == 0
  out[i][j] = input[i][index[i][j]] # dim == 1
  ```

* Examples

    Considering we want to extract one (or some) elements per row:

    ```python
    a = torch.arange(6).view(2, 3) # [[0,1,2], [3,4,5]]
    
    # we want [[2], [3]]
    b = torch.LongTensor([[2], [0]]) # [[2], [0]]
    a.gather(1, b) # note dim=1, not 0! because we collect element from dim1 !
    
    # we want [[2,1], [3,4]]
    b = torch.LongTensor([[2, 1], [0, 1]])
    a.gather(1, b)
    
    # we just want [[2, 1]] (index.shape[other_dim] < input.shape[other_dim])
    b = torch.LongTensor([[2, 0]])
    a.gather(1, b)
    ```

    

* Difference with `torch.take_along_dim & np.take_along_axis`.

  **In most normal use cases, it is the same.**

  One of the difference is the handling of `index.shape[other_dim] < input.shape[other_dim]`. `take_along_axis` tries to broadcast.

  ```python
  a = np.arange(6).reshape(2, 3)
  
  b = np.array([[2], [0]])
  np.take_along_axis(a, b, axis=1) # [[2], [3]]
  
  b = np.array([[2, 1], [0, 1]])
  np.take_along_axis(a, b, axis=1) # [[2, 1], [3, 4]]
  
  # BUT np will try to broadcast, if index[other_dim] < input[other_dim]!
  b = np.array([[2, 1]])
  np.take_along_axis(a, b, axis=1) # [[2, 1], [5, 4]]
  # this is the same as
  b = np.array([[2, 1], [2, 1]])
  np.take_along_axis(a, b, axis=1) # [[2, 1], [5, 4]]
  # if not broadcast-able, an error is thrown.
  a = np.arange(9).reshape(3, 3)
  b = np.array([[2, 1], [2, 1]])
  np.take_along_axis(a, b, axis=1) # IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,1) (2,2) 
  
  # while torch doesn't try to broadcast:
  a = torch.arange(9).reshape(3, 3)
  b = torch.LongTensor([[2, 1], [2, 1]])
  a.gather(1, b) # [[2, 1], [5, 4]]
  ```



### ``torch.scatter(input, dim, index, src)``

For the input specifications, it is very similar to `gather`.

* `assert input.dim == index.dim == src.dim`

  Only allows `input.shape[dim] != index.shape[dim]` (but must be broadcast-able, the most usual case is `index.shape[dim] == 1`)

  For the other dims, we requires `input.shape[other_dim] >= index.shape[other_dim]`, and only use `input.shape[:index.shape[other_dim]]`.

* weakly allows `src.shape >= index.shape` for all dims. In this case only `src[:index.shape]` is used.

  But the backward pass is only implemented for `src.shape == index.shape`.

  ```python
  # 1d example
  input[index[i]] = src[i] # dim == 0
  # shortcut: input[index] = src
  
  # 2d example
  input[index[i][j]][j] = src[i][j] # dim == 0
  input[i][index[i][j]] = src[i][j] # dim == 1
  ```

* **WARNING!!!** For `scatter`, index must be unique. Else the behavior is non-deterministic (only one of the duplicated indices is used) and lead to wrong gradient.

  However, For `scatter_add`, index can be duplicated and the answer is correctly summed over all duplicated indices..

* Examples:

  ```python
  input = torch.zeros((3, 3)).long()
  
  # we want to scatter [1,2,3] to the diag of input
  src = torch.arange(1,4)[:, None] # specify values, [[1], [2], [3]]
  index = torch.LongTensor([[0], [1], [2]]) # specify col in each row
  input.scatter(1, index, src)
  
  # duplicated index
  input = torch.zeros(5).long()
  src = torch.arange(1, 4)
  index = torch.LongTensor([1,1,1]) # duplicated index
  input.scatter(0, index, src) # dangerous! in cpu, seems the last src is adopted. [0, 3, 0, 0, 0]
  input.scatter_add(0, index, src) # safe. always [0, 6, 0, 0, 0]
  
  # column-wise merge
  input = torch.zeros(2, 2).long()
  src = torch.ones(2, 3).long()
  # we want input[:, 0] = src[:, 0]; input[:, 1] = src[:, 1] + src[:, 2]
  index = torch.LongTensor([0, 1, 1]) 
  # dim=1 since we collect from column(dim1).
  # we have to EXPLICITLY repeat index, since torch only use src[:index.shape] !!!
  input.scatter_add(1, index[None, :].repeat(2, 1), src) # [[1,2], [1,2]]
  input.scatter_add(1, index[None, :], src) # [[1,2], [0,0]], not the expected value
  ```

* Difference with `np.put_along_axis`:

  similar to `take_along_axis`, `np` tries to broadcast.

  Anyway, do not use these strange features is the most safe way...

  ```python
  input = np.zeros((3, 3))
  src = np.arange(1, 4)[:, None]
  index = np.arange(3)[:, None]
  
  # inplace, we have to print the input
  np.put_along_axis(input, index, src, axis=1)
  print(input)
  
  # However, there is no add version of put_along_axis. An alternative is np.add.at()
  ```

* Difference with `torch_scatter` package.

  the implementation in `torch_scatter` forces to broadcast index before scatter. (more like numpy)

  We can see from the source code:

  ```python
  def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
      if dim < 0:
          dim = other.dim() + dim
      if src.dim() == 1:
          for _ in range(0, dim):
              src = src.unsqueeze(0)
      for _ in range(src.dim(), other.dim()):
          src = src.unsqueeze(-1)
      src = src.expand_as(other)
      return src
  
  def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                  out: Optional[torch.Tensor] = None,
                  dim_size: Optional[int] = None) -> torch.Tensor:
      # this line is exactly what we did in `index[None, :].repeat(2, 1)`
      # equals `index = index.broadcast_to(src)`
      index = broadcast(index, src, dim)
      if out is None:
          size = list(src.size())
          if dim_size is not None:
              size[dim] = dim_size
          elif index.numel() == 0:
              size[dim] = 0
          else:
              size[dim] = int(index.max()) + 1
          out = torch.zeros(size, dtype=src.dtype, device=src.device)
          return out.scatter_add_(dim, index, src)
      else:
          return out.scatter_add_(dim, index, src)
  
  def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                   out: Optional[torch.Tensor] = None,
                   dim_size: Optional[int] = None) -> torch.Tensor:
  
      out = scatter_sum(src, index, dim, out, dim_size)
      dim_size = out.size(dim)
  
      index_dim = dim
      if index_dim < 0:
          index_dim = index_dim + src.dim()
      if index.dim() <= index_dim:
          index_dim = index.dim() - 1
  
      ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
      count = scatter_sum(ones, index, index_dim, None, dim_size)
      count[count < 1] = 1
      count = broadcast(count, out, dim)
      if out.is_floating_point():
          out.true_divide_(count)
      else:
          out.floor_divide_(count)
      return out    
  ```

  

