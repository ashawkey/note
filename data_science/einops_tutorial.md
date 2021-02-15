# einops

```bash
pip install einops
```



### operations

```python
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

# permutation 
y = rearrange(x, 'b c h w -> b h w c') 
# eqs:
y = x.permute(0, 2, 3, 1).contiguous()

# composition
y = rearrange(x, 'b h w c -> (b h) w c')
# eqs:
y = x.view(B*H, W, C)

# decomposition
y = rearrange(x, '(b h) w c -> b h w c')
# eqs:
y = x.view(B, H, W, C)

# shift
y = rearrange(x, '(b1 b2) h w c -> h (b1 b2 w) c') # 1 2 3 4
y = rearrange(x, '(b1 b2) h w c -> h (b2 b1 w) c') # 1 3 2 4

# mean/min/max
y = reduce(x, 'b h w c -> h w c', 'mean')
# eqs:
y = x.mean(dim=0)

# min-pooling
y = reduce(x, 'b (h h2) (w w2) c -> b h w c', 'min', h2=2, w2=2)

# stack
x = rearrange(xs, 'b h w c -> b h w c') # xs is [[h, w, c], ...] x b
# eqs:
x = torch.stack(xs, dim=0)

x = rearrange(xs, 'b h w c -> h w c b')
# eqs:
x = torch.stack(xs, dim=-1)

# squeeze/unsqueeze
x = rearrange(x, 'h w c -> 1 h w c')
# eqs:
x = x.unsqueeze(0)

x = rearrange(x, '1 h w c -> h w c')
# eqs:
x = x.squeeze(0)


```

