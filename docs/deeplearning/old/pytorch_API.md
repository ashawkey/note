### torch

```python
### Settings
is_tensor(obj)
set_default_dtype(d) # torch.float32
get_default_dtype()
set_default_tensor_type(t) # torch.FloatTensor , float32
set_printoptions() # threshold="nan" for print all

### Creations
''' shared parameters
@ dtype=None
@ device=None
@ layout=None : memory layout, eg. strided, sparse_coo
@ require_grad=False
'''
tensor(data) # always copy the data!
as_tensor(data) # no copy if device, dtype is the same.
from_numpy(ndarray) # no copy, share the same memory.
zeros(*size) # dtype is globally default as float32
zeros_like(input)
ones(*size)
ones_like(input)
arange(start=0, end, step=1)
linspace(start, end, steps=100)
logspace(start, end, steps=100)
eye(n)
empty(*size)
empty_like(input)
full(size, fill_value)
full_like(input, fill_value)


### Operations
cat(seq, dim=0) # concat in dim
stack(seq, dim=0) # stack in a new dim
chunk(tensor, chunks<int>, dim=0) # split into chunks
split(tensor, split_size, dim=0) # split by size
gather(input, dim, index<LongTensor>) # rearrange
'''
very tricky... Different from tensorflow.gather.
>>> t = torch.tensor([[1,2],[3,4]])
>>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
tensor([[ 1,  1],
        [ 4,  3]])
'''
index_select(input, dim, indices<LongTensor>) # select indices from input along dim.
"""
only support indices to be a vector(1-dim tensor).
so also silly.
"""

masked_select(input, mask<ByteTensor>)
nonzero(input) # indices of non-zero input
"""very important and useful!"""

reshape(x, shape)
squeeze(x, dim=None) # default: remove all 1. INPLACE!
unsqueeze(x, dim)
t(x) # transpose(x, 0, 1)
transpose(x, d0, d1)
permute(x, (d0, d1, ...))
take(input, indices<LongTensor>) # indices is 1d, flattened indices version of index_select
unbind(x, dim=0) # inversion of cat
where(condition<ByteTensor>, x, y) # x if True else y


### random sampling
manual_seed(seed)
bernoulli(input) # out_i ~ B(input_i)
normal(mean, std) # mean&std can be a tensor
rand(*size)
rand_like(input)
randint(low=0, high, size)
randint_like(input, low=0, high, size)
randn(*size)
randn_like(input)


### Math
abs(x)
add(x, n) # tensor + number
div(x, n)
mul(x, n)
ceil(x)
floor(x)
clamp(x, min, max)
pow(x, n)
exp(x)
...

### reduction
argmax(x, dim=None, keepdim=False) # default is flatten
argmin(x, dim=None, keepdim=False)
cumprod(x, dim) # cumulative product
cumsum(x, dim)
dist(x, y, p=2) # p-norm
norm(x, p=2)
unique(x, sorted=False, return_inverse=False) # unique value


### comparison
eq(x, y) # -> element-wise ByteTensor
ne(x, y)
equal(x, y) # -> Trur/False
isfinite(x)
isinf(x)
isnan(x)
sort(x, dim=None, descending=False)

# others
cross(x, y, dim=-1)
diag(x, diagonal=0)
diagonal(x) # diag(x, 0)

# BLAS & LAPACK
matmul(t1, t2) # broadcastable
''' behavior
@ [M]*[M] = [M], [M,N]*[N,K]=[M,K]
@ [M]*[M,N] = [1,M]*[M,N] = [N]
@ [M,N]*[N] = [M]

@ [j,m,n]*[j,n,p] = [j,m,p]
@ [j,m,n]*[n] = [j,m]
@ [j,1,n,m]*[k,m,p] = [j,k,n,p]
@ [j,m,n]*[n,p] = [j,m,p]

'''
bmm(b1, b2) # [B,X,Y] * [B,Y,Z] = [B,X,Z]
mm(m1, m2) # [N,M] * [M,P] = [N,P]
dot(t1, t2)
eig(mat, eigenvectors=False)
inverse(x)
det(x)


```

### torch.Tensor

![1540555801356](C:\Users\hawke\AppData\Roaming\Typora\typora-user-images\1540555801356.png)

default torch.Tensor is `FloatTensor`
```python
class Tensor:
    shape
    device
    dtype
    layout # strided for dense, sparse_coo for coo
	item() # tensor->scalar, only for one-element tensor
    tolist() # tensor->list
    abs()
    abs_() # inplace, faster
    clone()
    contiguous()
    cpu()
    cuda()
    to()
    '''
    to(dtype)
    to(device)
    '''
    repeat(*size) # np.tile
    '''
    >>> x = torch.tensor([1, 2, 3])
	>>> x.repeat(4, 2)
	tensor([[ 1,  2,  3,  1,  2,  3],
    	    [ 1,  2,  3,  1,  2,  3],
        	[ 1,  2,  3,  1,  2,  3],
        	[ 1,  2,  3,  1,  2,  3]])
    '''
    size()
    type()
    view(*size)
	view_as(other)
```

### torch.nn

```python
### parameters
Parameters()
	'''a kind of tensor able to be considered a module parameter'''
    data # the tensor
    requires_grad # bool
   
### Containers
Module() # base class for all nn modules.
	add_module(name, module) # add child module
    apply(fn) # apply function fn to all submodules recursively
    ''' net.apply(init_weight) '''
    children() # iterator on children
    modules() # ditto?
    cpu(), cuda()
    double(), float(), half(), ...
    train() # train mode.
    eval() # eval mode, stop Dropout and BatchNorm.
    forward(*input) # define the computation
    state_dict()
    load_state_dict(dict)
    parameters() # iterator on all Parameters
    register_buffer(name, tensor) # add a persistent buffer
    register_parameter(name, param) # add a parameter
    type(dtype) # cast all paramters to dtype.
    to() # device or dtype, inplace, always
    zero_grad() 
    
Sequential(*args)
	args: *list, or OrderedDict.
    # all inputs are aligned sequentially.
    
ModuleList(modules=None)
	modules: list
    # just a list, supporting append, extend.
    
ModuleDict(modules=None)
	# just a dict

ParameterList(...)
ParameterDict(...)

### Convolutions
Conv1d(Fin, Fout, ks, stride=1, padding=0, dilation=1, groups=1, bias=True)
'''
It's a VALID cross-correlation in fact.
padding: int or (int, int)
L_out = floor((L_in+2*padding-ks)/stride + 1)
'''
Conv2d(...)
"""
only accept (N,C,H,W).
"""
ConvTransposed1d(Fin, Fout, ks, stride=1, padding=0, output_padding=0,...)
"""
accept (N, F_in, L_in).
L_out = (L_in - 1)*stride - 2*paddint + ks + output_padding
"""
    
### Pooling
MaxPool1d(ks, stride=None, padding=0, dilation=1, ...)
MaxUnpool1d(ks, stride=None, padding=0) # lost-vals are set to 0.
AvgPool1d(...)
LPPool1d(p, ks, stride=None)
AdaptiveMaxPool1d(output_size) # [N, L] -> [N, out]
AdaptiveAvgPool1d(output_size)

### Padding
ReflectionPad1d(padding)
ReplicationPad1d(padding)
...

### activation
ELU(alpha=1.0, inplace=False)
SELU(inplace=False)

ReLU(inplace=False)
ReLU6(inplace=False) # min(max(0,x),6)

LeakyReLU(negative_slope=0.01, inplace=False)
PReLU(num_parameters=1, init=0.25)
"""
x = x if x>=0 else Ax; A is learnable.
num_parameters: length of A, set to Fin.
init: initial value of A.
"""
RReLU(lover=0.125, upper=0.333, inplace=False)
"""
similar to PReLU, where A ~ U(lower,upper)
"""
Sigmoid()
Softplus(beta=1)
Softmax()
Tanh()

### Normalization
BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, ...)
...

### RNN

### Linear
Linear(Fin, Fout, bias=True)
Bilinear(Fin1, Fin2, Fout, bias=True) # y = x_1 * A * x_2 + b

### dropout
Dropout(p=0.5, inplace=False)
Dropout2d(p=0.5, inplace=False)


### Loss
L1Loss(size_average=True, reduce=True, ...)
''' 
l = mean({|x_n - y_n|})
loss(input, target), shape=(N,*)
'''
MSELoss(...)
''' 
l = mean({(x_n - y_n)^2})
loss(input, target), shape=(N,*)
'''
CrossEntropyLoss()
'''
l(x,cls) = -x[class]+log(\sum_j exp(x[j]))
loss(input<N,nCls>, target<N>)
'''
NLLLoss()
'''
negative log likelihood.
l(x, y) = (\sum_n -w_yn*x_{n,yn})/\sum_n w_yn

logits = nn.LogSoftmax(x)
loss(logits<N,C>, target<N>)

equals to CrossEntropyLoss(x).
'''

BCELoss()
'''
l_n = -w_n[y_n * log x_n + (1-y_n)*log(1-x_n)]
l(input<N>,target<N>) = mean(l_n)

logits = nn.Sigmoid(x)
loss(logits, target)

eq. BCEWithLogitLoss(x)
'''
```

### torch.nn.functional

```python
# mostly correspond to layers.
interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None)
```


### torch.optim

```python
optimizer = optim.SGD(model.parameters(), lr=..., momentum=...)
optimizer.zero_grad()
loss.backward()
optimizer.step()
...

lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
'''
>>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)
'''
lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
'''
scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
'''
lr_scheduler.ExponentialLR(optimizer, gamma)

lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
'''
dynamically reduce lr when a metric has stopped improving.
mode: min means metric stops decreasing.

>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)
'''
```

### torch.autograd

```python
backward(tensors)
grad(outputs, inputs)

no_grad()
'''
reduce memory consumption, be sure not call backward().
1. with torch.no_grad():
2. @torch.no_grad()
'''
```

