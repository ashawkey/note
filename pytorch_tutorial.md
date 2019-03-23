# PyTorch

### Tensors

```python
## create a tensor
x = torch.Tensor(d1, d2)
r = torch.rand(d1, d2)
x # print(x)
x.size() # torch.Size, in fact tuple.

## operation
x + y
torch.add(x, y)
y.add_(x) # inplace
x[:, 1] # numpy-like slice is supported
z = x.view(-1, 8) # view means reshape. -1 is inferred.

## numpy
a = torch.ones(5)
b = a.numpy() # it will follow the change of a. SHALLOW.
c = np.ones(5)
d = torch.from_numpy(a) # it follows c.

## cuda
if torch.cuda.is_available():
    x = x.cuda()

```

```python
## operations
x.permute(*axes) # 3+ dim transpose
x.transpose(d1, d2) # only 2 dim
x.view(*shape) # reshape
x.contiguous() # make x store continuously in memory
x.is_contiguous() 
# after permute(), x is not contiguous, and CANNOT view.
x.permute().contiguous().view() # routine

torch.cat((A, B), dim=1)
```



### autograd

```python
## Variable
'''
Wrapper of Tensors. Support all operations of Tensors.
`.backward()` method will automatically compute all gradients.
'''
from torch.autograd import Variable
t = torch.ones(2, 2)
x = Variable(t, requires_grad=True)
x.data # t

## autograd
x = Variable(torch.ones(2, 2), requires_grad = True) # requires_grad default is False.
y = x + 2
z = y * y * 3
out = z.mean()
out.backward() # output nothing.
print(out.grad_fn) # Function 
print(x.grad) # 4.5


```

$$
out = \frac {3}{4} \sum (x + 2)^2 \\
\frac {\partial out}{\partial x_i} = \frac{3}{2}(x+2) = 4.5
$$



### Neural Network

Affine Layer: Fully connected layer.

> Affine Transformation = Linear Transformation + Translation
>
> $y = Wx + b$

```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # [in_channel, out_channel, filter_size]
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # [in, out]
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
	
    # override forward(), which will be used by backward()
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # [input, (px, py)]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # equals [input, (p, p)]
        x = x.view(-1, self.num_flat_features(x)) # flatten with batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # logits
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批量维度外的所有维度
        return reduce(lambda x,y:x*y, size)

net = Net()
print(net)

# list all learnable parameters
params = list(net.parameters())

# predict
input = Variable(torch.randn(1,1,32,32)) # input shape is [N, C, H, W]
# torch.nn only support 4-dim input, if necessary:
# input.unsqueeze(0) # tf.expand_dims()
output = net(input)
print(output)

# reset grad
net.zero_grad() # clear grad

# Loss
target = Variable(torch.arange(1, 11))
loss = nn.MSELoss(output, target)
loss.backward()

# optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad() # the same as net.zero_grad()
optimizer.step()
```

```python
nn.Module
'''
Base class for all networks
Method:
@ forward(self, x)
@ backward(self)
@ apply(fn)
	apply fn recursively to all sub-Modules.
	typically used to feed weight initializer to the net.
@ cpu()
@ cuda(device=None)
@ double()
	also float(), half()
@ children()
@ modules() # iterator 
@ parameters()
@ to(dtype/device)
	to(torch.device("cpu"))
	to(torch.double)
@ zero_grad()
'''
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)


nn.ModuleList
'''
for dynamic models.
just lika a python list.
append(), extend()
'''
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
    
    
nn.ModuleDict
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
    
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True)
'''
(x-mean(x))/(sqrt(var(x))+eps) * W + b
affine=True: add learnable weights
'''

nn.ReLU(inplace=False)

nn.ConvTranspose2d(in_cs, out_cs, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True)
'''
fractionally-strided convolution (deconvolution/transposed convolution) 
bias=True: whether to add a learnable Bias.
'''
```

```python
### init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```



#### Low level functional

```python
## Function
import torch.nn.functional as F
F.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ...)
'''
this is used more than nn.MaxPool2d, however.
kernel_size: int or (int,int)
stride is by default the kernel_size.
'''
F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
'''
this is the function called by
nn.Conv2d
need to provide weight tensor.
'''
F.relu(input, inplace=False)
F.softmax(input)
F.dropout(input, p=0.5, training=False, inplace=False)
F.cross_entropy(input, target, weight=None, size_average=True)
```



### dataset & torchvision

```python
# CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # 非标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.cuda() # run on GPU

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train
for epoch in range(2):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # 每2000个小批量打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# test
dataiter = iter(testloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# test single
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
# test whole
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


```



### Save Model

```python
class Net(nn.Module):
    pass

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
### save state_dict
torch.save(model.state_dict(), PATH)

# load
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval() # set dropout and BN to evaluation mode.

### save the whole model by pickle
torch.save(model, PATH)

# load
model = torch.load(PATH)
model.eval()
```



### torchvision transforms
