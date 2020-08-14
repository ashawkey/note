#### tips

* `axis`

  None is all, **0 is by col, 1 is by row**, [0, 1] is both row and col (the same as None for 2D data.)

* **Batch training**

  Compared to Stochastic training (each time pick one train data and fit it.)

  batch training pick BATCH SIZE train data randomly and fit them.

  **More robust and converge faster.**

  but how to choose a good BATCH SIZE is very confusing.

  **Tensorflow is suited for batch training, and the data shapes all start with a [None, ...], which is the batch dimension. **

* Batch normalization

  this is totally different from batch training. 

  BN means **Normalize the inputs of all layers**, instead of just normalize the first input.

  An implementation:

  ```python
  # think of this as the input layer
  model.add(Dense(64, input_dim=16, init=’uniform’))
  model.add(BatchNormalization())
  model.add(Activation(‘tanh’))
  model.add(Dropout(0.5))
  # think of this as the output layer
  model.add(Dense(2, init=’uniform’))
  model.add(BatchNormalization())
  model.add(Activation(‘softmax’))
  ```

  ...seems not that useful ?

* Losses

  * MSE (mean of L2_loss)

  * MAE (mean of L1_loss)

  * MAPE (percentage error)
    $$
    MAPE = \frac{1}{n}\sum_{i=1}^n|\frac{\hat y - y_i}{y_i}|
    $$

  * binary cross entropy

    binary classification with sigmoid.
    $$
    L = -\sum_{i=1}^n(y_ilog(\hat y) + (1-y_i)log(1-\hat y))
    $$

  * categorical cross entropy

    multiple classification with one_hot + softmax.
    $$
    L = -\sum_{i=1}^n\sum_{t=1}^c(y_{i,t}log\hat y_t)
    $$




#### tensorflow

```python
Session()
# with as 
InteractiveSession()
# start a default session, don't need with as.

## operations -----
add(x,y)
multiply(x,y)
pow(a,b)
greater(x,y,name=None)
equal(x,y)
cast(x,dtype)
where(condition, x, y, name)
'''
condition is a tensor of bool.
return indices.
#####
g = tf.where(tf.greater(x,30))
'''
gather()
'''
params, # tensor from where to gather
indices, # tensor of indices to gather
validate_indices=None,
name=None,
axis=0
#####
v = tf.gather(x, g)
'''
reduce_sum()
'''
input_tensor 
axis=None # None means reduce all, return single element.
keepdims=None # if True, avoid squeezing dims.
name=None
'''

## generators -----
zeros(shape)
ones(shape)
fill(shape, value)
lin_space(start,stop,num)
range(start, limit, delta, dtype)

random_uniform(shape, dtype, ...)
'''
shape, # [] for scalar
minval=0, maxval=None, # 1 for float
dtype=tf.float32,
seed=None, name=None
'''

case()
'''
pred_fn_pairs, # list or dict of pairs(in tuple form)
default=None, # callable
exclusive=False, # return immediately when one condition is True
strict=False, name='case'
#####
x = tf.random_uniform([],minval=-1,maxval=1)
y = tf.random_uniform([],minval=-1,maxval=1)
out = tf.case([(tf.greater(x,y),lambda: x-y),
               (tf.less(x,y),lambda: x+y)],
             default=lambda: 0.0
             )
'''

add_n(tensors)
'''
add up a list of tensor element-wise
'''

moments(x, axes)
'''
calculate mean and var along axes.
mean, variance = moments(x, [0])
'''

constant()

zeros_like(Tensor)


diag(diagonal, name)
matrix_determinant(mat)
unique()
'''
uni, idx = tf.unique(x)
'''
```

```python
g = tf.Graph()
with g.as_default():
    pass
sess = tf.Session(graph=g)

tf.reset_default_graph()
tf.get_default_graph()
```

#### train

```python
GradientDescentOptimizer(learning_rate)
```



#### layers

An encapsulation of `tf.nn `.

```python
conv2d()
'''
inputs # [N,H,W,C]
filters # int, out_channels
kernel_size # int-->[1,K,K,1]; or (int, int)-->[1,K,L,1]
strides=(1,1) # [1, 1, 1, 1]
padding="valid" # or "same"
data_format="channels_last" # NHWC
activation=None # tf.nn.relu
use_bias=True # auto plus bias
name=None
reuse=None # whether to reuse weights of a previous same-named layer.
#####
output # [N,HH,WW,CC]
'''
dense()
'''
inputs # tensor
units # int, dimension of output
activatoin=None # tf.nn.relu 
use_bias=True # 
trainable=True # whether to train kernel and bias.
name
reuse
#####
output = activation(inputs*kernel + bias)
'''
dropout()
'''
inputs # tensor
rate=0.5 # 50% of inpute unit will be dropped to 0, others scaled up by 1/0.5=2
training=False # whether to use dropout.
name
#####
We only dropout when training, and shouldn't when testing!
so the routine is to use a placeholder. (rate or training)
#####
return inputs/rate or 0.
'''
flatten(inputs)
'''
preserve the batch dimension.
(None, 4, 4) --> (None, 16)
'''
max_pooling2d()
'''
inputs
pool_size # int or (int, int)
strides # int or (int, int)
padding="valid"
data_format="channels_last",
name
'''
average_pooling2d()
```



#### nn

```python
## activation -----
relu(features, name=None)
'''
return max(features, 0)
'''
dropout(x, keep_prob)
'''
there is no *training* param, so use ph_keep_prob.
#####
if rand() < keep_prob: return x/keep_prob
else: return 0
'''

## convolution -----
conv2d()
'''
input # [batch, in_height, in_width, in_channels]
filter # [filter_height, filter_width, in_channels, out_channels]
strides # [1, stride, stride, 1] usually
padding # "SAME", "VALID"
data_format # default is "NHWC"
name=None
#####
output # [batch, out_height, out_width, out_channels]
'''

## pooling -----
avg_pool(value,ksize,strides,padding,data_format,name)
'''
ksize # [1, k, k, 1]
strides # [1, 1, 1, 1]
'''
max_pool(value, ksize, strides, padding, data_format, name)

## normalization -----
l2_normalize(x, axis=None, epsilon=1e-12, name=None)
'''
return x / sqrt(max(sum(x**2), epsilon))
'''

## losses -----
l2_loss(t, name=None)
'''
return sum(t**2)/2
'''

## classification -----
softmax(logits, axis=None, name)
'''
return tf.exp(logits)/tf.reduce_sum(tf.exp(logits), axis)
'''
softmax_cross_entropy_with_logits()
'''
this will call softmax with logits, so NEVER softmax the logits before it. (speed up)
#####
labels # one_hot tensor
logits # same as labels
'''
```

#### contrib

#### keras

```python
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import 

### layers -----
Conv2D() # or Convolution2D
'''
input_shape # [H,W,C], correspond to [N,H,W,C]
input_dim # H, correspond to [N,H]
#####
filters # int
kernel_size # int or (int, int)
strides # int or (int, int)
padding="valid" # or "same", CaSeInsEnsiTiVE
activation # "relu", "linear"
'''
Dense()
'''
units
activation
'''
MaxPooling2D() # or MaxPool2D
'''
pool_size
strides
padding
'''
Softmax()
Flatten()
ReLU()
Dropout(rate)

### optimizers -----
Adam()
'''
lr, ...
'''
Adagrad()
RMSprop()
SGD()
'''
lr
momentum # float>=0
dacay # float>=0
'''

### losses -----
KLD # or kld. KL divergence, relative entropy
MAE # or mae. Mean absolute error
MSE # or mse. Mean squared error
binary_crossentropy
categorical_crossentropy


### utils -----
to_categorical(y, num_classes=None)
```

```python
# example
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

model.fit(x, y, batch_size=32, epochs=10)
# model.evaluate(x,y,batch_size=None)
model.predict(x, batch_size=None)
model.save("full_model.h5")
# model = keras.models.load_model("xxx.h5")
```



#### summary







