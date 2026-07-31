# Keras Tutorial

> OK, I'm here for free TPU.

> The difference between `tf.keras` and `keras` is the Tensorflow specific enhancement to the framework.
>
> `keras` is an API specification that describes how a Deep Learning framework should implement certain part, related to the model definition and training. Is framework agnostic and supports different backends (Theano, Tensorflow, ...)
>
> `tf.keras` is the Tensorflow specific implementation of the Keras API specification. It adds the framework the support for many Tensorflow specific features like: perfect support for `tf.data.Dataset` as input objects, support for eager execution, ...
>
> In Tensorflow 2.0 `tf.keras` will be the default and I highly recommend to start working using `tf.keras`


### Model

##### Sequential 

```python
from keras.models import Sequential
from keras.layers import Dense

#model = Sequential()
#model.add(Dense(units=64, activation='relu', input_dim=100))

model = Sequential([
	Dense(32, input_shape=(784,)), # take [-1, 784] shaped input, same as input_dim=784
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
             )

model.fit(X, y, epochs=5, batch_size=32)
model.train_on_batch(Xb, yb)
loss_and_metrics = model.evaluate(X, y, batch_size=32)
classes = model.predict(X, y, batch_size=1)

```

##### Functional (More flexible)

```python
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer, loss, metrics)

### multiple output example
def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)

    input = Input(shape=(SIZE, SIZE, N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    grapheme_root = Dense(types['grapheme_root'],
                          activation = 'softmax', name='root')(x)
    vowel_diacritic = Dense(types['vowel_diacritic'],
                            activation = 'softmax', name='vowel')(x)
    consonant_diacritic = Dense(types['consonant_diacritic'],
                                activation = 'softmax', name='consonant')(x)

    # model
    model = Model(input, [grapheme_root, vowel_diacritic, consonant_diacritic])
    
    return model
```


### Data Loaders

```python
### built-in datasets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

### Use in-memory numpy ndarray
X, y = np.load(...)
model.fit(x=X, y=y)

### keras.utils.Sequence
class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size = 16, subset="train", shuffle=False, 
                 preprocess=None, info={}):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        
        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'
        
        ### shuffle
        self.on_epoch_end()
	
    ### Must
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    ### Must, return a Batch!
    def __getitem__(self, index): 
        X = np.empty((self.batch_size,128,800,3),dtype=np.float32)
        y = np.empty((self.batch_size,128,800,4),dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = Image.open(self.data_path + f).resize((800,128))
            if self.subset == 'train': 
                for j in range(4):
                    y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])
        ### Custom preprocess
        if self.preprocess!=None: 
            X = self.preprocess(X)
        if self.subset == 'train': 
            return X, y
        else: 
            return X
```


### Preprocess

```python
### Mostly, just use numpy (eg. albumentations) inside DataLoader.

### Image preprocessor
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# or
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```


### Callbacks

##### Built-ins

```python
# define callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
```

##### Custom Callbacks

```python
class MyCallback(keras.callbacks.Callback):
    def on_epoch_begin(self):
        pass
    def on_epoch_end(self):
        pass
    def on_batch_begin(self, batch, logs={}):
        pass
    def on_batch_end(self, batch, logs={}):
        pass
    def on_train_begin():
        pass
    def on_train_end():
        pass
```


### Custom Metrics

```python
import keras.backend as K ### here K == tf

# dice
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', dice_coef])
```


### Custom Losses

```python
### Just wrap metrics
import keras.backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# the wrapper should return a function that only accepts (y_true, y_pred)
def dice_loss(smooth):
    def _loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred, smooth=smooth)
   	return _loss

# use it
model.compile(loss=dice_loss(smooth=0.001))
```


### Custom Layers

```python
### Lambda

## Generalized mean pool - GeM

# define weights, functionals first
gm_exp = tf.Variable(3.0, dtype = tf.float32)
def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                        axis = [1, 2], 
                        keepdims = False) + 1.e-7)**(1./gm_exp)
    return pool
# use Lambda to wrap
lambda_layer = Lambda(generalized_mean_pool_2d)
lambda_layer.trainable_weights.extend([gm_exp])
x = lambda_layer(x_model.output)

### Layer class
from keras.layers import Layer
class MyLayer(Layer):
    def __init__(self, output_dim):
        super(MyLayer, self).__init__()
        self.output_dim = output_dim
    
    ## Must, define the weights
    def build(self, input_shape):
        # add_weight
        self.W = self.add_weight(name='W', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)
        # must call this at the end
        super(MyLayer, self).build(input_shape) 
    
    ## Must, forward()
    def call(self, x):
        return K.dot(x, self.W)
   	
    ## Must, for the next layer to know its input_shape
    def compute_output_shape(input_shape):
        return (input_shape[0], self.output_dim)
```


### Examples

```python
# Create Model
def create_model(input_shape):
    # Input Layer
    input = Input(shape = input_shape)
    
    # Create and Compile Model and show Summary
    x_model = efn.EfficientNetB3(weights = None, include_top = False, input_tensor = input, pooling = None, classes = None)
    
    # UnFreeze all layers
    for layer in x_model.layers:
        layer.trainable = True
    
    # GeM
    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])
    x = lambda_layer(x_model.output)
    
    # multi output
    grapheme_root = Dense(168, activation = 'softmax', name = 'root')(x)
    vowel_diacritic = Dense(11, activation = 'softmax', name = 'vowel')(x)
    consonant_diacritic = Dense(7, activation = 'softmax', name = 'consonant')(x)

    # model
    model = Model(inputs = x_model.input, outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])

    return model
```

