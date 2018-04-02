# Train the neural network

In this section, we will discuss how to train the previously defined network with data. We first import the libraries. The new ones are `mxnet.init` for more weight initialization methods, `matplotlib` for drawing, and `time` for benchmarking.

```{.python .input  n=1}
# Uncomment the following line if matplotlib is not installed.
# !pip install matplotlib

from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
import matplotlib.pyplot as plt
from time import time
```

## Get data

The handwritten digit MNIST dataset is one of the most commonly used datasets in deep learning. But it is too simple to get a 99% accuracy. Here we use a similar but slightly more complicated dataset called FashionMNIST. The goal is no longer to classify numbers, but clothing types instead.

The dataset can be automatically downloaded through Gluon's `data.vision` module. The following code downloads the training dataset and shows the first example.

```{.python .input  n=2}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
X, y = mnist_train[0]
('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)
```

Each example in this dataset is a $28\times 28$ size grey image, which is presented as NDArray with the shape format of `(height, width, channel)`.  The label is a `numpy` scalar.

Next, we visualize the first six examples.

```{.python .input  n=3}
text_labels = [
    't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
]
X, y = mnist_train[0:6]
# plot images
_, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))
for f,x,yi in zip(figs, X,y):
    # 3D->2D by removing the last channel dim
    f.imshow(x.reshape((28,28)).asnumpy())
    ax = f.axes
    ax.set_title(text_labels[int(yi)])
    ax.title.set_fontsize(20)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

`FashionMNIST` is a subclass of `gluon.data.Dataset`, which defines how to get the `i`-th example. In order to use it in training, we need to get a (randomized) batch of examples. It can be easily done by `gluon.data.DataLoader`.

```{.python .input  n=4}
batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train, batch_size, shuffle=True)
```

The returned `train_data` is an iterator that yields batches of data and labels pairs.

```{.python .input  n=5}
for data, label in train_data:
    print(data.shape, label.shape)
    break
```

Finally, we create a validation dataset and data loader.

```{.python .input  n=6}
mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid, batch_size, shuffle=False)
```

## Define the model

We reimplement the same LeNet introduced before. One difference here is that we changed the weight initialization method to `Xavier`, which is a popular choice for deep convolutional neural networks.

```{.python .input  n=7}
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10)
    )
net.initialize(init=init.Xavier())
```

Besides the neural network, we need to define the loss function and optimization method for training. We will use standard softmax cross entropy loss for classification problems. It first performs softmax on the output to obtain the predicted probability, and then compares the label with the cross entropy.

```{.python .input  n=8}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

The optimization method we picked is the standard stochastic gradient descent with constant learning rate of 0.1.

```{.python .input  n=9}
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
```

The `trainer` is created with all parameters (both weights and gradients) in `net`. Later on, we only need to call the `step` method to update its weights.

## Train

We create two auxiliary functions before training. Even though both are supported by built-in functions in Gluon, we found it's straightforward to implement them from scratch. One calculates the model accuracy, the other transforms the data into `(batch, channel, height, weight)` format and normalize its values.

```{.python .input  n=10}
def acc(output, label):
    # output: (batch, num_output) ndarray
    # label: (batch, ) ndarray
    return (output.argmax(axis=1)==label).mean().asscalar()

def transform(data, label):
    # data: (batch, height, weight, channel) ndarray
    # label: (batch, ) ndarray
    return (data.transpose((0,3,1,2)).astype('float32')/255,
            label.astype('float32'))
```

Now we can implement the complete training loop.

```{.python .input  n=10}
for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time()
    for data, label in train_data:
        data, label = transform(data, label)
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate traing metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)

    # calculate validation accuracy
    for data, label in valid_data:
        data, label = transform(data, label)
        valid_acc += acc(net(data), label)

    print("Epoch %d: Loss: %.3f, Train acc %.3f, Test acc %.3f,\
Time %.1f sec" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data),
        valid_acc/len(valid_data), time()-tic))
```

## Save the model

Finally, we save the trained parameters onto disk, so that we can use them later.

```{.python .input}
net.save_params('net.params')
```
