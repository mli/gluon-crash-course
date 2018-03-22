# Manipulate data with `ndarray`

We’ll start by introducing the `NDArray`, MXNet’s primary tool for storing and transforming data. If you’ve worked with `NumPy` before, you’ll notice that a NDArray is, by design, similar to NumPy’s multi-dimensional array. 

## Get started

To get started, let's import the `ndarray` package (`nd` is shortform) from MXNet.

```{.python .input  n=1}
# If MXNet is not installed. Uncomment the following line
# !pip install mxnet

from mxnet import nd
```

Next, let's see how to create a 2D array (also called a matrix) with values from two sets of numbers: 1, 2, 3 and 4, 5, 6. This might also be referred to as a tuple of a tuple of integers.

```{.python .input  n=2}
nd.array(((1,2,3),(5,6,7)))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  2.  3.]\n [ 5.  6.  7.]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can also create a very simple matrix with the same shape (2 rows by 3 columns), but fill it with 1s.

```{.python .input  n=3}
x = nd.ones((2,3))
x
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  1.  1.]\n [ 1.  1.  1.]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Often we’ll want to create arrays whose values are sampled randomly. For example, sampling values uniformly between -1 and 1. Here we create the same shape, but with random sampling.

```{.python .input  n=15}
y = nd.random.uniform(-1,1,(2,3))
y
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "\n[[ 0.08976638  0.69450343 -0.15269041]\n [ 0.24712741  0.29178822 -0.23123658]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can also fill an array of a given shape with a given value, such as `2.0`.
<!-- added to improve multiplication example -->

```{.python .input  n=16}
x = nd.full((2,3), 2.0)
x
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "\n[[ 2.  2.  2.]\n [ 2.  2.  2.]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

As with NumPy, the dimensions of each NDArray are accessible by accessing the `.shape` attribute. We can also query its `size`, which is equal to the product of the components of the shape. In addition, `.dtype` tells the data type of the stored values.

```{.python .input  n=17}
(x.shape, x.size, x.dtype)
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "((2, 3), 6, numpy.float32)"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Operations

NDArray supports a large number of standard mathematical operations. Such as element-wise multiplication:

```{.python .input  n=18}
x * y
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "\n[[ 0.17953277  1.38900685 -0.30538082]\n [ 0.49425483  0.58357644 -0.46247315]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Exponentiation:

```{.python .input  n=23}
y.exp()
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "\n[[ 1.09391868  2.0027144   0.85839546]\n [ 1.28034222  1.3388195   0.79355168]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

And grab a matrix’s transpose to compute a proper matrix-matrix product:

```{.python .input  n=24}
nd.dot(x, y.T)
```

```{.json .output n=24}
[
 {
  "data": {
   "text/plain": "\n[[ 1.2631588   0.61535811]\n [ 1.2631588   0.61535811]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Indexing

MXNet NDArrays support slicing in all the ridiculous ways you might imagine accessing your data. Here’s an example of reading a particular element, which returns a 1D array with shape `(1,)`.

```{.python .input  n=25}
y[1,2]
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "\n[-0.23123658]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Read the second and third columns from `y`.

```{.python .input  n=26}
y[:,1:3]
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "\n[[ 0.69450343 -0.15269041]\n [ 0.29178822 -0.23123658]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

and writing to a specific element

```{.python .input  n=27}
y[:,1:3] = 2
y
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "\n[[ 0.08976638  2.          2.        ]\n [ 0.24712741  2.          2.        ]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Multi-dimensional slicing is also supported.

```{.python .input  n=28}
y[1:2,0:2] = 4
y
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "\n[[ 0.08976638  2.          2.        ]\n [ 4.          4.          2.        ]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Converting between MXNet NDArray and NumPy

Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do not share memory.

```{.python .input  n=29}
a = x.asnumpy()
(type(a), a)
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "(numpy.ndarray, array([[ 2.,  2.,  2.],\n        [ 2.,  2.,  2.]], dtype=float32))"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=30}
nd.array(a)
```

```{.json .output n=30}
[
 {
  "data": {
   "text/plain": "\n[[ 2.  2.  2.]\n [ 2.  2.  2.]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 30,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```
