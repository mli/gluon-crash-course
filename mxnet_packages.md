# Install a different MXNet package

In this tutorial, we show how to install a different MXNet version through
`pip`. The plain `mxnet` which can be installed by `pip install mxnet`, is able
to execute almost all MXNet codes in CPUs. But there are other pre-compiled
packages to support more hardware and/or more efficient executions.

## Uninstall a previously installed version

To install a different version, we should uninstall the previously installed
version first. We can check it through

```bash
pip list | grep mxnet
```

If the previous command returns a non-empty result, such as `mxnet
(1.2.0b20180413)`, then we can remove it by

```bash
pip uninstall mxnet
```

## Choose another package

All MXNet packages can be found at
[pypi.org](https://pypi.org/search/?q=mxnet). Here we list three major variants.

### Nvidia GPUs

To run on Nvidia GPUs, we should install a package with `cu??` in the package
name, where `??` is CUDA version such as `80` and `91`. To install a specific
version, users should have [CUDA](https://developer.nvidia.com/cuda-downloads)
installed first. Then select the MXNet package that matches the CUDA version.
This can be checked by running `nvcc -V`.

```eval_rst

============  ===============
CUDA version  MXNet package
============  ===============
7.5           ``mxnet-cu75``
8.0           ``mxnet-cu80``
9.0           ``mxnet-cu90``
9.1           ``mxnet-cu91``
9.2           ``mxnet-cu92``
============  ===============

```

All `cu` packages ship with `cudnn` by default, so there is no need to install it
separately.

A common error with the `cu` packages is when CUDA shared objects fail to open
after using `import mxnet`. One example is:

```
OSError: libcudart.so.9.0: cannot open shared object file: No such file or directory
```

To solve, we just need to add CUDA into the library path. On Linux, we can run:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
```

### Intel CPUs

All MXNet packages support Intel CPUs, but the variants with `mkl` in the
package name can potentially improve the performance. These packages have
Intel's [MKL-DNN](https://github.com/intel/mkl-dnn) included. For example, for
convolutional neural network inference, `mxnet-mkl` often outperforms `mxnet` by
[more than 4x](https://mxnet.incubator.apache.org/faq/perf.html#intel-cpu).

You may install MXNet with MKL-DNN with the following:

```bash
pip install mxnet-mkl
```

### Nvidia GPU + Intel CPUs

We can have both CPU and GPU hardware accelerated by using CUDA and MKL-DNN.
Use one of the following package options:


```eval_rst

============  ==================
CUDA version  MXNet package
============  ==================
7.5           ``mxnet-cu75mkl``
8.0           ``mxnet-cu80mkl``
9.0           ``mxnet-cu90mkl``
9.1           ``mxnet-cu91mkl``
9.2           ``mxnet-cu92mkl``
============  ==================

```

## Upgrade to the newest version

MXNet makes a major release every one or two months. In addition, it releases
nightly builds every day. Some toolkits or tutorials require the newest version.
This can be installed or upgraded using the `--pre` flag.

Install the nightly build MXNet with CUDA 9.2:

```bash
pip install --pre mxnet-cu92
```

or upgrade the version with `-U`:

```bash
pip install --pre -U mxnet-cu92
```


## Other installation options

If you want to install MXNet in a different way, such as with Scala frontend or
Docker, refer to
[MXNet installation](http://mxnet.incubator.apache.org/install/index.html) for
more details.
