# MATRIX CAPSULES EM-Tensorflow

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0)
![completion](https://img.shields.io/badge/completion%20state-70%25-blue.svg?style=plastic)

A Tensorflow implementation of CapsNet based on paper [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)

> **Status:**
> 1. The code is working on MNIST with configuration:  A=32, B=8, C=16, D=16 on a NVIDIA P40 GPU. The definitions of A-D can be referred to the paper.
> 2. Some modification and optimization is implemented to prompt the numerical stability of GMM. Specific explanations can be found in the code.
> 3. With configuration A=32, B=4, D=4, D=4, each iteration of training takes around 0.7s on a GPU GTX 1080 and CPU i7-5820K CPU @ 3.30GHz. The final test error on MNIST is around 50%.

> **To Do List:**
> 1. The original configuration: A=B=C=D=32, is not supported on a Tesla P40 GPU, due to the limit of memory. Some optimization on implementation structure is required.
> 2. Coordinate Addition technique is about to be implemented.
> 3. Spread Loss is about to be implemented.
> 4. Experiments on smallNORB as in paper is about to be casted.

Any questions and comments to the code and the original algorithms are welcomed!!! My email: zhangsuofei at njupt.edu.cn

## Requirements
- Python >= 3.4
- Numpy
- Tensorflow >= 1.2.0

## Usage
**Step 1.**
Clone this repository with ``git``.

```
$ git clone https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow.git
$ cd Matrix-Capsules-EM-Tensorflow
```

**Step 2.**
Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), ``mv`` and extract it into ``data/mnist`` directory.(Be careful the backslash appeared around the curly braces when you copy the ``wget `` command to your terminal, remove it)

```
$ mkdir -p data/mnist
$ wget -c -P data/mnist http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
$ gunzip data/mnist/*.gz
```

** Step 3.**
Start the training:
```
$ python3 train.py
```

** Step 4.**
View the status of training:
```
$ tensorboard --logdir=./logdir
``` 
Open a chrome browser, visit the site: http://127.0.0.1:6006/

** Step 5.**
Start the test on MNIST:
```
$ python3 eval.py
```

** Step 6.**
View the status of test:
```
$ tensorboard --logdir=./test_logdir
```
Open a chrome browser, visit the site: http://127.0.0.1:6006/

### Reference
- [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow): the implementation of Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)