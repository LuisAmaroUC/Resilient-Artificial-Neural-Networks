## Setup

Install PyTorch C++ API

```sh
https://pytorch.org/cppdocs/installing.html
```

```sh
Download the datasets 
Fashion Mnist - https://github.com/pytorch/examples/tree/master/cpp/mnist
Mnist - http://yann.lecun.com/exdb/mnist/

$ cd mnist
$ mkdir build
$ cd build 
$ mkdir data                    // datasets in this directory
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
$ ./mnist                       
```


## How to Train and Test the Network

```sh
cd build
./trainMnist
./trainMnistDropout
```

```sh
cd build
./testMnist
./testMnistDropout
```

