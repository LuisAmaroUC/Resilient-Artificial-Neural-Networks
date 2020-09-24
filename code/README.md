
## Overall

* Train the network (with the desired level of dropout or stimulated dropout)
* Run the test code

## Setup

### Training and Tests

* Install pytorch(not necessarily the GPU version)

```sh
https://pytorch.org/cppdocs/installing.html
```


## How to Train the Network
* Go to build directory:

```sh
make
```
* Train:

```sh
./trainMnist
./trainMnistDropout
./trainStimulatedDropout
```
* Test:
```sh
./testMnist
./testMnistDropout
./testStimulatedDropout
```

