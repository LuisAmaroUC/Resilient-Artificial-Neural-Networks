
## Overall

* Train the network (with the desired level of dropout or stimulated dropout)
* Run the test code

## Setup

### Training and Python Tests

Install python3, then use pip to install some necessary modules (if needed, use the `--user` to avoid a system installation). Namely:

* Install numpy

```sh
pip install numpy
```

* Install pytorch(not necessarily the GPU version)

```sh
pip install pytorch
```

* Install keras




## How to Train the Network

* Run:

```sh
python3 train_simple_cnn.py
```

The script has command line arguments that can be used to control the level of dropout. They can be listed by calling:

```sh
python3 train_simple_cnn.py --help
```

