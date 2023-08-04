# Installation

## Requirements

OpenNMT-tf requires:

* Python 3.7 or above
* TensorFlow 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, or 2.13

For GPU support, please read the [TensorFlow documentation](https://www.tensorflow.org/install/gpu) for additional software and hardware requirements.

## pip (recommended)

Each tagged version of the project is automatically pushed to [PyPI](https://pypi.org/project/OpenNMT-tf/). We recommend installing this package with `pip` >= 19.3:

```bash
pip install --upgrade pip
pip install OpenNMT-tf
```

To confirm that the package is correctly installed, run:

```bash
onmt-main -v
```
