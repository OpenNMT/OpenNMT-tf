# Installation

## Requirements

OpenNMT-tf requires:

* Python 3.6 or above
* TensorFlow 2.4, 2.5, 2.6, or 2.7

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
