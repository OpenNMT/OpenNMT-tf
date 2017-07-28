## Features

* Distributed training
  * General documentation: https://www.tensorflow.org/deploy/distributed
* Learning rate decay
* Character convolution embedder
  * Blocked by https://github.com/tensorflow/tensorflow/issues/11715
* Other validation metrics

## Improvements

* Generalize to other dataset type
  * i.e. do not assume `tf.contrib.data.TextLineDataset` in the model's `_build_dataset` function
