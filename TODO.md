## Features

* Distributed training
  * General documentation: https://www.tensorflow.org/deploy/distributed
* Learning rate decay
* Character convolution embedder
  * Blocked by https://github.com/tensorflow/tensorflow/issues/11715
* Parallel encoder
  * i.e. encode the same input with multiple encoders and merge outputs and states
* Other validation metrics
* Self attention encoder
  * https://arxiv.org/abs/1706.03762

## Improvements

* Generalize to other dataset type
  * i.e. do not assume `tf.contrib.data.TextLineDataset` in the model's `_build_dataset` function
