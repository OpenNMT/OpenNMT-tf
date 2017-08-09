## Features

* Learning rate decay
* Character convolution embedder
  * Blocked by https://github.com/tensorflow/tensorflow/issues/11715
* Other validation metrics
  * BLEU
  * Accuracy

## Improvements

* Generalize to other dataset type
  * i.e. do not assume `tf.contrib.data.TextLineDataset` in the model's `_build_dataset` function
* Write modules tests

Also:

```
grep -r TODO opennmt/
```
