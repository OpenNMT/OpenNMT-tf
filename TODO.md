## Features

* Learning rate decay
* Other validation metrics
  * BLEU
  * Accuracy
* Self attention decoder
  * https://arxiv.org/abs/1706.03762
* Convolutional decoder
  * https://arxiv.org/abs/1705.03122

## Improvements

* Generalize to other dataset type
  * i.e. do not assume `tf.contrib.data.TextLineDataset` in the model's `_build_dataset` function
* Write modules tests
* Validate Python 2 & 3 compatibility

Also:

```
grep -r TODO opennmt/
```
