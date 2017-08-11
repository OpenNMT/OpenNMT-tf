## Features

* Learning rate decay
* Other validation metrics
  * BLEU
  * Accuracy
* Convolutional decoder
  * https://arxiv.org/abs/1705.03122
* Generic beam search
  * Maybe https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/beam_search.py

## Improvements

* Generalize to other dataset type
  * i.e. do not assume `tf.contrib.data.TextLineDataset` in the model's `_build_dataset` function
* Write modules tests
* Validate Python 2 & 3 compatibility
* Use in-graph Viterbi decoding
  * https://github.com/tensorflow/tensorflow/pull/12056

Also:

```
grep -r TODO opennmt/
```
