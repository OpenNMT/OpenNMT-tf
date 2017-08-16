## Features

* Learning rate decay
* Other validation metrics
  * BLEU
  * Accuracy
* Convolutional decoder
  * https://arxiv.org/abs/1705.03122

## Improvements

* Write modules tests
* Validate Python 2 & 3 compatibility
* Use in-graph Viterbi decoding
  * https://github.com/tensorflow/tensorflow/pull/12056
* Set TensorFlow target version. Waiting for:
  * https://github.com/tensorflow/tensorflow/commit/865b92da01582081576728504bedf932b367b26c

Also:

```
grep -r TODO *.py opennmt/
```
