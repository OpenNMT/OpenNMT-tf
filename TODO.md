## To add

* Other validation metrics
  * BLEU
  * Accuracy
* Convolutional sequence-to-sequence
  * https://arxiv.org/abs/1705.03122
* Words per second summary

## To improve

* Write modules tests
* Use in-graph Viterbi decoding
  * https://github.com/tensorflow/tensorflow/pull/12056
* Set TensorFlow target version. Waiting for:
  * https://github.com/tensorflow/tensorflow/commit/865b92da01582081576728504bedf932b367b26c

## To explore

* Model deployement (Is it just about exporting an inference graph?)
  * Inference from the C++ API
  * [TensorFlow Serving](https://www.tensorflow.org/serving/)

Also:

```
grep -r TODO *.py opennmt/
```
