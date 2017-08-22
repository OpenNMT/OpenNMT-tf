## To add

* Other validation metrics
  * BLEU
  * ROUGE
  * Accuracy
  * F-score
* Convolutional sequence-to-sequence
  * https://arxiv.org/abs/1705.03122
* Multi source embedder
* Positional encoding as in https://arxiv.org/abs/1706.03762

## To improve

* Write modules tests
* Gather default option values
* Use in-graph Viterbi decoding
  * https://github.com/tensorflow/tensorflow/pull/12056
* Set TensorFlow target version
  * Waiting for: https://github.com/tensorflow/tensorflow/commit/865b92da01582081576728504bedf932b367b26c

## To explore

* Model deployement and serving
  * Inference from the C++ API
  * [TensorFlow Serving](https://www.tensorflow.org/serving/)
* Integration with Docker or Kubernetes for distributed training
  * https://github.com/tensorflow/ecosystem

Also:

```
grep -r TODO *.py opennmt/
```
