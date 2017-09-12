## To add

* Other validation metrics compatible with `tf.metrics`
  * BLEU
  * ROUGE
  * Accuracy
  * F-score
* Convolutional sequence-to-sequence
  * https://arxiv.org/abs/1705.03122
* Positional encoding as in https://arxiv.org/abs/1706.03762

## To improve

* Write modules tests
* Expose alignment history after decoding

## To explore

* Integration with Docker or Kubernetes for distributed training
  * https://github.com/tensorflow/ecosystem

Also:

```
grep -r TODO *.py opennmt/
```
