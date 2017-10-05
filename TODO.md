## To add

* Other validation metrics compatible with `tf.metrics`
  * BLEU
  * ROUGE
  * Accuracy
  * F-score
* Convolutional sequence-to-sequence
  * https://arxiv.org/abs/1705.03122
* Positional encoding with sinusoid
  * https://arxiv.org/abs/1706.03762
* Multi-attention in the `AttentionalRNNDecoder`
* Documentation generation from docstrings
* Im2Text
  * https://github.com/OpenNMT/Im2Text

## To improve

* Write more tests
* Expose alignment history after decoding
  * https://github.com/tensorflow/tensorflow/issues/13154
* More frequent model export
  * https://github.com/tensorflow/tensorflow/commit/9e6ee10fa85053cf4c237970a58f4ca2fe2c497e

Also:

```
grep -r TODO *.py opennmt/
```
