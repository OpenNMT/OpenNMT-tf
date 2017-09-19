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
* Multi-source inputter
* Multi-attention in the `AttentionalRNNDecoder`
* Documentation generation from docstrings
* Im2Text
  * https://github.com/OpenNMT/Im2Text
* `setup.py`
* Pylint configuration

## To improve

* Write modules tests
* Expose alignment history after decoding
  * https://github.com/tensorflow/tensorflow/issues/13154

Also:

```
grep -r TODO *.py opennmt/
```
