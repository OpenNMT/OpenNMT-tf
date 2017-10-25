## To add

* Other validation metrics compatible with `tf.metrics`
  * BLEU
  * ROUGE
* Convolutional sequence-to-sequence
  * https://arxiv.org/abs/1705.03122
* Positional encoding with sinusoid
  * https://arxiv.org/abs/1706.03762
* Documentation generation from docstrings
* Im2Text
  * https://github.com/OpenNMT/Im2Text

## To improve

* Write more tests
* Expose alignment history after decoding
  * https://github.com/tensorflow/tensorflow/issues/13154

Also:

```
grep -r TODO *.py opennmt/
```
