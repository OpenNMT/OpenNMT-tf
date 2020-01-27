# Alignments

Target to source alignments are returned as part of the translation API. They are also used by some decoding features such as `replace_unknown_target`.

To improve their quality, OpenNMT-tf supports [guided alignment](https://arxiv.org/abs/1607.01628) training. It is a supervised approach to constrain one or more attention heads by alignments produced by external tools such as [fast_align](https://github.com/clab/fast_align).

Guided alignment should be configured in the YAML configuration file:

```yaml
data:
  train_alignments: train-alignment.txt

params:
  guided_alignment_type: ce
```

where `train-alignment.txt` is a text file that uses the "Pharaoh" alignment format, e.g.:

```text
0-0 1-1 2-4 3-2 4-3 5-5 6-6
0-0 1-1 2-2 2-3 3-4 4-5
0-0 1-2 2-1 3-3 4-4 5-5
```

where a pair `i-j` indicates that the `i`th word of the source sentence is aligned with the `j`th word of the target sentence (zero-indexed).
