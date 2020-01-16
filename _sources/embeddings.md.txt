# Embeddings

## Pretrained

Pretrained embeddings can be configured in the `data` section of the YAML configuration, e.g.:

```yaml
data:
  source_embedding:
    path: data/glove/glove-100000.txt
    with_header: True
    case_insensitive: True
    trainable: False

  # target_embedding: ...
```

The format of the embedding file and the options are described in the [load_pretrained_embeddings](https://opennmt.net/OpenNMT-tf/package/opennmt.inputters.load_pretrained_embeddings.html) function.

## Sharing

`SequenceToSequence` models take a `share_embeddings` argument in the constructor that accepts a [`EmbeddingsSharingLevel`](https://opennmt.net/OpenNMT-tf/package/opennmt.models.EmbeddingsSharingLevel) value to enable different level of embeddings sharing.

See for example the custom model definition [`transformer_shared_embedding.py`](https://github.com/OpenNMT/OpenNMT-tf/blob/master/config/models/transformer_shared_embedding.py) that shares all embeddings, including the output softmax weights, for a Transformer model.
