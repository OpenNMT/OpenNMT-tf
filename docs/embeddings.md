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

The format of the embedding file and the options are described in the [`opennmt.inputters.load_pretrained_embeddings`](https://opennmt.net/OpenNMT-tf/package/opennmt.inputters.load_pretrained_embeddings.html) function.

Pretrained embeddings are only loaded when initializing the model weights. If you continue the training with from a checkpoint, this configuration is ignored and the embeddings values are loaded from the checkpoint.

## Sharing

The following model types are configured to share embeddings by default:

* `TransformerBaseSharedEmbeddings`
* `TransformerBigSharedEmbeddings`

For custom models, see the values in [`EmbeddingsSharingLevel`](https://opennmt.net/OpenNMT-tf/package/opennmt.models.EmbeddingsSharingLevel) that can be passed to the [`SequenceToSequence`](https://opennmt.net/OpenNMT-tf/package/opennmt.models.SequenceToSequence.html) constructor.
