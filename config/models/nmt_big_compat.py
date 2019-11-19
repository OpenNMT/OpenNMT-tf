"""A Luong-style sequence to sequence model that is compatible with OpenNMT-tf v1
"NMTBig" model.
"""

import opennmt as onmt

class NMTBig(onmt.models.SequenceToSequence):
  def __init__(self):
    super(NMTBig, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
        encoder=onmt.encoders.RNNEncoder(
            num_layers=4,
            num_units=512,
            bidirectional=True,
            dropout=0.3,
            reducer=onmt.layers.ConcatReducer()),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=4,
            num_units=1024,
            bridge_class=onmt.layers.CopyBridge,
            attention_layer_activation=None,
            dropout=0.3))

  def auto_config(self, num_replicas=1):
    config = super(NMTBig, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "optimizer": "Adam",
            "learning_rate": 0.0002
        },
        "train": {
            "batch_size": 64,
            "maximum_features_length": 80,
            "maximum_labels_length": 80
        }
    })

model = NMTBig
