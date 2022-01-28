"""Defines a dual source Transformer architecture with serial attention layers
and parameter sharing between the encoders.

See for example https://arxiv.org/pdf/1809.00188.pdf.

The YAML configuration file should look like this:

data:
  train_features_file:
    - source_1.txt
    - source_2.txt
  train_labels_file: target.txt
  source_1_vocabulary: source_1_vocab.txt
  source_2_vocabulary: source_2_vocab.txt
  target_vocabulary: target_vocab.txt
"""

import opennmt


class DualSourceTransformer(opennmt.models.Transformer):
    def __init__(self):
        super().__init__(
            source_inputter=opennmt.inputters.ParallelInputter(
                [
                    opennmt.inputters.WordEmbedder(embedding_size=512),
                    opennmt.inputters.WordEmbedder(embedding_size=512),
                ]
            ),
            target_inputter=opennmt.inputters.WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            share_encoders=True,
        )


model = DualSourceTransformer
