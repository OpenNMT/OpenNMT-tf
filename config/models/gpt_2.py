import numpy as np
import opennmt as onmt
import tensorflow as tf

from opennmt.utils import misc


def gelu(x):
  """Gaussian Error Linear Unit activation function described in
  https://arxiv.org/abs/1606.08415.
  """
  return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


class GPT2Small(onmt.models.LanguageModel):
  """GPT-2 language model (small version) as described in:

  https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
  """

  def __init__(self):
    super(GPT2Small, self).__init__(
        decoder=onmt.decoders.SelfAttentionDecoderV2(
            num_layers=12,
            num_units=768,
            num_heads=12,
            ffn_inner_dim=3072,
            ffn_activation=gelu,
            position_encoder=onmt.layers.PositionEmbedder(maximum_position=1024),
            num_sources=0),
        embedding_size=768)

  def auto_config(self, num_devices=1):
    config = super(GPT2Small, self).auto_config(num_devices=num_devices)
    return misc.merge_dict(config, {
        "params": {
            "average_loss_in_time": True,
            "optimizer": "AdamOptimizer",
            "learning_rate": 2.5e-4,
            "decay_type": "cosine_annealing",
            "decay_params": {
                "max_step": 1000000,
                "warmup_steps": 2000,
            }
        },
        "train": {
            "bucket_width": 1,
            # Below options are from GPT-1.
            "batch_size": 64,
            "maximum_features_length": 512
        }
    })


def model():
  return GPT2Small()
