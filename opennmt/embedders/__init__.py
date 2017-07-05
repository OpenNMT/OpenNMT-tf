"""Module defining embedders.

Embedders implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""

from opennmt.embedders.embedder import MixedEmbedder
from opennmt.embedders.text_embedder import WordEmbedder
