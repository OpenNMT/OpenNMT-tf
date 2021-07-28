"""This module exposes classes and functions that help creating and processing data."""

from opennmt.data.dataset import (
    batch_dataset,
    batch_sequence_dataset,
    filter_examples_by_length,
    filter_irregular_batches,
    get_dataset_size,
    inference_pipeline,
    make_cardinality_multiple_of,
    random_shard,
    shuffle_dataset,
    training_pipeline,
)
from opennmt.data.noise import (
    Noise,
    WordDropout,
    WordNoiser,
    WordOmission,
    WordPermutation,
    WordReplacement,
)
from opennmt.data.text import (
    alignment_matrix_from_pharaoh,
    tokens_to_chars,
    tokens_to_words,
)
from opennmt.data.vocab import Vocab, create_lookup_tables
