"""Data manipulation module."""

from opennmt.data.dataset import batch_dataset
from opennmt.data.dataset import batch_sequence_dataset
from opennmt.data.dataset import filter_examples_by_length
from opennmt.data.dataset import filter_irregular_batches
from opennmt.data.dataset import get_dataset_size
from opennmt.data.dataset import inference_pipeline
from opennmt.data.dataset import random_shard
from opennmt.data.dataset import shuffle_dataset
from opennmt.data.dataset import training_pipeline

from opennmt.data.noise import Noise
from opennmt.data.noise import WordDropout
from opennmt.data.noise import WordNoiser
from opennmt.data.noise import WordOmission
from opennmt.data.noise import WordPermutation
from opennmt.data.noise import WordReplacement

from opennmt.data.text import alignment_matrix_from_pharaoh
from opennmt.data.text import tokens_to_chars
from opennmt.data.text import tokens_to_words

from opennmt.data.vocab import Vocab
