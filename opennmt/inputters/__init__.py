"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""

from opennmt.inputters.inputter import ParallelInputter, MixedInputter
from opennmt.inputters.text_inputter import WordEmbedder
from opennmt.inputters.text_inputter import CharConvEmbedder
from opennmt.inputters.text_inputter import CharRNNEmbedder
from opennmt.inputters.record_inputter import SequenceRecordInputter
