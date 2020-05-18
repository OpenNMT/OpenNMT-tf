"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""

from opennmt.inputters.inputter import ExampleInputter
from opennmt.inputters.inputter import ExampleInputterAdapter
from opennmt.inputters.inputter import Inputter
from opennmt.inputters.inputter import MixedInputter
from opennmt.inputters.inputter import MultiInputter
from opennmt.inputters.inputter import ParallelInputter

from opennmt.inputters.record_inputter import SequenceRecordInputter
from opennmt.inputters.record_inputter import create_sequence_records
from opennmt.inputters.record_inputter import write_sequence_record

from opennmt.inputters.text_inputter import CharEmbedder
from opennmt.inputters.text_inputter import CharConvEmbedder
from opennmt.inputters.text_inputter import CharRNNEmbedder
from opennmt.inputters.text_inputter import TextInputter
from opennmt.inputters.text_inputter import WordEmbedder
from opennmt.inputters.text_inputter import add_sequence_controls
from opennmt.inputters.text_inputter import load_pretrained_embeddings
