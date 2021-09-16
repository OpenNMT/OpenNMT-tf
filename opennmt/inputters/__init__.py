"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""

from opennmt.inputters.inputter import (
    ExampleInputter,
    ExampleInputterAdapter,
    Inputter,
    MixedInputter,
    MultiInputter,
    ParallelInputter,
)
from opennmt.inputters.record_inputter import (
    SequenceRecordInputter,
    create_sequence_records,
    write_sequence_record,
)
from opennmt.inputters.text_inputter import (
    CharConvEmbedder,
    CharEmbedder,
    CharRNNEmbedder,
    TextInputter,
    WordEmbedder,
    add_sequence_controls,
    load_pretrained_embeddings,
)
