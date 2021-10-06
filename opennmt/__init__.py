"""OpenNMT module."""

from opennmt.version import __version__, _check_tf_version

_check_tf_version()

from opennmt.config import convert_to_v2_config, load_config, load_model, merge_config
from opennmt.constants import (
    END_OF_SENTENCE_ID,
    END_OF_SENTENCE_TOKEN,
    PADDING_ID,
    PADDING_TOKEN,
    START_OF_SENTENCE_ID,
    START_OF_SENTENCE_TOKEN,
    UNKNOWN_TOKEN,
)
from opennmt.runner import Runner
