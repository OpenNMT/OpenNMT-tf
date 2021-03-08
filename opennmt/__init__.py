"""OpenNMT module."""

from opennmt.version import __version__, _check_tf_version

_check_tf_version()

from opennmt.config import convert_to_v2_config
from opennmt.config import load_config
from opennmt.config import load_model

from opennmt.constants import END_OF_SENTENCE_ID
from opennmt.constants import END_OF_SENTENCE_TOKEN
from opennmt.constants import PADDING_ID
from opennmt.constants import PADDING_TOKEN
from opennmt.constants import START_OF_SENTENCE_ID
from opennmt.constants import START_OF_SENTENCE_TOKEN
from opennmt.constants import UNKNOWN_TOKEN

from opennmt.runner import Runner
