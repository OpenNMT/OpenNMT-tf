"""Backward compatibility file, see :mod:`opennmt.optimizers.adafactor` instead."""

# pylint: disable=unused-import
from opennmt.optimizers.adafactor import AdafactorOptimizer
from opennmt.optimizers.adafactor import adafactor_decay_rate_adam
from opennmt.optimizers.adafactor import adafactor_decay_rate_pow
from opennmt.optimizers.adafactor import get_optimizer_from_params
