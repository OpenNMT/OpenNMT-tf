"""Module defining custom optimizers."""

from opennmt.utils.compat import is_tf2

if not is_tf2():
  from opennmt.optimizers.adafactor import AdafactorOptimizer
  from opennmt.optimizers.adafactor import get_optimizer_from_params \
      as get_adafactor_optimizer_from_params

  from opennmt.optimizers.mixed_precision_wrapper import MixedPrecisionOptimizerWrapper

from opennmt.optimizers.lr_schedules import CosineAnnealing
from opennmt.optimizers.lr_schedules import NoamDecay
from opennmt.optimizers.lr_schedules import RNMTPlusDecay
from opennmt.optimizers.lr_schedules import RsqrtDecay
from opennmt.optimizers.lr_schedules import ScheduleWrapper
