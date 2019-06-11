"""Module defining custom optimizers."""

from opennmt.optimizers.lr_schedules import CosineAnnealing
from opennmt.optimizers.lr_schedules import NoamDecay
from opennmt.optimizers.lr_schedules import RNMTPlusDecay
from opennmt.optimizers.lr_schedules import RsqrtDecay
from opennmt.optimizers.lr_schedules import ScheduleWrapper
