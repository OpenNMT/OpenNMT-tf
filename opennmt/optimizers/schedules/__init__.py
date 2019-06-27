"""Module defining learning rate schedules."""

from opennmt.optimizers.schedules.lr_schedules import CosineAnnealing
from opennmt.optimizers.schedules.lr_schedules import NoamDecay
from opennmt.optimizers.schedules.lr_schedules import RNMTPlusDecay
from opennmt.optimizers.schedules.lr_schedules import RsqrtDecay
from opennmt.optimizers.schedules.lr_schedules import ScheduleWrapper
from opennmt.optimizers.schedules.lr_schedules import make_learning_rate_schedule
