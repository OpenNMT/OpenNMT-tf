"""Module defining learning rate schedules."""

from opennmt.schedules.lr_schedules import CosineAnnealing
from opennmt.schedules.lr_schedules import NoamDecay
from opennmt.schedules.lr_schedules import RNMTPlusDecay
from opennmt.schedules.lr_schedules import RsqrtDecay
from opennmt.schedules.lr_schedules import ScheduleWrapper
from opennmt.schedules.lr_schedules import make_learning_rate_schedule
from opennmt.schedules.lr_schedules import register_learning_rate_schedule
