"""Module defining learning rate schedules."""

from opennmt.schedules.lr_schedules import (
    CosineAnnealing,
    InvSqrtDecay,
    NoamDecay,
    RNMTPlusDecay,
    RsqrtDecay,
    ScheduleWrapper,
    make_learning_rate_schedule,
    register_learning_rate_schedule,
)
