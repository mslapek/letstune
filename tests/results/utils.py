from __future__ import annotations

from typing import Any

import letstune
from letstune.patch37 import dataclass


class OptimizerParams(letstune.Params):
    alpha: int
    beta: float
    gamma: str


class ModelParams(letstune.Params):
    optimizer: OptimizerParams
    zeta: float


class AnotherOptimizerParams(letstune.Params):
    gamma: str
    epsilon: float


class UnionModelParams(letstune.Params):
    optimizer: OptimizerParams | AnotherOptimizerParams
    zeta: float


@dataclass(frozen=True)
class SimpleCheckpoint:
    training_id: int


class SimpleCheckpointFactory:
    def get_checkpoint(self, training_id: int) -> Any:
        return SimpleCheckpoint(training_id)


@dataclass(frozen=True)
class EpochCheckpoint:
    training_id: int
    epoch_id: int


class EpochCheckpointFactory:
    def get_checkpoint(self, training_id: int, epoch_id: int) -> Any:
        return EpochCheckpoint(training_id, epoch_id)
