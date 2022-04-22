import pytest

import letstune
from letstune.results import epoch, simple

from . import utils
from .utils import ModelParams, OptimizerParams

METRIC = letstune.Metric("accuracy")


def test_simple_tuning_must_be_non_empty() -> None:
    builder: simple.Builder[ModelParams] = simple.Builder(
        metric=METRIC, checkpoint_factory=utils.SimpleCheckpointFactory()
    )

    with pytest.raises(ValueError, match="tuning must have at least one training"):
        _ = builder.build()


def test_epoch_tuning_must_be_non_empty() -> None:
    builder: epoch.Builder[ModelParams] = epoch.Builder(
        metric=METRIC,
        checkpoint_factory=utils.EpochCheckpointFactory(),
        round_assigner=utils.RoundAssigner(dict()),
    )

    with pytest.raises(ValueError, match="tuning must have at least one training"):
        _ = builder.build()


def test_epoch_training_must_have_epochs() -> None:
    builder: epoch.Builder[ModelParams] = epoch.Builder(
        metric=METRIC,
        checkpoint_factory=utils.EpochCheckpointFactory(),
        round_assigner=utils.RoundAssigner(dict()),
    )

    epoch_builder = builder.add_training(
        ModelParams(
            optimizer=OptimizerParams(alpha=1, beta=3.3, gamma="hello"), zeta=1111.11
        )
    )

    with pytest.raises(ValueError, match="training must have at least one epoch"):
        epoch_builder.build()
