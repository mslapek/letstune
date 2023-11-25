import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence, TypeVar, overload
from uuid import uuid4

import numpy as np

import letstune.results.epoch
import letstune.results.simple
from letstune import results
from letstune.backend import repo
from letstune.backend.repo.sqlite import SqliteRepo
from letstune.backend.runner import log
from letstune.backend.runner.checkpoint import CheckpointFactory
from letstune.backend.runner.epoch import EpochRunner
from letstune.backend.runner.simple import SimpleRunner
from letstune.backend.scheduler.epoch import Config
from letstune.params import Params
from letstune.rand import RandomParamsGenerator
from letstune.trainer import EpochTrainer, SimpleTrainer

__all__ = ["tune"]

P = TypeVar("P", bound=Params)


@overload
def tune(
    trainer: EpochTrainer[P],
    params_number: int,
    *,
    dataset: Any = None,
    rounds: Sequence[timedelta] | dict[str, Any] | None = None,
    results_dir: Path | str | None = None,
    passthrough_errors: bool = False,
    verbose: bool = True,
) -> letstune.results.epoch.TuningResults[P]:
    ...


@overload
def tune(
    trainer: SimpleTrainer[P],
    params_number: int,
    *,
    dataset: Any = None,
    results_dir: Path | str | None = None,
    passthrough_errors: bool = False,
    verbose: bool = True,
) -> letstune.results.simple.TuningResults[P]:
    ...


def tune(
    trainer: EpochTrainer[P] | SimpleTrainer[P],
    params_number: int,
    *,
    dataset: Any = None,
    rounds: Sequence[timedelta] | dict[str, Any] | None = None,
    results_dir: Path | str | None = None,
    passthrough_errors: bool = False,
    verbose: bool = True,
) -> results.epoch.TuningResults[P] | results.simple.TuningResults[P]:
    """
    Do tuning on this computer (in this Python interpreter) with
    given ``trainer`` and return tuning results.

    **Params number**

    Parameter ``params_number`` declares target number of parameters and trainings.

    §

    If ``results_dir`` already has *k* parameters,
    then ``params_number`` - *k* will be generated.

    True number of trainings can be less than
    ``params_number`` due to parameter deduplication.

    **Dataset**

    Parameter ``dataset`` is passed-through
    to :meth:`letstune.SimpleTrainer.load_dataset`
    and :meth:`letstune.EpochTrainer.load_dataset`.

    **Rounds**

    Only for :class:`letstune.EpochTrainer`.

    Parameter ``rounds`` sets durations of rounds.

    `rounds` can be set to a :class:`list` of :class:`datetime.timedelta`::

        rounds = [
            timedelta(minutes=1),
            timedelta(minutes=2),
            timedelta(minutes=4),
        ]

    ``rounds`` also accepts a :class:`dict`::

        rounds = {
            'round_durations': [
                timedelta(minutes=1),
                timedelta(minutes=2),
                timedelta(minutes=4),
            ],
        }

    The dict accepts ``trainings_reduction`` key::

        rounds = {
            'round_durations': [
                timedelta(minutes=1),
                timedelta(minutes=2),
                timedelta(minutes=4),
            ],
            'trainings_reduction': 2.0,
        }

    To the next round are promoted top ``ceil(trainings_number / trainings_reduction)``
    trainings.

    The default value for ``trainings_reduction`` is ``4.0``.

    **Results directory**

    Parameter ``results_dir`` sets the directory,
    where tuning results will be stored.

    Directory contents:

    * ``letstune.db`` is an SQLite database with training metadata,
    * ``checkpoints`` is a directory with serialized trained models.

    If ``results_dir`` is not given, then directory ``ltruns`` is created, which
    will contain the results.

    **Passthrough errors**

    When ``passthrough_errors`` is ``True``, then all exceptions are *not* caught.

    Otherwise, they will be noted in tuning results and training will continue.
    That's the default option.

    **Verbose**

    If ``verbose`` is ``True``, then explanatory log messages will be printed
    to the standard output. That's the default option.

    **Returned tuning results**

    The type of returned tuning results depends on the type of ``trainer`` parameter.

    When given :class:`letstune.SimpleTrainer` [``P``],
    then :class:`letstune.results.simple.TuningResults` [``P``]
    will be returned.

    When given :class:`letstune.EpochTrainer` [``P``],
    then :class:`letstune.results.epoch.TuningResults` [``P``]
    will be returned.

    **Facade**

    This function is a facade to the *letstune* system -
    it is usually a good choice.

    If you need further customization, consider to
    copy this function and modify it according to your needs.
    """
    if results_dir is None:
        results_dir = Path() / "ltruns" / str(uuid4())
        results_dir = results_dir.resolve()
        if verbose:
            print(f"Creating new results_dir at {results_dir}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    repository = SqliteRepo(results_dir / "letstune.db")
    try:
        checkpoint_factory = CheckpointFactory(results_dir / "checkpoints")

        # params_cls can be a generator, not necessary a class
        generator: RandomParamsGenerator[P] = trainer.params_cls  # type: ignore
        params_cls = _fill_with_params(repository, generator, params_number)

        if isinstance(trainer, EpochTrainer):
            if rounds is None:
                raise ValueError("epoch trainings require rounds")
            config = _rounds_to_config(rounds)
            runner: EpochRunner[P] | SimpleRunner[P] = EpochRunner(
                repository=repository,
                checkpoint_factory=checkpoint_factory,
                trainer=trainer,
                config=config,
                dataset=dataset,
                params_cls=params_cls,
            )
        else:
            if rounds is not None:
                raise ValueError("simple trainings do not take rounds")
            config = None
            runner = SimpleRunner(
                repository=repository,
                checkpoint_factory=checkpoint_factory,
                trainer=trainer,
                dataset=dataset,
                params_cls=params_cls,
            )

        runner.logger = log.ConsoleLogger() if verbose else log.NULL_LOGGER
        runner.passthrough_errors = passthrough_errors
        runner.run()

        trainings = repository.get_all_trainings()
    finally:
        repository.close()

    if isinstance(trainer, EpochTrainer):
        assert config is not None

        return letstune.results.epoch.build(
            metric=trainer.metric,
            checkpoint_factory=checkpoint_factory,
            params_cls=params_cls,
            trainings=trainings,
            config=config,
        )
    else:
        return letstune.results.simple.build(
            metric=trainer.metric,
            checkpoint_factory=checkpoint_factory,
            params_cls=params_cls,
            trainings=trainings,
        )


def _deduplicate_params(params: Sequence[P]) -> list[P]:
    return list(dict.fromkeys(params))


def _fill_with_params(
    repository: repo.Repository,
    trainer: RandomParamsGenerator[P],
    params_number: int,
) -> type[P]:
    rng = np.random.default_rng(42)
    params = [trainer.get_random_params(rng) for _ in range(params_number)]
    params = _deduplicate_params(params)

    n = len(repository.get_all_trainings())
    for i in range(n, len(params)):
        p = params[i]
        repository.add_training(i, json.dumps(p.to_json()))

    return type(params[0])


def _rounds_to_config(cfg: Sequence[timedelta] | dict[str, Any]) -> Config:
    if not isinstance(cfg, dict):
        round_durations = _parse_round_durations(cfg, expect_dict=False)
        trainings_reduction = 4.0
    else:
        if "round_durations" not in cfg:
            raise ValueError(
                f"expected rounds to be a dict with 'round_durations', got {cfg!r}"
            )

        unknown_keys = set(cfg) - {"round_durations", "trainings_reduction"}
        if len(unknown_keys) > 0:
            unknown_key = next(iter(unknown_keys))
            raise ValueError(f"rounds got an unexpected key {unknown_key!r}")

        round_durations = cfg["round_durations"]
        trainings_reduction = cfg.get("trainings_reduction", 4.0)

        if not isinstance(trainings_reduction, (float, int)):
            raise TypeError(
                "expected rounds to be a dict where "
                "rounds['trainings_reduction'] is a float",
            )

        round_durations = _parse_round_durations(round_durations, expect_dict=True)

    return Config(
        round_durations=round_durations,
        trainings_reduction=trainings_reduction,
    )


def _parse_round_durations(
    round_durations: Any, expect_dict: bool
) -> tuple[timedelta, ...]:
    error_message = "expected rounds to be "

    if expect_dict:
        error_message += "a dict where rounds['round_durations'] is "

    error_message += "a sequence of timedelta"

    if not isinstance(round_durations, Sequence):
        raise TypeError(error_message)

    round_durations = tuple(round_durations)

    if not all(isinstance(d, timedelta) for d in round_durations):
        raise TypeError(error_message)

    return round_durations
