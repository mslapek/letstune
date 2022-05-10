import json
import pickle
from datetime import datetime, timezone
from typing import Any, Callable, Generic, Sequence, TypeVar

import letstune
from letstune import Params
from letstune.backend import repo
from letstune.backend.runner import log

T = TypeVar("T")

P = TypeVar("P", bound=Params)
Trainer = TypeVar("Trainer")
Task = TypeVar("Task")


class ErrorOccured:
    pass


class Runner(Generic[P, Trainer, Task]):
    passthrough_errors: bool = False
    logger: log.Logger = log.NULL_LOGGER
    _repository: repo.Repository
    _params_cls: type[P]
    _metric: letstune.Metric

    def __init__(
        self,
        dataset: Any,
        trainer: Trainer,
    ):
        self.__dataset = dataset
        self._trainer = clone_trainer(trainer)
        self._metric = trainer.metric  # type: ignore

    def _load_dataset(self) -> None:
        self._trainer.load_dataset(self.__dataset)  # type: ignore

    def _time(self) -> datetime:
        return datetime.now(timezone.utc)

    def _parse_params(self, params_json: str) -> P:
        return self._params_cls.from_json(json.loads(params_json))

    def _catch_error(
        self, training_id: int, f: Callable[..., T], *args: Any
    ) -> T | ErrorOccured:
        error_msg = None
        result: T | ErrorOccured = ErrorOccured()
        try:
            result = f(*args)
        except Exception as e:
            if self.passthrough_errors:
                raise
            error_msg = repr(e)

        if error_msg is not None:
            self._repository.set_error(training_id, error_msg)

        return result

    def _run_round(self) -> bool:
        trainings = self._repository.get_all_trainings()
        params = {t.training_id: t.params for t in trainings}
        tasks = self._get_next_tasks(trainings)
        tasks_present = len(tasks) != 0

        log = dict(
            event="round",
            tasks_count=len(tasks),
        )

        if tasks_present:
            self.logger.log(**log, sub_event="start")

        for t in tasks:
            p = self._parse_params(params[t.training_id])  # type: ignore
            self._run_task(t, p)

        if tasks_present:
            self.logger.log(**log, sub_event="end")

        return tasks_present

    def _get_next_tasks(self, trainings: Sequence[repo.Training]) -> list[Task]:
        raise NotImplementedError

    def _run_task(self, task: Task, params: P) -> None:
        raise NotImplementedError


def clone_trainer(t: T) -> T:
    return pickle.loads(pickle.dumps(t))  # type: ignore
