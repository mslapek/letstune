from typing import Any, Generic, Protocol, Sequence, TypeVar

from letstune.backend import repo
from letstune.backend.runner import _base
from letstune.backend.scheduler.simple import Task, get_next_tasks
from letstune.params import Params
from letstune.trainer import SimpleTrainer

__all__ = ["CheckpointFactory", "SimpleRunner"]

P = TypeVar("P", bound=Params)


class CheckpointFactory(Protocol):
    """Type representing checkpoint factory."""

    def get_checkpoint(self, training_id: int) -> Any:
        ...


class SimpleRunner(_base.Runner[P, SimpleTrainer[P], Task], Generic[P]):
    def __init__(
        self,
        *,
        repository: repo.Repository,
        checkpoint_factory: CheckpointFactory,
        trainer: SimpleTrainer[P],
        dataset: Any,
        params_cls: type[P],
    ):
        super().__init__(
            dataset=dataset,
            trainer=trainer,
        )

        self._params_cls = params_cls
        self._repository = repository
        self._checkpoint_factory = checkpoint_factory

    def run(self) -> None:
        self._load_dataset()
        self._run_round()

    def _run_task(self, task: Task, params: P) -> None:
        log = dict(event="train", training_id=task.training_id)
        self.logger.log(**log, sub_event="start")

        start_time = self._time()
        result = self._catch_error(task.training_id, self._trainer.train, params)
        end_time = self._time()

        if isinstance(result, _base.ErrorOccured):
            self.logger.log(**log, sub_event="stop", status="failed")
            return

        model, metric_values = result

        chk = self._checkpoint_factory.get_checkpoint(task.training_id)
        self._trainer.save(chk, model)
        self._repository.add_epoch(
            task.training_id,
            repo.EpochStats(
                epoch_id=0,
                metric_values=metric_values,
                start_time=start_time,
                end_time=end_time,
            ),
        )

        self.logger.log(
            **log,
            sub_event="stop",
            metric_value=metric_values.get(self._metric.name),
        )

    def _get_next_tasks(self, trainings: Sequence[repo.Training]) -> list[Task]:
        return get_next_tasks(trainings)
