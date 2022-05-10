from datetime import timedelta
from typing import Any, Generic, Protocol, Sequence, TypeVar

from letstune.backend import repo
from letstune.backend.runner import _base
from letstune.backend.scheduler.epoch import Config, Task, get_next_tasks
from letstune.params import Params
from letstune.trainer import EpochTrainer

P = TypeVar("P", bound=Params)


class CheckpointFactory(Protocol):
    """Type representing checkpoint factory."""

    def get_checkpoint(self, training_id: int, epoch_id: int) -> Any:
        ...


class EpochRunner(_base.Runner[P, EpochTrainer[P], Task], Generic[P]):
    def __init__(
        self,
        *,
        repository: repo.Repository,
        checkpoint_factory: CheckpointFactory,
        trainer: EpochTrainer[P],
        config: Config,
        dataset: Any,
        params_cls: type[P],
    ):
        super().__init__(
            dataset=dataset,
            trainer=trainer,
        )

        self._config = config

        self._params_cls = params_cls
        self._repository = repository
        self._checkpoint_factory = checkpoint_factory

    def run(self) -> None:
        self._load_dataset()
        while self._run_round():
            pass

    def _run_task(self, task: Task, params: P) -> None:
        duration = timedelta()
        next_epoch = task.next_epoch

        if next_epoch == 0:
            result = self._catch_error(
                task.training_id, self._trainer.create_model, params
            )
        else:
            chk = self._checkpoint_factory.get_checkpoint(
                task.training_id, next_epoch - 1
            )
            result = self._catch_error(
                task.training_id, self._trainer.load, chk, params
            )

        log = dict(
            event="training",
            training_id=task.training_id,
            planned_duration=task.duration,
        )

        if isinstance(result, _base.ErrorOccured):
            self.logger.log(**log, sub_event="start", status="failure")
            return

        self.logger.log(**log, sub_event="start")

        while duration < task.duration:
            epoch_duration = self._train_epoch(task.training_id, next_epoch)
            if epoch_duration is None:
                self.logger.log(**log, sub_event="end", status="failure")
                return

            duration += epoch_duration
            next_epoch += 1

        self.logger.log(**log, sub_event="end")

    def _train_epoch(self, training_id: int, epoch: int) -> timedelta | None:
        log = dict(
            event="epoch",
            training_id=training_id,
            epoch=epoch,
        )

        self.logger.log(**log, sub_event="start")
        start_time = self._time()
        metric_values = self._catch_error(training_id, self._trainer.train_epoch, epoch)
        end_time = self._time()

        if isinstance(metric_values, _base.ErrorOccured):
            self.logger.log(**log, sub_event="end", status="failure")
            return None

        chk = self._checkpoint_factory.get_checkpoint(training_id, epoch)
        self._trainer.save(chk)
        self._repository.add_epoch(
            training_id,
            repo.EpochStats(
                epoch_id=epoch,
                metric_values=metric_values,
                start_time=start_time,
                end_time=end_time,
            ),
        )

        self.logger.log(
            **log,
            sub_event="end",
            metric_value=metric_values.get(self._metric.name),
        )

        return end_time - start_time

    def _get_next_tasks(self, trainings: Sequence[repo.Training]) -> list[Task]:
        return get_next_tasks(self._config, self._metric, trainings)
