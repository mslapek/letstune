from datetime import timedelta

from letstune.backend.scheduler.epoch import Config, Task, _get_next_tasks, _Training

METRIC_VALUE = 0.0
# Algorithm doesn't need metric value


def test_no_round_durations() -> None:
    tasks = _get_next_tasks(
        Config(round_durations=tuple()),
        [],
    )

    assert len(tasks) == 0


def test_no_trainings() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [],
    )

    assert len(tasks) == 0


def test_all_new() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(),
                next_epoch=0,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(),
                next_epoch=0,
                metric_value=METRIC_VALUE,
            ),
        ],
    )

    assert tasks == [
        Task(
            training_id=5,
            next_epoch=0,
            duration=timedelta(minutes=1),
        ),
        Task(
            training_id=8,
            next_epoch=0,
            duration=timedelta(minutes=1),
        ),
    ]


def test_some_still_new() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=100,
                cum_duration=timedelta(seconds=130),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=5,
                cum_duration=timedelta(),
                next_epoch=0,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=16,
                cum_duration=timedelta(seconds=30),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(),
                next_epoch=0,
                metric_value=METRIC_VALUE,
            ),
        ],
    )

    assert tasks == [
        Task(
            training_id=5,
            next_epoch=0,
            duration=timedelta(minutes=1),
        ),
        Task(
            training_id=16,
            next_epoch=2,
            duration=timedelta(seconds=30),
        ),
        Task(
            training_id=8,
            next_epoch=0,
            duration=timedelta(minutes=1),
        ),
    ]


def test_all_in_round_0() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(seconds=30),
                next_epoch=8,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(seconds=59),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
        ],
    )

    assert tasks == [
        Task(
            training_id=5,
            next_epoch=8,
            duration=timedelta(seconds=30),
        ),
        Task(
            training_id=8,
            next_epoch=2,
            duration=timedelta(seconds=1),
        ),
    ]


def test_all_in_round_0_skip_errors() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(seconds=30),
                next_epoch=8,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(seconds=59),
                next_epoch=2,
                metric_value=METRIC_VALUE,
                error=True,
            ),
        ],
    )

    assert tasks == [
        Task(
            training_id=5,
            next_epoch=8,
            duration=timedelta(seconds=30),
        ),
    ]


def test_all_in_round_1() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(seconds=150),
                next_epoch=8,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=105,
                cum_duration=timedelta(seconds=160),
                next_epoch=18,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=9,
                cum_duration=timedelta(seconds=165),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=3,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=108,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=109,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=103,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
        ],
    )

    assert tasks == [
        Task(
            training_id=5,
            next_epoch=8,
            duration=timedelta(seconds=30),
        ),
        Task(
            training_id=105,
            next_epoch=18,
            duration=timedelta(seconds=20),
        ),
    ]


def test_all_done() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(minutes=4),
                next_epoch=8,
                metric_value=METRIC_VALUE,
            ),
        ]
        * 100,
    )

    assert len(tasks) == 0


def test_some_still_in_round_0() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(seconds=150),
                next_epoch=8,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=105,
                cum_duration=timedelta(seconds=160),
                next_epoch=18,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=9,
                cum_duration=timedelta(seconds=50),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=3,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=108,
                cum_duration=timedelta(seconds=55),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=109,
                cum_duration=timedelta(seconds=10),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=103,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
        ],
    )

    assert tasks == [
        Task(
            training_id=9,
            next_epoch=2,
            duration=timedelta(seconds=10),
        ),
        Task(
            training_id=108,
            next_epoch=2,
            duration=timedelta(seconds=5),
        ),
        Task(
            training_id=109,
            next_epoch=2,
            duration=timedelta(seconds=50),
        ),
    ]


def test_errors_only_in_round_0() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ]
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(seconds=150),
                next_epoch=8,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=105,
                cum_duration=timedelta(seconds=160),
                next_epoch=18,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=9,
                cum_duration=timedelta(seconds=50),
                next_epoch=2,
                metric_value=METRIC_VALUE,
                error=True,
            ),
            _Training(
                training_id=3,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=108,
                cum_duration=timedelta(seconds=55),
                next_epoch=2,
                metric_value=METRIC_VALUE,
                error=True,
            ),
            _Training(
                training_id=109,
                cum_duration=timedelta(seconds=10),
                next_epoch=2,
                metric_value=METRIC_VALUE,
                error=True,
            ),
            _Training(
                training_id=103,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
        ],
    )

    assert tasks == [
        Task(training_id=5, next_epoch=8, duration=timedelta(seconds=30)),
        Task(
            training_id=105,
            next_epoch=18,
            duration=timedelta(seconds=20),
        ),
    ]


def test_different_reduction() -> None:
    tasks = _get_next_tasks(
        Config(
            round_durations=[
                timedelta(minutes=1),
                timedelta(minutes=2),
            ],
            trainings_reduction=2.0,
        ),
        [
            _Training(
                training_id=5,
                cum_duration=timedelta(seconds=150),
                next_epoch=8,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=105,
                cum_duration=timedelta(seconds=160),
                next_epoch=18,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=8,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=9,
                cum_duration=timedelta(seconds=165),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=3,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=108,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=109,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
            _Training(
                training_id=103,
                cum_duration=timedelta(seconds=119),
                next_epoch=2,
                metric_value=METRIC_VALUE,
            ),
        ],
    )

    assert tasks == [
        Task(
            training_id=5,
            next_epoch=8,
            duration=timedelta(seconds=30),
        ),
        Task(
            training_id=105,
            next_epoch=18,
            duration=timedelta(seconds=20),
        ),
        Task(
            training_id=8,
            next_epoch=2,
            duration=timedelta(seconds=61),
        ),
        Task(
            training_id=9,
            next_epoch=2,
            duration=timedelta(seconds=15),
        ),
    ]
