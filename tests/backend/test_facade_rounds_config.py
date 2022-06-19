import re
from datetime import timedelta
from typing import Any

import pytest

from letstune.backend.facade import _rounds_to_config
from letstune.backend.scheduler.epoch import Config


def test_rounds_list() -> None:
    rounds = [
        timedelta(minutes=1),
        timedelta(minutes=4),
        timedelta(minutes=16),
        timedelta(minutes=64),
    ]

    config = _rounds_to_config(rounds)

    assert config == Config(
        round_durations=(
            timedelta(minutes=1),
            timedelta(minutes=4),
            timedelta(minutes=16),
            timedelta(minutes=64),
        ),
        trainings_reduction=4.0,
    )


def test_rounds_dict() -> None:
    rounds = {
        "round_durations": [
            timedelta(minutes=1),
            timedelta(minutes=4),
            timedelta(minutes=16),
            timedelta(minutes=64),
        ]
    }

    config = _rounds_to_config(rounds)

    assert config == Config(
        round_durations=(
            timedelta(minutes=1),
            timedelta(minutes=4),
            timedelta(minutes=16),
            timedelta(minutes=64),
        ),
        trainings_reduction=4.0,
    )


def test_rounds_dict_with_trainings_reduction() -> None:
    rounds = {
        "round_durations": [
            timedelta(minutes=1),
            timedelta(minutes=4),
            timedelta(minutes=16),
            timedelta(minutes=64),
        ],
        "trainings_reduction": 2.5,
    }

    config = _rounds_to_config(rounds)

    assert config == Config(
        round_durations=(
            timedelta(minutes=1),
            timedelta(minutes=4),
            timedelta(minutes=16),
            timedelta(minutes=64),
        ),
        trainings_reduction=2.5,
    )


@pytest.mark.parametrize(
    "rounds,message",
    [
        (
            {},
            "expected rounds to be a dict with 'round_durations', got {}",
        ),
        (
            {"trainings_reduction": 2.5},
            "expected rounds to be a dict with 'round_durations', "
            "got {'trainings_reduction': 2.5}",
        ),
        (
            {"qwerty": 2.5},
            "expected rounds to be a dict with 'round_durations', got {'qwerty': 2.5}",
        ),
    ],
)
def test_missing_round_durations(rounds: dict[str, Any], message: str) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(message),
    ):
        _ = _rounds_to_config(rounds)


def test_unexpected_key_in_dict() -> None:
    rounds = {
        "round_durations": [
            timedelta(minutes=2),
        ],
        "qwerty": 2.5,
    }

    with pytest.raises(
        ValueError,
        match=re.escape("rounds got an unexpected key 'qwerty'"),
    ):
        _ = _rounds_to_config(rounds)


def test_wrong_trainings_reduction_type() -> None:
    rounds = {
        "round_durations": [timedelta(minutes=5)],
        "trainings_reduction": ["hello"],
    }

    with pytest.raises(
        TypeError,
        match=re.escape(
            "expected rounds to be a dict where "
            "rounds['trainings_reduction'] is a float",
        ),
    ):
        _ = _rounds_to_config(rounds)


def test_dict_wrong_round_durations_content_type() -> None:
    rounds = {
        "round_durations": ["hello"],
        "trainings_reduction": 2.5,
    }

    with pytest.raises(
        TypeError,
        match=re.escape(
            "expected rounds to be a dict where "
            "rounds['round_durations'] is a sequence of timedelta",
        ),
    ):
        _ = _rounds_to_config(rounds)


def test_list_wrong_round_durations_content_type() -> None:
    rounds = ["hello"]

    with pytest.raises(
        TypeError,
        match=re.escape(
            "expected rounds to be a sequence of timedelta",
        ),
    ):
        _ = _rounds_to_config(rounds)  # type: ignore


def test_dict_wrong_round_durations_type() -> None:
    rounds = {
        "round_durations": 345,
        "trainings_reduction": 2.5,
    }

    with pytest.raises(
        TypeError,
        match=re.escape(
            "expected rounds to be a dict where "
            "rounds['round_durations'] is a sequence of timedelta",
        ),
    ):
        _ = _rounds_to_config(rounds)


def test_list_wrong_round_durations_type() -> None:
    rounds = 345

    with pytest.raises(
        TypeError,
        match=re.escape(
            "expected rounds to be a sequence of timedelta",
        ),
    ):
        _ = _rounds_to_config(rounds)  # type: ignore
