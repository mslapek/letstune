import re
from typing import Any

import numpy as np
import pytest

from letstune.backend.runner._base import normalize_metric_values


def is_normalized(metric_values: Any) -> bool:
    if type(metric_values) is not dict:
        return False

    for k, v in metric_values.items():
        if not (type(k) == str and type(v) == float):
            return False

    return True


def test_empty_dict() -> None:
    d: dict[Any, Any] = dict()

    with pytest.raises(ValueError, match="got empty dict instead of metric values"):
        _ = normalize_metric_values(d)


def test_dict_with_floats() -> None:
    metric_values = {
        "qwerty": 3.5,
        "foo": -633.3,
    }

    normalized_metric_values = normalize_metric_values(metric_values)

    assert is_normalized(normalized_metric_values)
    assert normalized_metric_values == {
        "qwerty": 3.5,
        "foo": -633.3,
    }


def test_dict_with_ints() -> None:
    metric_values = {
        "qwerty": 3.5,
        "foo": -101,
    }

    normalized_metric_values = normalize_metric_values(metric_values)

    assert is_normalized(normalized_metric_values)
    assert normalized_metric_values == {
        "qwerty": 3.5,
        "foo": -101.0,
    }


def test_dict_with_numpy_floats() -> None:
    metric_values = {
        "qwerty": 3.5,
        "foo": np.float32(3.8),
        "bar": np.float64(-8.8),
    }

    normalized_metric_values = normalize_metric_values(metric_values)

    assert is_normalized(normalized_metric_values)
    assert normalized_metric_values == {
        "qwerty": 3.5,
        "foo": float(np.float32(3.8)),
        "bar": -8.8,
    }


def test_invalid_type() -> None:
    metric_values = [6.6, 3.3]

    with pytest.raises(
        TypeError,
        match=re.escape(
            "expected metric_values to be a dict of floats, got [6.6, 3.3]"
        ),
    ):
        _ = normalize_metric_values(metric_values)


def test_invalid_key_type() -> None:
    metric_values = {80: 8.8}

    with pytest.raises(
        TypeError,
        match=re.escape("expected metric_values to be a dict of floats, got {80: 8.8}"),
    ):
        _ = normalize_metric_values(metric_values)


def test_invalid_value_type() -> None:
    metric_values = {"foo": "bar"}

    with pytest.raises(
        TypeError,
        match=re.escape(
            "expected metric_values to be a dict of floats, got {'foo': 'bar'}"
        ),
    ):
        _ = normalize_metric_values(metric_values)
