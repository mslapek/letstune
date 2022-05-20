#!/usr/bin/env dash
set -ex

if [ -z "$TEST_TENSORFLOW" ]; then
  export TEST_TENSORFLOW=0
fi

if [ -z "$TEST_XGBOOST" ]; then
  export TEST_XGBOOST=0
fi

isort --check letstune tests examples
black --check letstune tests examples
flake8 letstune tests examples

## mypy ##

mypy_args="letstune tests examples/sklearn"

if [ "$TEST_TENSORFLOW" -ne 0 ]; then
  mypy_args="${mypy_args} examples/keras"
fi

if [ "$TEST_XGBOOST" -ne 0 ]; then
  mypy_args="${mypy_args} examples/xgboost"
fi

mypy $mypy_args

## end mypy ##

## pytest ##

pytest_args="--doctest-modules --ignore=examples"

if [ "$TEST_TENSORFLOW" -eq 0 ]; then
  pytest_args="${pytest_args} --ignore=tests/test_keras.py --ignore=letstune/keras.py"
fi

pytest $pytest_args

## end pytest ##

echo "Lint OK!"
